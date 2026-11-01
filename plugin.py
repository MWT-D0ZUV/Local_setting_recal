"""Local setting recall: SDK-only, read-only memory access, no generation call."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .core import Query, choose_queries, inject, knowledge_text, message_text
from .local_documents import DocumentIndex
from .batching import BatchStore
from .manual_documents import ManualDocuments
from .vector_search import VectorSearch
from .management_ui import ManagementMixin

LOG = logging.getLogger('Local.setting_recall')


def field(default: Any, label: str, description: str = '', **kw: Any) -> Any:
    return Field(default=default, description=description, json_schema_extra={'label': label}, **kw)


class PluginSection(PluginConfigBase):
    __ui_label__ = '启用与会话'
    enabled: bool = field(True, '启用自动记忆召回')
    config_version: str = field('1.1.2', '配置版本')
    session_allowlist: list[str] = field([], '允许会话 ID', '空列表为所有会话；填写宿主 session_id，不是 QQ 群号。')


class Rule(PluginConfigBase):
    name: str = field('新规则', '规则名称', max_length=60)
    enabled: bool = field(True, '启用此规则')
    keywords: list[str] = field([], '触发关键词', '任意一个命中即可；英文/世界代码匹配完整 ASCII 词，中文按包含匹配。')
    query: str = field('', '检索内容', '描述要查的主题和角色，勿仅填写文件名。命中的关键词会自动附在查询末尾。', max_length=500)


def default_rules() -> list[Rule]:
    return [
        Rule(name='世界版本与人物关系', keywords=['C8', 'F1', 'C5', 'A11', 'L1', 'T0', '古明地觉', '古明地恋', '觉姐姐', '恋恋', '归言录', '平行世界'],
             query='古明地念 跨世界古明地觉姐妹观 东方亲友档案 古明地恋 不同世界 关系边界'),
        Rule(name='创生故事与心相后室', keywords=['后室', '心相迷宫', '心相后室', '创生', '永夏海滩', '常夏海滩', '红字', '第五层'],
             query='古明地念 创生故事 心相迷宫 历程 旅人 古明地恋'),
        Rule(name='形象服饰与能力边界', keywords=['帽檐', '帽饰', '心之瞳', '觉之瞳', '第三只眼', '触手', '读心', '服饰', '穿什么', '长什么样'],
             query='古明地念 完整形象服饰 身体能力 边界 双手 管线触手 帽饰'),
        Rule(name='旅者共同体与跨世界散步', keywords=['MWT', '旅者', '心相书房', '共同体', '跨世界', '箱庭'],
             query='古明地念 MWT旅者 精神共同体 心相书房 跨世界散步 机制'),
        Rule(name='地灵殿与幻想乡日常', keywords=['地灵殿', '幻想乡', '阿燐', '阿空', '火焰猫燐', '灵乌路空', '秦心', '魔理沙', '串门'],
             query='古明地念 地灵殿 幻想乡 串门 日常纪事 亲友互动'),
    ]


class TriggerSection(PluginConfigBase):
    __ui_label__ = '通用触发与自定义规则'
    generic_enabled: bool = field(True, '启用通用设定触发', '未命中自定义规则时，用当前用户消息检索新主题。')
    generic_keywords: list[str] = field(['设定', '档案', '世界观', '背景故事', '人设', '角色关系', '你还记得', '查一下记忆', '查询记忆'], '通用触发词')
    query_prefix: str = field('古明地念 设定资料', '通用查询前缀', max_length=120)
    rules: list[Rule] = Field(default_factory=default_rules, json_schema_extra={'label': '主题规则（可新增、删除、停用）'})


class StyleSection(PluginConfigBase):
    __ui_label__ = '发言风格参考'
    enabled: bool = field(True, '启用风格参考', '只检索已加载的 Q 开头 MD/TXT 语录；F 开头适合事实，N 开头适合经历。文件命名说明见 README。')
    on_every_reply: bool = field(True, '为批次内消息检索风格', '每条消息最多执行一次检索流程；风格查询附带当前消息情境。Planner/Replyer 复用同一结果。')
    keywords: list[str] = field(['说话风格', '发言风格', '语气', '措辞', '语录', '风格参考', '像你平时那样', '口吻'], '显式风格触发词')
    query: str = field('古明地念 日常闲谈 互动记忆语录 简短自然表述 说话语气', '风格检索内容', '更新语录文档后可以修改此查询；空字符串停止风格查询。', max_length=500)


class RetrievalSection(PluginConfigBase):
    __ui_label__ = '检索预算与缓存'
    max_fact_queries: int = field(1, '每条消息最多主题查询数', '同一消息的多个触发只取优先主题，检索结果在两个阶段复用。', ge=1, le=1)
    result_limit: int = field(4, '每个查询最多结果数', ge=1, le=8)
    max_query_chars: int = field(220, '查询最大字符数', ge=80, le=500)
    max_reference_chars: int = field(2200, '每阶段注入最大字符数', '字符数不是 token 数；总参考文本上限。', ge=600, le=5000)
    query_timeout_seconds: float = field(3.0, '单次查询超时秒数', ge=0.1, le=4.5)
    total_budget_seconds: float = field(5.0, '阶段总预算秒数', '包含消息读取和等待查询槽位，超时继续原回复。', ge=0.2, le=6.0)
    cache_ttl_seconds: float = field(300.0, '命中缓存秒数', '仅宿主回退查询缓存；手动加载文档立即失效。消息去重记录独立持久保存。', ge=0, le=3600)
    recent_reference_max_age_seconds: float = field(300.0, '跨消息参考复用有效期', '只复用同会话、同版本、原消息主题相近的结果；0 关闭跨消息复用，不影响同一消息的结果。', ge=0, le=3600)
    empty_cache_seconds: float = field(20.0, '无结果或失败冷却秒数', ge=0, le=120)
    max_message_age_seconds: float = field(86400.0, '最多补读近期消息秒数', '仅在模型请求前收集宿主已存消息，不扫描更早聊天。', ge=60, le=604800)
    log_enabled: bool = field(True, '记录召回计数日志', '只记录阶段、查询数、命中数和字符数，不记录会话、消息或记忆正文。')


class DocumentSection(PluginConfigBase):
    __ui_label__ = '手动加载参考文档'
    enabled: bool = field(True, '检索已加载的文档', '在插件文档管理页粘贴绝对目录、浏览勾选并点击加载；不定时扫描。F=事实，N=经历，Q=语录。')
    directory: str = field('', '默认参考目录', '可粘贴本地绝对路径；保存路径不会读取文件，之后到管理页点击浏览和加载。留空使用私有 documents 目录。', max_length=1000)
    excluded_patterns: list[str] = field(['00*'], '不显示和读取的文件名模式', '默认排除 00 开头的指南，打开文件前过滤。')
    fallback_to_host: bool = field(False, '本地无结果时查询宿主记忆', '可能取回共享库中的旧文档；默认关闭。宿主查询保持会话范围。')
    auto_keywords: bool = field(True, '识别已加载文档的标题与关键词', '仅对手动加载的快照生效。建议文档写“关键词：词一,词二”。')
    max_documents: int = field(40, '最多加载文档数', ge=1, le=100)
    max_file_bytes: int = field(131072, '单文档字节上限', ge=1024, le=1048576)
    max_total_bytes: int = field(1048576, '全部文档字节上限', ge=1024, le=4194304)


class BatchSection(PluginConfigBase):
    __ui_label__ = '按消息批次检索'
    size: int = field(20, '每批真人文本消息数', '不足一批不检索；设置 1 为逐条。只在 Planner/Replyer 请求前收集已存消息。', ge=1, le=100)
    use_host_threshold: bool = field(False, '跟随宿主新增消息门槛', '读取 a_memorix.integration.heuristic_memory_recall_min_new_messages；不是上下文窗口大小。读取失败使用上面的数值。')
    job_timeout_seconds: float = field(120, '整批后台处理超时秒数', '后台逐条处理，不阻塞模型请求；失败或中断也计为已尝试，不自动重复请求。', ge=5, le=600)


class EmbeddingSection(PluginConfigBase):
    __ui_label__ = '语义向量检索'
    enabled: bool = field(True, '使用 embedding 检索', '先在文档管理页点击构建向量；未构建或失效时使用关键词。会将已选文档片段及查询发送到宿主配置的 embedding 服务。')
    task_name: str = field('embedding', '宿主 embedding 任务名称', '复用宿主任务及密钥；这是任务别名，不是具体模型名。', max_length=100)
    min_similarity: float = field(0.4, '最低余弦相似度', '不是正确率；没有通用适用阈值，应结合自己的资料调整。', ge=0, le=1)
    relative_margin: float = field(0.08, '候选与最佳结果的最大相似度差', '在最低相似度之外，再排除明显弱于最佳候选的结果；不是正确率。', ge=0, le=1)
    batch_size: int = field(8, '手动建库每批片段数', ge=1, le=32)
    timeout_seconds: float = field(20, 'embedding 调用超时秒数', ge=1, le=60)


class Config(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    documents: DocumentSection = Field(default_factory=DocumentSection)
    triggers: TriggerSection = Field(default_factory=TriggerSection)
    style: StyleSection = Field(default_factory=StyleSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    batch: BatchSection = Field(default_factory=BatchSection)
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)


class RecallEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
        self.locks: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}
        self.slots = asyncio.Semaphore(2)
        self.documents = DocumentIndex()
        self.vectors = None

    async def search(self, ctx: Any, sid: str, query: Query, deadline: float) -> str:
        if self.cfg.documents.enabled:
            if self.vectors is not None and self.cfg.embedding.enabled:
                try:
                    semantic = await self.vectors.search(ctx, self.documents, query, self.cfg.retrieval.result_limit)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    semantic = ''
                if semantic:
                    return semantic
            local = self.documents.search(query.context if query.style and query.context else query.text,
                                          min(self.cfg.retrieval.result_limit, 1 if query.style else 2),
                                          document_name=query.document_name, query_style=query.style)
            if local or not self.cfg.documents.fallback_to_host:
                return local
        key = (sid, query.text)
        def cached() -> str | None:
            entry = self.cache.get(key)
            if entry and entry[0] > time.monotonic():
                self.cache.move_to_end(key)
                return entry[1]
            if entry:
                del self.cache[key]
            return None
        value = cached()
        if value is not None:
            return value
        lock, refs = self.locks.get(key, (asyncio.Lock(), 0))
        self.locks[key] = (lock, refs + 1)
        async def run() -> str:
            async with lock:
                hit = cached()
                if hit is not None:
                    return hit
                async with self.slots:
                    remaining = min(self.cfg.retrieval.query_timeout_seconds, deadline - time.monotonic())
                    if remaining <= 0:
                        return ''
                    try:
                        payload = await asyncio.wait_for(ctx.call_capability(
                            'knowledge.search', timeout_ms=max(1, int(remaining * 1000)),
                            query=query.text, limit=self.cfg.retrieval.result_limit,
                            mode='search', chat_id=sid, respect_filter=True,
                        ), timeout=remaining)
                        result = knowledge_text(payload)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # No exception text: providers may embed private query/result text.
                        result = ''
                    ttl = self.cfg.retrieval.cache_ttl_seconds if result else self.cfg.retrieval.empty_cache_seconds
                    if ttl > 0:
                        self.cache[key] = (time.monotonic() + ttl, result)
                        self.cache.move_to_end(key)
                        while len(self.cache) > 256:
                            self.cache.popitem(last=False)
                    return result
        try:
            return await asyncio.wait_for(run(), timeout=max(0.001, deadline - time.monotonic()))
        except TimeoutError:
            return ''
        finally:
            current = self.locks.get(key)
            if current and current[0] is lock:
                if current[1] <= 1:
                    del self.locks[key]
                else:
                    self.locks[key] = (lock, current[1] - 1)


class SettingRecallPlugin(ManagementMixin, MaiBotPlugin):
    config_model = Config
    config_reload_subscriptions = ('bot', 'model')

    def __init__(self) -> None:
        super().__init__()
        self._engine = None
        self.manual = None
        self._store = None
        self._batch_tasks = {}
        self._session_locks = OrderedDict()
        self._job_slots = asyncio.Semaphore(2)
        self._threshold = 20
        self.init_management()

    def set_plugin_config(self, config_data: Any) -> None:
        super().set_plugin_config(config_data)
        self._engine = RecallEngine(self.config)
        if self.manual is not None:
            self._engine.documents = self.manual.restore(self.config.documents)
            self._engine.vectors = VectorSearch(Path(self.ctx.paths.data_dir), self.config.embedding)

    async def _update_threshold(self):
        value = self.config.batch.size
        if self.config.batch.use_host_threshold:
            try:
                result = await self.ctx.config.get('a_memorix.integration.heuristic_memory_recall_min_new_messages', value)
                if type(result) in (int, float) and 1 <= result <= 100:
                    value = int(result)
            except Exception:
                pass
        self._threshold = value

    async def on_load(self) -> None:
        root = Path(self.ctx.paths.data_dir)
        self.manual = ManualDocuments(root)
        self._store = BatchStore(root / 'recall-batches.sqlite3')
        self._engine = RecallEngine(self.config)
        self._engine.documents = self.manual.restore(self.config.documents)
        self._engine.vectors = VectorSearch(root, self.config.embedding)
        await self._update_threshold()
        LOG.info('Local.setting_recall v1.1.2 loaded: manual documents, batch=%d', self._threshold)

    async def on_unload(self) -> None:
        await self.shutdown_management()
        tasks = list(self._batch_tasks.values())
        self._engine = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._batch_tasks.clear()

    async def on_config_update(self, scope: str, config_data: Any, version: str) -> None:
        if scope == 'self':
            self.set_plugin_config(config_data)
        elif scope == 'model' and self._engine and self._engine.vectors:
            self._engine.vectors.invalidate()
        await self._update_threshold()

    async def reload_documents(self, names: list[str]):
        engine = self._engine
        if engine is None:
            return
        directory = getattr(self, '_browse_directory', '') or self.manual.directory or self.config.documents.directory
        staged = await asyncio.to_thread(self.manual.load, directory, names, self.config.documents)
        if self._engine is engine:
            engine.documents = staged
            engine.cache.clear()
            if engine.vectors:
                engine.vectors.invalidate()
        return {'documents': len(getattr(staged, 'names', names)), 'chunks': len(staged.chunks)}

    async def build_vectors(self):
        engine = self._engine
        if engine is None or not engine.cfg.embedding.enabled:
            raise ValueError('请先启用 embedding')
        return await engine.vectors.build(self.ctx, engine.documents)

    def _version(self, engine):
        import hashlib
        return hashlib.sha256(('relevance-v2:' + repr(engine.documents.signature) + engine.cfg.model_dump_json()).encode()).hexdigest()

    async def _process_batch(self, engine, sid, batch, rows):
        per_message = {}
        try:
            async with self._job_slots:
                async with asyncio.timeout(engine.cfg.batch.job_timeout_seconds):
                    for mid, text in rows:
                        if self._engine is not engine:
                            break
                        queries = choose_queries(text, engine.cfg, replyer=True)
                        facts = [q for q in queries if not q.style]
                        styles = [q for q in queries if q.style]
                        if engine.cfg.documents.enabled and engine.cfg.documents.auto_keywords:
                            auto = engine.documents.trigger_queries(text, limit=1)
                            facts = (auto + facts)[:1]
                        # Facts require an explicit rule, generic trigger or document alias.
                        hits = []
                        for query in facts[:1] + styles[:1]:
                            deadline = time.monotonic() + engine.cfg.embedding.timeout_seconds + engine.cfg.retrieval.query_timeout_seconds
                            body = await engine.search(self.ctx, sid, query, deadline)
                            if body:
                                hits.append((query, body))
                        per_message[mid] = hits
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.warning('Local.setting_recall batch stopped; attempted messages will not be retried')
        finally:
            # Documents/config changed midway: never publish old results for new version.
            if self._store:
                self._store.finish(batch, [], per_message if self._engine is engine else {})
                if engine.cfg.retrieval.log_enabled:
                    LOG.info('Local.setting_recall batch finished attempted=%d matched=%d published=%s',
                             len(per_message), sum(bool(hits) for hits in per_message.values()), self._engine is engine)

    async def _collect(self, sid, engine):
        cfg = engine.cfg
        cursor = self._store.cursor(sid)
        if cursor is None:
            payload = await self.ctx.call_capability('message.get_recent', timeout_ms=1800,
                chat_id=sid, limit=self._threshold * 2, filter_mai=True,
                hours=cfg.retrieval.max_message_age_seconds / 3600)
        else:
            payload = await self.ctx.call_capability('message.get_by_time_in_chat', timeout_ms=1800,
                chat_id=sid, start_time=max(cursor - 180, time.time() - cfg.retrieval.max_message_age_seconds),
                end_time=time.time() + 0.001, limit=2000, limit_mode='earliest', filter_mai=True, filter_command=True)
        messages = payload.get('messages') if isinstance(payload, dict) and payload.get('success') is True else payload
        if not isinstance(messages, list):
            return []
        before = self._store.pending(sid)
        pending = self._store.observe(sid, messages, max_age=cfg.retrieval.max_message_age_seconds, initial_limit=self._threshold)
        if cfg.retrieval.log_enabled and pending != before:
            LOG.info('Local.setting_recall collected pending=%d threshold=%d', pending, self._threshold)
        return messages

    async def _handle(self, kwargs: dict[str, Any], *, replyer: bool) -> dict[str, Any]:
        engine = self._engine
        if engine is None or self._store is None or not engine.cfg.plugin.enabled:
            return {'action': 'continue'}
        cfg = engine.cfg
        sid = kwargs.get('session_id')
        if (not isinstance(sid, str) or not sid.strip()
                or (cfg.plugin.session_allowlist and sid not in cfg.plugin.session_allowlist)
                or type(kwargs.get('item_schema_version')) is not int or kwargs['item_schema_version'] != 1
                or not isinstance(kwargs.get('items'), list)):
            return {'action': 'continue'}
        try:
            lock = self._session_locks.setdefault(sid, asyncio.Lock())
            self._session_locks.move_to_end(sid)
            # Only prune unlocked entries with no queued waiters.
            while len(self._session_locks) > 256:
                key, old = next(iter(self._session_locks.items()))
                if old.locked() or getattr(old, '_waiters', None):
                    break
                self._session_locks.pop(key)
            async with asyncio.timeout(cfg.retrieval.total_budget_seconds):
                async with lock:
                    messages = await self._collect(sid, engine)
                    current = self._batch_tasks.get(sid)
                    if (current is None or current.done()) and len([t for t in self._batch_tasks.values() if not t.done()]) < 32:
                        batch, rows = self._store.claim(sid, self._threshold, self._version(engine))
                        if rows:
                            if cfg.retrieval.log_enabled:
                                LOG.info('Local.setting_recall batch started messages=%d', len(rows))
                            task = asyncio.create_task(self._process_batch(engine, sid, batch, rows))
                            self._batch_tasks[sid] = task
                    for key, task in list(self._batch_tasks.items()):
                        if task.done():
                            self._batch_tasks.pop(key, None)
                    mid = kwargs.get('reply_message_id') if replyer else None
                    if not replyer:
                        valid = [m for m in messages if message_text(m, sid, max_age=cfg.retrieval.max_message_age_seconds)]
                        if valid:
                            mid = valid[-1]['message_id']
                        elif self._store.cursor(sid) is not None:
                            mid = self._latest_id(sid)
                    blocks = self._store.result_for(sid, str(mid), self._version(engine)) if mid else []
                    if not blocks and mid:
                        current_text = next((message_text(m, sid, max_age=cfg.retrieval.max_message_age_seconds)
                            for m in messages if isinstance(m, dict) and str(m.get('message_id')) == str(mid)), '')
                        if current_text:
                            blocks = self._store.recent_reference(sid, self._version(engine), current_text,
                                max_age=cfg.retrieval.recent_reference_max_age_seconds)
                    if blocks and self._engine is engine and cfg.retrieval.log_enabled:
                        LOG.info('Local.setting_recall reference reused stage=%s blocks=%d', 'replyer' if replyer else 'planner', len(blocks))
                    return inject(kwargs, blocks, cfg.retrieval.max_reference_chars) if self._engine is engine else {'action':'continue'}
        except asyncio.CancelledError:
            raise
        except Exception:
            if cfg.retrieval.log_enabled:
                LOG.warning('Local.setting_recall request collection failed; continuing original request')
            return {'action':'continue'}

    def _latest_id(self, sid):
        with self._store.connect() as db:
            row = db.execute('SELECT mid FROM messages WHERE sid=? ORDER BY stamp DESC,mid DESC LIMIT 1',(sid,)).fetchone()
        return row[0] if row else None

    @HookHandler('maisaka.planner.before_request', name='setting_recall_planner',
                 description='收集真人文本批次并复用已完成的检索', mode=HookMode.BLOCKING,
                 order=HookOrder.EARLY, timeout_ms=7500, error_policy=ErrorPolicy.SKIP)
    async def planner_recall(self, **kwargs: Any) -> dict[str, Any]:
        return await self._handle(kwargs, replyer=False)

    @HookHandler('maisaka.replyer.before_model_request', name='setting_recall_replyer',
                 description='复用回复对象所属批次的检索资料', mode=HookMode.BLOCKING,
                 order=HookOrder.EARLY, timeout_ms=7500, error_policy=ErrorPolicy.SKIP)
    async def replyer_recall(self, **kwargs: Any) -> dict[str, Any]:
        return await self._handle(kwargs, replyer=True)


def create_plugin() -> SettingRecallPlugin:
    return SettingRecallPlugin()
