"""Offline behavioral tests; real SDK imports and synthetic mocked capabilities only."""
import asyncio
import copy
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.core import MARKER, Query, choose_queries, inject, keyword_matches, knowledge_text, message_text
from Local_setting_recall.plugin import Config, RecallEngine, Rule, SettingRecallPlugin
HIT = '你知道这些知识: 1. Synthetic C8 has a blue gate.'
CONTINUE = {'action': 'continue'}


def config():
    cfg = Config()
    cfg.style.enabled = False
    cfg.retrieval.log_enabled = False
    cfg.triggers.generic_enabled = False
    cfg.triggers.rules = [Rule(name='Synthetic world', keywords=['C8'], query='Synthetic world facts')]
    if hasattr(cfg, 'documents'):
        cfg.documents.enabled = False
    return cfg


def message(text='C8', sid='session-a', mid='message-a', **extra):
    return {'session_id': sid, 'message_id': mid, 'timestamp': time.time(), 'processed_plain_text': text,
            'raw_message': [{'type': 'text', 'data': text}], **extra}


def request(**extra):
    return {'session_id': 'session-a', 'item_schema_version': 1,
            'items': [{'item_type': 'SystemMessageItem', 'parts': [{'type': 'text', 'text': 'system'}], 'meta': {'keep': True}},
                      {'item_type': 'UserMessageItem', 'parts': [{'type': 'text', 'text': 'current'}], 'meta': {'original': True}}],
            'model': 'synthetic', 'temperature': 0.25, 'provider_options': {'keep': ['all']}, **extra}


def inserted(result):
    return [i for i in result.get('modified_kwargs', {}).get('items', [])
            if isinstance(i, dict) and i.get('meta', {}).get('setting_recall_marker') == MARKER]


def plugin_with(callback, cfg=None):
    plugin = SettingRecallPlugin()
    plugin.set_plugin_config((cfg or config()).model_dump(mode='python'))
    mock = AsyncMock(side_effect=callback)
    plugin._set_context(type('FakeContext', (), {'call_capability': mock})())
    return plugin, mock


class TriggerTests(unittest.TestCase):
    def test_ascii_boundaries(self):
        for text in ('C8', 'c8', 'see C8!', '世界C8关系'):
            self.assertTrue(keyword_matches(text, 'C8'), text)
        for text in ('C80', 'sc8x', 'C8_test', 'prefixC8', 'AC8', 'c80'):
            self.assertFalse(keyword_matches(text, 'C8'), text)

    def test_new_disabled_empty_query_rules(self):
        cfg = config()
        cfg.triggers.rules = [Rule(name='disabled', keywords=['C8'], query='disabled', enabled=False),
                              Rule(name='empty', keywords=['C8'], query=' '),
                              Rule(name='new', keywords=['newtopic'], query='new details')]
        self.assertEqual(choose_queries('C8', cfg, replyer=False), [])
        queries = choose_queries('tell me newtopic', cfg, replyer=False)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].label, 'new')
        self.assertIn('new details', queries[0].text)
        self.assertIn('newtopic', queries[0].text)

    def test_generic_current_text_and_switch(self):
        cfg = config()
        cfg.triggers.generic_enabled = True
        text = '查询记忆 一个新角色的档案'
        self.assertIn(text, choose_queries(text, cfg, replyer=False)[0].text)
        cfg.triggers.generic_enabled = False
        self.assertEqual(choose_queries(text, cfg, replyer=False), [])

    def test_specific_rule_precedes_generic(self):
        cfg = config()
        cfg.triggers.generic_enabled = True
        self.assertEqual(len(choose_queries('查询记忆 C8 的档案', cfg, replyer=False)), 1)

    def test_style_every_reply_can_be_disabled(self):
        cfg = config()
        cfg.style.enabled = True
        cfg.style.on_every_reply = True
        self.assertEqual(choose_queries('ordinary', cfg, replyer=False), [])
        self.assertTrue(choose_queries('ordinary', cfg, replyer=True)[0].style)
        cfg.style.on_every_reply = False
        self.assertEqual(choose_queries('ordinary', cfg, replyer=True), [])
        self.assertTrue(choose_queries('请参考说话风格', cfg, replyer=False)[0].style)
        cfg.style.query = ''
        self.assertEqual(choose_queries('请参考说话风格', cfg, replyer=True), [])

    def test_session_freshness_notifications(self):
        now = time.time()
        valid = message(timestamp=now)
        self.assertEqual(message_text(valid, 'session-a', now=now), 'C8')
        for change in ({'session_id': 'b'}, {'message_id': ''}, {'timestamp': now-181}, {'timestamp': now+11},
                       {'timestamp': 'bad'}, {'is_notify': True}, {'is_command': True}):
            with self.subTest(change=change):
                self.assertEqual(message_text({**valid, **change}, 'session-a', now=now), '')

    def test_nontext_segment_not_used_as_keyword(self):
        msg = message('', raw_message={'type': 'image', 'data': {'url': 'C8'}})
        self.assertEqual(message_text(msg, 'session-a'), '')
        msg['raw_message'] = {'type': 'seglist', 'data': [{'type': 'image', 'data': 'C8'}, {'type': 'text', 'data': 'hello'}]}
        self.assertEqual(message_text(msg, 'session-a').strip(), 'hello')

    def test_host_success_text(self):
        self.assertIn('Synthetic C8', knowledge_text(HIT))
        self.assertIn('Synthetic C8', knowledge_text({'success': True, 'content': HIT}))

    def test_literal_newline_conversion(self):
        self.assertEqual(knowledge_text('你知道这些知识: 1. first\\n2. second'), '1. first\n2. second')

    def test_empty_failure_unrecognized_skipped(self):
        for payload in (None, {}, ['entry'], 'unrelated text', '你不太了解有关C8的知识', '你知道这些知识: ',
                        '你知道这些知识: 1. ', {'success': False, 'content': HIT}, {'success': True, 'content': 'unrelated'}):
            with self.subTest(payload=payload):
                self.assertEqual(knowledge_text(payload), '')

    def test_injection_preserves_kwargs_meta_nontext_tool_adjacency(self):
        kwargs = request()
        kwargs['items'].extend([{'item_type': 'AssistantMessageItem', 'parts': [{'type': 'tool_call', 'id': 'call-a'}], 'meta': {'provider': 'opaque'}},
                                {'item_type': 'ToolResultItem', 'parts': [{'type': 'image', 'data': {'blob': 'opaque'}}], 'meta': {'tool_call_id': 'call-a'}}])
        before = copy.deepcopy(kwargs)
        result = inject(kwargs, [(Query('facts', 'q'), '1. fact')], 1200)
        changed = result['modified_kwargs']
        self.assertEqual(kwargs, before)
        self.assertEqual({k:v for k,v in changed.items() if k!='items'}, {k:v for k,v in before.items() if k!='items'})
        self.assertEqual(changed['items'][0], before['items'][0])
        self.assertEqual(changed['items'][2:], before['items'][1:])
        self.assertEqual(len(inserted(result)), 1)

    def test_only_plugin_marker_is_replaced(self):
        kwargs = request()
        own = {'item_type': 'UserMessageItem', 'parts': [], 'meta': {'setting_recall_marker': MARKER}}
        foreign = {'item_type': 'UserMessageItem', 'parts': [], 'meta': {'setting_recall_marker': 'another.plugin'}}
        kwargs['items'].extend([own, foreign])
        result = inject(kwargs, [(Query('facts', 'q'), '1. fact')], 1000)
        self.assertEqual(len(inserted(result)), 1)
        self.assertIn(foreign, result['modified_kwargs']['items'])
        cleaned = inject(kwargs, [], 1000)['modified_kwargs']['items']
        self.assertNotIn(own, cleaned)
        self.assertIn(foreign, cleaned)

    def test_item_id_survives_marker_removal_for_replace_and_delete(self):
        import uuid
        first = inject(request(), [(Query('facts', 'q'), 'old-reference-body')], 1000)
        old = inserted(first)[0]
        item_id = old['meta']['item_id']
        self.assertTrue(item_id.startswith(MARKER + ':'))
        uuid.UUID(item_id[len(MARKER) + 1:])
        del old['meta']['setting_recall_marker']
        foreign = {'item_type': 'UserMessageItem', 'parts': [],
                   'meta': {'item_id': MARKER + '_external:foreign'}}
        kwargs = first['modified_kwargs']
        kwargs['items'].append(foreign)
        second = inject(kwargs, [(Query('facts', 'q'), 'new-reference-body')], 1000)
        items = second['modified_kwargs']['items']
        self.assertNotIn(old, items)
        self.assertIn(foreign, items)
        self.assertEqual(len(inserted(second)), 1)
        self.assertNotEqual(inserted(second)[0]['meta']['item_id'], item_id)
        self.assertIn('new-reference-body', inserted(second)[0]['parts'][0]['text'])
        cleaned = inject(kwargs, [], 1000)['modified_kwargs']['items']
        self.assertEqual(cleaned, request()['items'] + [foreign])
        self.assertNotIn('setting_recall_marker', old['meta'])

    def test_invalid_schema_untouched(self):
        for change in ({'item_schema_version': True}, {'item_schema_version': 2}, {'items': 'bad'}, {'item_schema_version': None}):
            self.assertEqual(inject(request(**change), [(Query('f','q'), '1. fact')], 1000), CONTINUE)

    def test_query_reference_limits_and_style_budget(self):
        cfg = config()
        cfg.retrieval.max_query_chars = 80
        cfg.triggers.rules[0].query = 'Q'*200
        self.assertLessEqual(len(choose_queries('C8', cfg, replyer=False)[0].text), 80)
        blocks = [(Query('fact','q'), 'F'*10000), (Query('style','s',True), 'S'*10000)]
        text = inserted(inject(request(), blocks, 600))[0]['parts'][0]['text']
        self.assertLessEqual(len(text), 600)
        self.assertIn('S', text)
        self.assertIn('F', text)


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_session_query_isolation(self):
        engine = RecallEngine(config())
        ctx = type('Fake', (), {'call_capability': AsyncMock(return_value=HIT)})()
        for sid,q in [('a','one'),('a','one'),('b','one'),('a','two')]:
            self.assertTrue(await engine.search(ctx, sid, Query('f',q), time.monotonic()+3))
        self.assertEqual(ctx.call_capability.await_count, 3)

    async def test_same_key_concurrency_singleflight(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def reply(*args, **kwargs):
            started.set()
            await release.wait()
            return HIT
        engine = RecallEngine(config())
        ctx = type('Fake', (), {'call_capability': AsyncMock(side_effect=reply)})()
        tasks = [asyncio.create_task(engine.search(ctx,'a',Query('f','q'),time.monotonic()+3)) for _ in range(2)]
        try:
            await asyncio.wait_for(started.wait(),1)
            release.set()
            self.assertTrue(all(await asyncio.gather(*tasks)))
        finally:
            release.set()
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(ctx.call_capability.await_count,1)

    async def test_empty_error_cooldown(self):
        for payload in ('你不太了解有关主题的知识', RuntimeError('synthetic')):
            engine = RecallEngine(config())
            mock = AsyncMock(side_effect=payload) if isinstance(payload,Exception) else AsyncMock(return_value=payload)
            ctx = type('Fake', (), {'call_capability': mock})()
            for _ in range(2):
                self.assertEqual(await engine.search(ctx,'a',Query('f','q'),time.monotonic()+1),'')
            self.assertEqual(mock.await_count,1)

    async def test_timeout_fails_open_and_cancels(self):
        cfg = config()
        cfg.retrieval.query_timeout_seconds = 0.1
        cancelled = asyncio.Event()
        async def blocked(*args, **kwargs):
            try: await asyncio.Event().wait()
            finally: cancelled.set()
        ctx = type('Fake', (), {'call_capability': AsyncMock(side_effect=blocked)})()
        engine = RecallEngine(cfg)
        self.assertEqual(await asyncio.wait_for(engine.search(ctx,'a',Query('f','q'),time.monotonic()+1),0.8),'')
        self.assertTrue(cancelled.is_set())
        self.assertEqual(await engine.search(ctx,'a',Query('f','q'),time.monotonic()+1),'')
        self.assertEqual(ctx.call_capability.await_count,1)

    async def test_expired_hit_refresh_zero_ttl(self):
        cfg = config()
        engine = RecallEngine(cfg)
        engine.cache[('a','q')] = (time.monotonic()-1,'old')
        ctx = type('Fake', (), {'call_capability': AsyncMock(return_value=HIT)})()
        self.assertNotEqual(await engine.search(ctx,'a',Query('f','q'),time.monotonic()+1),'old')
        cfg.retrieval.cache_ttl_seconds = 0
        engine.cache.clear()
        for _ in range(2): await engine.search(ctx,'a',Query('f','q'),time.monotonic()+1)
        self.assertEqual(ctx.call_capability.await_count,3)

    async def test_formal_call_scope(self):
        engine = RecallEngine(config())
        ctx = type('Fake', (), {'call_capability': AsyncMock(return_value=HIT)})()
        await engine.search(ctx,'session-a',Query('f','topic'),time.monotonic()+1)
        args, kw = ctx.call_capability.call_args
        self.assertEqual(args,('knowledge.search',))
        self.assertEqual(kw['query'],'topic')
        self.assertEqual(kw['chat_id'],'session-a')
        self.assertEqual(kw['mode'],'search')
        self.assertIs(kw['respect_filter'],True)
        self.assertGreater(kw['timeout_ms'],0)


class HookTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_missing_session_allowlist(self):
        variants = [(config(),request(session_id='')),(config(),request(session_id=None))]
        disabled=config();disabled.plugin.enabled=False;variants.append((disabled,request()))
        restricted=config();restricted.plugin.session_allowlist=['session-b'];variants.append((restricted,request()))
        for cfg,kw in variants:
            plugin,mock=plugin_with(None,cfg)
            self.assertEqual(await plugin.planner_recall(**kw),CONTINUE)
            mock.assert_not_awaited()

    async def test_self_config_update_drops_query_cache(self):
        plugin,_=plugin_with(None)
        old=plugin._engine;old.cache[('session-a','q')]=(time.monotonic()+100,'old')
        await plugin.on_config_update('self',config().model_dump(),'new')
        self.assertIsNot(plugin._engine,old)
        self.assertFalse(plugin._engine.cache)

    async def test_store_failure_keeps_original_request(self):
        from unittest.mock import Mock
        plugin,mock=plugin_with(RuntimeError('synthetic'))
        plugin._store=Mock();plugin._store.cursor.return_value=None
        self.assertEqual(await plugin.planner_recall(**request()),CONTINUE)
        mock.assert_awaited_once()


if __name__ == '__main__': unittest.main(verbosity=2)
