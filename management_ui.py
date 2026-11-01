"""Official plugin WebUI APIs; slow operations use one tracked background job."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from maibot_sdk import API


class ManagementMixin:
    def init_management(self) -> None:
        self._management_task: asyncio.Task | None = None
        self._management_job = {'name': '', 'state': 'idle', 'message': '尚无管理任务'}
        self._browse_directory = ''

    async def shutdown_management(self) -> None:
        task = self._management_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._management_task = None

    def _start_management(self, name: str, operation: Any) -> dict[str, Any]:
        if self._management_task and not self._management_task.done():
            raise ValueError('已有管理任务正在执行，请等待完成')
        self._management_job = {'name': name, 'state': 'running', 'message': name + '进行中'}

        async def run() -> None:
            try:
                result = await operation()
                self._management_job = {'name': name, 'state': 'done', 'message': name + '完成', 'result': result}
            except asyncio.CancelledError:
                self._management_job = {'name': name, 'state': 'cancelled', 'message': name + '已取消'}
                raise
            except Exception as error:
                self._management_job = {'name': name, 'state': 'error', 'message': str(error)[:1000]}

        self._management_task = asyncio.create_task(run(), name='setting-recall-management')
        return {'accepted': True, 'job': dict(self._management_job)}

    @API('management_status', description='查看手动资料和后台任务状态')
    async def management_status(self) -> dict[str, Any]:
        manual = self.manual
        engine = self._engine
        vectors = getattr(engine, 'vectors', None)
        return {
            'directory': self._browse_directory or self.config.documents.directory or (manual.directory if manual else ''),
            'loaded_directory': manual.directory if manual else '',
            'files': list(manual.listing) if manual else [],
            'selected': list(manual.selected) if manual else [],
            'documents': manual.index.document_count if manual else 0,
            'chunks': len(manual.index.chunks) if manual else 0,
            'status': manual.status if manual else '插件尚未就绪',
            'busy': bool(self._management_task and not self._management_task.done()),
            'job': dict(self._management_job),
            'embedding': {
                'enabled': bool(self.config.embedding.enabled),
                'model': getattr(vectors, 'model_name', ''),
                'dimension': getattr(vectors, 'dimension', 0),
                'vectors': len(getattr(vectors, 'vectors', {})),
                **(vectors.status(engine.documents) if vectors and engine else {}),
            },
        }

    @API('management_browse', description='浏览指定目录，只列文件名不读正文')
    async def management_browse(self, directory: str = '') -> dict[str, Any]:
        if not self.manual:
            raise ValueError('插件尚未就绪')
        directory = directory.strip() or self.config.documents.directory
        if not directory:
            directory = str(Path(self.ctx.paths.data_dir) / 'documents')

        async def browse() -> dict[str, Any]:
            manual, engine = self.manual, self._engine
            names = await asyncio.to_thread(manual.browse, directory, self.config.documents)
            self._browse_directory = manual.browse_directory
            return {'files': len(names)}

        return self._start_management('浏览目录', browse)

    @API('management_load', description='读取明确选中的文档并替换私有快照')
    async def management_load(self, names: list[str]) -> dict[str, Any]:
        if not self.manual:
            raise ValueError('插件尚未就绪')
        if not isinstance(names, list) or len(names) > 40 or any(not isinstance(name, str) for name in names):
            raise ValueError('请选择不超过40个文件')
        selected = list(names)
        return self._start_management('加载选中文档', lambda: self.reload_documents(selected))

    @API('management_vectors', description='为已加载文档手动构建向量索引')
    async def management_vectors(self) -> dict[str, Any]:
        return self._start_management('构建向量索引', self.build_vectors)
