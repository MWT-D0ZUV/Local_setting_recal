"""Explicit file browsing/loading and persisted private snapshots; no polling."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import fnmatch
import hashlib
import json
import os
import re
import uuid

from .local_documents import DocumentIndex, split_document


class ManualDocuments:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.index = DocumentIndex()
        self.directory = ''
        self.browse_directory = ''
        self.selected: list[str] = []
        self.listing: list[str] = []
        self.status = '尚未手动加载文档'
        self._snapshot = self.data_dir / 'manual_documents.json'

    @staticmethod
    def _directory(value: str) -> Path:
        root = Path(value.strip()).expanduser()
        if not value.strip() or not root.is_absolute():
            raise ValueError('请填写服务器本机的绝对文档目录')
        if root.is_symlink() or root.resolve().parent != root.parent.resolve():
            raise ValueError('资料目录不能通过链接指向其他位置')
        if not root.is_dir():
            raise ValueError('资料目录不存在或不是文件夹')
        return root

    @staticmethod
    def _allowed(name: str, cfg: Any) -> bool:
        return (bool(name) and name not in {'.', '..'} and Path(name).name == name
                and '/' not in name and '\\' not in name and ':' not in name
                and not name.startswith('.') and Path(name).suffix.lower() in {'.txt', '.md'}
                and not any(fnmatch.fnmatchcase(name.casefold(), p.casefold())
                            for p in getattr(cfg, 'excluded_patterns', ['00*'])))

    @staticmethod
    def _build(documents: list[dict[str, str]]) -> DocumentIndex:
        index = DocumentIndex()
        for document in documents:
            name, content = document['name'], document['content']
            index.chunks.extend(split_document(name, content))
            words = [Path(name).stem]
            words.extend(re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)[:10])
            for line in re.findall(r'^(?:keywords|关键词|触发词)\s*[:：]\s*(.+)$', content, re.MULTILINE | re.IGNORECASE):
                words.extend(w.strip(' \t[]\"\'') for w in re.split(r'[,，;；、]', line))
            index.aliases.append((name, [word.strip()[:120] for word in words if len(word.strip()) >= 2]))
        index.document_count = len(documents)
        index.frequencies = Counter(term for chunk in index.chunks for term in chunk.terms)
        index.signature = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        return index

    def browse(self, directory: str, cfg: Any) -> list[str]:
        """List only names/stat; no original content is opened, including excluded files."""
        root = self._directory(directory)
        names = []
        for path in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
            if not self._allowed(path.name, cfg):
                continue
            if (path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve()):
                continue
            if path.stat().st_size > cfg.max_file_bytes:
                continue
            names.append(path.name)
            if len(names) >= min(40, cfg.max_documents):
                break
        self.browse_directory = str(root)
        self.listing = names
        self.status = f'目录中有 {len(names)} 份可选资料；浏览未加载正文'
        return list(names)

    def load(self, directory: str, names: list[str], cfg: Any) -> DocumentIndex:
        """Read exactly the selected direct children; empty selection clears loaded data."""
        root = self._directory(directory)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError('文件选择必须是文件名列表')
        selected = list(dict.fromkeys(names))
        if len(selected) > min(40, cfg.max_documents):
            raise ValueError('选择的文档数量超出上限')
        documents: list[dict[str, str]] = []
        total = 0
        for name in selected:
            if not self._allowed(name, cfg):
                raise ValueError('选择包含不支持或已排除的文件')
            path = root / name
            if path.is_symlink() or path.resolve().parent != root.resolve() or not path.is_file():
                raise ValueError('选择文件不存在或路径超出目录')
            size = path.stat().st_size
            if size > cfg.max_file_bytes or total + size > cfg.max_total_bytes:
                raise ValueError('选择的文件内容超出大小上限')
            # Bounded read also protects against a file growing between stat/read.
            with path.open('rb') as stream:
                raw = stream.read(cfg.max_file_bytes + 1)
            if len(raw) > cfg.max_file_bytes or total + len(raw) > cfg.max_total_bytes:
                raise ValueError('读取时文件内容超出大小上限')
            content = raw.decode('utf-8-sig')
            total += len(raw)
            documents.append({'name': name, 'content': content})
        index = self._build(documents)
        payload = {'schema_version': 1, 'directory': str(root), 'documents': documents}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._snapshot.with_name('manual_documents.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, self._snapshot)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.index = index
        self.directory = str(root)
        self.selected = selected
        self.status = f'已手动加载 {len(selected)} 份文档，共 {len(index.chunks)} 个片段'
        return index

    def restore(self, cfg: Any) -> DocumentIndex:
        """Restore only the private snapshot, never list/open the source directory."""
        self.index = DocumentIndex()
        self.selected = []
        configured = str(getattr(cfg, 'directory', '') or '').strip()
        self.directory = configured
        try:
            if self._snapshot.is_symlink():
                raise ValueError('快照不能是链接')
            if not self._snapshot.exists():
                return self.index
            # JSON escaping may grow to six bytes per source character.
            max_snapshot = min(int(cfg.max_total_bytes) * 6 + 65536, 7 * 1024 * 1024)
            with self._snapshot.open('rb') as stream:
                raw = stream.read(max_snapshot + 1)
            if len(raw) > max_snapshot:
                raise ValueError('本地快照超出上限')
            payload = json.loads(raw)
            if payload.get('schema_version') != 1 or not isinstance(payload.get('documents'), list):
                raise ValueError('本地快照格式不正确')
            source = str(payload.get('directory', ''))
            # The configured directory is a UI suggestion; only explicit load replaces a snapshot.
            documents: list[dict[str, str]] = []
            total = 0
            for record in payload['documents']:
                if len(documents) >= min(40, cfg.max_documents):
                    raise ValueError('快照文档数超限')
                if not isinstance(record, dict):
                    raise ValueError('快照文档无效')
                name, content = record.get('name'), record.get('content')
                if not isinstance(name, str) or not isinstance(content, str):
                    raise ValueError('快照文档字段无效')
                if not self._allowed(name, cfg):
                    continue
                size = len(content.encode('utf-8'))
                if size > cfg.max_file_bytes or total + size > cfg.max_total_bytes:
                    raise ValueError('快照正文超限')
                documents.append({'name': name, 'content': content})
                total += size
            self.index = self._build(documents)
            self.directory = source
            self.selected = [record['name'] for record in documents]
            self.status = f'已恢复 {len(documents)} 份手动加载快照；原目录未扫描'
        except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
            self.status = '资料快照不可用，请重新手动加载'
        return self.index
