"""Optional manual embedding index. No Host internals or automatic rebuilding."""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import math
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .core import Query
from .local_documents import DocumentIndex


def _chunk_key(chunk: Any) -> str:
    return hashlib.sha256((chunk.name + '\0' + chunk.section + '\0' + str(chunk.number) + '\0' + chunk.text).encode('utf-8')).hexdigest()


def _unit(raw: Any) -> list[float]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 16384:
        raise ValueError('invalid embedding dimension')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in raw):
        raise ValueError('embedding contains invalid numbers')
    values = [float(x) for x in raw]
    norm = math.sqrt(math.fsum(x*x for x in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('embedding norm is invalid')
    return [x/norm for x in values]


class VectorSearch:
    """One private persisted index; queries never persist and rebuilds are explicit."""

    def __init__(self, data_dir: Path, cfg: Any) -> None:
        self.cfg = cfg
        self.path = Path(data_dir) / 'setting-recall-vectors.json'
        self.model_name = ''
        self.task_name = ''
        self.dimension = 0
        self.signature = ''
        self.vectors: dict[str, list[float]] = {}
        self.query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._generation = 0
        self._invalidated = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.is_symlink() or self.path.stat().st_size > 64 * 1024 * 1024:
                return
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            if raw.get('schema') != 1 or raw.get('task_name') != self.cfg.task_name:
                return
            entries = raw.get('vectors')
            dim = raw.get('dimension')
            if (type(dim) is not int or not isinstance(entries, dict) or len(entries) > 2048
                    or dim * len(entries) > 2_000_000):
                return
            if not isinstance(raw.get('signature'), str) or not isinstance(raw.get('model_name'), str) or not raw['model_name']:
                return
            vectors = {}
            for key, value in entries.items():
                if not re.fullmatch(r'[a-f0-9]{64}', key):
                    return
                vector = _unit(value)
                if len(vector) != dim:
                    return
                vectors[key] = vector
            self.model_name, self.dimension = raw['model_name'], dim
            self.task_name, self.signature = raw['task_name'], raw['signature']
            self.vectors = vectors
        except (OSError, ValueError, TypeError, AttributeError, OverflowError):
            self.vectors = {}

    def invalidate(self) -> None:
        """Model config changes cannot be identified from SDK model_name alone."""
        self._generation += 1
        self._invalidated = True
        self.query_cache.clear()

    def status(self, index: DocumentIndex) -> dict[str, Any]:
        ready = bool(self.vectors and not self._invalidated and self.signature == index.signature
                     and self.task_name == self.cfg.task_name)
        return {'ready': ready, 'chunks': len(self.vectors), 'model_name': self.model_name,
                'dimension': self.dimension, 'needs_manual_build': not ready}

    async def _embed(self, ctx: Any, texts: list[str]) -> list[dict[str, Any]]:
        timeout = min(120.0, max(0.1, float(getattr(self.cfg, 'timeout_seconds', 20.0))))
        result = await asyncio.wait_for(ctx.llm.embed(texts=texts, task_name=self.cfg.task_name, max_concurrent=2), timeout)
        if not isinstance(result, dict) or result.get('success') is not True:
            raise ValueError('embedding capability failed')
        results = result.get('results')
        if not isinstance(results, list) or len(results) != len(texts):
            raise ValueError('embedding batch length mismatch')
        return results

    async def build(self, ctx: Any, index: DocumentIndex) -> dict[str, Any]:
        if not self.cfg.enabled:
            raise ValueError('vector search is disabled')
        chunks = tuple(index.chunks)
        signature = index.signature
        maximum = min(2048, max(1, int(getattr(self.cfg, 'max_chunks', 2048))))
        if not signature or not chunks or len(chunks) > maximum:
            raise ValueError('empty or oversized document index')
        batch_size = min(32, max(1, int(getattr(self.cfg, 'batch_size', 8))))
        generation = self._generation
        async with self._lock:
            vectors: dict[str, list[float]] = {}
            model = ''
            dimension = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start+batch_size]
                results = await self._embed(ctx, [c.name + '\n' + c.section + '\n' + c.text for c in batch])
                for chunk, result in zip(batch, results):
                    if not isinstance(result, dict) or not isinstance(result.get('model_name'), str) or not result['model_name']:
                        raise ValueError('missing embedding model identity')
                    vector = _unit(result.get('embedding'))
                    if not model:
                        model, dimension = result['model_name'], len(vector)
                        if len(chunks) * dimension > 2_000_000:
                            raise ValueError('vector index exceeds local memory budget')
                    if result['model_name'] != model or len(vector) != dimension:
                        raise ValueError('embedding model or dimension changed during build')
                    vectors[_chunk_key(chunk)] = vector
                if generation != self._generation or index.signature != signature:
                    raise ValueError('index or model changed during build')
            payload = {'schema': 1, 'task_name': self.cfg.task_name, 'model_name': model,
                       'dimension': dimension, 'signature': signature, 'vectors': vectors}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                temporary.replace(self.path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self.vectors, self.signature = vectors, signature
            self.task_name, self.model_name, self.dimension = self.cfg.task_name, model, dimension
            self.query_cache.clear()
            self._generation += 1
            self._invalidated = False
            return {'chunks': len(vectors), 'model_name': model, 'dimension': dimension}

    async def search(self, ctx: Any, index: DocumentIndex, query: Query, limit: int = 4) -> str:
        if not self.cfg.enabled or not self.status(index)['ready'] or not query.text.strip():
            return ''
        from .relevance import candidate_allowed
        candidates = [c for c in index.chunks if candidate_allowed(c, style=query.style,
                      document_name=query.document_name, context=query.context or query.text)]
        if not candidates:
            return ''
        query_input = (('当前消息：' + query.context[:220] + '\n检索方向：' + query.text[:120])
                       if query.context else query.text)
        async with self._lock:
            if not self.status(index)['ready']:
                return ''
            signature, generation = self.signature, self._generation
            query_key = hashlib.sha256((self.task_name + '\0' + self.model_name + '\0' + query_input).encode('utf-8')).hexdigest()
            vector = self.query_cache.get(query_key)
            if vector is None:
                result = (await self._embed(ctx, [query_input]))[0]
                if not isinstance(result, dict) or result.get('model_name') != self.model_name:
                    self.invalidate()
                    raise ValueError('query embedding model differs from document model')
                vector = _unit(result.get('embedding'))
                if len(vector) != self.dimension:
                    self.invalidate()
                    raise ValueError('query embedding dimension differs from document index')
                if generation != self._generation or index.signature != signature:
                    return ''
                self.query_cache[query_key] = vector
                self.query_cache.move_to_end(query_key)
                while len(self.query_cache) > 128:
                    self.query_cache.popitem(last=False)
            if generation != self._generation or index.signature != signature:
                return ''
            scored = []
            threshold = min(1.0, max(-1.0, float(self.cfg.min_similarity)))
            for chunk in candidates:
                stored = self.vectors.get(_chunk_key(chunk))
                if stored is None:
                    continue
                score = math.fsum(a*b for a,b in zip(vector, stored))
                if score >= threshold:
                    scored.append((score, chunk))
            scored.sort(key=lambda item: (-item[0], item[1].name, item[1].number))
            if scored:
                cutoff = max(threshold, scored[0][0] - float(getattr(self.cfg, 'relative_margin', 0.08)))
                scored = [entry for entry in scored if entry[0] >= cutoff]
            return '\n\n'.join(f'[本地向量参考：{c.name} | {c.section} | 片段{c.number} | sha256:{c.digest}]\n{c.text}'
                                 for _, c in scored[:min(1 if query.style else 2, max(1, int(limit)))] )
