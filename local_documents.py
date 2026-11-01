"""Small private TXT/Markdown reference index; standard library only."""
from __future__ import annotations

import hashlib
import fnmatch
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Query, keyword_matches


@dataclass(frozen=True)
class Chunk:
    name: str
    section: str
    number: int
    digest: str
    text: str
    terms: Counter[str]


def tokens(text: str) -> Counter[str]:
    lowered = text.casefold()
    result = re.findall(r'[a-z0-9_]+', lowered)
    for run in re.findall(r'[\u3400-\u9fff]+', lowered):
        result.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run) == 1:
            result.append(run)
    return Counter(result)


def split_document(name: str, content: str) -> list[Chunk]:
    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    chunks: list[Chunk] = []
    section = '正文'
    buffer: list[str] = []
    def flush() -> None:
        text = '\n'.join(buffer).strip()
        buffer.clear()
        if not text:
            return
        # Bounded overlapping windows preserve paragraphs without swallowing whole files.
        for start in range(0, len(text), 780):
            part = text[start:start+900]
            if part.strip():
                chunks.append(Chunk(name, section, len(chunks)+1, digest, part, tokens(name + ' ' + section + ' ' + part)))
            if start + 900 >= len(text):
                break
    for line in content.splitlines():
        if re.match(r'^#{1,6}\s+\S', line):
            flush()
            section = re.sub(r'^#{1,6}\s+', '', line).strip()[:120]
        else:
            buffer.append(line)
    flush()
    return chunks


class DocumentIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.signature = ''
        self.last_check = float('-inf')
        self.errors = 0
        self.document_count = 0
        self.aliases: list[tuple[str, list[str]]] = []
        self.frequencies: Counter[str] = Counter()

    def refresh(self, directory: Path | None, inline: list[Any], cfg: Any, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_check < cfg.refresh_seconds:
            return
        self.last_check = now
        docs: list[tuple[str, str]] = []
        self.errors = 0
        total = 0
        if directory is not None:
            try:
                if directory.is_symlink() or directory.resolve().parent != directory.parent.resolve():
                    raise OSError('document root escapes private data directory')
                paths = sorted(directory.iterdir(), key=lambda p: p.name.casefold()) if directory.exists() else []
            except OSError:
                self.chunks = []
                self.aliases = []
                self.frequencies = Counter()
                self.signature = ''
                self.document_count = 0
                self.errors = 1
                return
            # Direct children only, with symlink/reparse escapes rejected by resolve().
            resolved = directory.resolve()
            for path in paths:
                if len(docs) >= cfg.max_documents:
                    break
                if (any(fnmatch.fnmatchcase(path.name.casefold(), pattern.casefold())
                        for pattern in getattr(cfg, 'excluded_patterns', ['00*']))
                        or path.name.startswith('.') or path.suffix.lower() not in {'.txt', '.md'}):
                    continue
                try:
                    if path.is_symlink() or not path.is_file() or path.resolve().parent != resolved:
                        continue
                    size = path.stat().st_size
                    if size > cfg.max_file_bytes or total + size > cfg.max_total_bytes:
                        self.errors += 1
                        continue
                    content = path.read_text(encoding='utf-8-sig')
                    docs.append((path.name, content))
                    total += size
                except (OSError, UnicodeError):
                    self.errors += 1
        for item in inline:
            if len(docs) >= cfg.max_documents:
                break
            if not item.enabled or not item.content.strip():
                continue
            size = len(item.content.encode('utf-8'))
            if size > cfg.max_file_bytes or total + size > cfg.max_total_bytes:
                self.errors += 1
                continue
            docs.append(('配置/' + item.name[:80], item.content))
            total += size
        signature = hashlib.sha256(repr(docs).encode('utf-8')).hexdigest()
        if signature == self.signature:
            return
        chunks: list[Chunk] = []
        aliases: list[tuple[str, list[str]]] = []
        for name, content in docs:
            chunks.extend(split_document(name, content))
            words = [Path(name).stem]
            words.extend(re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)[:10])
            for line in re.findall(r'^(?:keywords|关键词|触发词)\s*[:：]\s*(.+)$', content, re.MULTILINE | re.IGNORECASE):
                words.extend(w.strip(' \t[]\"\'') for w in re.split(r'[,，;；、]', line))
            aliases.append((name, [word[:120] for word in words if len(word.strip()) >= 2]))
        self.chunks = chunks
        self.aliases = aliases
        self.frequencies = Counter(term for chunk in chunks for term in chunk.terms)
        self.signature = signature
        self.document_count = len(docs)

    def trigger_queries(self, text: str, limit: int = 2) -> list[Query]:
        result: list[Query] = []
        for name, words in self.aliases:
            from .relevance import is_quotes
            if is_quotes(name):
                continue
            matches = [word for word in words if keyword_matches(text, word)]
            if matches:
                result.append(Query('本地/' + name[:50], (name + ' ' + ' '.join(matches[:4]))[:220], document_name=name, context=text))
            if len(result) >= limit:
                break
        return result

    def search(self, query: str, limit: int = 4, *, document_name: str = '', document_patterns: list[str] | None = None, query_style: bool | None = None) -> str:
        query_terms = tokens(query)
        if not self.chunks or not query_terms:
            return ''
        # Exact world codes are hard constraints in this private reference channel.
        codes = set(re.findall(r'(?<![a-z0-9_])[a-z]\d{1,2}(?![a-z0-9_])', query.casefold()))
        frequencies = self.frequencies
        # Names and boilerplate alone are not topical evidence. Explicit title
        # triggers already carry an exact source constraint and bypass this gate.
        evidence = set(query_terms) - set(tokens('古明地念 古明地觉 古明地恋 设定资料 日常'))
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            if query_style is not None:
                from .relevance import candidate_allowed
                if not candidate_allowed(chunk, style=query_style, document_name=document_name, context=query):
                    continue
            if document_patterns and not any(fnmatch.fnmatchcase(chunk.name.casefold(), pat.casefold()) for pat in document_patterns):
                continue
            if document_name and chunk.name != document_name:
                continue
            if not document_name and not (evidence & set(tokens(chunk.text))):
                continue
            if codes:
                available = set(re.findall(r'(?<![a-z0-9_])[a-z]\d{1,2}(?![a-z0-9_])', chunk.text.casefold()))
                if not codes.issubset(available):
                    continue
            shared = set(query_terms) & set(chunk.terms)
            # Require more than the universal character name or title token.
            if not document_name and len(shared) < 2:
                continue
            score = sum((math.log((len(self.chunks)+1)/(frequencies[term]+0.5))+1)
                        * min(chunk.terms[term], 2) for term in shared)
            score /= 0.6 + min(len(chunk.text), 900) / 1500
            scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name, pair[1].number))
        selected: list[Chunk] = []
        fingerprints: set[str] = set()
        for _, chunk in scored:
            key = hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()
            if key not in fingerprints:
                selected.append(chunk)
                fingerprints.add(key)
            if len(selected) >= limit:
                break
        return '\n\n'.join(f'[本地文档：{chunk.name} | {chunk.section} | 片段{chunk.number} | sha256:{chunk.digest}]\n{chunk.text}'
                             for chunk in selected)
