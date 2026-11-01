"""Conservative source and topic gates; no model or document I/O."""
from __future__ import annotations
import fnmatch
import re


def world_codes(text: str) -> set[str]:
    return set(re.findall(r'(?<![a-z0-9_])[a-z][0-9]{1,2}(?![a-z0-9_])', text.casefold()))


def is_quotes(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name.casefold(), p) for p in ('q*.md', 'q*.txt'))


def candidate_allowed(chunk, *, style: bool, document_name: str = '', context: str = '') -> bool:
    if document_name and chunk.name != document_name:
        return False
    if style != is_quotes(chunk.name):
        return False
    # A document description is not an utterance or evidence of a setting fact.
    intro = chunk.text[:400]
    if '文档分类' in intro and '用途说明' in intro:
        return False
    codes = world_codes(context)
    return not codes or codes.issubset(world_codes(chunk.text))


def topic_terms(text: str) -> set[str]:
    from .local_documents import tokens
    boilerplate = '古明地念 古明地觉 古明地恋 设定资料 日常 你我我们 这个那个 就是这样 什么 可以 知道 觉得 一下 时候 还有 现在 记得 对的 谢谢 哈哈'
    return set(tokens(text)) - set(tokens(boilerplate))


def same_topic(current: str, original: str, body: str) -> bool:
    codes = world_codes(current)
    if codes and (not codes.issubset(world_codes(original)) or not codes.issubset(world_codes(body))):
        return False
    wanted, source = topic_terms(current), topic_terms(original)
    shared = wanted & source
    # A short exact world code is useful evidence; a generic word in a long
    # cached document is not. Compare original messages, never result bodies.
    if codes and codes.issubset(shared):
        return True
    return len(shared) >= 2 and len(shared) / max(1, len(wanted)) >= 0.5
