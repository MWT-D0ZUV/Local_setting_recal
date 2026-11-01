"""Local-document tests use temporary synthetic TXT/Markdown only; no real memories."""
import asyncio
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.core import Query
from Local_setting_recall.local_documents import DocumentIndex
from Local_setting_recall.plugin import Config, DocumentSection, RecallEngine
from types import SimpleNamespace

def InlineDocument(name, content, enabled=True):
    return SimpleNamespace(name=name, content=content, enabled=enabled)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='synthetic-docs-', dir=Path(__file__).parent)
        self.directory = Path(self.temporary.name)
        self.cfg = DocumentSection()
        self.index = DocumentIndex()

    def tearDown(self):
        self.temporary.cleanup()

    def refresh(self, inline=None):
        self.index.refresh(self.directory, inline or [], self.cfg, force=True)

    def put(self, name, text, encoding='utf-8'):
        path = self.directory / name
        path.write_text(text, encoding=encoding)
        return path

    def test_create_edit_delete_is_reflected_on_refresh(self):
        self.refresh()
        self.assertEqual(self.index.search('C8 blue'), '')
        path = self.put('world.md', '# C8 World\nC8 blue gate is open.')
        self.refresh()
        self.assertIn('blue gate', self.index.search('C8 blue'))
        path.write_text('# C8 World\nC8 green gate is closed.', encoding='utf-8')
        self.refresh()
        edited = self.index.search('C8 green')
        self.assertIn('green gate', edited)
        self.assertNotIn('blue gate', edited)
        path.unlink()
        self.refresh()
        self.assertEqual(self.index.search('C8 green'), '')
        self.assertEqual(self.index.document_count, 0)

    def test_inline_enable_edit_disable_and_blank(self):
        doc = InlineDocument(name='inline-world', content='# C8 World\nC8 blue gate is open.')
        self.refresh([doc])
        self.assertIn('配置/inline-world', self.index.search('C8 blue'))
        doc.content = '# C8 World\nC8 green gate is closed.'
        self.refresh([doc])
        self.assertIn('green gate', self.index.search('C8 green'))
        doc.enabled = False
        self.refresh([doc])
        self.assertEqual(self.index.search('C8 green'), '')
        self.refresh([InlineDocument(name='blank', content=' ')])
        self.assertEqual(self.index.document_count, 0)

    def test_source_section_hash_and_fragment_are_preserved(self):
        path = self.put('source.md', '# Gate catalog\nC8 blue gate has synthetic details.')
        self.refresh()
        first = self.index.search('C8 blue')
        self.assertIn('source.md', first)
        self.assertIn('Gate catalog', first)
        self.assertIn('片段1', first)
        self.assertRegex(first, r'sha256:[a-f0-9]{12}')
        old_digest = self.index.chunks[0].digest
        path.write_text('# Gate catalog\nC8 blue gate now has edited details.', encoding='utf-8')
        self.refresh()
        self.assertNotEqual(self.index.chunks[0].digest, old_digest)

    def test_utf8_bom_decodes_without_bom_in_content(self):
        self.put('bom.TXT', '# 蓝门档案\nC8 蓝门属于合成测试资料。', encoding='utf-8-sig')
        self.refresh()
        result = self.index.search('C8 蓝门')
        self.assertIn('蓝门属于', result)
        self.assertNotIn('\ufeff', result)
        self.assertEqual(self.index.errors, 0)

    def test_child_folders_hidden_and_unsupported_files_are_ignored(self):
        self.put('valid.md', 'C8 blue gate is visible.')
        self.put('.hidden.md', 'C8 secret hidden marker.')
        self.put('ignored.json', 'C8 secret JSON marker.')
        sub = self.directory / 'child'
        sub.mkdir()
        (sub / 'nested.txt').write_text('C8 secret nested marker.', encoding='utf-8')
        self.refresh()
        self.assertEqual(self.index.document_count, 1)
        self.assertEqual({chunk.name for chunk in self.index.chunks}, {'valid.md'})
        self.assertNotIn('secret', self.index.search('C8 blue'))

    def test_default_00_exclusion_happens_before_opening_files(self):
        self.assertEqual(self.cfg.excluded_patterns, ['00*'])
        self.put('00-synthetic-guide.md', 'C8 excluded guide marker.')
        self.put('001-synthetic-note.TXT', 'C8 excluded note marker.')
        valid = self.put('01-synthetic-world.md', 'C8 blue gate is visible.')
        opened = []
        original_read = Path.read_text

        def guarded_read(path, *args, **kwargs):
            self.assertFalse(path.name.startswith('00'), 'excluded file was opened')
            opened.append(path)
            return original_read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', new=guarded_read):
            self.refresh()
        self.assertEqual(opened, [valid])
        self.assertEqual(self.index.document_count, 1)
        self.assertEqual({chunk.name for chunk in self.index.chunks}, {valid.name})
        self.assertEqual(self.index.trigger_queries('00-synthetic-guide'), [])

    def test_unreadable_or_linked_root_clears_previous_index(self):
        self.put('world.md', 'C8 blue gate is visible.')
        for failure in ('linked-root', 'listing-failed'):
            with self.subTest(failure=failure):
                self.refresh()
                self.assertTrue(self.index.search('C8 blue'))
                if failure == 'linked-root':
                    failure_patch = patch.object(Path, 'is_symlink', return_value=True)
                else:
                    failure_patch = patch.object(Path, 'iterdir', side_effect=OSError('synthetic listing failure'))
                with failure_patch:
                    self.refresh()
                self.assertEqual(self.index.chunks, [])
                self.assertEqual(self.index.aliases, [])
                self.assertEqual(dict(self.index.frequencies), {})
                self.assertEqual(self.index.signature, '')
                self.assertEqual(self.index.document_count, 0)
                self.assertEqual(self.index.errors, 1)
                self.assertEqual(self.index.search('C8 blue'), '')

    def test_exact_world_code_c8_does_not_return_c5_or_c80(self):
        self.put('worlds.md', '# C5 world\nC5 blue gate belongs to red team.\n# C80 world\nC80 blue gate belongs to green team.\n# C8 world\nC8 blue gate belongs to silver team.')
        self.refresh()
        result = self.index.search('C8 blue gate')
        self.assertIn('silver team', result)
        self.assertNotIn('red team', result)
        self.assertNotIn('green team', result)

    def test_unrelated_and_single_token_queries_are_empty(self):
        self.put('gates.md', 'C8 blue gate belongs to synthetic town.')
        self.refresh()
        self.assertEqual(self.index.search('unrelated vanished topic'), '')
        self.assertEqual(self.index.search('blue'), '')

    def test_oversize_file_and_invalid_utf8_do_not_break_good_document(self):
        self.cfg.max_file_bytes = 1024
        self.put('oversize.md', 'x' * 1100)
        (self.directory / 'bad.txt').write_bytes(b'\xff\xfe\xfd')
        self.put('good.md', 'C8 blue gate is valid.')
        self.refresh()
        self.assertIn('is valid', self.index.search('C8 blue'))
        self.assertEqual(self.index.document_count, 1)
        self.assertEqual(self.index.errors, 2)

    def test_new_heading_automatically_triggers_query(self):
        self.put('neutral.md', '# 星砂档案\n星砂角色具有蓝色帽子。')
        self.refresh()
        queries = self.index.trigger_queries('请说说星砂档案', 2)
        self.assertTrue(queries)
        self.assertTrue(any('星砂' in getattr(query, 'text', str(query)) for query in queries))
        self.assertFalse(self.index.trigger_queries('普通无关问候', 2))

    def test_auto_query_is_bound_to_exact_source_document(self):
        self.put('target.md', '# Shared title\nblue gate target-only detail.')
        self.put('other.md', '# Shared title\nblue gate other-only detail.')
        self.refresh()
        queries = self.index.trigger_queries('tell me target', 2)
        self.assertEqual(len(queries), 1)
        query = queries[0]
        self.assertEqual(query.document_name, 'target.md')
        result = self.index.search(query.text, document_name=query.document_name)
        self.assertIn('target-only', result)
        self.assertNotIn('other-only', result)
        self.assertEqual(self.index.search('Shared title', document_name='missing.md'), '')

    def test_names_boilerplate_and_filename_are_not_topic_evidence(self):
        self.put('nuclear physics.md', '# 古明地念设定资料\n古明地念 古明地觉 古明地恋 日常设定资料：角色喜欢蓝色帽子。')
        self.refresh()
        for query in ('古明地念 nuclear physics', '古明地觉 nuclear physics', '古明地恋 设定资料 日常', '古明地念'):
            with self.subTest(query=query):
                self.assertEqual(self.index.search(query), '')
        self.assertIn('蓝色帽子', self.index.search('古明地念 蓝色帽子'))

    def test_keyword_line_triggers_with_ascii_boundaries(self):
        self.put('neutral.txt', '关键词：ZX7, 星砂旅人\n星砂旅人 ZX7 lives at the silver gate.')
        self.refresh()
        self.assertTrue(self.index.trigger_queries('请介绍ZX7', 2))
        self.assertTrue(self.index.trigger_queries('星砂旅人是谁', 2))
        self.assertFalse(self.index.trigger_queries('ZX70 unrelated', 2))
        self.assertFalse(self.index.trigger_queries('prefixZX7 unrelated', 2))


class DocumentFallbackTests(unittest.IsolatedAsyncioTestCase):
    def make_engine(self):
        cfg = Config()
        cfg.documents.enabled = True
        cfg.documents.fallback_to_host = False
        cfg.retrieval.log_enabled = False
        return RecallEngine(cfg)

    async def test_local_hit_precedes_host_even_when_fallback_enabled(self):
        engine = self.make_engine()
        engine.cfg.documents.fallback_to_host = True
        engine.documents.refresh(None, [InlineDocument(name='Synthetic', content='C8 blue gate is local.')], engine.cfg.documents, force=True)
        ctx = type('Fake', (), {'call_capability': AsyncMock(return_value='你知道这些知识: 1. host data')})()
        result = await engine.search(ctx, 'a', Query('f', 'C8 blue'), time.monotonic()+1)
        self.assertIn('本地文档', result)
        self.assertIn('is local', result)
        ctx.call_capability.assert_not_awaited()

    async def test_style_search_uses_q_documents_while_facts_search_all_documents(self):
        engine = self.make_engine()
        ctx = type('Fake', (), {'call_capability': AsyncMock()})()
        with tempfile.TemporaryDirectory(prefix='synthetic-style-', dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            docs = {'Q01-style.md': 'blue gate Q_MARKDOWN_DETAIL',
                    'Q02-style.txt': 'blue gate Q_TEXT_DETAIL',
                    'F01-facts.md': 'blue gate FACT_DETAIL'}
            for name, content in docs.items():
                (directory / name).write_text(content, encoding='utf-8')
            engine.documents.refresh(directory, [], engine.cfg.documents, force=True)
            style = await engine.search(ctx, 'a', Query('style', 'blue gate', style=True), time.monotonic()+1)
            self.assertTrue('Q_MARKDOWN_DETAIL' in style or 'Q_TEXT_DETAIL' in style)
            self.assertEqual(style.count('本地文档：'), 1)
            self.assertNotIn('FACT_DETAIL', style)
            facts = await engine.search(ctx, 'a', Query('facts', 'blue gate'), time.monotonic()+1)
            self.assertIn('FACT_DETAIL', facts)
            self.assertNotIn('Q_MARKDOWN_DETAIL', facts)
            self.assertNotIn('Q_TEXT_DETAIL', facts)
        ctx.call_capability.assert_not_awaited()

    async def test_engine_respects_query_source_constraint(self):
        engine = self.make_engine()
        docs = [InlineDocument(name='target', content='blue gate target-only detail.'),
                InlineDocument(name='other', content='blue gate other-only detail.')]
        engine.documents.refresh(None, docs, engine.cfg.documents, force=True)
        ctx = type('Fake', (), {'call_capability': AsyncMock()})()
        query = Query('source', 'blue gate', document_name='配置/target')
        result = await engine.search(ctx, 'a', query, time.monotonic()+1)
        self.assertIn('target-only', result)
        self.assertNotIn('other-only', result)
        ctx.call_capability.assert_not_awaited()

    async def test_no_local_hit_with_fallback_off_makes_no_host_call(self):
        engine = self.make_engine()
        ctx = type('Fake', (), {'call_capability': AsyncMock()})()
        self.assertEqual(await engine.search(ctx, 'a', Query('f', 'C8 blue'), time.monotonic()+1), '')
        ctx.call_capability.assert_not_awaited()

    async def test_no_local_hit_with_fallback_on_queries_host(self):
        engine = self.make_engine()
        engine.cfg.documents.fallback_to_host = True
        ctx = type('Fake', (), {'call_capability': AsyncMock(return_value='你知道这些知识: 1. synthetic host data')})()
        result = await engine.search(ctx, 'a', Query('f', 'C8 blue'), time.monotonic()+1)
        self.assertIn('synthetic host data', result)
        ctx.call_capability.assert_awaited_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
