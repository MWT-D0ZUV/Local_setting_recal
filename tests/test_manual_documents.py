"""Manual selection uses synthetic documents only, with source read guards."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.manual_documents import ManualDocuments
from Local_setting_recall.management_ui import ManagementMixin


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.data = self.root / 'private'
        self.cfg = SimpleNamespace(directory=str(self.source), excluded_patterns=['00*'], max_documents=40,
                                   max_file_bytes=10000, max_total_bytes=100000)
        (self.source / '00-guide.md').write_text('excluded secret', encoding='utf-8')
        (self.source / 'F01-facts.md').write_text('# Blue Gate\n关键词：蓝门\nC8 blue gate is open.', encoding='utf-8')
        (self.source / 'Q01-quotes.txt').write_text('Q style example', encoding='utf-8')
        self.manual = ManualDocuments(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_browse_opens_no_body_and_keeps_loaded_snapshot(self):
        self.manual.load(str(self.source), ['F01-facts.md'], self.cfg)
        old = self.manual.index
        other = self.root / 'other'
        other.mkdir()
        (other / 'N01.md').write_text('N example', encoding='utf-8')
        with patch.object(Path, 'open', side_effect=AssertionError('browse opened body')):
            names = self.manual.browse(str(other), self.cfg)
        self.assertEqual(names, ['N01.md'])
        self.assertIs(self.manual.index, old)
        self.assertEqual(self.manual.directory, str(self.source))
        self.assertEqual(self.manual.browse_directory, str(other))
        self.assertEqual(self.manual.selected, ['F01-facts.md'])

    def test_only_selected_body_is_opened_and_aliases_work(self):
        original = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            if path.parent == self.source:
                opened.append(path.name)
                if path.name != 'F01-facts.md':
                    raise AssertionError('unselected body opened')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            index = self.manual.load(str(self.source), ['F01-facts.md'], self.cfg)
        self.assertEqual(opened, ['F01-facts.md'])
        self.assertEqual(index.document_count, 1)
        self.assertIn('blue gate', index.search('C8 blue'))
        self.assertTrue(index.trigger_queries('蓝门', limit=1))
        self.assertTrue(any('Blue Gate' in words for _, words in index.aliases))

    def test_restore_never_reads_source_and_default_path_is_only_suggestion(self):
        self.manual.load(str(self.source), ['F01-facts.md'], self.cfg)
        restored = ManualDocuments(self.data)
        self.cfg.directory = str(self.root / 'new-default')
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.parent == self.source:
                raise AssertionError('restore opened source')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded), patch.object(Path, 'iterdir', side_effect=AssertionError('restore scanned')):
            restored.restore(self.cfg)
        self.assertEqual(restored.directory, str(self.source))
        self.assertEqual(restored.selected, ['F01-facts.md'])
        self.assertIn('blue gate', restored.index.search('C8 blue'))

    def test_excluded_and_escape_names_rejected_before_read(self):
        for name in ['00-guide.md', '../F01-facts.md', 'child/F01.md', 'child\\F01.md']:
            with self.assertRaises(ValueError):
                self.manual.load(str(self.source), [name], self.cfg)
        self.assertEqual(self.manual.index.document_count, 0)
        names = self.manual.browse(str(self.source), self.cfg)
        self.assertNotIn('00-guide.md', names)

    def test_failed_load_is_atomic_and_empty_load_clears_persisted_snapshot(self):
        first = self.manual.load(str(self.source), ['F01-facts.md'], self.cfg)
        before = (self.data / 'manual_documents.json').read_bytes()
        with self.assertRaises(ValueError):
            self.manual.load(str(self.source), ['missing.md'], self.cfg)
        self.assertIs(self.manual.index, first)
        self.assertEqual((self.data / 'manual_documents.json').read_bytes(), before)
        self.manual.load(str(self.source), [], self.cfg)
        self.assertEqual(self.manual.selected, [])
        restored = ManualDocuments(self.data)
        self.assertEqual(restored.restore(self.cfg).document_count, 0)


class ManagementTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_job_is_single_and_shutdown_cancels(self):
        management = ManagementMixin()
        management.init_management()
        event = asyncio.Event()
        async def operation():
            await event.wait()
        accepted = management._start_management('synthetic', operation)
        self.assertTrue(accepted['accepted'])
        with self.assertRaises(ValueError):
            management._start_management('second', operation)
        await asyncio.sleep(0)
        await management.shutdown_management()
        self.assertEqual(management._management_job['state'], 'cancelled')

    async def test_background_failure_is_observable(self):
        management = ManagementMixin()
        management.init_management()
        async def operation():
            raise ValueError('synthetic failure')
        management._start_management('synthetic', operation)
        await management._management_task
        self.assertEqual(management._management_job['state'], 'error')
        self.assertEqual(management._management_job['message'], 'synthetic failure')


if __name__ == '__main__':
    unittest.main()
