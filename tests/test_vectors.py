import asyncio
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.core import Query
from Local_setting_recall.local_documents import DocumentIndex, split_document
from Local_setting_recall.vector_search import VectorSearch


def index_for(*docs):
    index = DocumentIndex()
    index.chunks = [c for name,text in docs for c in split_document(name,text)]
    index.signature = hashlib.sha256(repr(docs).encode()).hexdigest()
    return index


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.model = 'synthetic-embedding'
        self.bad = None
    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.bad == 'failure':
            return {'success':False,'error':'synthetic'}
        results = []
        for text in kwargs['texts']:
            vector = [1.0,0.0] if 'blue' in text or 'c8' in text.lower() else [0.0,1.0]
            if self.bad == 'nan': vector = [float('nan'),1.0]
            if self.bad == 'dimension': vector = [1.0,0.0,0.0]
            results.append({'embedding':vector,'model_name':self.model})
        if self.bad == 'length': results=[]
        return {'success':True,'results':results}


class VectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = SimpleNamespace(enabled=True,task_name='embedding',min_similarity=0.4,batch_size=2,timeout_seconds=1)
        self.llm=FakeLLM();self.ctx=SimpleNamespace(llm=self.llm)
        self.index=index_for(('F01.md','# Facts\nblue lake C8.'),('Q01.md','# Quotes\nblue tone.'),('F02.md','# Other\nred meadow C5.'))
        self.vector=VectorSearch(Path(self.temp.name),self.cfg)

    async def test_search_does_not_automatically_build(self):
        self.assertEqual(await self.vector.search(self.ctx,self.index,Query('x','blue')),'')
        self.assertEqual(self.llm.calls,[])
        self.assertFalse(self.vector.path.exists())

    async def test_manual_build_is_bounded_and_persists_only_document_vectors(self):
        result=await self.vector.build(self.ctx,self.index)
        self.assertEqual(result,{'chunks':3,'model_name':'synthetic-embedding','dimension':2})
        self.assertEqual([len(c['texts']) for c in self.llm.calls],[2,1])
        raw=json.loads(self.vector.path.read_text())
        self.assertEqual(raw['task_name'],'embedding')
        self.assertEqual(len(raw['vectors']),3)
        self.assertTrue(self.vector.status(self.index)['ready'])
        loaded=VectorSearch(Path(self.temp.name),self.cfg)
        self.assertTrue(loaded.status(self.index)['ready'])
        self.assertEqual(loaded.dimension,2)

    async def test_semantic_search_and_exact_source_constraint(self):
        await self.vector.build(self.ctx,self.index)
        result=await self.vector.search(self.ctx,self.index,Query('x','blue'))
        self.assertIn('F01.md',result);self.assertNotIn('Q01.md',result);self.assertNotIn('F02.md',result)
        result=await self.vector.search(self.ctx,self.index,Query('x','blue',style=True,document_name='Q01.md'))
        self.assertNotIn('F01.md',result);self.assertIn('Q01.md',result)

    async def test_style_only_quotes_and_exact_world_codes(self):
        await self.vector.build(self.ctx,self.index)
        result=await self.vector.search(self.ctx,self.index,Query('x','blue',style=True))
        self.assertIn('Q01.md',result);self.assertNotIn('F01.md',result)
        result=await self.vector.search(self.ctx,self.index,Query('x','C8 blue'))
        self.assertIn('F01.md',result);self.assertNotIn('F02.md',result);self.assertNotIn('Q01.md',result)

    async def test_document_edits_require_manual_build(self):
        await self.vector.build(self.ctx,self.index);count=len(self.llm.calls)
        changed=index_for(('F01.md','changed blue'))
        self.assertEqual(await self.vector.search(self.ctx,changed,Query('x','blue')),'')
        self.assertEqual(len(self.llm.calls),count)
        self.assertFalse(self.vector.status(changed)['ready'])

    async def test_query_cache_hashes_and_caps_memory(self):
        await self.vector.build(self.ctx,self.index)
        for _ in range(2): await self.vector.search(self.ctx,self.index,Query('x','private blue needle'))
        count=len(self.llm.calls)
        self.assertEqual(count,3)
        self.assertTrue(all(len(key)==64 for key in self.vector.query_cache))
        self.assertNotIn('private blue needle',self.vector.path.read_text())
        for number in range(135):
            await self.vector.search(self.ctx,self.index,Query('x',f'blue {number}'))
        self.assertEqual(len(self.vector.query_cache),128)

    async def test_failed_build_does_not_replace_good_index(self):
        await self.vector.build(self.ctx,self.index)
        before=self.vector.path.read_bytes()
        for bad in ['failure','nan','length']:
            self.llm.bad=bad
            with self.assertRaises(ValueError): await self.vector.build(self.ctx,self.index)
            self.assertEqual(self.vector.path.read_bytes(),before)

    async def test_query_model_or_dimension_change_rejected(self):
        await self.vector.build(self.ctx,self.index)
        self.llm.model='different-model'
        with self.assertRaises(ValueError): await self.vector.search(self.ctx,self.index,Query('x','blue'))
        self.assertFalse(self.vector.status(self.index)['ready'])
        self.llm.model='synthetic-embedding';await self.vector.build(self.ctx,self.index)
        self.llm.bad='dimension'
        with self.assertRaises(ValueError): await self.vector.search(self.ctx,self.index,Query('x','blue'))
        self.assertFalse(self.vector.status(self.index)['ready'])

    async def test_task_change_invalidates_persisted_index(self):
        await self.vector.build(self.ctx,self.index)
        other=VectorSearch(Path(self.temp.name),SimpleNamespace(enabled=True,task_name='another',min_similarity=.4))
        self.assertFalse(other.status(self.index)['ready'])

    async def test_disabled_build_and_search_are_not_network_calls(self):
        self.cfg.enabled=False
        with self.assertRaises(ValueError): await self.vector.build(self.ctx,self.index)
        self.assertEqual(await self.vector.search(self.ctx,self.index,Query('x','blue')),'')
        self.assertEqual(self.llm.calls,[])

    async def test_invalid_persisted_vectors_are_ignored(self):
        await self.vector.build(self.ctx,self.index)
        raw=json.loads(self.vector.path.read_text())
        raw['vectors'][next(iter(raw['vectors']))]=[True,1]
        self.vector.path.write_text(json.dumps(raw))
        broken=VectorSearch(Path(self.temp.name),self.cfg)
        self.assertFalse(broken.status(self.index)['ready'])

    async def test_invalidated_inflight_query_does_not_refill_cache(self):
        await self.vector.build(self.ctx,self.index)
        started=asyncio.Event();release=asyncio.Event();original=self.llm.embed
        async def delayed(**kwargs):
            started.set();await release.wait();return await original(**kwargs)
        self.llm.embed=delayed
        task=asyncio.create_task(self.vector.search(self.ctx,self.index,Query('x','blue')))
        await started.wait();self.vector.invalidate();release.set()
        self.assertEqual(await task,'')
        self.assertEqual(len(self.vector.query_cache),0)

    async def test_document_change_during_manual_build_keeps_previous_file(self):
        await self.vector.build(self.ctx,self.index)
        before=self.vector.path.read_bytes();original=self.llm.embed
        async def mutate(**kwargs):
            result=await original(**kwargs);self.index.signature='new-version';return result
        self.llm.embed=mutate
        with self.assertRaises(ValueError): await self.vector.build(self.ctx,self.index)
        self.assertEqual(self.vector.path.read_bytes(),before)
        self.assertFalse(self.vector.status(self.index)['ready'])

    async def test_model_event_invalidates_until_explicit_build(self):
        await self.vector.build(self.ctx,self.index)
        self.vector.invalidate();count=len(self.llm.calls)
        self.assertEqual(await self.vector.search(self.ctx,self.index,Query('x','blue')),'')
        self.assertEqual(len(self.llm.calls),count)
        await self.vector.build(self.ctx,self.index)
        self.assertTrue(self.vector.status(self.index)['ready'])


if __name__=='__main__': unittest.main()
