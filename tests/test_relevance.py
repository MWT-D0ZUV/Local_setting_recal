"""Regression tests for observed production over-retrieval; synthetic data only."""
import asyncio
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.core import Query, choose_queries
from Local_setting_recall.local_documents import DocumentIndex, split_document
from Local_setting_recall.plugin import Config, SettingRecallPlugin
from Local_setting_recall.batching import BatchStore
from Local_setting_recall.relevance import candidate_allowed, same_topic
from Local_setting_recall.vector_search import VectorSearch, _chunk_key


def make_index(*documents):
    index=DocumentIndex()
    index.chunks=[c for name,text in documents for c in split_document(name,text)]
    index.signature=hashlib.sha256(repr(documents).encode()).hexdigest()
    return index


class RelevanceTests(unittest.TestCase):
    def test_cached_body_overlap_cannot_replace_message_topic(self):
        self.assertFalse(same_topic('不用质疑，就是这样w（摸）','旅者共同体的跨世界散步机制','不用质疑，就是这样w（摸） 共同体 传送'))
        self.assertTrue(same_topic('蓝色大门在哪里','蓝色大门附近有什么','蓝色大门通向庭院'))
        self.assertFalse(same_topic('C5 blue gate','C8 blue gate','C8 C5 blue gate'))

    def test_cache_has_ttl_and_original_context_and_no_requery(self):
        with tempfile.TemporaryDirectory() as root:
            store=BatchStore(Path(root)/'batch.db')
            messages=[{'message_id':'1','session_id':'s','timestamp':str(time.time()),'raw_message':[{'type':'text','data':'蓝色大门在哪里'}]}]
            store.observe('s',messages,max_age=300,initial_limit=20)
            batch,_=store.claim('s',1,'v')
            store.finish(batch,[],{'1':[(Query('facts','大门资料',context='蓝色大门在哪里'),'蓝色大门，摸头话题，无用的重词')]})
            self.assertTrue(store.recent_reference('s','v','蓝色大门在哪里'))
            self.assertEqual(store.recent_reference('s','v','摸头话题'),[])
            self.assertEqual(store.recent_reference('s','v','蓝色大门在哪里',max_age=0),[])
            with store.connect() as db:db.execute('UPDATE batches SET created=?',(time.time()-301,))
            self.assertEqual(store.recent_reference('s','v','蓝色大门在哪里'),[])
            self.assertTrue(store.result_for('s','1','v'))

    def test_overview_and_quote_fact_mixing_are_rejected(self):
        index=make_index(('Q01.md','# 说明\n> 文档分类：quotes\n> 用途说明：这里描述语录用途'),('Q02.md','# 安慰\n别担心，我陪着你。'),('F01.md','# 大门\nC8 蓝色大门。'))
        overview,quote,fact=index.chunks
        self.assertFalse(candidate_allowed(overview,style=True))
        self.assertTrue(candidate_allowed(quote,style=True))
        self.assertFalse(candidate_allowed(quote,style=False))
        self.assertTrue(candidate_allowed(fact,style=False,context='C8'))
        self.assertFalse(candidate_allowed(fact,style=False,context='C8 C5'))

    def test_style_current_context_is_first_and_q_alias_not_fact(self):
        cfg=Config();text='不用质疑，就是这样w（摸）'
        queries=choose_queries(text,cfg,replyer=True)
        self.assertEqual(len(queries),1)
        self.assertTrue(queries[0].style)
        self.assertTrue(queries[0].text.startswith('当前情境：'+text))
        self.assertEqual(queries[0].context,text)
        index=DocumentIndex();index.aliases=[('Q01.md',['安慰']),('F01.md',['蓝色大门'])]
        self.assertEqual(index.trigger_queries('请安慰我'),[])
        self.assertEqual(index.trigger_queries('蓝色大门在哪里')[0].context,'蓝色大门在哪里')


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_smalltalk_never_searches_facts_even_with_ready_vectors(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=Config();plugin=SettingRecallPlugin();plugin.set_plugin_config(cfg.model_dump())
            plugin._ctx=SimpleNamespace(paths=SimpleNamespace(data_dir=Path(root)))
            await plugin.on_load()
            try:
                calls=[]
                plugin._engine.vectors.status=lambda index:{'ready':True}
                async def search(ctx,sid,query,deadline):calls.append(query);return ''
                plugin._engine.search=search
                await plugin._process_batch(plugin._engine,'s','synthetic', [('1','不用质疑，就是这样w（摸）')])
                self.assertEqual(len(calls),1);self.assertTrue(calls[0].style)
                calls.clear()
                await plugin._process_batch(plugin._engine,'s','synthetic',[('2','C8 的人物关系是什么？')])
                self.assertTrue(any(not q.style for q in calls))
            finally:await plugin.on_unload()

    async def test_vector_relative_gap_and_metadata_gating(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=Config().embedding
            index=make_index(('F01.md','# gate\nblue gate'),('F02.md','# teleport\nred portal'),('Q01.md','# quotes\nblue tone'),('Q02.md','# meta\n文档分类：quotes\n用途说明：blue catalog'))
            vector=VectorSearch(Path(root),cfg)
            vector.signature=index.signature;vector.task_name=cfg.task_name;vector.model_name='synthetic';vector.dimension=2
            vector.vectors={_chunk_key(c):([.95,math.sqrt(1-.95**2)] if c.name=='F01.md' else [.6,.8]) for c in index.chunks}
            calls=[]
            async def embed(**kwargs):calls.append(kwargs);return {'success':True,'results':[{'embedding':[1,0],'model_name':'synthetic'}]}
            ctx=SimpleNamespace(llm=SimpleNamespace(embed=embed))
            result=await vector.search(ctx,index,Query('fact','blue'))
            self.assertIn('F01.md',result);self.assertNotIn('F02.md',result);self.assertNotIn('Q01.md',result)
            style=await vector.search(ctx,index,Query('style','blue',True))
            self.assertIn('Q01.md',style);self.assertNotIn('Q02.md',style)
            count=len(calls)
            self.assertEqual(await vector.search(ctx,index,Query('world','C8 C5')), '')
            self.assertEqual(len(calls),count)


if __name__=='__main__':unittest.main()
