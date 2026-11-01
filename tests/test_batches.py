import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.batching import BatchStore
from Local_setting_recall.core import Query
from Local_setting_recall.plugin import Config, SettingRecallPlugin


def message(mid,sid='session-a',text='后室',stamp=None):
    return {'message_id':str(mid),'session_id':sid,'timestamp':str(time.time() if stamp is None else stamp),
            'raw_message':[{'type':'text','data':text}],'is_command':False,'is_notify':False}


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'batches.sqlite3';self.store=BatchStore(self.path)

    def test_pending_threshold_and_persistent_no_repeat(self):
        rows=[message(1),message(2)]
        self.assertEqual(self.store.observe('session-a',rows,max_age=300,initial_limit=20),2)
        self.assertEqual(self.store.claim('session-a',3,'v1'),('',[]))
        batch,claimed=self.store.claim('session-a',2,'v1')
        self.assertEqual(len(claimed),2)
        self.store.finish(batch,[],{'2':[(Query('fact','query'),'saved reference')]})
        reopened=BatchStore(self.path)
        self.assertEqual(reopened.observe('session-a',rows,max_age=300,initial_limit=20),0)
        self.assertEqual(reopened.claim('session-a',2,'v1'),('',[]))
        self.assertEqual(reopened.result_for('session-a','2','v1')[0][1],'saved reference')

    def test_exact_target_scope_and_document_version(self):
        self.store.observe('session-a',[message('1'),message('2')],max_age=300,initial_limit=20)
        batch,_=self.store.claim('session-a',2,'v1')
        self.store.finish(batch,[],{'1':[(Query('one','q'),'first')],'2':[(Query('two','q'),'second')]})
        self.assertEqual(self.store.result_for('session-a','1','v1')[0][1],'first')
        self.assertEqual(self.store.result_for('session-a','2','v1')[0][1],'second')
        self.assertEqual(self.store.result_for('session-b','1','v1'),[])
        self.assertEqual(self.store.result_for('session-a','3','v1'),[])
        self.assertEqual(self.store.result_for('session-a','1','v2'),[])

    def test_interrupted_claim_is_not_automatically_retried_after_restart(self):
        self.store.observe('session-a',[message('1'),message('2')],max_age=300,initial_limit=20)
        batch,_=self.store.claim('session-a',2,'v1')
        reopened=BatchStore(self.path)
        self.assertEqual(reopened.pending('session-a'),0)
        self.assertEqual(reopened.result_for('session-a','1','v1'),[])
        self.assertEqual(reopened.claim('session-a',2,'v1'),('',[]))
        with reopened.connect() as db:
            self.assertEqual(db.execute('SELECT text FROM messages').fetchall(),[('',),('',)])

    def test_bootstrap_bounded_and_cursor_tracks_newest(self):
        now=time.time()
        rows=[message(i,stamp=now-10+i) for i in range(5)]
        self.assertEqual(self.store.observe('session-a',rows,max_age=300,initial_limit=2),2)
        _,claimed=self.store.claim('session-a',2,'v1')
        self.assertEqual([mid for mid,_ in claimed],['3','4'])
        self.assertEqual(self.store.cursor('session-a'),now-6)

    def test_overlap_duplicate_and_late_arrival_can_be_recorded_once(self):
        now=time.time();first=message('new',stamp=now)
        self.store.observe('session-a',[first],max_age=300,initial_limit=20)
        late=message('late',stamp=now-5)
        self.assertEqual(self.store.observe('session-a',[late,first],max_age=300,initial_limit=20),2)
        self.assertEqual(self.store.observe('session-a',[late,first],max_age=300,initial_limit=20),2)
        batch,claimed=self.store.claim('session-a',2,'v1')
        self.assertEqual([x[0] for x in claimed],['late','new'])
        self.store.finish(batch,[])
        self.assertEqual(self.store.cursor('session-a'),now)

    def test_recent_reference_is_cached_topic_and_code_scoped(self):
        self.store.observe('session-a',[message('1')],max_age=300,initial_limit=20)
        batch,_=self.store.claim('session-a',1,'v1')
        self.store.finish(batch,[],{'1':[(Query('fact','C8 blue gate'),'C8 blue gate silver detail')]})
        self.assertTrue(self.store.recent_reference('session-a','v1','C8 blue gate'))
        self.assertEqual(self.store.recent_reference('session-a','v1','C5 blue gate'),[])
        self.assertEqual(self.store.recent_reference('session-b','v1','C8 blue gate'),[])
        self.assertEqual(self.store.recent_reference('session-a','v2','C8 blue gate'),[])
        self.assertEqual(self.store.recent_reference('session-a','v1','unrelated physics'),[])

    def test_ineligible_messages_do_not_fill_batch(self):
        rows=[message('ok'),message('other',sid='session-b'),message('stale',stamp=time.time()-1000)]
        command=message('cmd');command['is_command']=True;rows.append(command)
        self.assertEqual(self.store.observe('session-a',rows,max_age=300,initial_limit=20),1)
        self.assertIsNone(self.store.cursor('session-b'))


class BatchHookTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.plugin=SettingRecallPlugin();self.cfg=Config()
        self.cfg.batch.size=2;self.cfg.documents.enabled=False;self.cfg.style.enabled=False;self.cfg.embedding.enabled=False
        self.plugin.set_plugin_config(self.cfg.model_dump())
        self.rows=[message('1'),message('2')]
        self.caps=[]
        async def capability(name,**kwargs):
            self.caps.append((name,kwargs))
            if name in ['message.get_recent','message.get_by_time_in_chat']:
                return list(self.rows)
            raise AssertionError('unexpected capability '+name)
        self.plugin._ctx=SimpleNamespace(paths=SimpleNamespace(data_dir=Path(self.temp.name)),call_capability=capability)
        await self.plugin.on_load()
        self.addAsyncCleanup(self.plugin.on_unload)
        self.kwargs={'session_id':'session-a','items':[],'item_schema_version':1,'foreign':{'keep':True}}

    async def test_hook_returns_before_slow_batch_and_reentry_does_not_repeat(self):
        started=asyncio.Event();release=asyncio.Event();calls=[]
        async def search(ctx,sid,query,deadline):
            calls.append(query.text);started.set();await release.wait();return 'fact '+query.text
        self.plugin._engine.search=search
        first=await asyncio.wait_for(self.plugin.planner_recall(**self.kwargs),1)
        self.assertEqual(first,{'action':'continue'})
        await asyncio.wait_for(started.wait(),1)
        second=await self.plugin.planner_recall(**self.kwargs)
        self.assertEqual(second,{'action':'continue'})
        self.assertEqual(len(calls),1)
        release.set();await asyncio.gather(*self.plugin._batch_tasks.values())
        self.assertEqual(len(calls),2)
        result=await self.plugin.replyer_recall(**{**self.kwargs,'reply_message_id':'2'})
        self.assertIn('modified_kwargs',result)
        self.assertEqual(result['modified_kwargs']['foreign'],{'keep':True})
        self.assertEqual(len(calls),2)
        missing=await self.plugin.replyer_recall(**{**self.kwargs,'reply_message_id':'missing'})
        self.assertEqual(missing,{'action':'continue'})

    async def test_fewer_than_threshold_has_no_search(self):
        self.rows=self.rows[:1]
        async def search(*args):raise AssertionError('not enough messages')
        self.plugin._engine.search=search
        self.assertEqual(await self.plugin.planner_recall(**self.kwargs),{'action':'continue'})
        self.assertEqual(self.plugin._batch_tasks,{})
        self.assertEqual(self.plugin._store.pending('session-a'),1)

    async def test_cancelled_batch_is_attempted_and_not_repeated(self):
        started=asyncio.Event();calls=[]
        async def search(*args):
            calls.append(1);started.set();await asyncio.Event().wait()
        self.plugin._engine.search=search
        await self.plugin.planner_recall(**self.kwargs)
        await started.wait();task=self.plugin._batch_tasks['session-a'];task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        self.assertEqual(self.plugin._store.pending('session-a'),0)
        await self.plugin.planner_recall(**self.kwargs)
        self.assertEqual(len(calls),1)

    async def test_engine_replacement_discards_inflight_result(self):
        started=asyncio.Event();release=asyncio.Event()
        async def search(*args):
            started.set();await release.wait();return 'old result'
        self.plugin._engine.search=search
        await self.plugin.planner_recall(**self.kwargs);await started.wait()
        old=self.plugin._engine
        self.plugin._engine=SimpleNamespace(cfg=self.cfg,documents=SimpleNamespace(signature='new'))
        release.set();await asyncio.gather(*self.plugin._batch_tasks.values())
        self.assertEqual(self.plugin._store.result_for('session-a','2',self.plugin._version(old)),[])

if __name__=='__main__':unittest.main()
