"""Persistent bounded message accounting; no full chat history is indexed."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from .core import Query, message_text

class BatchStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS messages (
                  sid TEXT NOT NULL, mid TEXT NOT NULL, stamp REAL NOT NULL,
                  text TEXT NOT NULL, state INTEGER NOT NULL DEFAULT 0,
                  batch TEXT NOT NULL DEFAULT '', PRIMARY KEY(sid,mid));
                CREATE TABLE IF NOT EXISTS batches (
                  id TEXT PRIMARY KEY, sid TEXT NOT NULL, created REAL NOT NULL,
                  version TEXT NOT NULL, result TEXT NOT NULL DEFAULT '[]');
                CREATE INDEX IF NOT EXISTS pending ON messages(sid,state,stamp);
            ''')
            # An interrupted batch is an attempted query, never silently retried.
            db.execute('UPDATE messages SET state=2,text="" WHERE state=1')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path, timeout=2)
        try:
            with db:
                yield db
        finally:
            db.close()

    def observe(self, sid: str, messages: list, *, max_age: float, initial_limit: int) -> int:
        valid=[]
        for msg in messages:
            text=message_text(msg,sid,max_age=max_age)
            if text:
                valid.append((sid,str(msg['message_id']),float(msg['timestamp']),text))
        valid.sort(key=lambda row:(row[2],row[1]))
        with self.connect() as db:
            first=db.execute('SELECT 1 FROM messages WHERE sid=? LIMIT 1',(sid,)).fetchone() is None
            cutoff=max(0,len(valid)-initial_limit) if first else 0
            for i,row in enumerate(valid):
                db.execute('INSERT OR IGNORE INTO messages(sid,mid,stamp,text,state) VALUES(?,?,?,?,?)',
                           (*row[:3],row[3] if i>=cutoff else '',0 if i>=cutoff else 2))
            return db.execute('SELECT count(*) FROM messages WHERE sid=? AND state=0',(sid,)).fetchone()[0]

    def claim(self,sid:str,size:int,version:str) -> tuple[str,list[tuple[str,str]]]:
        import uuid
        with self.connect() as db:
            rows=db.execute('SELECT mid,text FROM messages WHERE sid=? AND state=0 ORDER BY stamp,mid LIMIT ?', (sid,size)).fetchall()
            if len(rows)<size:
                return '',[]
            batch=uuid.uuid4().hex
            db.execute('INSERT INTO batches(id,sid,created,version) VALUES(?,?,?,?)',(batch,sid,time.time(),version))
            db.executemany('UPDATE messages SET state=1,batch=? WHERE sid=? AND mid=?',[(batch,sid,mid) for mid,_ in rows])
            return batch,rows

    def finish(self,batch:str,blocks:list[tuple[Query,str]], per_message:dict|None=None) -> None:
        def encode(rows):
            return [{'label':q.label,'text':q.text,'style':q.style,'document_name':q.document_name,'context':q.context[:4000],'body':body[:5000]} for q,body in rows]
        data=json.dumps({'shared':encode(blocks),'messages':{mid:encode(rows) for mid,rows in (per_message or {}).items()}},ensure_ascii=False)
        with self.connect() as db:
            db.execute('UPDATE batches SET result=? WHERE id=?',(data,batch))
            db.execute('UPDATE messages SET state=2,text="" WHERE batch=?',(batch,))
            # Keep last 500 IDs per active session and only 128 recent sessions.
            db.execute('DELETE FROM messages WHERE state=2 AND rowid IN (SELECT rowid FROM (SELECT rowid,row_number() OVER(PARTITION BY sid ORDER BY stamp DESC) AS n FROM messages) WHERE n>500)')
            db.execute('DELETE FROM messages WHERE sid NOT IN (SELECT sid FROM messages GROUP BY sid ORDER BY max(stamp) DESC LIMIT 128)')
            db.execute('DELETE FROM batches WHERE id NOT IN (SELECT DISTINCT batch FROM messages WHERE batch<>"")')
            db.execute('DELETE FROM batches WHERE id IN (SELECT id FROM batches ORDER BY created DESC LIMIT -1 OFFSET 256)')

    def result_for(self,sid:str,mid:str,version:str) -> list[tuple[Query,str]]:
        with self.connect() as db:
            row=db.execute('SELECT b.result FROM messages m JOIN batches b ON m.batch=b.id WHERE m.sid=? AND m.mid=? AND b.version=? AND m.state=2',(sid,mid,version)).fetchone()
        if not row:
            return []
        try:
            data=json.loads(row[0])
            if not isinstance(data,dict):
                return []
            rows=data.get('messages',{}).get(mid,[])+data.get('shared',[])
            return [(Query(v['label'],v['text'],bool(v['style']),v.get('document_name',''),v.get('context','')),v['body']) for v in rows]
        except (ValueError,KeyError,TypeError):
            return []

    def recent_reference(self,sid:str,version:str,text:str,*,max_age:float=300) -> list[tuple[Query,str]]:
        from .relevance import same_topic, topic_terms
        if max_age <= 0:
            return []
        with self.connect() as db:
            row=db.execute('SELECT result FROM batches WHERE sid=? AND version=? AND result<>? AND created>=? ORDER BY created DESC LIMIT 1',
                           (sid,version,'[]',time.time()-max_age)).fetchone()
        if not row:
            return []
        try:
            data=json.loads(row[0])
            ranked=[]
            for rows in data.get('messages',{}).values():
                for value in rows:
                    original=value.get('context') or value['text']
                    if same_topic(text, original, value['body']):
                        ranked.append((len(topic_terms(text) & topic_terms(original)),value))
            picked=[];seen=set()
            for _,v in sorted(ranked,key=lambda x:x[0],reverse=True):
                if v['body'] in seen:
                    continue
                q=Query(v['label'],v['text'],bool(v['style']),v.get('document_name',''),v.get('context',''))
                if any(a.style == q.style for a,_ in picked):
                    continue
                picked.append((q,v['body']));seen.add(v['body'])
            return picked
        except (ValueError,KeyError,TypeError,AttributeError):
            return []

    def cursor(self,sid:str) -> float | None:
        with self.connect() as db:
            row=db.execute('SELECT max(stamp) FROM messages WHERE sid=?',(sid,)).fetchone()
        return row[0] if row and row[0] is not None else None

    def pending(self,sid:str) -> int:
        with self.connect() as db:
            return db.execute('SELECT count(*) FROM messages WHERE sid=? AND state=0',(sid,)).fetchone()[0]
