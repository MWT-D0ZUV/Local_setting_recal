"""Regression cases for Host 1.3.18 MessageDict.raw_message list payloads."""
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Local_setting_recall.core import message_text
from Local_setting_recall.plugin import SettingRecallPlugin

class HostMessageTests(unittest.TestCase):
    def test_host_list_uses_only_original_text_and_preserves_input(self):
        import copy
        msg = {'session_id':'a', 'message_id':'m', 'timestamp':str(time.time()),
               'raw_message':[{'type':'at','data':{'target_user_id':'bot'}},
                              {'type':'text','data':'C8'},
                              {'type':'reply','data':{'target_message_content':'C5 secret'}},
                              {'type':'forward','data':[{'type':'text','data':'C5 secret'}]},
                              {'type':'image','data':'C5 image'},
                              {'type':'text','data':'blue gate'}],
               'processed_plain_text':'C5 quoted or generated text'}
        before = copy.deepcopy(msg)
        self.assertEqual(message_text(msg,'a'), 'C8 blue gate')
        self.assertEqual(msg,before)
        msg['raw_message'] = msg['raw_message'][2:5]
        self.assertEqual(message_text(msg,'a'),'')

    def test_legacy_forms_and_bounds(self):
        base = {'session_id':'a','message_id':'m','timestamp':str(time.time())}
        for raw in ({'type':'text','data':'C8'}, {'type':'seglist','data':[{'type':'text','data':'C8'}]}, [{'type':'text','data':'C8'}]):
            self.assertEqual(message_text({**base,'raw_message':raw},'a'),'C8')
        raw = {'type':'text','data':'C8'}
        for _ in range(20):
            raw = {'type':'seglist','data':[raw]}
        self.assertEqual(message_text({**base,'raw_message':raw},'a'),'')
        self.assertEqual(len(message_text({**base,'raw_message':[{'type':'text','data':'X'*5000}]},'a')),4000)

    def test_both_request_hooks_are_collected_by_real_sdk(self):
        hooks = [c for c in SettingRecallPlugin().get_components() if c.get('name') in ('setting_recall_planner','setting_recall_replyer')]
        self.assertEqual({c['name'] for c in hooks}, {'setting_recall_planner','setting_recall_replyer'})

if __name__ == '__main__':
    unittest.main()
