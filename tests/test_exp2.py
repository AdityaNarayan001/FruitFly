import copy,json,tempfile,time,unittest
from pathlib import Path
import numpy as np
from fruitflybrain.exp2.core import *
from fruitflybrain.exp2.encoder import NeuralEncoder
from fruitflybrain.exp2.session import Session,policy_rng
from fruitflybrain.demo import prepare

class Exp2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=prepare(Path(cls.tmp.name)/'graph');cls.enc=NeuralEncoder(cls.path,'cpu')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_invalid_arena_and_reachable_sites(self):
        c=default_config();validate_config(c)
        for seed in range(30):validate_config({**c,'layout':make_maze(seed)})
        for point in ((0,0),(1,1)):
            bad=copy.deepcopy(c);bad['layout']['start']=list(point)
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(c);bad['layout']['walls'][2][1]=1;bad['layout']['walls'][1][2]=1
        with self.assertRaises(ValueError):validate_config(bad)
    def test_rewards_swaps_and_terminal_target(self):
        c=default_config();w=World(c,('A','B'));w.pos=[1,2];self.assertEqual(w.step(0),1);self.assertTrue(w.done)
        w=World(c,('B','A'));w.pos=[1,2];self.assertEqual(w.step(0),-.25)
        for i in range(0,20,2):self.assertNotEqual(patterns(42,i),patterns(42,i+1))
        q=Learner(c);s=q.state([1,2],(0,1));q.q[:]=100;q.update(s,0,1,s,[0],True)
        self.assertAlmostEqual(q.q[s][0],75.25)
    def test_neural_probes_and_image_control(self):
        self.assertTrue(self.enc.meta['synthetic']);self.assertGreater(self.enc.meta['template_distance'],0)
        for assignment in (('A','B'),('B','A')):self.assertEqual(self.enc.tokens(assignment),self.enc.tokens(assignment,'image'))
        self.assertNotEqual(self.enc.records['A']['image_sha256'],self.enc.records['B']['image_sha256'])
    def test_learning_reward_reversal_and_frozen_evaluation(self):
        for target in ('A','B'):
            c=default_config();c['rewarded_pattern']=target;a=Learner(c);b=Learner(c)
            for ep in range(400):
                p=patterns(c['seed'],ep)
                for policy,mode in ((a,'neural'),(b,'image')):rollout(c,p,policy,self.enc.tokens(p,mode),policy_rng(c['seed'],ep),epsilon(ep),learn=True)
            np.testing.assert_array_equal(a.q,b.q);before=digest(a.q)
            rows=[rollout(c,patterns(1000000000,j),a,self.enc.tokens(patterns(1000000000,j)),1000000000+j*17) for j in range(30)]
            randoms=[rollout(c,patterns(1000000000,j),a,self.enc.tokens(patterns(1000000000,j)),1000000000+j*17,random=True) for j in range(30)]
            self.assertEqual(digest(a.q),before);self.assertEqual(summarize(rows)['success_rate'],1.)
            self.assertGreater(summarize(rows)['success_rate'],summarize(randoms)['success_rate']);self.assertEqual(summarize(rows)['mean_success_efficiency'],1.)
    def wait(self,session,predicate,timeout=10):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            s=session.snapshot()
            if s['status']=='error':self.fail(s['message'])
            if predicate(s):return s
            time.sleep(.01)
        self.fail('Timed out waiting for state')
    def test_session_manual_checkpoint_restore_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Session(self.path,tmp,'cpu',encoder=self.enc)
            try:
                self.wait(s,lambda x:x['status']=='paused');before=digest(s.neural.q)
                s.command({'action':'manual','direction':0});self.assertEqual(digest(s.neural.q),before)
                self.assertEqual(s.snapshot()['completed'],0);s.command({'action':'reset_trial'})
                s.command({'action':'speed','value':2000});s.command({'action':'train','episodes':8})
                snap=self.wait(s,lambda x:x.get('completed')==8 and x['status']=='paused')
                run=snap['run_id'];q=s.neural.q.copy();s.command({'action':'evaluate','episodes':4})
                snap=self.wait(s,lambda x:x.get('evaluation') is not None);self.assertTrue(snap['evaluation']['frozen_verified']);np.testing.assert_array_equal(q,s.neural.q)
                s.command({'action':'load','run_id':run});snap=s.snapshot();self.assertNotEqual(snap['run_id'],run);self.assertEqual(snap['completed'],8);self.assertEqual(len(snap['history']),8);self.assertIsNotNone(snap['parent_checkpoint']);np.testing.assert_array_equal(q,s.neural.q)
                with self.assertRaises(ValueError):s.command({'action':'load','run_id':'../../outside'})
                s.command({'action':'step'});s.command({'action':'reset_trial'});np.testing.assert_array_equal(q,s.neural.q)
            finally:s.close()
            manifests=[json.loads(p.read_text()) for p in Path(tmp).glob('*/manifest.json')]
            self.assertTrue(all(m['status']=='closed' for m in manifests))
if __name__=='__main__':unittest.main()
