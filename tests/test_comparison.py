import copy,unittest
import numpy as np
from fruitflybrain.comparison.models import Network,Table,observe,all_observations,fingerprint
from fruitflybrain.comparison.circuit import rewire
from fruitflybrain.comparison.__main__ import collect,evaluate,paired_interval
from fruitflybrain.exp2.core import World,default_config

class ComparisonTests(unittest.TestCase):
    def weights(self):
        return np.array([[.3,0,.7,0],[0,.4,0,.6],[.6,.4,0,0]],float)
    def batch(self):
        return collect(default_config(),5,64)
    def test_reward_identity_does_not_enter_observations_or_behavior_data(self):
        c=default_config();other=copy.deepcopy(c);other['rewarded_pattern']='B'
        a=collect(c,3,500);b=collect(other,3,500)
        for key in a:
            if key!='reward':np.testing.assert_array_equal(a[key],b[key])
        world=World(c,('A','B'));obs,index,_=observe(world)
        np.testing.assert_array_equal(obs,all_observations()[index])
        world.c['rewarded_pattern']='B';np.testing.assert_array_equal(obs,observe(world)[0])
    def test_degree_preserving_control_and_strength_multisets(self):
        w=self.weights();mask,random,stats=rewire(w>0,w,seed=9)
        self.assertGreater(stats['different_mask_entries'],0)
        np.testing.assert_array_equal(mask.sum(0),(w>0).sum(0));np.testing.assert_array_equal(mask.sum(1),(w>0).sum(1))
        for a,b in zip(w,random):np.testing.assert_array_equal(np.sort(a),np.sort(b))
    def test_masked_gradient_matches_finite_difference(self):
        m=Network(self.weights(),3);ids=np.array([171,702,1301]);actions=np.array([0,1,2]);target=np.array([.5,-.2,.3])
        _,grads=m.gradients(ids,actions,target)
        for array,gradient,index in [(m.w,grads[0],(0,0)),(m.readout,grads[1],(1,1)),(m.bias,grads[2],(2,))]:
            old=array[index];eps=1e-6;array[index]=old+eps;hi=m.gradients(ids,actions,target)[0]
            array[index]=old-eps;lo=m.gradients(ids,actions,target)[0];array[index]=old
            self.assertAlmostEqual(gradient[index],(hi-lo)/(2*eps),places=8)
    def test_frozen_and_plastic_parameters_and_eval_integrity(self):
        frozen=Network(self.weights(),2,False);plastic=Network(self.weights(),2,True);batch=self.batch()
        for _ in range(20):frozen.update(batch);plastic.update(batch)
        np.testing.assert_array_equal(frozen.w,self.weights());self.assertFalse(np.array_equal(plastic.w,self.weights()))
        self.assertTrue((plastic.w[~plastic.mask]==0).all());self.assertTrue((plastic.w>=0).all())
        before=plastic.state_hash();rows=evaluate(plastic,default_config(),17,4)
        self.assertEqual(before,plastic.state_hash());self.assertEqual(len(rows),8)
    def test_terminal_q_target_ignores_next_values(self):
        q=Table();q.q[:]=100
        q.update({'state':np.array([1]),'action':np.array([0]),'reward':np.array([1.]),'next_state':np.array([2]),'terminal':np.array([True]),'next_legal':np.ones((1,4),bool)})
        self.assertEqual(q.q[1,0],75.25)
    def test_target_network_lag_and_refresh(self):
        m=Network(self.weights(),7);before=fingerprint(m.target_w,m.target_readout,m.target_bias)
        for _ in range(99):m.update(self.batch())
        self.assertEqual(before,fingerprint(m.target_w,m.target_readout,m.target_bias))
        m.update(self.batch());np.testing.assert_array_equal(m.target_w,m.w);np.testing.assert_array_equal(m.target_readout,m.readout)
    def test_seed_level_bootstrap_zero_difference(self):
        result=paired_interval([0]*10);self.assertEqual(result['ci95'],[0.,0.]);self.assertEqual(result['seed_units'],10)

if __name__=='__main__':unittest.main()
