import unittest
import os
from pathlib import Path
import numpy as np
from fruitflybrain.graph import Graph,fixture
from fruitflybrain.model import Reference,Parameters

class SolverTests(unittest.TestCase):
    def network(self,weight=8):
        return Graph.edges([100,200],[0],[1],[weight])

    def test_zero_input_and_reset(self):
        m=Reference(fixture());a=np.zeros((30,8),np.float32)
        r=m.run(a,True);self.assertEqual(int(r['counts'].sum()),0);np.testing.assert_array_equal(r['voltage'],-52)
        a[:,0]=10;m.run(a);m.reset();self.assertEqual(m.tick,0);np.testing.assert_array_equal(m.run(np.zeros_like(a))['voltage'],-52)

    def test_causal_edge_delay(self):
        m=Reference(self.network());a=np.zeros((6,2),np.float32);a[0,0]=8
        z=m.run(a,True)['spikes'];self.assertEqual(z[0,0],1);self.assertEqual(z[:2,1].sum(),0);self.assertEqual(z[2,1],1)

    def test_inhibition(self):
        m=Reference(self.network(-8));a=np.zeros((4,2),np.float32);a[0,0]=8
        r=m.run(a,True);self.assertLess(r['voltages'][2,1],-52);self.assertEqual(r['counts'][1],0)

    def test_refractory(self):
        g=Graph.edges([1],[],[],[]);r=Reference(g).run(np.full((10,1),100,np.float32),True)
        np.testing.assert_array_equal(np.flatnonzero(r['spikes'][:,0]),[0,3,6,9])

    def test_chunk_state_equivalence(self):
        a=np.zeros((50,8),np.float32);a[:,0]=1.0
        m=Reference(fixture());whole=m.run(a,True)
        m.reset();parts=[m.run(x,True) for x in np.array_split(a,5)]
        np.testing.assert_array_equal(whole['spikes'],np.concatenate([r['spikes'] for r in parts]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):Parameters(delay_ms=.1).validate()
        with self.assertRaises(ValueError):Reference(fixture()).run(np.full((1,8),np.nan))
        with self.assertRaises(ValueError):Graph.edges([1,1],[],[],[])
        with self.assertRaises(ValueError):Graph.edges([1.0,2.0],[],[],[])

    @unittest.skipUnless(os.environ.get('FFB_TEST_CUDA')=='1','Set FFB_TEST_CUDA=1 on a GPU host')
    def test_cuda_matches_full_reference_trajectories(self):
        from fruitflybrain.cuda import Cuda
        rng=np.random.default_rng(1)
        graphs=[fixture(),self.network(),self.network(-8),Graph.edges([1],[],[],[]),Graph.edges(np.arange(20),rng.integers(0,20,120),rng.integers(0,20,120),rng.uniform(-2,2,120))]
        for g in graphs:
            for delay in (1,2,3):
                p=Parameters(delay_ms=delay)
                drive=rng.uniform(0,.7,(100,g.n)).astype(np.float32);drive[0,0]=8
                want=Reference(g,p).run(drive,True)
                with Cuda(g,p) as m:
                    got=m.run(drive,True)
                    np.testing.assert_array_equal(got['spikes'],want['spikes'])
                    np.testing.assert_allclose(got['voltages'],want['voltages'],rtol=0,atol=2e-4)
                    m.reset();parts=[m.run(x,True) for x in np.array_split(drive,5)]
                    np.testing.assert_array_equal(got['spikes'],np.concatenate([r['spikes'] for r in parts]))

if __name__=='__main__':unittest.main()
