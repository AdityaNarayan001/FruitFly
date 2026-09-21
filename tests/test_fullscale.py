import os,unittest
import numpy as np
from fruitflybrain.fullscale.model import CPU,CUDA


def fixture(cls=CPU,plastic=True):
    ptr=np.array([0,2,4,6,8],np.int32);idx=np.array([1,3,0,2,0,3,1,2],np.int32)
    w=np.array([.2,-.25,.15,.3,-.3,.15,.1,-.35],np.float32)
    return cls(ptr,idx,w,np.array([0,1,-1,3],np.int32),np.array([1,-1,0,1],np.float32),17,plastic)


class FullScaleTests(unittest.TestCase):
    def test_gradients(self):
        m=fixture();obs=np.linspace(-.8,1,24,dtype=np.float32);m.step(obs);h=m.h.copy();m.step(obs);grad=m.gradients(2)
        numerical=[];eps=2e-3
        for e in range(len(m.w)):
            old=m.w[e];m.w[e]=old+eps;m.h=h.copy();a=m.step(obs)[2]
            m.w[e]=old-eps;m.h=h.copy();b=m.step(obs)[2]
            numerical.append((a-b)/(2*eps));m.w[e]=old
        np.testing.assert_allclose(grad,numerical,rtol=2e-3,atol=2e-6)
    def test_history_preview_reset(self):
        m=fixture();o=np.ones(24,np.float32);q=m.step(o);h=m.h.copy();cached=m.cache;preview=m.step(o*2,preview=True)
        np.testing.assert_array_equal(m.h,h);self.assertIs(m.cache,cached)
        np.testing.assert_allclose(m.step(o*2),preview)
        self.assertFalse(np.allclose(h,m.h));m.reset();np.testing.assert_array_equal(q,m.step(o))
    def test_plastic_and_frozen(self):
        for plastic in (False,True):
            m=fixture(plastic=plastic);initial=m.parameters()
            for i in range(100):m.step(np.ones(24,np.float32));m.learn(i%4,10*(-1)**i)
            self.assertEqual(np.any(m.w!=initial[0]),plastic);self.assertTrue(np.any(m.r!=initial[1]))
            self.assertTrue(np.all(m.w>=np.minimum(0,2*m.base)));self.assertTrue(np.all(m.w<=np.maximum(0,2*m.base)))
            self.assertTrue(np.max(np.abs(m.h))<=1)
    @unittest.skipUnless(os.environ.get('FFB_FULLSCALE_CUDA')=='1','CUDA opt in')
    def test_cuda_matches_reference(self):
        for plastic in (False,True):
            a=fixture(plastic=plastic);b=fixture(CUDA,plastic=plastic);rng=np.random.default_rng(42)
            try:
                for i in range(30):
                    if i%7==0:a.reset();b.reset()
                    o=rng.normal(size=24).astype(np.float32)
                    np.testing.assert_allclose(a.step(o),b.step(o),rtol=2e-4,atol=2e-6)
                    np.testing.assert_allclose(a.step(-o,preview=True),b.step(-o,preview=True),rtol=2e-4,atol=2e-6)
                    target=float(rng.normal());a.learn(i%4,target);b.learn(i%4,target)
                for x,y in zip(a.parameters(),b.parameters()):np.testing.assert_allclose(x,y,rtol=2e-4,atol=2e-6)
                np.testing.assert_allclose(a.h,b.state()[0],rtol=2e-4,atol=2e-6)
                saved=b.parameters();b.set_parameters(*saved);b.reset();a.reset()
                np.testing.assert_allclose(a.step(o),b.step(o),rtol=2e-4,atol=2e-6)
            finally:b.close()

if __name__=='__main__':unittest.main()

class ExperienceTests(unittest.TestCase):
    def test_shared_experience_and_truncation(self):
        from fruitflybrain.fullscale.__main__ import collect
        from fruitflybrain.exp2.core import default_config,World
        from fruitflybrain.comparison.models import observe
        c=default_config();c['max_steps']=20
        a=collect(c,3,500);c['rewarded_pattern']='B';b=collect(c,3,500)
        for key in ('state','action','next_state','terminal','next_legal','reset','truncated'):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(a['truncated'].any());self.assertFalse(np.any(a['terminal']&a['truncated']))
        np.testing.assert_array_equal(a['reset'][1:],(a['terminal']|a['truncated'])[:-1])
        w=World(c,('A','B'));first=observe(w)[0];w.c['rewarded_pattern']='A'
        np.testing.assert_array_equal(first,observe(w)[0])
