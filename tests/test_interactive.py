import json,tempfile,time,unittest,threading,urllib.request,urllib.error
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from fruitflybrain.graph import Graph
from fruitflybrain.model import Reference
from fruitflybrain.stimuli import validate_stimulus,frame,make_drive,DEFAULT

def create_atlas(path):
    import pyarrow as pa,pyarrow.feather as f
    from fruitflybrain.atlas import Atlas
    ids=np.array([100,101,200,201,300,301],np.int64)
    g=Graph.edges(ids,[0,1,2,3],[2,3,4,5],[8,8,4,4],{'dataset':'test'})
    g.save(path)
    rows=[]
    for i,n in enumerate(ids):rows.append({'bodyId':int(n),'type':['L1','L1','T4a','T4a','DNtest','DNtest'][i],
        'somaSide':'L' if i%2==0 else 'R','superclass':'descending_neuron' if i>=4 else 'ol_intrinsic',
        'assignedOlHex1':float(i) if i<2 else None,'assignedOlHex2':float(i) if i<2 else None,
        'somaLocation':[i*10,i*2,i*3],'tosomaLocation':None})
    f.write_feather(pa.Table.from_pylist(rows),path/'neurons.feather')
    return g,Atlas(g,path/'neurons.feather')

class InteractiveTests(unittest.TestCase):
    def setUp(self):
        try:import pyarrow
        except ImportError:self.skipTest('Run on the pinned GPU-host Python environment')
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.graph,self.atlas=create_atlas(self.root/'graph')
    def tearDown(self):
        if hasattr(self,'temp'):self.temp.cleanup()
    def test_eye_specific_image_sampling(self):
        stim=validate_stimulus({'kind':'uniform','eye':'L','gain':.7})
        drive=make_drive(self.atlas,stim,0,3,1,[])
        np.testing.assert_allclose(drive[:,0],.7);np.testing.assert_array_equal(drive[:,1:],0)
        self.assertEqual(self.atlas.public()['mapping']['mapped_neurons'],2)
    def test_pulse_timing_and_sign(self):
        pulse={'start_tick':2,'end_tick':4,'indices':np.array([2]),'amplitude':-1.0}
        drive=make_drive(self.atlas,validate_stimulus(DEFAULT),0,5,1,[pulse])
        np.testing.assert_array_equal(drive[:,2],[0,0,-1,-1,0])
    def test_paint_preview_is_actual_input(self):
        grid=np.zeros(768);grid[0]=1
        s=validate_stimulus({'kind':'paint','paint':grid.tolist(),'gain':1})
        self.assertEqual(frame(s,100)[0,0],1)
        d=make_drive(self.atlas,s,100,1,1,[]);self.assertEqual(d[0,0],1)
    def test_reject_bad_controls(self):
        for s in [{'gain':float('nan')},{'paint':[1]},{'direction':0},{'extra':1}]:
            with self.assertRaises(ValueError):validate_stimulus(s)
    def test_live_state_and_saved_commands(self):
        from fruitflybrain.live import LiveSession
        live=LiveSession(self.root/'graph',self.root/'runs',Reference)
        def wait_for(fn):
            end=time.time()+5
            while time.time()<end:
                state=live.snapshot()
                if state['status']=='error':self.fail(state['message'])
                if fn(state):return state
                time.sleep(.02)
            self.fail('Live worker timeout')
        try:
            wait_for(lambda s:s['status']=='paused')
            live.command({'action':'probe','body_id':'100'})
            live.command({'action':'pulse','group':'L1_L','amplitude':2,'duration_ms':50})
            s=wait_for(lambda s:s['status']=='paused' and s['sim_ms']==150)
            self.assertGreater(sum(x['spikes'] for x in s['history']),0)
            self.assertEqual(s['probe']['id'],'100')
            live.command({'action':'finish'})
            files=list((self.root/'runs').glob('*/manifest.json'));self.assertEqual(len(files),1)
            m=json.loads(files[0].read_text());self.assertEqual(m['status'],'complete')
            events=[json.loads(x) for x in (files[0].parent/'commands.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['command']['action']=='pulse' and e['tick']==0 for e in events))
            live.command({'action':'reset'});self.assertEqual(live.snapshot()['sim_ms'],0)
            live.command({'action':'step','duration_ms':1})
            s=wait_for(lambda s:s['status']=='paused' and s['sim_ms']==1)
            self.assertTrue(np.isfinite(s['point_rates_hz']).all())
            live.command({'action':'play','duration_ms':5000})
            wait_for(lambda s:s['status']=='paused' and s['sim_ms']==5001)
            live.command({'action':'play','duration_ms':4999})
            wait_for(lambda s:s['status']=='complete' and s['sim_ms']==10000)
            self.assertIsNone(live.run_dir)
        finally:live.close()
    def test_post_requires_session_and_same_origin(self):
        from fruitflybrain.server import Handler
        class Fake:
            def command(self,v):return {'ok':True}
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,runs=self.root,live=Fake(),token='secret'))
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            for headers,code in [({},403),({'X-FFB-Token':'secret','Origin':'http://other.invalid'},403),({'X-FFB-Token':'secret','Origin':base},200)]:
                req=urllib.request.Request(base+'/api/control',data=b'{"action":"pause"}',headers={'Content-Type':'application/json',**headers})
                try:r=urllib.request.urlopen(req);status=r.status
                except urllib.error.HTTPError as e:status=e.code
                self.assertEqual(status,code)
        finally:server.shutdown();server.server_close();t.join()
