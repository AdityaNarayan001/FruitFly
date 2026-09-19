"""Single-worker interactive CUDA session with tick-stamped command recording."""
import copy,hashlib,json,queue,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .atlas import Atlas
from .backends import label
from .graph import Graph
from .model import Parameters
from .stimuli import DEFAULT,validate_stimulus,frame,make_drive
from .provenance import preserve_source,ROOT,source_manifest,atomic_json,environment,sha256

class LiveSession:
    def __init__(self,graph_path,runs,model_factory=None,backend="cuda"):
        self.graph_path=Path(graph_path);self.runs=Path(runs);self.factory=model_factory;self.backend=backend
        self.lock=threading.Lock();self.commands=queue.Queue(maxsize=32);self.atlas=None;self.model=None
        self.state={'status':'loading','message':'Loading graph and annotations','sim_ms':0,'history':[]}
        self.thread=threading.Thread(target=self._worker,daemon=True,name='fruitfly-cuda');self.thread.start()
    def snapshot(self):
        with self.lock:return copy.deepcopy(self.state)
    def command(self,value):
        if not isinstance(value,dict):raise ValueError('Expected a command object')
        if self.snapshot()['status'] in ('loading','error'):raise ValueError('Simulator is not ready')
        done=threading.Event();result={}
        try:self.commands.put_nowait((value,done,result))
        except queue.Full:raise ValueError('Command queue is busy')
        if not done.wait(10):raise TimeoutError('Command acknowledgement timed out')
        if 'error' in result:raise ValueError(result['error'])
        return result
    def close(self):
        if self.thread.is_alive():
            done=threading.Event();self.commands.put(({'action':'_shutdown'},done,{}));done.wait(10);self.thread.join(10)
    def _worker(self):
        self.tick=0;self.budget=0;self.stim=validate_stimulus(DEFAULT);self.pulses=[];self.history=[];self.raster=[]
        self.run_dir=None;self.events=None;self.frames=None;self.last=None;self.probe=None;self.finished_id=None;self.chunk_size=20
        try:
            self.graph=Graph.load(self.graph_path);self.atlas=Atlas(self.graph,self.graph_path/'neurons.feather');self.params=Parameters()
            if self.factory is None:
                from .backends import factory
                self.factory=factory(self.backend)
            self.model=self.factory(self.graph,self.params);self.source=source_manifest()
            deployed=ROOT/'SOURCE_MANIFEST.json'
            if deployed.exists() and json.loads(deployed.read_text())['sha256']!=self.source['sha256']:raise ValueError('Source manifest mismatch')
            self.env=environment();self.last_counts=np.zeros(self.graph.n,np.int32);self.last_voltage=np.full(self.graph.n,-52,np.float32)
            self.total_counts=np.zeros(self.graph.n,np.int64);self._publish('paused')
            while True:
                try:item=self.commands.get(timeout=0 if self.budget else .25)
                except queue.Empty:item=None
                if item:
                    value,done,result=item
                    try:
                        if value.get('action')=='_shutdown':self._finish('service_shutdown');result['ok']=True;done.set();break
                        self._execute(value);result.update(ok=True,applied_at_ms=self.tick,run_id=self.run_dir.name if self.run_dir else None)
                    except (ValueError,TypeError,KeyError) as e:result['error']=str(e)
                    finally:done.set()
                    continue
                if self.budget:
                    size=min(20,self.budget,10000-self.tick)
                    if size<=0:self._finish('session_limit');self._publish('complete');continue
                    self._advance(size);self.budget-=size
                    if self.tick>=10000:self._finish('session_limit');self._publish('complete')
                    else:self._publish('running' if self.budget else 'paused')
        except BaseException as e:
            if self.run_dir:
                try:self._finish('error',str(e))
                except Exception:pass
            with self.lock:self.state.update(status='error',message=f'{type(e).__name__}: {e}')
        finally:
            if self.model and hasattr(self.model,'close'):self.model.close()
    def _new_run(self):
        source_snapshot=preserve_source(self.runs,self.source)
        self.model.reset();self.tick=0;self.pulses=[];self.history=[];self.raster=[];self.last=None
        self.last_counts.fill(0);self.last_voltage.fill(self.params.rest_mv);self.total_counts.fill(0)
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-interactive-'+uuid.uuid4().hex[:6]
        self.run_dir=self.runs/name;self.run_dir.mkdir(parents=True,exist_ok=False)
        self.events=open(self.run_dir/'commands.jsonl','w',buffering=1);self.frames=open(self.run_dir/'frames.jsonl','w',buffering=1)
        self.input_hash=hashlib.sha256();self.compute_s=0
        self.manifest={'run_id':name,'status':'running','mode':'interactive_exploration','interpretation':'Synthetic teaching fixture; not MaleCNS' if self.atlas.synthetic else 'Exploratory L1 column drive and direct neural pulses; not validated fly vision or learning',
            'start_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':self.source['sha256'],
            'source_snapshot':source_snapshot,
            'graph':self.graph.metadata,'n_neurons':self.graph.n,'n_edges':self.graph.m,'environment':self.env,
            'backend':label(self.factory),'seed':9000,
            'mapping':self.atlas.mapping,'initial_probe_id':str(self.graph.ids[self.probe]) if self.probe is not None else None,'initial_stimulus':copy.deepcopy(self.stim),'parameters':vars(self.params),
            'groups':{k:self.graph.ids[v].tolist() for k,v in self.atlas.groups.items()},
            'display_sample_ids':self.graph.ids[self.atlas.display_ids].tolist(),'raster_ids':self.graph.ids[self.atlas.raster_ids].tolist(),
            'recording':'All-neuron total counts, 20 ms population summaries, fixed sampled spike raster, final sampled activity and command log',
            'learning':False,'max_session_ms':10000}
        library=ROOT/'build/libffb.so'
        if library.exists() and self.factory.__name__=='Cuda':self.manifest['native_library_sha256']=sha256(library)
        atomic_json(self.run_dir/'source_manifest.json',self.source);atomic_json(self.run_dir/'manifest.json',self.manifest)
    def _ensure_run(self):
        if self.run_dir is None:self._new_run()
    def _log(self,value):
        if self.events:self.events.write(json.dumps({'tick':self.tick,'sim_ms':self.tick,'utc':datetime.now(timezone.utc).isoformat(),'command':value},allow_nan=False)+'\n')
    def _execute(self,v):
        action=v.get('action')
        if action=='stimulus':
            self.stim=validate_stimulus(v.get('stimulus',{}));self._log({'action':'stimulus','stimulus':self.stim})
        elif action in ('play','step'):
            duration=float(v.get('duration_ms',1000 if action=='play' else 20))
            if not np.isfinite(duration) or not 1<=duration<=5000 or duration!=int(duration):raise ValueError('Duration must be 1-5000 whole ms')
            self._ensure_run();self.budget=int(min(duration,10000-self.tick));self._log({'action':action,'duration_ms':self.budget})
        elif action=='pause':self.budget=0;self._log({'action':'pause'})
        elif action=='pulse':
            amplitude=float(v.get('amplitude',.9));duration=float(v.get('duration_ms',100))
            if not np.isfinite([amplitude,duration]).all() or not -2<=amplitude<=2 or not 1<=duration<=500 or duration!=int(duration):raise ValueError('Pulse range: -2 to 2 mV/tick, 1-500 whole ms')
            if 'body_id' in v:
                info=self.atlas.neuron(v['body_id']);ids=np.array([info['index']],np.int32);target={'body_id':info['id']}
            else:
                key=v.get('group')
                if key not in self.atlas.groups:raise ValueError('Select a listed neuron group')
                ids=self.atlas.groups[key];target={'group':key}
            self._ensure_run();self.pulses.append({'start_tick':self.tick,'end_tick':self.tick+int(duration),'amplitude':amplitude,'indices':ids,**target})
            self.budget=max(self.budget,min(int(duration)+100,10000-self.tick))
            self._log({'action':'pulse','amplitude':amplitude,'duration_ms':int(duration),'target_ids':self.graph.ids[ids].tolist(),**target})
        elif action=='probe':self.probe=self.atlas.neuron(v['body_id'])['index'];self._log({'action':'probe','body_id':str(v['body_id'])})
        elif action=='reset':
            self._log({'action':'reset'});self._finish('reset');self.model.reset();self.tick=0;self.budget=0;self.stim=validate_stimulus(DEFAULT)
            self.pulses=[];self.history=[];self.raster=[];self.last=None;self.finished_id=None;self.last_counts.fill(0);self.last_voltage.fill(-52)
        elif action=='finish':self._log({'action':'finish'});self._finish('user_finished')
        else:raise ValueError('Unknown action')
        self._publish('running' if self.budget else 'paused')
    def _advance(self,size):
        self.chunk_size=size;drive=make_drive(self.atlas,self.stim,self.tick,size,1,self.pulses);self.input_hash.update(drive.tobytes())
        start=time.perf_counter();out=self.model.run(drive,record=True);self.compute_s+=time.perf_counter()-start
        self.last_counts=out['counts'];self.last_voltage=out['voltage'];self.total_counts+=out['counts'];rates=out['counts']/(size/1000)
        groups={k:float(rates[v].mean()) for k,v in self.atlas.groups.items()}
        regions=[float(rates[self.atlas.region==i].mean()) if np.any(self.atlas.region==i) else 0.0 for i in range(6)]
        t,neuron=np.nonzero(out['spikes'][:,self.atlas.raster_ids]);events=np.column_stack((t+self.tick,neuron)).tolist()
        self.raster.extend(events);self.raster=[e for e in self.raster if e[0]>=self.tick+size-1000][-6000:]
        self.tick+=size;self.pulses=[p for p in self.pulses if p['end_tick']>self.tick]
        self.last={'sim_ms':self.tick,'mean_rate_hz':float(rates.mean()),'active_fraction':float(np.mean(rates>0)),
            'spikes':int(out['counts'].sum()),'groups':groups,'regions':regions,'mean_voltage_mv':float(out['voltage'].mean()),
            'input_mean_mv':float(drive.mean()),'probe_voltage_mv':float(out['voltage'][self.probe]) if self.probe is not None else None}
        self.history.append(self.last);self.history=self.history[-250:];self.frames.write(json.dumps({**self.last,'raster_events':events},allow_nan=False)+'\n')
    def _publish(self,status,expose=True):
        sample={'backend':label(self.factory),'dataset':self.graph.metadata.get('dataset','unknown'),'synthetic':self.atlas.synthetic,'status':status,'message':'Exploratory controls enabled','sim_ms':self.tick,'remaining_ms':self.budget,
            'run_id':self.run_dir.name if self.run_dir else self.finished_id,'stimulus':self.stim,
            'preview':frame(self.stim,max(0,self.tick-1)).round(4).ravel().tolist(),'history':self.history,'raster':self.raster,
            'latest':self.last,'point_rates_hz':(self.last_counts[self.atlas.display_ids]/(self.chunk_size/1000)).tolist(),
            'pulse_targets':[{k:v for k,v in p.items() if k!='indices'} for p in self.pulses],
            'probe':self._probe(),'mapping_status':self.atlas.mapping['status']}
        if expose:
            with self.lock:self.state=copy.deepcopy(sample)
        if self.run_dir:atomic_json(self.run_dir/'telemetry.json',sample)
    def _probe(self):
        if self.probe is None:return None
        return {**self.atlas.neuron(int(self.graph.ids[self.probe])),'voltage_mv':float(self.last_voltage[self.probe]),'chunk_spikes':int(self.last_counts[self.probe])}
    def _finish(self,reason,error=None):
        self.budget=0
        if not self.run_dir:return
        self._publish('failed' if error else 'complete',expose=False);np.save(self.run_dir/'spike_counts.npy',self.total_counts,allow_pickle=False)
        self.events.close();self.frames.close();self.events=self.frames=None
        atomic_json(self.run_dir/'metrics.json',{'simulated_ms':self.tick,'compute_seconds':self.compute_s,'total_spikes':int(self.total_counts.sum()),'input_sha256':self.input_hash.hexdigest(),'reason':reason,'learning':False})
        self.manifest.update(status='failed' if error else 'complete',end_utc=datetime.now(timezone.utc).isoformat(),end_reason=reason)
        if error:self.manifest['error']=error
        self.manifest['outputs']={p.name:sha256(p) for p in self.run_dir.iterdir() if p.is_file() and p.name!='manifest.json'}
        atomic_json(self.run_dir/'manifest.json',self.manifest);self.finished_id=self.run_dir.name;self.run_dir=None;self._publish('failed' if error else 'complete')
