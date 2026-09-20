"""Single-owner experiment worker, complete records and frozen evaluation."""
import copy,json,queue,threading,time,uuid,re,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .core import default_config,validate_config,make_maze,patterns,epsilon,digest,World,Learner,rollout,summarize
from .encoder import NeuralEncoder
from ..provenance import atomic_json,source_manifest,preserve_source,environment,sha256,ROOT

def utc():return datetime.now(timezone.utc).isoformat()
def policy_rng(seed,episode):return seed*100003+episode*31337+11

class Session:
    def __init__(self,graph,runs,backend,encoder=None):
        self.graph_path=Path(graph);self.runs=Path(runs);self.backend=backend;self.encoder=encoder
        self.lock=threading.Lock();self.queue=queue.Queue(maxsize=32)
        self.state={'status':'loading','message':'Measuring the two fixed neural pattern probes...'}
        self.thread=threading.Thread(target=self._worker,daemon=True);self.thread.start()
    def snapshot(self):
        with self.lock:return copy.deepcopy(self.state)
    def command(self,value):
        if not isinstance(value,dict):raise ValueError('Expected command object')
        if self.snapshot()['status'] in ('loading','error') and value.get('action')!='shutdown':raise ValueError('Experiment is not ready')
        done=threading.Event();result={}
        try:self.queue.put_nowait((value,done,result))
        except queue.Full:raise ValueError('Command queue busy')
        if not done.wait(20):raise TimeoutError('Command acknowledgement timed out; inspect state before retrying')
        if 'error' in result:raise ValueError(result['error'])
        return result
    def close(self):
        if self.thread.is_alive():
            try:self.command({'action':'shutdown'})
            except (ValueError,TimeoutError):pass
            self.thread.join(25)
    def _worker(self):
        self.run_dir=None;self.log=None;self.episodes_file=None;self.remaining=0;self.evaluation=None;self.speed=100;self.last_publish=0.;self.next_step=0.
        try:
            if self.encoder is None:self.encoder=NeuralEncoder(self.graph_path,self.backend)
            self.source=source_manifest();self.env=environment();self._new(default_config());self._publish()
            while True:
                try:item=self.queue.get(timeout=0 if self.remaining or self.evaluation else .15)
                except queue.Empty:item=None
                if item:
                    v,done,result=item
                    try:
                        if v.get('action')=='shutdown':self._close_run('service_shutdown');result['ok']=True;done.set();break
                        self._execute(v);self._publish();result.update(ok=True,run_id=self.run_dir.name)
                    except (ValueError,TypeError,KeyError,OSError) as e:result['error']=str(e)
                    finally:done.set()
                    self._publish();continue
                now=time.monotonic()
                if self.evaluation:self._eval_one()
                elif self.remaining and now>=self.next_step:self._step();self.next_step=now+1/self.speed
                else:time.sleep(.002)
                if time.monotonic()-self.last_publish>.09:self._publish()
        except BaseException as e:
            try:self._close_run('failed',str(e))
            except Exception:pass
            with self.lock:self.state={'status':'error','message':f'{type(e).__name__}: {e}'}
    def _new(self,config,restore=None,parent=None):
        c=validate_config(config);snapshot=preserve_source(self.runs,self.source)
        self._close_run('new_experiment')
        self.c=c;self.neural=Learner(c,restore[0] if restore else None);self.image=Learner(c,restore[1] if restore else None)
        self.completed=int(restore[2]) if restore else 0;self.history=[];self.evaluations=[];self.remaining=0;self.evaluation=None;self.manual_trials=0
        self.initial_completed=self.completed;self.message='Ready. Train the external controller; neural weights stay fixed.'
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-exp2-'+uuid.uuid4().hex[:6]
        self.run_dir=self.runs/name;self.run_dir.mkdir(parents=True)
        self.log=(self.run_dir/'commands.jsonl').open('w',buffering=1);self.episodes_file=(self.run_dir/'episodes.jsonl').open('w',buffering=1)
        atomic_json(self.run_dir/'config.json',c);atomic_json(self.run_dir/'source_manifest.json',self.source);self.encoder.save(self.run_dir)
        self.manifest={'run_id':name,'experiment':'exp_2','status':'open','created_utc':utc(),'source_sha256':self.source['sha256'],
            'source_snapshot':snapshot,'encoder_sha256':self.encoder.identity,'dataset':self.encoder.meta['dataset'],'backend':self.encoder.backend,
            'environment':self.env,'parent_checkpoint':parent,'initial_completed_episodes':self.completed,
            'learning':'External tabular Q-learning; fixed connectome dynamics and cached calibrated probes',
            'observation':'Grid position, legal neighboring moves, two globally visible pattern signals; no reward label or path oracle',
            'config':c,'protocol':'docs/research/exp_2.json v0.1'}
        if self.backend=='cuda':self.manifest['native_library_sha256']=sha256(ROOT/'build/libffb.so')
        atomic_json(self.run_dir/'manifest.json',self.manifest);self._reset_trial();self._checkpoint()
    def _close_run(self,reason,error=None):
        if not self.run_dir:return
        if self.evaluation:self._end_eval(cancelled=True)
        self._checkpoint();self.manifest.update(status='failed' if error else 'closed',closed_utc=utc(),reason=reason,error=error)
        self.log.close();self.episodes_file.close()
        self.manifest['outputs']={p.name:sha256(p) for p in self.run_dir.iterdir() if p.is_file() and p.name!='manifest.json'}
        atomic_json(self.run_dir/'manifest.json',self.manifest);self.run_dir=None
    def _reset_trial(self):
        self.world=World(self.c,patterns(self.c['seed'],self.completed));self.mode='training'
        self.rng=np.random.default_rng(policy_rng(self.c['seed'],self.completed));self.start_q=self.neural.q.copy();self.last_action=None;self.td=0.
    def _checkpoint(self):
        if not self.run_dir:return
        # Checkpoints represent completed-episode boundaries, not partial updates.
        q=self.start_q if self.mode=='training' and self.world.steps and not self.world.done else self.neural.q
        meta={'config':self.c,'completed':self.completed,'encoder_sha256':self.encoder.identity,
              'source_sha256':self.source['sha256'],'run_id':self.run_dir.name,'boundary':'completed episodes; partial trajectory excluded'}
        tmp=self.run_dir/'checkpoint.tmp'
        with tmp.open('wb') as f:np.savez_compressed(f,neural=q,image=self.image.q,metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8))
        tmp.replace(self.run_dir/'checkpoint.npz');atomic_json(self.run_dir/'evaluations.json',self.evaluations)
        atomic_json(self.run_dir/'training_summary.json',{'completed':self.completed,'manual_trials':self.manual_trials,'history':self.history,
                    'q_sha256':digest(q),'image_q_sha256':digest(self.image.q)})
    def _execute(self,v):
        action=v.get('action')
        if self.log:self.log.write(json.dumps({'utc':utc(),'episode':self.completed,'step':self.world.steps,'command':v},allow_nan=False)+'\n')
        if action=='train':
            if self.evaluation:raise ValueError('Pause evaluation first')
            n=v.get('episodes',100)
            if type(n) is not int or not 1<=n<=5000:raise ValueError('Train 1-5000 episodes per request')
            if self.mode=='manual':raise ValueError('Reset trial before training after manual movement')
            self.remaining=min(5000,self.remaining+n);self.message='Training external Q values. Pattern assignments swap between trials.'
        elif action=='pause':
            self.remaining=0
            if self.evaluation:self._end_eval(cancelled=True)
            self._checkpoint();self.message='Paused. Q values and the current trial are retained.'
        elif action=='speed':
            n=float(v.get('value',100))
            if not np.isfinite(n) or not 1<=n<=2000:raise ValueError('Speed must be 1-2000 actions/s')
            self.speed=n
        elif action=='step':
            if self.evaluation or self.remaining:raise ValueError('Pause before single stepping')
            if self.mode=='manual':raise ValueError('Reset trial before a learning step')
            self._step()
        elif action=='manual':
            if self.evaluation or self.remaining:raise ValueError('Pause before manual movement')
            if self.mode=='training' and self.world.steps:raise ValueError('Reset trial before switching to manual movement')
            self.mode='manual';self.last_action=v.get('direction');self.world.step(self.last_action)
            if self.world.done:
                self.manual_trials+=1;self.episodes_file.write(json.dumps({'mode':'manual',**self.world.summary()})+'\n');self._checkpoint()
            self.message='Manual intervention: no Q updates and no training metric.'
        elif action=='reset_trial':
            if self.evaluation or self.remaining:raise ValueError('Pause first')
            if self.mode=='training' and self.world.steps and not self.world.done:self.neural.q[:]=self.start_q
            self._reset_trial();self._checkpoint();self.message='Fresh trial; completed-episode learning retained. Incomplete updates rolled back.'
        elif action=='new':self._new(v.get('config',default_config()))
        elif action=='save':self._checkpoint();self.message='Saved completed-episode checkpoint. Partial trial state is excluded.'
        elif action=='load':
            if self.remaining or self.evaluation:raise ValueError('Pause before restoring a checkpoint')
            run=str(v.get('run_id',''))
            if not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Invalid run ID')
            path=self.runs/run/'checkpoint.npz'
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(z['metadata'].tobytes());q=z['neural'];b=z['image']
                if meta['encoder_sha256']!=self.encoder.identity:raise ValueError('Checkpoint encoder/dataset/backend differs from this service')
                config=validate_config(meta['config']);Learner(config,q);Learner(config,b)
                count=meta['completed']
                if type(count) is not int or count<0:raise ValueError('Invalid completed episode count')
                parent={'run_id':run,'sha256':sha256(path)}
                summary_path=self.runs/run/'training_summary.json';eval_path=self.runs/run/'evaluations.json'
                history=json.loads(summary_path.read_text()).get('history',[]) if summary_path.exists() else []
                evaluations=json.loads(eval_path.read_text()) if eval_path.exists() else []
                self._new(config,(q.copy(),b.copy(),count),parent)
                self.history=[row for row in history if row.get('episode',count+1)<=count]
                self.evaluations=evaluations
                self.message='Restored checkpoint and recorded history into a new linked experiment. Fresh trial; neural weights remain fixed.'
                self._checkpoint()
        elif action=='evaluate':
            if self.remaining or self.evaluation:raise ValueError('Pause training before evaluation')
            n=v.get('episodes',30)
            if type(n) is not int or not 4<=n<=100 or n%2:raise ValueError('Evaluation requires an even 4-100 episodes per condition')
            self._checkpoint();q=self.start_q if self.mode=='training' and self.world.steps and not self.world.done else self.neural.q
            self.evaluation={'n':n,'index':0,'rows':[],'neural':Learner(self.c,q),'image':Learner(self.c,self.image.q),
                             'hashes':[digest(q),digest(self.image.q)],'live_hashes':[digest(self.neural.q),digest(self.image.q)],'started_utc':utc()}
            self.message='Frozen-policy evaluation: familiar and unseen mazes, three agents.'
        else:raise ValueError('Unknown action')
    def _step(self):
        w=self.world;ep=self.completed;tokens=self.encoder.tokens(w.patterns);s=self.neural.state(w.pos,tokens)
        a=self.neural.choose(s,w.legal(),self.rng,epsilon(ep));r=w.step(a)
        self.td=self.neural.update(s,a,r,self.neural.state(w.pos,tokens),w.legal(),w.done);self.last_action=a
        if w.done:
            b=rollout(self.c,w.patterns,self.image,self.encoder.tokens(w.patterns,'image'),policy_rng(self.c['seed'],ep),epsilon(ep),learn=True)
            row={'episode':ep+1,'epsilon':epsilon(ep),'mode':'training',**w.summary(),'path':copy.deepcopy(w.path),'image_control':b}
            self.history.append(row);self.episodes_file.write(json.dumps(row)+'\n');self.completed+=1;self.remaining=max(0,self.remaining-1)
            self.message=f'Trial {self.completed}: {w.outcome}, {w.steps} actions. Checkpoint saved.'
            self._reset_trial();self._checkpoint()
    def _eval_one(self):
        e=self.evaluation;i=e['index'];n=e['n'];group=i//(3*n);trial=(i%(3*n))//3;agent=('neural','image','random')[i%3]
        namespace=1_000_000_000+self.c['seed']*10000
        config=copy.deepcopy(self.c)
        if group:config['layout']=make_maze(namespace+trial//2)
        assignment=patterns(namespace,trial);tokens=self.encoder.tokens(assignment,'image' if agent=='image' else 'neural')
        policy=e['image'] if agent=='image' else e['neural']
        row=rollout(config,assignment,policy,tokens,namespace+trial*17,random=agent=='random')
        e['rows'].append({'environment':'unseen' if group else 'familiar','agent':agent,'trial':trial,'seed':namespace+trial*17,'layout':config['layout'],**row});e['index']+=1
        if e['index']==6*n:self._end_eval()
    def _end_eval(self,cancelled=False):
        e=self.evaluation
        assert e['hashes']==[digest(e['neural'].q),digest(e['image'].q)],'Evaluation mutated frozen weights'
        assert e['live_hashes']==[digest(self.neural.q),digest(self.image.q)],'Evaluation mutated training weights'
        groups={}
        for env in ('familiar','unseen'):
            groups[env]={agent:summarize([r for r in e['rows'] if r['agent']==agent and r['environment']==env]) for agent in ('neural','image','random')}
        self.evaluations.append({'started_utc':e['started_utc'],'finished_utc':utc(),'cancelled':cancelled,'completed_training_episodes':self.completed,
            'policy_sha256':e['hashes'],'frozen_verified':True,'seed_namespace':1_000_000_000+self.c['seed']*10000,'summary':groups,'trials':e['rows']})
        self.evaluation=None;self.message='Evaluation saved. Frozen weights verified; familiar and unseen scores are separate.';self._checkpoint()
    def _publish(self):
        self.last_publish=time.monotonic();tokens=self.encoder.tokens(self.world.patterns);s=self.neural.state(self.world.pos,tokens)
        latest=self.evaluations[-1] if self.evaluations else None
        data={'status':'evaluating' if self.evaluation else 'training' if self.remaining else 'paused','message':self.message,
            'run_id':self.run_dir.name,'parent_checkpoint':self.manifest.get('parent_checkpoint'),'config':self.c,'encoder':self.encoder.public(),'completed':self.completed,'initial_completed':self.initial_completed,
            'remaining_episodes':self.remaining,'epsilon':epsilon(self.completed),'speed':self.speed,'mode':self.mode,'world':self.world.public(),
            'action_values':self.neural.q[s].tolist(),'last_action':self.last_action,'td_error':self.td,'tokens':list(tokens),
            'training_summary':summarize(self.history[-20:]),'history':self.history[-500:],
            'evaluation_progress':{'done':self.evaluation['index'],'total':self.evaluation['n']*6} if self.evaluation else None,
            'evaluation':{k:v for k,v in latest.items() if k!='trials'} if latest else None}
        with self.lock:self.state=copy.deepcopy(data)
