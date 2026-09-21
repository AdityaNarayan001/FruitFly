"""Full retained-connectome maze benchmark, isolated from existing live services."""
import argparse,json,time,uuid,copy
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .anatomy import load,transpose
from .model import CUDA
from ..comparison.models import observe,all_observations,Table,fingerprint
from ..comparison.__main__ import trials,emit,paired_interval
from ..exp2.core import World,default_config,patterns,summarize
from ..provenance import ROOT,atomic_json,sha256,source_manifest,preserve_source,environment,canonical_hash


AGENTS=('q_table','full_fixed','full_plastic','rewired_plastic','random')


def collect(config,seed,n):
    rng=np.random.default_rng(4_000_000+seed);episode=0;world=World(config,patterns(5_000_000+seed,episode));reset=True
    data={k:[] for k in ('state','action','reward','next_state','terminal','next_legal','reset','truncated')}
    for _ in range(n):
        _,s,_=observe(world);action=int(rng.choice(world.legal()));reward=world.step(action);_,ns,legal=observe(world)
        truncated=world.outcome=='timeout';terminal=world.done and not truncated
        for k,value in zip(data,(s,action,reward,ns,terminal,legal,reset,truncated)):data[k].append(value)
        reset=world.done
        if world.done:episode+=1;world=World(config,patterns(5_000_000+seed,episode))
    return {k:np.asarray(v,dtype=bool if k in ('terminal','next_legal','reset','truncated') else np.float32 if k=='reward' else np.int64) for k,v in data.items()}


def evaluate(agent,config,seed,n,neural=False):
    rows=[];seen=np.zeros(agent.n,bool) if neural else None;max_activity=0.
    for split,i,c,assignment,rng_seed in trials(seed,n,config):
        w=World(c,assignment);rng=np.random.default_rng(rng_seed)
        if neural:agent.reset()
        while not w.done:
            obs,index,mask=observe(w);legal=np.flatnonzero(mask)
            q=agent.step(obs) if neural else agent.q[index] if agent else None
            if q is None:action=int(rng.choice(legal))
            else:
                values=q[legal];best=np.flatnonzero(np.isclose(values,values.max(),rtol=0,atol=1e-12));action=int(legal[rng.choice(best)])
            w.step(action)
        if neural:
            h,activity=agent.state();seen|=activity;max_activity=max(max_activity,float(np.abs(h).max()))
            if not np.isfinite(h).all() or max_activity>1.000001:raise FloatingPointError('Invalid neural activity')
        rows.append({'split':split,'trial':i,'layout':c['layout'],'path':w.path,**w.summary()})
    return rows,{'active_neurons_above_1e-6':int(seen.sum()) if neural else None,'max_final_absolute_activity':max_activity if neural else None}


def train_neural(agent,data,run,details):
    obs=all_observations().astype(np.float32);seen=np.zeros(agent.n,bool);deltas=[];start=time.perf_counter()
    for i in range(len(data['state'])):
        if data['reset'][i]:
            if i:seen|=agent.state()[1]
            agent.reset()
        q=agent.step(obs[data['state'][i]])
        if data['terminal'][i]:target=float(data['reward'][i])
        else:
            nxt=agent.step(obs[data['next_state'][i]],preview=True)
            target=float(data['reward'][i])+.95*float(nxt[data['next_legal'][i]].max())
        deltas.append(agent.learn(int(data['action'][i]),target))
        if (i+1)%1000==0:emit(run,'training',**details,transitions=i+1,seconds=time.perf_counter()-start)
    h,activity=agent.state();seen|=activity
    if not np.isfinite(h).all() or np.abs(h).max()>1.000001:raise FloatingPointError('Invalid training activity')
    return {'training_seconds':time.perf_counter()-start,'active_neurons_above_1e-6':int(seen.sum()),
            'mean_absolute_clipped_td':float(np.mean(np.abs(deltas))),'transitions':len(deltas)}


def preflight(run,g,base,channels,signs,meta):
    agent=CUDA(g.indptr,g.indices,base,channels,signs,991)
    rng=np.random.default_rng(144);start=time.perf_counter()
    try:
        for i in range(500):
            obs=rng.uniform(-1,1,24).astype(np.float32);agent.step(obs);agent.step(-obs,preview=True);agent.learn(i%4,float(rng.uniform(-1,1)))
        seconds=time.perf_counter()-start;h,seen=agent.state();w,r,b=agent.parameters()
        assert np.isfinite(h).all() and np.abs(h).max()<=1.000001
        assert np.all(w>=np.minimum(0,2*base)) and np.all(w<=np.maximum(0,2*base))
        changed=int(np.count_nonzero(w!=base));assert changed>0
        result={'engineering_only':True,'maze_outcomes_observed':False,'training_transitions':500,'seconds':seconds,
          'milliseconds_per_step_preview_update':1000*seconds/500,'nodes':g.n,'edge_slots':g.m,'active_neurons_above_1e-6':int(seen.sum()),
          'max_absolute_activity':float(np.abs(h).max()),'changed_edges':changed,'finite_parameters':all(np.isfinite(a).all() for a in (w,r,b)),
          'sign_bounds_verified':True,'anatomy':meta}
        atomic_json(run/'preflight.json',result);print(json.dumps(result),flush=True)
    finally:agent.close()


def run_benchmark(run,g,base,channels,signs,rewired,meta,p,seeds,targets):
    conditions=[];checks=[];transposes={False:transpose(g.indices,g.n),True:transpose(rewired,g.n)}
    with (run/'trials.jsonl').open('w',buffering=1) as log:
        for seed in seeds:
            for target in targets:
                c=default_config();c.update(seed=seed,rewarded_pattern=target)
                data=collect(c,seed,p['transitions']);np.savez_compressed(run/f'experience-{seed}-{target}.npz',**data)
                for name in AGENTS:
                    neural=name in ('full_fixed','full_plastic','rewired_plastic');randomized=name=='rewired_plastic'
                    detail={'seed':seed,'rewarded_pattern':target,'agent':name};emit(run,'initializing',**detail)
                    agent=CUDA(g.indptr,rewired if randomized else g.indices,base,channels,signs,seed,plastic=name!='full_fixed',transpose=transposes[randomized]) if neural else Table() if name=='q_table' else None
                    try:
                        if neural:
                            initial=agent.parameters();initial_hash=fingerprint(*initial)
                            trained=train_neural(agent,data,run,detail);parameters=agent.parameters();w,r,b=parameters
                            changed=int(np.count_nonzero(w!=base));same=np.array_equal(w,base)
                            if name=='full_fixed' and not same:raise AssertionError('Frozen internal graph changed')
                            if name!='full_fixed' and not changed:raise AssertionError('No internal learning occurred')
                            if not all(np.isfinite(a).all() for a in parameters):raise FloatingPointError('Nonfinite parameters')
                            if not (np.all(w>=np.minimum(0,2*base)) and np.all(w<=np.maximum(0,2*base))):raise AssertionError('Synaptic sign bounds violated')
                            before=fingerprint(*parameters);np.savez(run/f'policy-{seed}-{target}-{name}.npz',w=w,readout=r,bias=b)
                            check={**detail,**trained,'initial_policy_hash':initial_hash,'final_policy_hash':before,
                              'changed_internal_edges':changed,'internal_weight_l2_change':float(np.linalg.norm(w-base)),
                              'max_absolute_row_sum':float(np.bincount(np.repeat(np.arange(g.n),np.diff(g.indptr)),weights=np.abs(w),minlength=g.n).max()),
                              'zero_weights_still_zero':bool(np.all(w[base==0]==0)),'sign_bounds_verified':True,
                              'readout_changed':not np.array_equal(initial[1],r)}
                            count=(int(np.count_nonzero(base)) if name!='full_fixed' else 0)+4*g.n+4
                        elif agent:
                            start=time.perf_counter();agent.update(data);trained={'training_seconds':time.perf_counter()-start}
                            before=agent.state_hash();agent.save(run/f'policy-{seed}-{target}-{name}.npz');check={**detail,**trained};count=agent.parameters
                        else:trained={'training_seconds':0.};check={**detail};before=None;count=0
                        emit(run,'evaluating',**detail);tick=time.perf_counter();rows,activity=evaluate(agent,c,seed,p['evaluation_episodes'],neural)
                        eval_seconds=time.perf_counter()-tick
                        if neural:
                            if before!=fingerprint(*agent.parameters()):raise AssertionError('Evaluation changed parameters')
                            # Re-open the actual saved file and replay the first trial at a clean boundary.
                            with np.load(run/f'policy-{seed}-{target}-{name}.npz',allow_pickle=False) as z:agent.set_parameters(z['w'],z['readout'],z['bias'])
                            replay,_=evaluate(agent,c,seed,1,True)
                            if replay[0]!=rows[0]:raise AssertionError('Checkpoint replay changed first familiar trial')
                            check.update(checkpoint_replay_verified=True,evaluation_frozen_verified=True,evaluation_activity=activity)
                        elif agent and before!=agent.state_hash():raise AssertionError('Evaluation changed Q table')
                        for row in rows:log.write(json.dumps({**detail,**row})+'\n')
                        for split in ('familiar','unseen'):
                            conditions.append({**detail,'split':split,'training_seconds':trained['training_seconds'],'evaluation_seconds':eval_seconds,
                               'parameters':count,**summarize([r for r in rows if r['split']==split])})
                        checks.append(check);atomic_json(run/'results.json',{'anatomy':meta,'conditions':conditions,'checks':checks})
                        emit(run,'condition_complete',**detail,success={r['split']:r['success_rate'] for r in conditions[-2:]})
                    finally:
                        if neural:agent.close()
    return {'anatomy':meta,'conditions':conditions,'checks':checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--runs',type=Path,default=ROOT/'runs/exp_2_fullscale');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--targets',choices=['A','B'],nargs='+')
    args=parser.parse_args();p=json.loads((ROOT/'configs/exp_2_fullscale.json').read_text())
    fixed={'ticks_per_observation':2,'initial_incoming_norm':.45,'leak':.5,'synaptic_learning_rate':.01,'readout_learning_rate':.25,'gamma':.95,'tabular_alpha':.25,'td_clip':1.,'weight_multiple_limit':2.}
    if any(p[k]!=v for k,v in fixed.items()):raise ValueError('Protocol differs from implemented model')
    seeds=args.seeds or p['seeds'];targets=args.targets or p['rewarded_patterns']
    if not set(seeds)<=set(p['seeds']) or len(set(seeds))!=len(seeds) or len(set(targets))!=len(targets):raise ValueError('Invalid run partition')
    args.runs.mkdir(parents=True,exist_ok=True)
    run=args.runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+('-preflight-' if args.preflight else '-fullscale-')+uuid.uuid4().hex[:6]);run.mkdir()
    source=source_manifest();snapshot=preserve_source(args.runs,source)
    manifest={'status':'running','source_sha256':source['sha256'],'source_snapshot':snapshot,'protocol_sha256':canonical_hash(p),'protocol':p,
       'seeds':seeds,'rewarded_patterns':targets,'engineering_only':args.preflight,'environment':environment(),
       'native_build':json.loads((ROOT/'build/fullscale_build.json').read_text())}
    atomic_json(run/'manifest.json',manifest);atomic_json(run/'source_manifest.json',source);atomic_json(run/'protocol.json',p)
    print('RUN_DIR='+str(run.resolve()),flush=True)
    try:
        emit(run,'verifying_complete_graph');g,base,channels,signs,rewired,meta=load(args.graph,p,run)
        if args.preflight:preflight(run,g,base,channels,signs,meta)
        else:run_benchmark(run,g,base,channels,signs,rewired,meta,p,seeds,targets)
        manifest.update(status='complete',outputs={str(f.relative_to(run)):sha256(f) for f in run.rglob('*') if f.is_file() and f.name not in ('manifest.json','progress.json')})
        atomic_json(run/'manifest.json',manifest);emit(run,'complete')
    except BaseException as e:
        manifest.update(status='failed',error=f'{type(e).__name__}: {e}');atomic_json(run/'manifest.json',manifest);raise

if __name__=='__main__':main()
