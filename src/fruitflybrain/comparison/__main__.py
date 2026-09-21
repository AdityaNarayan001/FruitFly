"""Run isolated matched comparisons, preserving the interactive Exp 2 service."""
import argparse,copy,json,time,uuid,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .circuit import extract
from .models import Table,Network,observe,fingerprint
from ..exp2.core import World,Learner,default_config,patterns,epsilon,rollout,make_maze,summarize,digest
from ..exp2.encoder import NeuralEncoder
from ..exp2.session import policy_rng
from ..provenance import ROOT,atomic_json,canonical_hash,source_manifest,preserve_source,environment,sha256


def emit(run,stage,**details):
    record={'stage':stage,**details};atomic_json(run/'progress.json',record);print(json.dumps(record),flush=True)


def trials(seed,n,config):
    for split in ('familiar','unseen'):
        for i in range(n):
            c=copy.deepcopy(config)
            if split=='unseen':c['layout']=make_maze(2_000_000+seed*1000+i//2)
            # Each unseen layout appears with both assignments.
            assignment=patterns(1_000_000+seed*1000,i)
            yield split,i,c,assignment,3_000_000+seed*1000+i


def paired_interval(values):
    a=np.array(values,dtype=float)
    if len(a)<2:return {'mean':float(a.mean()),'ci95':None,'seed_units':len(a)}
    rng=np.random.default_rng(921);samples=rng.choice(a,(10000,len(a)),replace=True).mean(1)
    return {'mean':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'seed_units':len(a)}


def legacy(run,graph,backend,p):
    out=run/'legacy';out.mkdir();start=time.perf_counter();encoder=NeuralEncoder(graph,backend);encoder.save(out)
    probe_seconds=time.perf_counter()-start
    token_equal=all(encoder.tokens(a)==encoder.tokens(a,'image') for a in [('A','B'),('B','A')])
    if not token_equal:raise ValueError('Legacy tokens differ; exact-equivalence assertion is inapplicable')
    all_rows=[];identity=[]
    with (out/'trials.jsonl').open('w',buffering=1) as log:
        for seed in p['seeds']:
            for target in p['rewarded_patterns']:
                c=default_config();c.update(seed=seed,rewarded_pattern=target)
                image=Learner(c);fly=Learner(c);tick=time.perf_counter()
                for episode in range(p['legacy_training_episodes']):
                    assignment=patterns(seed,episode)
                    for agent,mode in ((image,'image'),(fly,'neural')):
                        rollout(c,assignment,agent,encoder.tokens(assignment,mode),policy_rng(seed,episode),epsilon(episode),learn=True)
                    if not np.array_equal(image.q,fly.q):raise AssertionError('Legacy paired Q updates diverged')
                before=(digest(image.q),digest(fly.q));condition=[]
                for split,i,ec,assignment,rng_seed in trials(seed,p['evaluation_episodes'],c):
                    for name,agent,mode in [('q_table',image,'image'),('fly_probe_q',fly,'neural'),('random',image,'image')]:
                        row={'seed':seed,'rewarded_pattern':target,'split':split,'trial':i,'agent':name,
                             **rollout(ec,assignment,agent,encoder.tokens(assignment,mode),rng_seed,random=name=='random')}
                        log.write(json.dumps(row)+'\n');condition.append(row)
                if before!=(digest(image.q),digest(fly.q)):raise AssertionError('Evaluation modified legacy Q table')
                np.savez_compressed(out/f'policy-{seed}-{target}.npz',image_q=image.q,fly_q=fly.q)
                identity.append({'seed':seed,'rewarded_pattern':target,'q_sha256':before[0],'equal_every_training_episode':True,
                    'frozen_verified':True,'training_and_evaluation_seconds':time.perf_counter()-tick})
                for split in ('familiar','unseen'):
                    for name in ('q_table','fly_probe_q','random'):
                        all_rows.append({'seed':seed,'rewarded_pattern':target,'split':split,'agent':name,
                            **summarize([r for r in condition if r['split']==split and r['agent']==name])})
                emit(run,'legacy',seed=seed,rewarded_pattern=target,identical_q=True)
    result={'token_equality':token_equal,'probe_seconds':probe_seconds,'encoder':encoder.meta,'policy_identity':identity,'conditions':all_rows}
    atomic_json(out/'results.json',result)
    return result


def collect(config,seed,n):
    rng=np.random.default_rng(4_000_000+seed);episode=0;world=World(config,patterns(5_000_000+seed,episode))
    data={k:[] for k in ('state','action','reward','next_state','terminal','next_legal')}
    for _ in range(n):
        _,s,_=observe(world);action=int(rng.choice(world.legal()));reward=world.step(action);_,ns,legal=observe(world)
        for k,value in zip(data,(s,action,reward,ns,world.done,legal)):data[k].append(value)
        if world.done:episode+=1;world=World(config,patterns(5_000_000+seed,episode))
    return {k:np.asarray(v,dtype=bool if k in ('terminal','next_legal') else np.float64 if k=='reward' else np.int64) for k,v in data.items()}


def evaluate(agent,config,seed,n):
    before=agent.state_hash() if agent else None
    # Pure fixed feedforward observations allow exact caching during frozen tests.
    cached=agent.values(np.arange(2*9*9*16)) if agent else None
    records=[]
    for split,i,c,assignment,rng_seed in trials(seed,n,config):
        world=World(c,assignment);rng=np.random.default_rng(rng_seed)
        while not world.done:
            _,index,mask=observe(world);legal=np.flatnonzero(mask)
            if agent:
                q=cached[index,legal];best=np.flatnonzero(np.isclose(q,q.max(),rtol=0,atol=1e-12));action=int(legal[rng.choice(best)])
            else:action=int(rng.choice(legal))
            world.step(action)
        records.append({'split':split,'trial':i,**world.summary()})
    if agent and before!=agent.state_hash():raise AssertionError('Frozen evaluation changed parameters or optimizer state')
    return records


def circuit(run,graph,p):
    out=run/'circuit';out.mkdir();weights,rewired,meta=extract(graph,p['kc_count'],p['circuit_seed'])
    atomic_json(out/'anatomy.json',meta);np.savez_compressed(out/'anatomy.npz',weights=weights,mask=weights>0,rewired=rewired,rewired_mask=rewired>0,
        kc_body_ids=np.array(meta['kc_body_ids']),mbon_body_ids=np.array(meta['mbon_body_ids']))
    conditions=[];checks=[]
    with (out/'trials.jsonl').open('w',buffering=1) as log:
        for seed in p['seeds']:
            for target in p['rewarded_patterns']:
                config=default_config();config.update(seed=seed,rewarded_pattern=target)
                data=collect(config,seed,p['transitions']);name=f'{seed}-{target}'
                np.savez_compressed(out/f'experience-{name}.npz',**data)
                agents={'q_table':Table(),'fly_fixed_readout':Network(weights,seed,False),
                        'fly_plastic':Network(weights,seed,True),'rewired_plastic':Network(rewired,seed,True)}
                starts={k:v.state_hash() for k,v in agents.items()};rng=np.random.default_rng(6_000_000+seed)
                timings={k:0. for k in agents};elapsed_updates=0
                for checkpoint in p['checkpoints']:
                    while elapsed_updates<checkpoint:
                        ids=rng.integers(len(data['state']),size=p['batch_size']);batch={k:v[ids] for k,v in data.items()}
                        for key,agent in agents.items():
                            tick=time.perf_counter();agent.update(batch);timings[key]+=time.perf_counter()-tick
                        elapsed_updates+=1
                    for key,agent in [*agents.items(),('random',None)]:
                        tick=time.perf_counter();rows=evaluate(agent,config,seed,p['evaluation_episodes']);eval_seconds=time.perf_counter()-tick
                        for row in rows:log.write(json.dumps({'seed':seed,'rewarded_pattern':target,'updates':checkpoint,'agent':key,**row})+'\n')
                        for split in ('familiar','unseen'):
                            conditions.append({'seed':seed,'rewarded_pattern':target,'updates':checkpoint,'agent':key,'split':split,
                                'training_seconds':timings.get(key,0.),'evaluation_seconds':eval_seconds,
                                'parameters':agent.parameters if agent else 0,
                                **summarize([r for r in rows if r['split']==split])})
                        if agent:agent.save(out/f'policy-{name}-{key}-{checkpoint}.npz')
                    emit(run,'circuit',seed=seed,rewarded_pattern=target,updates=checkpoint,
                         familiar={r['agent']:r['success_rate'] for r in conditions[-10:] if r['split']=='familiar'})
                frozen=agents['fly_fixed_readout'];plastic=agents['fly_plastic'];randomized=agents['rewired_plastic']
                if not np.array_equal(frozen.w,weights):raise AssertionError('Frozen anatomy changed')
                if np.array_equal(plastic.w,weights):raise AssertionError('Plastic connections did not train')
                for network in (plastic,randomized):
                    if (network.w[~network.mask]!=0).any() or (network.w<0).any():raise AssertionError('Learned network violated anatomical/sign mask')
                checks.append({'seed':seed,'rewarded_pattern':target,'frozen_internal_weights_verified':True,
                    'plastic_edges_changed':int(np.count_nonzero(plastic.w!=weights)),
                    'plastic_weight_l2_change':float(np.linalg.norm(plastic.w-weights)),
                    'missing_edges_remain_zero':True,'evaluation_frozen_verified':True,
                    'initial_policy_hashes':starts,'final_policy_hashes':{k:v.state_hash() for k,v in agents.items()},
                    'experience_sha256':sha256(out/f'experience-{name}.npz')})
                atomic_json(out/'results.json',{'anatomy':meta,'conditions':conditions,'verification':checks})
    return {'anatomy':meta,'conditions':conditions,'verification':checks}


def aggregate(result,p):
    report={'protocol':p,'scope':'Engineering comparison; new plastic controller is reduced rate model, not full fly brain', 'legacy':[], 'circuit':[], 'paired_differences':[]}
    for part in ('legacy','circuit'):
        if part not in result:continue
        rows=result[part]['conditions'];agents=sorted({r['agent'] for r in rows})
        for split in ('familiar','unseen'):
            for updates in ([None] if part=='legacy' else p['checkpoints']):
                for agent in agents:
                    selected=[r for r in rows if r['agent']==agent and r['split']==split and (updates is None or r['updates']==updates)]
                    report[part].append({'agent':agent,'split':split,'updates':updates,
                        'success_rate':float(np.mean([r['success_rate'] for r in selected])),
                        'successes':sum(r['successes'] for r in selected),'trials':sum(r['n'] for r in selected),
                        'mean_return':float(np.mean([r['mean_return'] for r in selected])),
                        'mean_steps':float(np.mean([r['mean_steps'] for r in selected])),
                        'mean_success_efficiency':float(np.mean([r['mean_success_efficiency'] for r in selected if r['mean_success_efficiency'] is not None])) if any(r['mean_success_efficiency'] is not None for r in selected) else None,
                        'mean_training_seconds':float(np.mean([r.get('training_seconds',0.) for r in selected])) if part=='circuit' else None,
                        'parameters':selected[0].get('parameters')})
        pairs=[('fly_probe_q','q_table')] if part=='legacy' else [('fly_plastic','q_table'),('fly_plastic','rewired_plastic'),('fly_plastic','fly_fixed_readout')]
        for split in ('familiar','unseen'):
            for a,b in pairs:
                differences=[]
                for seed in p['seeds']:
                    chosen=[r for r in rows if r['seed']==seed and r['split']==split and (part=='legacy' or r['updates']==p['checkpoints'][-1])]
                    differences.append(float(np.mean([r['success_rate'] for r in chosen if r['agent']==a])-np.mean([r['success_rate'] for r in chosen if r['agent']==b])))
                report['paired_differences'].append({'part':part,'split':split,'a':a,'b':b,**paired_interval(differences)})
    return report


def markdown(report):
    text=['# Exp 2 matched comparison','',report['scope'],'','These are results under a fixed experience/update budget. They do not rank every possible ML/RL model.','']
    for part in ('legacy','circuit'):
        text += [f'## {part.capitalize()} comparison','','| Agent | Split | Updates | Success | Mean actions | Training seconds |','| --- | --- | ---: | ---: | ---: | ---: |']
        for r in report[part]:
            if part=='circuit' and r['updates']!=report['protocol']['checkpoints'][-1]:continue
            seconds='—' if r['mean_training_seconds'] is None else f"{r['mean_training_seconds']:.2f}"
            text.append(f"| {r['agent']} | {r['split']} | {r['updates'] if r['updates'] is not None else '400 episodes'} | {r['success_rate']:.1%} ({r['successes']}/{r['trials']}) | {r['mean_steps']:.2f} | {seconds} |")
        text.append('')
    text += ['## Paired success-rate differences','','Intervals resample training seeds, averaging the two reward assignments inside each seed. Units are percentage points.','']
    for r in report['paired_differences']:
        ci=r['ci95'];interval=f'[{100*ci[0]:.2f}, {100*ci[1]:.2f}]' if ci else 'not estimated in smoke run'
        text.append(f"- {r['part']}, {r['split']}: {r['a']} minus {r['b']}: {100*r['mean']:.2f} pp; 95% interval {interval}.")
    text += ['','The plastic network trains modeled KC-to-MBON strengths and an external action readout. It does not reproduce natural plasticity, simulate the entire connectome, or demonstrate biological learning.','The recorded-circuit and randomized-circuit controls use the same input projection, degree sequence, number of parameters, experience and replay batches.','']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True);parser.add_argument('--runs',type=Path,default=ROOT/'runs/exp_2_comparison')
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu',help='Full-graph probe backend; reduced network always uses host CPU')
    parser.add_argument('--part',choices=['all','legacy','circuit'],default='all');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();p=json.loads((ROOT/'configs/exp_2_comparison.json').read_text())
    if args.smoke:p.update(seeds=[0],rewarded_patterns=['A'],legacy_training_episodes=4,evaluation_episodes=4,transitions=200,checkpoints=[0,2],development_smoke=True)
    args.runs.mkdir(parents=True,exist_ok=True);run=args.runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-comparison-'+uuid.uuid4().hex[:6]);run.mkdir()
    source=source_manifest();snapshot=preserve_source(args.runs,source)
    manifest={'status':'running','source_sha256':source['sha256'],'source_snapshot':snapshot,'protocol_sha256':canonical_hash(p),
              'config':p,'environment':environment(),'backend':args.backend,'part':args.part}
    atomic_json(run/'manifest.json',manifest);atomic_json(run/'source_manifest.json',source);atomic_json(run/'protocol.json',p)
    print('RUN_DIR='+str(run.resolve()),flush=True)
    try:
        results={}
        if args.part in ('all','legacy'):results['legacy']=legacy(run,args.graph,args.backend,p)
        if args.part in ('all','circuit'):results['circuit']=circuit(run,args.graph,p)
        report=aggregate(results,p);atomic_json(run/'report.json',report);(run/'report.md').write_text(markdown(report))
        manifest.update(status='complete',outputs={str(f.relative_to(run)):sha256(f) for f in run.rglob('*') if f.is_file() and f.name!='manifest.json'})
        atomic_json(run/'manifest.json',manifest);emit(run,'complete',report=str(run/'report.md'))
    except BaseException as e:
        manifest.update(status='failed',error=f'{type(e).__name__}: {e}');atomic_json(run/'manifest.json',manifest);raise

if __name__=='__main__':main()
