#!/usr/bin/env python3
"""Verify complete fixed-budget run partitions, then aggregate seed-paired outcomes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fruitflybrain.provenance import sha256,atomic_json,canonical_hash
from fruitflybrain.fullscale.__main__ import AGENTS
from fruitflybrain.comparison.__main__ import paired_interval
from fruitflybrain.exp2.core import World,default_config,MOVES,summarize


def verify(path):
    m=json.loads((path/'manifest.json').read_text())
    if m['status']!='complete' or m['engineering_only']:raise ValueError('Incomplete or engineering-only run: '+str(path))
    for name,digest in m['outputs'].items():
        if sha256(path/name)!=digest:raise ValueError('Artifact checksum mismatch: '+str(path/name))
    if m['protocol']!=json.loads((path/'protocol.json').read_text()) or canonical_hash(m['protocol'])!=m['protocol_sha256']:raise ValueError('Protocol identity mismatch')
    source=json.loads((path/'source_manifest.json').read_text())
    if source['sha256']!=m['source_sha256'] or canonical_hash(source['files'])!=source['sha256']:raise ValueError('Source identity mismatch')
    r=json.loads((path/'results.json').read_text());raw=[json.loads(line) for line in (path/'trials.jsonl').read_text().splitlines()]
    expected={(s,t,a,split,i) for s in m['seeds'] for t in m['rewarded_patterns'] for a in AGENTS for split in ('familiar','unseen') for i in range(m['protocol']['evaluation_episodes'])}
    keys=[(v['seed'],v['rewarded_pattern'],v['agent'],v['split'],v['trial']) for v in raw]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('Missing or duplicate raw trials')
    for row in raw:
        c=default_config();c.update(layout=row['layout'],rewarded_pattern=row['rewarded_pattern']);w=World(c,row['patterns'])
        if row['path'][0]!=w.pos:raise ValueError('Wrong trajectory start')
        for pos in row['path'][1:]:
            move=(pos[0]-w.pos[0],pos[1]-w.pos[1]);action=MOVES.index(move)
            if action not in w.legal():raise ValueError('Illegal evaluation action')
            w.step(action)
        if not w.done or w.summary()!={k:row[k] for k in w.summary()}:raise ValueError('Trajectory outcome mismatch')
    for row in r['conditions']:
        selected=[v for v in raw if all(v[k]==row[k] for k in ('seed','rewarded_pattern','agent','split'))]
        actual=summarize(selected)
        if any(actual[k]!=row[k] for k in actual):raise ValueError('Reported summary disagrees with raw outcomes')
    expected_checks={(s,t,a) for s in m['seeds'] for t in m['rewarded_patterns'] for a in AGENTS}
    check_keys=[(v['seed'],v['rewarded_pattern'],v['agent']) for v in r['checks']]
    if set(check_keys)!=expected_checks or len(check_keys)!=len(expected_checks):raise ValueError('Missing or duplicate condition checks')
    for check in r['checks']:
        if check['agent'] in ('full_fixed','full_plastic','rewired_plastic'):
            if not all(check[k] for k in ('evaluation_frozen_verified','checkpoint_replay_verified','sign_bounds_verified','zero_weights_still_zero')):raise ValueError('Failed neural verification')
            if (check['changed_internal_edges']>0)!=(check['agent']!='full_fixed'):raise ValueError('Wrong plasticity condition')
            if check['transitions']!=m['protocol']['transitions'] or check['max_absolute_row_sum']>.900001:raise ValueError('Wrong training budget or unstable row')
    return m,r,{'run_id':path.name,'artifacts_verified':len(m['outputs']),'raw_trials_replayed':len(raw),'source_sha256':m['source_sha256'],
                'protocol_sha256':m['protocol_sha256'],'native_binary_sha256':m['native_build']['sha256'],'host':m['environment']['host']}



def audit_parameters(paths,graph_path):
    from fruitflybrain.graph import Graph
    from fruitflybrain.fullscale.model import normalized_weights
    from fruitflybrain.comparison.models import fingerprint
    from scipy.sparse import csr_matrix,hstack,vstack
    from scipy.sparse.csgraph import breadth_first_order
    graph=Graph.load(graph_path);base,_=normalized_weights(graph)
    mapping_hashes={sha256(path/'input_mapping.npz') for path in paths}
    null_hashes={sha256(path/'rewired_indices.npy') for path in paths}
    if len(mapping_hashes)!=1 or len(null_hashes)!=1:raise ValueError('Different input mapping or wiring null')
    checked=0;experience={};base_norm=float(np.linalg.norm(base));relative_changes=[]
    for path in paths:
        result=json.loads((path/'results.json').read_text())
        if result['anatomy']['graph_sha256']!=graph.metadata['graph_sha256']:raise ValueError('Audit graph mismatch')
        for c in result['checks']:
            if c['agent'] not in ('full_fixed','full_plastic','rewired_plastic'):continue
            with np.load(path/f"policy-{c['seed']}-{c['rewarded_pattern']}-{c['agent']}.npz",allow_pickle=False) as z:
                w=z['w'];r=z['readout'];b=z['bias']
            if (w.shape,r.shape,b.shape)!=((graph.m,),(4,graph.n),(4,)):raise ValueError('Policy size mismatch')
            if not all(np.isfinite(v).all() for v in (w,r,b)):raise ValueError('Nonfinite policy')
            if not (np.all(w>=np.minimum(0,2*base)) and np.all(w<=np.maximum(0,2*base))):raise ValueError('Policy sign/strength bounds violated')
            if int(np.count_nonzero(w!=base))!=c['changed_internal_edges']:raise ValueError('Wrong changed-edge count')
            if fingerprint(w,r,b)!=c['final_policy_hash']:raise ValueError('Policy fingerprint mismatch')
            relative_changes.append({'seed':c['seed'],'rewarded_pattern':c['rewarded_pattern'],'agent':c['agent'],'relative_internal_l2_change':float(np.linalg.norm(w-base))/base_norm})
            checked+=1
        manifest=json.loads((path/'manifest.json').read_text())
        for seed in manifest['seeds']:
            for target in manifest['rewarded_patterns']:
                with np.load(path/f'experience-{seed}-{target}.npz',allow_pickle=False) as z:
                    arrays={k:z[k] for k in z.files}
                if len(arrays['state'])!=manifest['protocol']['transitions']:raise ValueError('Experience budget mismatch')
                if np.any(arrays['terminal'] & arrays['truncated']):raise ValueError('Timeout treated as terminal')
                if not np.array_equal(arrays['reset'][1:],(arrays['terminal']|arrays['truncated'])[:-1]):raise ValueError('Experience reset flags mismatch')
                experience[seed,target]=fingerprint(*[arrays[k] for k in sorted(arrays) if k!='reward'])
    for seed in manifest['protocol']['seeds']:
        if experience[seed,'A']!=experience[seed,'B']:raise ValueError('Behavior experience depends on reward identity')
    with np.load(paths[0]/'input_mapping.npz',allow_pickle=False) as z:channels=z['channels']
    matrix=csr_matrix(((graph.weights!=0).astype(np.int8),graph.indices,graph.indptr),shape=(graph.n,graph.n));matrix.eliminate_zeros()
    expanded=hstack([matrix.T,csr_matrix((graph.n,1),dtype=np.int8)],format='csr')
    sensory=np.flatnonzero(channels>=0)
    source=csr_matrix((np.ones(len(sensory),np.int8),(np.zeros(len(sensory),int),sensory)),shape=(1,graph.n+1))
    expanded=vstack([expanded,source],format='csr')
    reachable=len(breadth_first_order(expanded,graph.n,directed=True,return_predecessors=False))-1
    return {'full_policy_arrays_verified':checked,'experience_reward_identity_independent':True,
      'initial_internal_weight_l2_norm':base_norm,'relative_internal_changes':relative_changes,
      'mapping_sha256':next(iter(mapping_hashes)),'null_wiring_sha256':next(iter(null_hashes)),
      'structurally_reachable_from_input_via_nonzero_edges':reachable,'unreachable_from_input':graph.n-reachable,
      'reachability_interpretation':'Post-run descriptive graph analysis; ignores sign cancellation and threshold. No policy or training changes.'}


def aggregate(paths):
    verified=[verify(path) for path in paths];p=verified[0][0]['protocol'];meta=verified[0][1]['anatomy'];rows=[];checks=[]
    for m,r,_ in verified:
        if m['protocol']!=p or r['anatomy']!=meta or m['source_sha256']!=verified[0][0]['source_sha256']:raise ValueError('Runs use different scientific protocols, anatomy or sources')
        rows+=r['conditions'];checks+=r['checks']
    expected={(s,t,a,split) for s in p['seeds'] for t in p['rewarded_patterns'] for a in AGENTS for split in ('familiar','unseen')}
    keys=[(v['seed'],v['rewarded_pattern'],v['agent'],v['split']) for v in rows]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('Supply each protocol partition exactly once')
    report={'scope':'Full retained graph; engineered bounded recurrent rate model and two-tick TD gradients. Not biological physiology.',
      'protocol':p,'anatomy':meta,'runs':[v[2] for v in verified],'aggregate':[],'paired_differences':[],'checks':checks,'conditions':rows}
    for agent in AGENTS:
        for split in ('familiar','unseen'):
            selected=[v for v in rows if v['agent']==agent and v['split']==split]
            report['aggregate'].append({'agent':agent,'split':split,'successes':sum(v['successes'] for v in selected),'trials':sum(v['n'] for v in selected),
              'success_rate':float(np.mean([v['success_rate'] for v in selected])),
              'mean_return':float(np.mean([v['mean_return'] for v in selected])),'mean_steps':float(np.mean([v['mean_steps'] for v in selected])),
              'timeouts':sum(v['timeouts'] for v in selected),'mean_training_seconds':float(np.mean([v['training_seconds'] for v in selected])),
              'parameters':selected[0]['parameters']})
    for a,b in [('full_plastic','q_table'),('full_plastic','full_fixed'),('full_plastic','rewired_plastic')]:
        for split in ('familiar','unseen'):
            delta=[]
            for seed in p['seeds']:
                values={agent:np.mean([r['success_rate'] for r in rows if r['seed']==seed and r['agent']==agent and r['split']==split]) for agent in (a,b)}
                delta.append(float(values[a]-values[b]))
            report['paired_differences'].append({'a':a,'b':b,'split':split,'seed_differences':delta,**paired_interval(delta)})
    return report


def markdown(r):
    endpoints=[d for d in r['paired_differences'] if d['b']=='q_table']
    outcome=' '.join(('Tabular Q performed better on '+d['split']+' success under this fixed budget.' if d['ci95'][1]<0 else 'The full plastic model performed better on '+d['split']+' success under this fixed budget.' if d['ci95'][0]>0 else 'The '+d['split']+' success difference from tabular Q was inconclusive.') for d in endpoints)
    text=['# Full retained-connectome maze comparison','',r['scope'],'',outcome,'',
      'Fixed before outcomes: five seeds, both reward choices, 10,000 shared transitions, one chronological pass, 50 familiar and 50 unseen frozen trials per condition. No cached neural responses. Activity resets only at trial boundaries.','',
      '| Controller | Familiar success | Unseen success | Mean training time | Trainable parameters |',
      '| --- | ---: | ---: | ---: | ---: |']
    for a in AGENTS:
        rows={v['split']:v for v in r['aggregate'] if v['agent']==a};f=rows['familiar'];u=rows['unseen']
        text.append(f"| {a} | {f['success_rate']:.1%} ({f['successes']}/{f['trials']}) | {u['success_rate']:.1%} ({u['successes']}/{u['trials']}) | {f['mean_training_seconds']:.2f} s | {f['parameters']:,} |")
    text+=['','## Paired uncertainty','','Percentage-point differences, resampling five seed means after averaging reward A/B. These descriptive intervals have limited resolution.','']
    for d in r['paired_differences']:
        text.append(f"- {d['split']}: {d['a']} minus {d['b']}: {100*d['mean']:.2f} pp; 95% interval [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}].")
    text+=['','## What was simulated and trained','',
      f"Every decision advanced {r['anatomy']['neurons']:,} nodes and {r['anatomy']['edge_slots']:,} edge slots. Of these, {r['anatomy']['nonzero_signed_edges']:,} had nonzero signed initial weights; zero weights stayed zero. Engineered input drove {r['anatomy']['input_neurons']:,} annotated sensory neurons.",'']
    for a in ('full_fixed','full_plastic','rewired_plastic'):
        cs=[c for c in r['checks'] if c['agent']==a];edges=[c['changed_internal_edges'] for c in cs];active=[c['active_neurons_above_1e-6'] for c in cs]
        text.append(f"- {a}: {min(edges):,}-{max(edges):,} internal edge strengths changed; {min(active):,}-{max(active):,} neurons exceeded the activity threshold during training. All nodes were simulated, including quiet ones.")
    if 'independent_audit' in r:
        audit=r['independent_audit'];changes=[v['relative_internal_l2_change'] for v in audit['relative_internal_changes'] if v['agent']=='full_plastic']
        text += ['',f"Full anatomical plasticity changed the global internal-weight L2 norm by {100*min(changes):.6f}% to {100*max(changes):.6f}% relative to initialization (norm of the change divided by initial norm). Count of changed edges alone does not establish a large functional change.",f"{audit['structurally_reachable_from_input_via_nonzero_edges']:,} nodes are structurally reachable from driven inputs through nonzero modeled edges; {audit['unreachable_from_input']} are not. Thresholded activity can be lower because of attenuation/cancellation."]
    text+=['','All neural evaluations preserved parameter hashes; reloaded checkpoints reproduced the first familiar trial. Raw saved paths were replayed independently to verify each outcome.','',
      '## Limits','',
      'This is one artificial recurrent model, one fixed sensory projection, one fixed multigraph wiring null and a fixed training budget. The generic rate dynamics, sign normalization, privileged grid localization, four-action readout and two-tick truncated gradient rule are engineered. There is no validated natural sensory/motor mapping or dopamine mechanism. A weak result cannot rule out other dynamics, learning rules or budgets. Q-learning and the neural models are not separately hyperparameter-optimized. Equal data does not imply equal compute or parameter counts.','',
      'Familiar trials repeat one layout and two cue assignments, so trial counts are not independent training replications; the interval unit is the five seeds. The null preserves sign-stratified source degree counts and each target row weight multiset, but allows parallel pairs and self-edges. Unseen mazes vary walls, while start/food coordinates stay fixed. Food identity is hidden from the policy; raw visible cue assignment is supplied. Timeout transitions bootstrap before resetting. No test-result tuning was performed.','',
      '## Exact evidence','']
    for run in r['runs']:text.append(f"- {run['host']}: `{run['run_id']}`, {run['artifacts_verified']} immutable artifacts checked, {run['raw_trials_replayed']} trial trajectories replayed.")
    text += ['',f"Scientific source SHA-256: `{r['runs'][0]['source_sha256']}`.",f"Graph SHA-256: `{r['anatomy']['graph_sha256']}`.",
      'Raw data/checkpoints are outside Git in `runs/gx10-a/exp_2_fullscale` and `runs/gx10-b/exp_2_fullscale`. The source and protocol are versioned; the full external dataset is required to reproduce GPU behavior.','']
    return '\n'.join(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,default=ROOT/'docs/research/fullscale_results');p.add_argument('--graph',type=Path,default=ROOT/'data/malecns-graph-v1')
    args=p.parse_args();report=aggregate(args.runs);report['independent_audit']=audit_parameters(args.runs,args.graph);atomic_json(args.output.with_suffix('.json'),report);args.output.with_suffix('.md').write_text(markdown(report))
    print(json.dumps({'runs':report['runs'],'aggregate':report['aggregate'],'paired_differences':report['paired_differences']},indent=2))
