#!/usr/bin/env python3
"""Post-hoc zero-recurrence diagnostic, separate from the prospective comparison."""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fruitflybrain.fullscale.__main__ import evaluate
from fruitflybrain.exp2.core import default_config,summarize
from fruitflybrain.provenance import atomic_json,sha256,source_manifest,preserve_source,environment,canonical_hash
from fruitflybrain.comparison.models import fingerprint


class Disconnected:
    """Exact rate equations with W=0; full neuron state, unchanged learned readout.

    The zero sparse product is omitted algebraically. This is an ablation only,
    never a replacement for or claim of running the primary full-graph test.
    """
    def __init__(self,channels,signs,r,b):
        self.channels=channels;self.signs=signs;self.r=r;self.b=b;self.n=len(channels);self.reset()
    def reset(self):self.h=np.zeros(self.n,np.float32);self.seen=np.zeros(self.n,bool)
    def step(self,obs):
        drive=np.where(self.channels>=0,np.asarray(obs,np.float32)[np.maximum(self.channels,0)]*self.signs,0)
        t=np.tanh(drive)
        for _ in range(2):self.h=.5*self.h+.5*t;self.seen |= np.abs(self.h)>1e-6
        return self.r@self.h+self.b
    def state(self):return self.h.copy(),self.seen.copy()


def diagnose(paths):
    rows=[];checks=[];details=[];expected=set()
    for path in paths:
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['status']!='complete':raise ValueError('Wait for the primary run to finish')
        expected|={(s,t) for s in manifest['seeds'] for t in manifest['rewarded_patterns']}
        p=manifest['protocol'];primary=[json.loads(line) for line in (path/'trials.jsonl').read_text().splitlines()]
        with np.load(path/'input_mapping.npz',allow_pickle=False) as z:channels=z['channels'];signs=z['signs']
        for seed in manifest['seeds']:
            for target in manifest['rewarded_patterns']:
                saved=path/f'policy-{seed}-{target}-full_plastic.npz'
                if sha256(saved)!=manifest['outputs'][saved.name]:raise ValueError('Checkpoint hash mismatch')
                with np.load(saved,allow_pickle=False) as z:r=z['readout'];b=z['bias']
                before=fingerprint(r,b)
                agent=Disconnected(channels,signs,r,b);c=default_config();c.update(seed=seed,rewarded_pattern=target)
                start=time.perf_counter();evaluated,_=evaluate(agent,c,seed,p['evaluation_episodes'],True);seconds=time.perf_counter()-start
                if before!=fingerprint(agent.r,agent.b):raise AssertionError('Diagnostic modified saved readout')
                reference={(v['split'],v['trial']):v for v in primary if v['seed']==seed and v['rewarded_pattern']==target and v['agent']=='full_plastic'}
                same_paths=same_outcomes=0
                for row in evaluated:
                    before=reference[row['split'],row['trial']];same_paths+=row['path']==before['path'];same_outcomes+=row['outcome']==before['outcome']
                    details.append({'seed':seed,'rewarded_pattern':target,'agent':'zero_recurrence_saved_readout',**row})
                for split in ('familiar','unseen'):
                    rows.append({'seed':seed,'rewarded_pattern':target,'split':split,**summarize([v for v in evaluated if v['split']==split])})
                checks.append({'seed':seed,'rewarded_pattern':target,'identical_paths':same_paths,'identical_outcomes':same_outcomes,'trials':len(evaluated),'seconds':seconds,'source_checkpoint_sha256':sha256(saved)})
                print(json.dumps(checks[-1]),flush=True)
    keys=[(c['seed'],c['rewarded_pattern']) for c in checks]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Missing or duplicate conditions')
    return {'status':'post_hoc_diagnostic','defined_after':'Partial primary outcomes showed matching anatomical/frozen/rewired success rates; no training changed.',
      'model':'All recurrent weights removed; fixed sensory mapping and saved learned full-neuron action readout retained. CPU float32 rate recurrence; zero sparse multiplication omitted.',
      'interpretation':'Sensitivity of this saved controller to recurrence. Does not replace the full-graph primary run or demonstrate biological circuitry.',
      'conditions':rows,'checks':checks,'identical_paths':sum(c['identical_paths'] for c in checks),
      'identical_outcomes':sum(c['identical_outcomes'] for c in checks),'trials':sum(c['trials'] for c in checks)},details


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,default=ROOT/'runs/exp_2_fullscale_diagnostic/diagnostic.json');p.add_argument('--trials-output',type=Path,default=None)
    args=p.parse_args()
    if args.trials_output is None:args.trials_output=args.output.with_name(args.output.stem+'-trials.jsonl')
    if args.output.exists() or args.trials_output.exists():p.error('Choose fresh output paths; records are immutable')
    protocol=json.loads((ROOT/'configs/exp_2_fullscale_diagnostic.json').read_text());source=source_manifest();snapshot=preserve_source(args.trials_output.parent,source)
    report,details=diagnose(args.runs);args.trials_output.parent.mkdir(parents=True,exist_ok=True)
    args.trials_output.write_text(''.join(json.dumps(r)+'\n' for r in details));report['trials_sha256']=sha256(args.trials_output);report['trials_file']=os.path.relpath(args.trials_output,args.output.parent);report['diagnostic_source_sha256']=source['sha256'];report['source_snapshot']=snapshot;report['protocol']=protocol;report['environment']=environment();report['input_runs']=[str(r.resolve()) for r in args.runs]
    atomic_json(args.output,report);print(json.dumps({k:report[k] for k in ('identical_paths','identical_outcomes','trials')},indent=2))
