import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import sys
import time
import uuid
from pathlib import Path
import numpy as np
from .provenance import preserve_source,ROOT,atomic_json,source_manifest,environment,sha256,canonical_hash
from .graph import Graph,fixture
from .model import Parameters

def run(config_path,runs_dir,backend="cuda"):
    from .backends import factory,label
    Model=factory(backend)
    config=json.loads(Path(config_path).read_text())
    if config.get('mode')!='engineering':
        raise ValueError('Only engineering runs are enabled. EXP-001 requires a frozen mapping/model amendment.')
    if config.get('seed')!=9000:raise ValueError('Engineering runs must use reserved fixture seed 9000')
    p=Parameters(**config.get('parameters',{})).validate()
    graph=fixture() if config['graph']=='fixture' else Graph.load(config['graph'])
    steps=round(config['duration_ms']/p.dt_ms);chunk=int(config.get('chunk_ticks',20))
    if not (1<=steps<=100000 and 1<=chunk<=200 and config['duration_ms']==steps*p.dt_ms):raise ValueError('Invalid run dimensions')
    source=source_manifest();snapshot=ROOT/'SOURCE_MANIFEST.json'
    if snapshot.exists() and json.loads(snapshot.read_text())['sha256']!=source['sha256']:raise ValueError('Deployed source was modified')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    dest=Path(runs_dir)/run_id;dest.mkdir(parents=True,exist_ok=False)
    atomic_json(dest/'source_manifest.json',source);atomic_json(dest/'config.json',config)
    manifest={'run_id':run_id,'status':'running','mode':'engineering','interpretation':'Direct-stimulation infrastructure run; not a visual-motion result',
              'start_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source['sha256'],
              'source_snapshot':preserve_source(runs_dir,source),
              'config_sha256':canonical_hash(config),'graph':graph.metadata,'n_neurons':graph.n,'n_edges':graph.m,
              'parameters':asdict(p),'backend':label(Model),'seed':9000,'environment':environment(),
              'native_library_sha256':sha256(ROOT/'build/libffb.so') if backend=='cuda' else None,'command':sys.argv,
              'recording':'Per-neuron total spike counts, chunk aggregates and first 8 neurons; no complete spike raster'}
    build=ROOT/'build/build.json'
    if build.exists():manifest['native_build']=json.loads(build.read_text())
    atomic_json(dest/'manifest.json',manifest)
    rng=np.random.default_rng(9000)
    stimulated=np.array([0,1],np.int32) if config['graph']=='fixture' else np.sort(rng.choice(graph.n,min(100,graph.n),replace=False))
    manifest['stimulated_body_ids']=graph.ids[stimulated].tolist()
    manifest['probe_body_ids']=graph.ids[:8].tolist()
    atomic_json(dest/'manifest.json',manifest)
    counts=np.zeros(graph.n,np.int64);samples=[];start=time.perf_counter();compute_s=0.0
    input_hasher=__import__('hashlib').sha256()
    try:
        with Model(graph,p) as model:
            # Warm-up kernel launch and reset; excluded from per-chunk execution time.
            model.run(np.zeros((1,graph.n),np.float32));model.reset()
            for first in range(0,steps,chunk):
                size=min(chunk,steps-first);drive=np.zeros((size,graph.n),np.float32)
                for k in range(size):
                    # Direct square-pulse input, explicitly not a retinal/vision encoder.
                    on=((first+k)*p.dt_ms)%200>=50
                    if on:drive[k,stimulated]=float(config.get('drive_mv_per_tick',.9))
                input_hasher.update(drive.tobytes())
                t=time.perf_counter();out=model.run(drive);compute_s+=time.perf_counter()-t
                counts+=out['counts'];seconds=size*p.dt_ms/1000
                sample={'sim_ms':(first+size)*p.dt_ms,'mean_rate_hz':float(out['counts'].mean()/seconds),
                        'max_rate_hz':float(out['counts'].max()/seconds),'active_fraction':float(np.mean(out['counts']>0)),
                        'total_spikes':int(out['counts'].sum()),'sample_rates_hz':(out['counts'][:8]/seconds).tolist(),
                        'mean_voltage_mv':float(out['voltage'].mean()),'drive_on':bool(((first+size-1)*p.dt_ms)%200>=50)}
                samples.append(sample)
                atomic_json(dest/'telemetry.json',{'run_id':run_id,'status':'running','samples':samples,'sim_ms':sample['sim_ms'],'wall_s':time.perf_counter()-start})
        np.save(dest/'spike_counts.npy',counts,allow_pickle=False)
        metrics={'simulated_seconds':steps*p.dt_ms/1000,'execution_seconds':compute_s,
                 'simulated_seconds_per_execution_second':steps*p.dt_ms/1000/compute_s,
                 'wall_seconds_including_setup_and_telemetry':time.perf_counter()-start,'total_spikes':int(counts.sum()),
                 'silent_neuron_fraction':float(np.mean(counts==0)),'input_sha256':input_hasher.hexdigest(),
                 'benchmark_scope':'Includes input/output copies and allocations; excludes import, warm-up and JSON telemetry',
                 'claim':'Engineering benchmark only; not directional processing or learning'}
        atomic_json(dest/'metrics.json',metrics);manifest['status']='complete'
        atomic_json(dest/'telemetry.json',{'run_id':run_id,'status':'complete','samples':samples,'sim_ms':steps*p.dt_ms,'wall_s':metrics['wall_seconds_including_setup_and_telemetry']})
    except BaseException as e:
        manifest['status']='failed';manifest['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        manifest['end_utc']=datetime.now(timezone.utc).isoformat()
        manifest['outputs']={f.name:sha256(f) for f in dest.iterdir() if f.is_file() and f.name!='manifest.json'}
        atomic_json(dest/'manifest.json',manifest)
    print(json.dumps({'run_dir':str(dest),'metrics':metrics},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description='FruitFlyBrain engineering and data tools')
    sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run');r.add_argument('--config',required=True);r.add_argument('--runs',required=True);r.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    s=sub.add_parser('serve');s.add_argument('--runs',required=True);s.add_argument('--port',type=int,default=8765);s.add_argument('--graph');s.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    d=sub.add_parser('demo');d.add_argument('--output',default='data/demo-v1')
    d=sub.add_parser('doctor');d.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    for name in ('download','inspect','import'):
        t=sub.add_parser(name);t.add_argument('--data',required=True)
        if name=='import':t.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='run':run(a.config,a.runs,a.backend)
    elif a.action=='demo':
        from .demo import prepare
        print(prepare(a.output))
    elif a.action=='doctor':
        from .backends import factory,label
        from .model import Reference
        g=fixture();drive=np.zeros((50,g.n),np.float32);drive[:,0]=.9
        expected=Reference(g).run(drive,record=True)
        with factory(a.backend)(g) as model:actual=model.run(drive,record=True)
        np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
        np.testing.assert_allclose(actual['voltages'],expected['voltages'],rtol=2e-5,atol=2e-4)
        print(json.dumps({'ok':True,'backend':label(factory(a.backend)),'neurons':g.n,'spikes':int(actual['counts'].sum())}))
    elif a.action=='serve':
        from .server import serve
        serve(a.runs,a.port,a.graph,a.backend)
    elif a.action=='import':
        from .importer import prepare
        prepare(a.data,a.output)
    else:
        from . import datasets
        getattr(datasets,a.action)(a.data)

if __name__=='__main__':main()
