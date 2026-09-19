#!/usr/bin/env python3
"""Versioned one-way deployment. Never rsync --delete datasets or run outputs."""
import argparse,json,subprocess,sys,tarfile,tempfile,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fruitflybrain.provenance import source_manifest,atomic_json

def ssh(host,cmd):return subprocess.run(['ssh','-o','BatchMode=yes',host,cmd],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','bootstrap','collect']);p.add_argument('host',choices=['gx10-a','gx10-b']);a=p.parse_args()
    if a.action=='bootstrap':
        ssh(a.host,'mkdir -p FruitFlyBrain/data FruitFlyBrain/runs && python3 -m venv FruitFlyBrain/.venv && FruitFlyBrain/.venv/bin/python -m pip install -r FruitFlyBrain/current/requirements.lock && cd FruitFlyBrain/current && ../..//.venv/bin/python scripts/build_cuda.py')
        return
    if a.action=='collect':
        out=ROOT/'runs'/a.host;out.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az',f'{a.host}:FruitFlyBrain/runs/',str(out)+'/'],check=True);return
    manifest=source_manifest();digest=manifest['sha256']
    # Upload staging is temporary; research PDFs stay only in the Mac's PLAN.
    with tempfile.TemporaryDirectory(prefix='fruitflybrain-upload-') as tmp:
        stage=Path(tmp)/'source';stage.mkdir()
        for name in manifest['files']:
            dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        atomic_json(stage/'SOURCE_MANIFEST.json',manifest)
        archive=Path(tmp)/f'{digest}.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for path in sorted(stage.rglob('*')):
                if path.is_file():tf.add(path,arcname=str(path.relative_to(stage)))
        target=f'FruitFlyBrain/releases/{digest}'
        ssh(a.host,f'mkdir -p {target} FruitFlyBrain/snapshots FruitFlyBrain/data FruitFlyBrain/runs')
        subprocess.run(['rsync','-az',str(stage)+'/',f'{a.host}:{target}/'],check=True)
        subprocess.run(['rsync','-az',str(archive),f'{a.host}:FruitFlyBrain/snapshots/'],check=True)
        verify="import json,hashlib,pathlib; p=pathlib.Path('.'); m=json.loads((p/'SOURCE_MANIFEST.json').read_text()); assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m['files'].items()); assert not (p/'PLAN').exists(); print('verified',m['sha256'])"
        ssh(a.host,f'cd {target} && python3 -c {shlex.quote(verify)} && cd ../.. && ln -sfn releases/{digest} current')
    print(json.dumps({'host':a.host,'source_sha256':digest,'remote_release':target,'local_staging':'automatically removed'},indent=2))

if __name__=='__main__':main()
