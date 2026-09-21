#!/usr/bin/env python3
"""Verify comparison artifacts; explicitly audit the first-version progress-hash fix."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fruitflybrain.provenance import sha256,atomic_json

def verify(path,repair=False):
    file=path/'manifest.json';m=json.loads(file.read_text())
    if m['status']!='complete':raise ValueError('Run is not complete')
    bad=[name for name,digest in m['outputs'].items() if sha256(path/name)!=digest]
    if bad:
        if not repair or bad!=['progress.json']:raise ValueError('Checksum mismatches: '+repr(bad))
        progress=json.loads((path/'progress.json').read_text())
        if progress.get('stage')!='complete':raise ValueError('Unexpected progress state')
        audit=path/'finalization_repair.json'
        if audit.exists():raise ValueError('An audit record already exists')
        atomic_json(audit,{'reason':'Initial runner hashed progress.json before publishing its final complete state. Scientific outputs are unchanged.',
            'original_manifest':m,'original_manifest_sha256':sha256(file),'final_progress_sha256':sha256(path/'progress.json')})
        m['outputs'].pop('progress.json');m['outputs'][audit.name]=sha256(audit)
        m['finalization_repair']='finalization_repair.json';atomic_json(file,m)
    print(f'Verified {len(m["outputs"])} immutable artifacts: {path}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--repair-progress-manifest',action='store_true');a=p.parse_args();verify(a.run,a.repair_progress_manifest)
