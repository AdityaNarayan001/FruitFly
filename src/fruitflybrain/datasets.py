import json
import urllib.request
import shutil
from datetime import datetime,timezone
from pathlib import Path
from .provenance import ROOT,sha256,atomic_json

def download(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    registry=json.loads((ROOT/'configs/datasets.json').read_text())
    lockpath=dest/'downloads.json'
    lock=json.loads(lockpath.read_text()) if lockpath.exists() else {'dataset':registry['dataset'],'license':registry['license'],'files':{}}
    for key,url in registry['files'].items():
        name=url.rsplit('/',1)[1];path=dest/name;prior=lock['files'].get(key)
        expected=registry.get('sha256',{}).get(key)
        if path.exists():
            actual=sha256(path)
            if not prior or actual!=prior['sha256'] or (expected and actual!=expected):raise ValueError(f'Existing data lacks matching integrity record: {path}')
            print('verified',name,flush=True);continue
        print('downloading',name,flush=True)
        partial=path.with_suffix('.partial')
        with urllib.request.urlopen(url,timeout=90) as response,open(partial,'wb') as out:
            shutil.copyfileobj(response,out,1024*1024)
            headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Content-Length','x-goog-hash')}
        if headers['Content-Length'] and partial.stat().st_size!=int(headers['Content-Length']):raise ValueError('Incomplete download')
        actual=sha256(partial)
        if expected and actual!=expected:raise ValueError(f'Pinned dataset checksum mismatch: {key}')
        partial.replace(path)
        lock['files'][key]={'name':name,'url':url,'sha256':actual,'bytes':path.stat().st_size,'headers':headers,'downloaded_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(lockpath,lock)
        print('saved',name,path.stat().st_size,flush=True)
    return lock

def inspect(dest):
    import pyarrow.feather as feather
    dest=Path(dest);lock=json.loads((dest/'downloads.json').read_text())
    for key,rec in lock['files'].items():
        t=feather.read_table(dest/rec['name'],memory_map=True)
        print(key,t.num_rows,str(t.schema),json.dumps(t.slice(0,2).to_pylist(),default=str),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['download','inspect']);p.add_argument('directory');a=p.parse_args()
    globals()[a.action](a.directory)
