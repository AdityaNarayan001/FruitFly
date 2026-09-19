"""Deterministic teaching fixture, explicitly not an extracted connectome."""
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from .graph import Graph
from .provenance import atomic_json,sha256

def prepare(path):
    path=Path(path)
    if path.exists():
        g=Graph.load(path)
        if g.metadata.get('dataset')!='synthetic_demo:v1':raise ValueError('Demo path contains another dataset')
        expected=g.metadata['supplemental_files']['neurons.feather']
        if sha256(path/'neurons.feather')!=expected:raise ValueError('Demo annotation checksum mismatch')
        return path
    rows=[];pre=[];post=[];weights=[]
    # 2 sides x 4 illustrative layers x 8x4 columns = 256 nodes.
    for side_idx,side in enumerate(('L','R')):
        for layer,typ in enumerate(('L1','T4a','T5a','DNdemo')):
            for y in range(4):
                for x in range(8):
                    i=len(rows)
                    rows.append({'bodyId':1_000_000+i,'type':typ,'somaSide':side,
                        'superclass':'descending_neuron' if layer==3 else 'ol_intrinsic',
                        'assignedOlHex1':float(x) if layer==0 else None,'assignedOlHex2':float(y) if layer==0 else None,
                        'somaLocation':[side_idx*140+x*10,y*15,layer*25],'tosomaLocation':None})
                    if layer:
                        pre.append(i-32);post.append(i);weights.append(1.2)
                        if x>0:pre.append(i-33);post.append(i);weights.append(.3)
    ids=np.asarray([r['bodyId'] for r in rows],np.int64)
    graph=Graph.edges(ids,pre,post,weights,{'dataset':'synthetic_demo:v1','dataset_kind':'synthetic',
        'interpretation':'Invented 256-node teaching fixture. All positions, labels and edges are synthetic; no MaleCNS data.',
        'license':'Project-authored teaching fixture','population_rule':'All synthetic nodes'})
    path.parent.mkdir(parents=True,exist_ok=True);graph.save(path)
    feather.write_feather(pa.Table.from_pylist(rows),path/'neurons.feather')
    meta=json.loads((path/'graph.json').read_text());meta['supplemental_files']={'neurons.feather':sha256(path/'neurons.feather')}
    atomic_json(path/'graph.json',meta)
    return path
