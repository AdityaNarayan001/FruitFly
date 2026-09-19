from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .provenance import atomic_json, sha256, canonical_hash

@dataclass
class Graph:
    ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    metadata: dict
    @property
    def n(self): return len(self.ids)
    @property
    def m(self): return len(self.indices)

    def validate(self):
        if any(a.ndim!=1 or not a.flags.c_contiguous for a in (self.ids,self.indptr,self.indices,self.weights)):raise ValueError('Graph arrays must be contiguous vectors')
        if self.ids.dtype!=np.int64 or self.indptr.dtype!=np.int64 or self.indices.dtype!=np.int32 or self.weights.dtype!=np.float32:
            raise ValueError('Graph arrays require int64 IDs/pointers, int32 indices, float32 weights')
        if not self.n or self.n>=2**31 or len(np.unique(self.ids))!=self.n: raise ValueError('Invalid neuron IDs')
        if len(self.indptr)!=self.n+1 or self.indptr[0]!=0 or self.indptr[-1]!=self.m or np.any(np.diff(self.indptr)<0): raise ValueError('Invalid CSR pointers')
        if len(self.weights)!=self.m or np.any(self.indices<0) or np.any(self.indices>=self.n) or not np.isfinite(self.weights).all(): raise ValueError('Invalid edges')
        return self

    @classmethod
    def edges(cls, ids, pre, post, weights, metadata=None):
        if np.asarray(ids).dtype.kind not in 'iu':raise ValueError('Neuron IDs must be supplied as integers')
        ids=np.asarray(ids,dtype=np.int64); pre=np.asarray(pre,dtype=np.int32); post=np.asarray(post,dtype=np.int32)
        weights=np.asarray(weights,dtype=np.float32)
        if len(pre)!=len(post) or len(pre)!=len(weights) or np.any(post<0) or np.any(post>=len(ids)): raise ValueError('Invalid edge arrays')
        order=np.lexsort((pre,post)); ptr=np.zeros(len(ids)+1,np.int64)
        ptr[1:]=np.cumsum(np.bincount(post,minlength=len(ids)))
        return cls(ids,ptr,pre[order],weights[order],metadata or {}).validate()

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=False)
        for name in ('ids','indptr','indices','weights'): np.save(path/f'{name}.npy',getattr(self,name),allow_pickle=False)
        meta=dict(self.metadata,n_neurons=self.n,n_edges=self.m)
        meta['arrays']={p.name:sha256(p) for p in sorted(path.glob('*.npy'))}
        meta['graph_sha256']=canonical_hash(meta['arrays'])
        atomic_json(path/'graph.json',meta)

    @classmethod
    def load(cls,path,verify=True):
        path=Path(path); meta=json.loads((path/'graph.json').read_text())
        if verify:
            for name,digest in meta['arrays'].items():
                if sha256(path/name)!=digest: raise ValueError(f'Graph checksum mismatch: {name}')
        return cls(*(np.load(path/f'{n}.npy',allow_pickle=False) for n in ('ids','indptr','indices','weights')),meta).validate()

def fixture():
    return Graph.edges(np.arange(8,dtype=np.int64),[0,1,2,3,4,5,6],[2,3,4,5,6,7,7],
                       [8,8,5,5,-4,4,2],{'dataset':'synthetic_fixture','interpretation':'Engineering only; not MaleCNS or a motion detector'})
