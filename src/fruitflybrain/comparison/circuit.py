"""Recorded KC -> MBON edges, with an explicit reduced-model boundary."""
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from ..graph import Graph
from ..provenance import canonical_hash,sha256


def rewire(mask, weights, seed=9173):
    """Bipartite swaps preserve both degrees; shuffle each row's strength multiset."""
    rng=np.random.default_rng(seed); result=mask.copy(); edges=np.argwhere(result)
    accepted=0
    for attempt in range(max(1,len(edges))*40):
        if attempt>=len(edges)*30 and not np.array_equal(mask,result):break
        i,j=rng.integers(len(edges),size=2); a,b=edges[i]; c,d=edges[j]
        if a==c or b==d or result[a,d] or result[c,b]:continue
        result[a,b]=result[c,d]=False;result[a,d]=result[c,b]=True
        edges[i]=[a,d];edges[j]=[c,b];accepted+=1
    out=np.zeros_like(weights)
    for row in range(len(mask)):
        values=weights[row,mask[row]].copy();rng.shuffle(values);out[row,result[row]]=values
    if not np.array_equal(mask.sum(0),result.sum(0)) or not np.array_equal(mask.sum(1),result.sum(1)):
        raise AssertionError('Rewiring changed degrees')
    return result,out,{'accepted_swaps':accepted,'different_mask_entries':int(np.count_nonzero(mask!=result))}


def extract(path,count=256,seed=20260921):
    path=Path(path);g=Graph.load(path)
    if g.metadata.get('dataset')!='male-cns:v1.0':raise ValueError('This anatomical comparison requires real MaleCNS; synthetic fallback is forbidden')
    for name in ('neurons.feather','anatomical_contacts.npy'):
        expected=g.metadata.get('supplemental_files',{}).get(name)
        if not expected or sha256(path/name)!=expected:raise ValueError('Missing or mismatched anatomical file: '+name)
    t=feather.read_table(path/'neurons.feather',columns=['bodyId','class']).to_pydict()
    labels={int(i):c for i,c in zip(t['bodyId'],t['class'])}
    kc={i for i,b in enumerate(g.ids) if labels.get(int(b))=='Kenyon_Cell'}
    mb=[i for i,b in enumerate(g.ids) if labels.get(int(b))=='MBON']
    contacts=np.load(path/'anatomical_contacts.npy',mmap_mode='r',allow_pickle=False)
    all_edges=[]
    for post in mb:
        for edge in range(g.indptr[post],g.indptr[post+1]):
            pre=int(g.indices[edge])
            if pre in kc and g.weights[edge]>0 and contacts[edge]>0:
                all_edges.append((post,pre,float(contacts[edge])))
    connected=np.array(sorted({e[1] for e in all_edges}),dtype=int)
    if count>len(connected) or count<2:raise ValueError('Invalid selected KC count')
    chosen=np.sort(np.random.default_rng(seed).choice(connected,count,replace=False))
    lookup={int(v):i for i,v in enumerate(chosen)};rows={v:i for i,v in enumerate(mb)}
    weights=np.zeros((len(mb),count),np.float64)
    for post,pre,value in all_edges:
        if pre in lookup:weights[rows[post],lookup[pre]]+=value
    keep=weights.sum(1)>0;weights=weights[keep];mb=np.array(mb)[keep]
    weights/=weights.sum(1,keepdims=True);mask=weights>0
    random_mask,random_weights,stats=rewire(mask,weights)
    if not stats['different_mask_entries']:raise ValueError('Anatomical control could not be randomized')
    meta={'dataset':g.metadata['dataset'],'graph_sha256':g.metadata['graph_sha256'],
          'selection_seed':seed,'kc_body_ids':g.ids[chosen].astype(str).tolist(),
          'mbon_body_ids':g.ids[mb].astype(str).tolist(),'n_kc':count,'n_mbon':len(mb),'edges':int(mask.sum()),
          'selection':'Uniform sample of connected annotated Kenyon_Cell cells; recorded positive KC -> MBON pairs only',
          'initialization':'Contact counts, normalized to unit incoming sum; positive sign from imported modeling assumption',
          'model':'Reduced feedforward rate network; no recurrence, dopamine dynamics or full-brain LIF simulation',
          'rewiring':stats}
    meta['sha256']=canonical_hash(meta)
    return weights,random_weights,meta
