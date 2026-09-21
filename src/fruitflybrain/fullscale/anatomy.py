"""Load every retained neuron and edge; make a sign-stratified wiring null."""
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from ..graph import Graph
from ..provenance import sha256,atomic_json
from .model import normalized_weights


def transpose(indices,n):
    order=np.argsort(indices,kind='stable').astype(np.int32)
    ptr=np.zeros(n+1,np.int32);ptr[1:]=np.cumsum(np.bincount(indices,minlength=n))
    return ptr,order


def load(path,p,out=None):
    path=Path(path);g=Graph.load(path)
    if g.metadata.get('dataset')!='male-cns:v1.0' or (g.n,g.m)!=(166700,25582938):
        raise ValueError('Full benchmark requires the complete prepared MaleCNS graph; no fallback')
    for name in ('neurons.feather','anatomical_contacts.npy'):
        if sha256(path/name)!=g.metadata.get('supplemental_files',{}).get(name):raise ValueError('Unverified '+name)
    table=feather.read_table(path/'neurons.feather',columns=['bodyId','superclass']).to_pydict()
    labels=dict(zip(table['bodyId'],table['superclass']))
    sensory={'ol_sensory','cb_sensory','vnc_sensory','sensory_ascending','sensory_descending'}
    mask=np.array([labels.get(int(i)) in sensory for i in g.ids]);rng=np.random.default_rng(p['mapping_seed'])
    channels=np.full(g.n,-1,np.int32);channels[mask]=rng.integers(0,24,mask.sum(),dtype=np.int32)
    signs=np.zeros(g.n,np.float32);signs[mask]=rng.choice(np.array([-1,1],np.float32),mask.sum())
    base,post=normalized_weights(g);rewired=g.indices.copy();rng=np.random.default_rng(p['rewiring_seed'])
    stats={}
    for name,which in [('positive',base>0),('negative',base<0)]:
        before=rewired[which].copy();after=before.copy();rng.shuffle(after);rewired[which]=after
        if not np.array_equal(np.bincount(before,minlength=g.n),np.bincount(after,minlength=g.n)):raise AssertionError('Source degrees changed')
        stats[name+'_edges']=int(which.sum())
    pairs=post.astype(np.int64)*g.n+rewired;pairs.sort()
    stats.update(parallel_excess_edges=int(np.count_nonzero(np.diff(pairs)==0)),self_edges=int(np.count_nonzero(post==rewired)),
                 source_indices_changed=int(np.count_nonzero(rewired!=g.indices)),degree_counts_verified=True,
                 description='Sign-stratified configuration multigraph; parallel pairs and self-edges allowed; one fixed null')
    meta={'graph_sha256':g.metadata['graph_sha256'],'neurons':g.n,'edge_slots':g.m,'nonzero_signed_edges':int(np.count_nonzero(base)),
      'input_neurons':int(mask.sum()),'sensory_superclasses':sorted(sensory),'channels_per_feature':np.bincount(channels[mask],minlength=24).tolist(),
      'mapping_seed':p['mapping_seed'],'rewiring_seed':p['rewiring_seed'],'rewiring':stats,'initial_max_row_abs_sum':float(np.bincount(post,weights=np.abs(base),minlength=g.n).max()),
      'full_graph':True,'model':'Bounded recurrent signed rate deviations, all retained nodes; artificial inputs/outputs and TD gradients'}
    if out:
        atomic_json(out/'anatomy.json',meta)
        np.savez(out/'input_mapping.npz',body_ids=g.ids,channels=channels,signs=signs)
        np.save(out/'rewired_indices.npy',rewired)
    return g,base,channels,signs,rewired,meta
