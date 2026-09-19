"""Import all annotation rows with a nonempty superclass, without strength pruning."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from .graph import Graph
from .provenance import atomic_json,sha256

SIGNS={'acetylcholine':1,'gaba':-1,'glutamate':-1,'dopamine':0,'serotonin':0,'octopamine':0}

def prepare(data,output):
    data=Path(data);final_output=Path(output)
    if final_output.exists():raise FileExistsError('Graph outputs are immutable; select a new output directory')
    output=final_output.with_name(final_output.name+'.partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    lock=json.loads((data/'downloads.json').read_text())
    for item in lock['files'].values():
        if sha256(data/item['name'])!=item['sha256']:raise ValueError('Source data checksum mismatch')
    ann=feather.read_table(data/lock['files']['annotations']['name'])
    rows=ann.to_pylist();selected=[r for r in rows if r.get('superclass') not in (None,'')]
    selected.sort(key=lambda r:r['bodyId']);ids=np.asarray([r['bodyId'] for r in selected],np.int64)
    if not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Empty/duplicate retained IDs')
    nt=feather.read_table(data/lock['files']['neurotransmitters']['name'])
    nt=nt.filter(pc.is_in(nt['body'],value_set=pa.array(ids)))
    ntrows=nt.select(['body','consensus_nt']).to_pylist()
    if len({r['body'] for r in ntrows})!=len(ntrows):raise ValueError('Duplicate transmitter IDs')
    labels={r['body']:(r['consensus_nt'] or 'unknown').lower() for r in ntrows}
    signs=np.asarray([SIGNS.get(labels.get(int(i),'unknown'),0) for i in ids],np.float32)
    print(f'Retained {len(ids):,} annotated neurons; loading connection table',flush=True)
    table=feather.read_table(data/lock['files']['connections']['name'],memory_map=True)
    pieces=[];retained_contacts=0
    for batch in table.to_batches(max_chunksize=1_000_000):
        pre=batch.column('body_pre').to_numpy();post=batch.column('body_post').to_numpy();weight=batch.column('weight').to_numpy()
        if pre.dtype!=np.int64 or post.dtype!=np.int64 or weight.dtype!=np.int64:raise ValueError('Expected exact int64 source fields')
        pi=np.searchsorted(ids,pre);qi=np.searchsorted(ids,post)
        valid=(pi<len(ids))&(qi<len(ids))
        pi=np.minimum(pi,len(ids)-1);qi=np.minimum(qi,len(ids)-1)
        valid &= (ids[pi]==pre)&(ids[qi]==post)
        if np.any(weight[valid]<=0):raise ValueError('Nonpositive contact counts')
        pieces.append((pi[valid].astype(np.int32),qi[valid].astype(np.int32),weight[valid]))
    pre=np.concatenate([p[0] for p in pieces]);post=np.concatenate([p[1] for p in pieces]);contacts=np.concatenate([p[2] for p in pieces]);del pieces
    order=np.lexsort((pre,post));pre=pre[order];post=post[order];contacts=contacts[order]
    duplicate=(pre[1:]==pre[:-1])&(post[1:]==post[:-1])
    if duplicate.any():raise ValueError('Duplicate neuron-pair rows require an explicit aggregation decision')
    ptr=np.zeros(len(ids)+1,np.int64);ptr[1:]=np.cumsum(np.bincount(post,minlength=len(ids)))
    weights=(contacts.astype(np.float32)*np.float32(.275)*signs[pre]).astype(np.float32)
    degree=np.bincount(pre,minlength=len(ids))+np.bincount(post,minlength=len(ids))
    audit={'dataset':lock['dataset'],'license':lock['license'],'source_files':lock['files'],
           'population_rule':'All annotations with nonempty superclass, including tbc labels; retain isolated nodes',
           'source_annotation_rows':len(rows),'excluded_annotation_rows':len(rows)-len(ids),
           'source_segment_edge_rows':table.num_rows,'retained_contact_count':int(contacts.sum()),
           'excluded_segment_edge_rows':table.num_rows-len(pre),'isolated_retained_neurons':int(np.sum(degree==0)),
           'transmitter_neuron_counts':dict(Counter(labels.get(int(i),'unknown') for i in ids)),
           'status_counts':dict(Counter(str(r.get('status')) for r in selected)),
           'superclass_counts':dict(Counter(str(r.get('superclass')) for r in selected)),
           'fast_synapse_sign_policy':SIGNS,'zero_fast_weight_edges':int(np.sum(weights==0)),
           'weight_scale_mv_per_contact':.275,'zero_fast_weights_are_retained':True,
           'interpretation':'Anatomical import plus assumed fast-synapse signs; no retinotopic mapping or biological validation',
           'prepared_utc':datetime.now(timezone.utc).isoformat()}
    g=Graph(ids,ptr,pre,weights,audit).validate();g.save(output)
    np.save(output/'anatomical_contacts.npy',contacts,allow_pickle=False)
    # Keep Arrow's ordered dictionary metadata intact; rebuilding Python rows
    # changes the statusLabel dictionary's ordered flag in the public release.
    retained_ann=ann.take(pc.index_in(pa.array(ids),value_set=ann['bodyId']))
    feather.write_feather(retained_ann,output/'neurons.feather')
    audit=json.loads((output/'graph.json').read_text())
    audit['supplemental_files']={name:sha256(output/name) for name in ('neurons.feather','anatomical_contacts.npy')}
    atomic_json(output/'graph.json',audit)
    output.rename(final_output)
    print(json.dumps({k:audit[k] for k in ('n_neurons','n_edges','retained_contact_count','isolated_retained_neurons','zero_fast_weight_edges','graph_sha256')},indent=2),flush=True)
