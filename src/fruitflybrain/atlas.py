"""Annotations and explicit, exploratory L1 column-coordinate stimulation."""
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from .provenance import sha256,canonical_hash

REGIONS=['Visual input','Optic lobe','Visual projection','Central brain','Descending','Ventral / other']

class Atlas:
    def __init__(self,graph,path):
        self.graph=graph;self.synthetic=graph.metadata.get("dataset_kind")=="synthetic";path=Path(path)
        expected=graph.metadata.get('supplemental_files',{}).get('neurons.feather')
        if expected and sha256(path)!=expected:raise ValueError('Neuron annotation integrity mismatch')
        self.digest=sha256(path)
        table=feather.read_table(path);rows=table.to_pylist()
        by_id={int(r['bodyId']):r for r in rows}
        if len(by_id)!=graph.n or set(by_id)!=set(graph.ids.tolist()):raise ValueError('Atlas IDs do not match graph')
        self.rows=[by_id[int(i)] for i in graph.ids];self.lookup={int(v):i for i,v in enumerate(graph.ids)}
        self.types=np.array([r.get('type') or 'untyped' for r in self.rows])
        self.sides=np.array([r.get('somaSide') or '?' for r in self.rows])
        self.classes=np.array([r.get('superclass') or 'unknown' for r in self.rows])
        self.groups={}
        for side in ('L','R'):
            for label,mask in [('R1-R6',self.types=='R1-R6'),('L1',self.types=='L1'),('T4',np.char.startswith(self.types,'T4')),('T5',np.char.startswith(self.types,'T5')),('Descending',self.classes=='descending_neuron')]:
                ids=np.flatnonzero(mask&(self.sides==side)).astype(np.int32)
                if len(ids):self.groups[f'{label}_{side}']=ids
        self.region=np.full(graph.n,5,np.int32)
        for j,prefixes in enumerate([('ol_sensory',),('ol_intrinsic',),('visual_',),('cb_',),('descending_',)]):
            mask=np.zeros(graph.n,bool)
            for prefix in prefixes:mask|=np.char.startswith(self.classes,prefix)
            self.region[mask]=j
        mapped=[];xy=[];sides=[]
        for i,r in enumerate(self.rows):
            x,y=r.get('assignedOlHex1'),r.get('assignedOlHex2')
            if r.get('type')=='L1' and r.get('somaSide') in ('L','R') and x is not None and y is not None and np.isfinite([x,y]).all():
                mapped.append(i);xy.append([x,y]);sides.append(r['somaSide'])
        self.input_ids=np.asarray(mapped,np.int32);self.column_xy=np.asarray(xy,np.float32).reshape(-1,2);self.input_sides=np.asarray(sides)
        self.uv=self.column_xy.copy();self.bounds={}
        for side in ('L','R'):
            mask=self.input_sides==side
            if mask.any():
                lo=self.uv[mask].min(0);hi=self.uv[mask].max(0);self.bounds[side]={'min':lo.tolist(),'max':hi.tolist()}
                self.uv[mask]=(self.uv[mask]-lo)/np.maximum(hi-lo,1)
        self.mapping={'status':'exploratory_column_drive','input_type':'L1',
            'mapped_neurons':len(mapped),'total_L1':int(np.sum(self.types=='L1')),
            'coordinate_source':'MaleCNS assignedOlHex1/assignedOlHex2 annotations',
            'transform':'Each eye independently min-max normalized; no optical calibration or mirror convention inferred',
            'bounds':self.bounds,'annotation_sha256':self.digest,
            'limitation':'Direct stimulation of L1 column neurons; not a photoreceptor model or validated visual direction mapping'}
        if self.synthetic:
            self.mapping.update(status='synthetic_demo',coordinate_source='Invented teaching coordinates',limitation='All labels, positions and connections are invented for the portable demo; no biological inference')
        self.mapping['sha256']=canonical_hash({'ids':graph.ids[self.input_ids].tolist(),'uv':self.uv.tolist(),'sides':self.input_sides.tolist(),'policy':self.mapping})
        rng=np.random.default_rng(9000)
        candidates=[];positions=[];position_sources=[]
        for i,r in enumerate(self.rows):
            point=r.get('somaLocation');src='somaLocation'
            if point is None:point=r.get('tosomaLocation');src='tosomaLocation'
            if point is not None and len(point)==3 and np.isfinite(point).all():
                candidates.append(i);positions.append(point);position_sources.append(src)
        # Stratified by anatomical class, keeping enough points from small populations.
        candidates=np.asarray(candidates,np.int32);positions=np.asarray(positions,np.float32).reshape(-1,3)
        chosen=[]
        for region in range(6):
            pool=np.flatnonzero(self.region[candidates]==region)
            if len(pool):chosen.extend(rng.choice(pool,min(1000,len(pool)),replace=False).tolist())
        chosen=np.array(sorted(chosen),np.int32)
        self.display_ids=candidates[chosen];raw=positions[chosen]
        if len(raw):
            lo=raw.min(0);hi=raw.max(0);self.positions=((raw-(lo+hi)/2)/max(float((hi-lo).max()/2),1)).astype(np.float32)
        else:self.positions=np.zeros((0,3),np.float32)
        probes=[]
        for ids in self.groups.values():
            probes.extend(ids[np.linspace(0,len(ids)-1,min(4,len(ids)),dtype=int)].tolist())
        self.raster_ids=np.array(list(dict.fromkeys(probes))[:64],np.int32)

    def public(self):
        return {'dataset':self.graph.metadata.get('dataset','unknown'),'synthetic':self.synthetic,'neurons':self.graph.n,'edges':self.graph.m,'mapping':self.mapping,
                'groups':[{'key':k,'count':len(v)} for k,v in self.groups.items()],
                'regions':REGIONS,'display_count':len(self.display_ids),
                'points':self.positions.round(5).tolist(),'point_ids':self.graph.ids[self.display_ids].astype(str).tolist(),
                'point_types':self.types[self.display_ids].tolist(),'point_sides':self.sides[self.display_ids].tolist(),'point_regions':self.region[self.display_ids].tolist(),
                'raster_ids':self.graph.ids[self.raster_ids].astype(str).tolist(),'raster_types':self.types[self.raster_ids].tolist(),
                'view_note':'Synthetic node positions; not anatomy or connectome data.' if self.synthetic else 'Sampled annotated cell landmarks in reconstructed coordinates; not neuron meshes, complete connectivity or live tissue.'}

    def search(self,query):
        q=str(query).strip().lower()[:80]
        if not q:return []
        return [{'id':str(self.graph.ids[i]),'type':self.types[i],'side':self.sides[i],'class':self.classes[i]} for i in range(self.graph.n)
                if q in str(self.graph.ids[i]) or q in self.types[i].lower()][:30]

    def neuron(self,body_id):
        idx=self.lookup.get(int(body_id))
        if idx is None:raise ValueError('Unknown retained neuron ID')
        r=self.rows[idx];start,end=self.graph.indptr[idx:idx+2]
        edges=np.arange(start,end);best=edges[np.argsort(-np.abs(self.graph.weights[edges]))[:6]]
        return {'id':str(body_id),'index':idx,'type':str(self.types[idx]),'side':str(self.sides[idx]),'class':str(self.classes[idx]),
                'incoming_edges':int(end-start),'soma_location':r.get('somaLocation'),'column':[r.get('assignedOlHex1'),r.get('assignedOlHex2')],
                'strongest_inputs':[{'id':str(self.graph.ids[self.graph.indices[e]]),'type':str(self.types[self.graph.indices[e]]),'weight_mv':float(self.graph.weights[e])} for e in best]}
