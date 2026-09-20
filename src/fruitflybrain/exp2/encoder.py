"""Measured reset probes. Cached features are never advertised as online brain state."""
import hashlib
from pathlib import Path
import numpy as np
from ..atlas import Atlas
from ..backends import factory,label
from ..graph import Graph
from ..model import Parameters
from ..provenance import canonical_hash
from ..stimuli import sample_frame

PATTERNS=('A','B')
def image_for(name):
    y,x=np.mgrid[:24,:32]
    if name not in PATTERNS:raise ValueError('Unknown pattern')
    return (((x//8 if name=='A' else y//6)%2)*.9+.05).astype(np.float32)

def pixel_features(image):
    return image.reshape(4,6,4,8).mean(axis=(1,3)).ravel().astype(np.float64)

class NeuralEncoder:
    def __init__(self,graph_path,backend):
        self.g=Graph.load(graph_path);self.atlas=Atlas(self.g,Path(graph_path)/'neurons.feather');self.params=Parameters()
        self.features={};self.records={};self.counts={};self.backend=label(factory(backend))
        self.images={k:image_for(k) for k in PATTERNS}
        input_ids=self.atlas.input_ids
        if not len(input_ids):raise ValueError('This dataset has no usable L1 input coordinates')
        bins=np.minimum((self.atlas.uv*4).astype(int),3)
        assignments=bins[:,1]*4+bins[:,0]+(self.atlas.input_sides=='R')*16
        with factory(backend)(self.g,self.params) as model:
            for name,image in self.images.items():
                model.reset();drive=np.zeros((40,self.g.n),np.float32)
                drive[:,input_ids]=sample_frame(image,self.atlas.uv)*.6
                out=model.run(drive);rates=out['counts'].astype(np.float64)/.04
                counts=np.bincount(assignments,minlength=32)
                features=np.bincount(assignments,weights=rates[input_ids],minlength=32)/np.maximum(counts,1)
                self.features[name]=features;self.counts[name]=out['counts']
                self.records[name]={'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'pixels':image.round(4).tolist(),'features_hz':features.tolist(),
                    'total_spikes':int(out['counts'].sum()),'network_rate_hz':float(rates.mean()),
                    'mean_voltage_mv':float(out['voltage'].mean()),'min_voltage_mv':float(out['voltage'].min()),
                    'max_voltage_mv':float(out['voltage'].max()),'groups_hz':{k:float(rates[v].mean()) for k,v in self.atlas.groups.items()}}
        distance=float(np.linalg.norm(self.features['A']-self.features['B']))
        if not np.isfinite(distance) or distance<1e-6:raise ValueError('Neural probes do not distinguish these patterns; training is blocked')
        self.pixel_templates=np.array([pixel_features(self.images[n]) for n in PATTERNS])
        self.templates=np.array([self.features[n] for n in PATTERNS])
        self.meta={'dataset':self.g.metadata.get('dataset'),'synthetic':self.atlas.synthetic,'neurons':self.g.n,'edges':self.g.m,
            'graph':self.g.metadata,'mapping':self.atlas.mapping,'parameters':vars(self.params),'backend':self.backend,
            'probe_duration_ms':40,'gain_mv_per_tick':.6,'feature_policy':'32 L1 spatial mean-rate bins; 4x4 per eye',
            'decoder':'nearest of two calibrated fixed neural templates','template_distance':distance,
            'execution':'Cached deterministic reset probes; no continuous neural state during maze movement',
            'trainable':'External tabular Q values only; neural graph/dynamics/templates fixed','patterns':self.records}
        self.identity=canonical_hash(self.meta);self.meta['sha256']=self.identity
    def tokens(self,assignment,mode='neural'):
        # Image identity selects an already measured response to that exact image.
        if mode=='neural':vectors=[self.features[n] for n in assignment];templates=self.templates
        elif mode=='image':vectors=[pixel_features(self.images[n]) for n in assignment];templates=self.pixel_templates
        else:raise ValueError('Unknown encoder mode')
        return tuple(int(np.argmin(np.linalg.norm(templates-v,axis=1))) for v in vectors)
    def save(self,path):
        from ..provenance import atomic_json
        atomic_json(Path(path)/'neural_probes.json',self.meta)
        np.savez_compressed(Path(path)/'neural_probes.npz',counts_A=self.counts['A'],counts_B=self.counts['B'],
                            features_A=self.features['A'],features_B=self.features['B'])
    def public(self):
        return {k:self.meta[k] for k in ('dataset','synthetic','neurons','edges','backend','probe_duration_ms','template_distance','execution','trainable','patterns','sha256')}
