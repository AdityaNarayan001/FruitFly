"""Portable sparse CPU backend; float32 dynamics matching the reference equations."""
import math
import numpy as np
from scipy.sparse import csr_matrix
from .model import Parameters,validate_drive

class Cpu:
    def __init__(self,graph,parameters=Parameters()):
        self.g=graph.validate();self.p=parameters.validate()
        self.matrix=csr_matrix((graph.weights,graph.indices,graph.indptr),shape=(graph.n,graph.n),copy=False)
        self.reset()
    def reset(self):
        self.tick=0;n=self.g.n
        self.v=np.full(n,self.p.rest_mv,np.float32);self.s=np.zeros(n,np.float32)
        self.ref=np.zeros(n,np.int32);self.history=np.zeros((self.p.delay_ticks,n),np.float32)
    def run(self,drive,record=False):
        drive=validate_drive(drive,self.g.n)
        if record and drive.size>5_000_000:raise ValueError('Full-state recording capped; use chunks')
        counts=np.zeros(self.g.n,np.int32);spikes=[];volts=[]
        decay=np.float32(math.exp(-self.p.dt_ms/self.p.tau_s_ms));ratio=np.float32(self.p.dt_ms/self.p.tau_m_ms)
        for current in drive:
            delayed=self.history[self.tick%self.p.delay_ticks]
            self.s=decay*self.s+self.matrix.dot(delayed)
            active=self.ref==0;self.ref[~active]-=1
            self.v[active]+=ratio*(np.float32(self.p.rest_mv)-self.v[active])+self.s[active]+current[active]
            z=(active&(self.v>=self.p.threshold_mv)).astype(np.float32)
            self.v[z>0]=self.p.rest_mv;self.ref[z>0]=self.p.refractory_ticks
            self.history[self.tick%self.p.delay_ticks]=z;self.tick+=1;counts+=z.astype(np.int32)
            if record:spikes.append(z.copy());volts.append(self.v.copy())
        if not np.isfinite(self.v).all():raise FloatingPointError('Nonfinite voltage')
        out={'counts':counts,'voltage':self.v.copy()}
        if record:out.update(spikes=np.asarray(spikes),voltages=np.asarray(volts))
        return out
    def close(self):pass
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
