from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class Parameters:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_s_ms: float = 5.0
    rest_mv: float = -52.0
    threshold_mv: float = -45.0
    refractory_ms: float = 2.0
    delay_ms: float = 2.0

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()): raise ValueError('Parameters must be finite')
        if not (0 < self.dt_ms <= min(self.tau_m_ms,self.tau_s_ms)): raise ValueError('Invalid integration time constants')
        if self.threshold_mv <= self.rest_mv: raise ValueError('Threshold must exceed rest')
        if self.refractory_ms < 0 or self.delay_ms < self.dt_ms: raise ValueError('Invalid refractory/delay')
        for value in (self.refractory_ms,self.delay_ms):
            if not math.isclose(value/self.dt_ms, round(value/self.dt_ms)): raise ValueError('Delays must be whole ticks')
        return self

    @property
    def delay_ticks(self): return round(self.delay_ms/self.dt_ms)
    @property
    def refractory_ticks(self): return round(self.refractory_ms/self.dt_ms)

class Reference:
    """Small-network reference with explicit sequential incoming-edge reduction."""
    def __init__(self, graph, parameters=Parameters()):
        self.g = graph.validate(); self.p = parameters.validate(); self.reset()

    def reset(self):
        self.tick=0; n=self.g.n
        self.v=np.full(n,self.p.rest_mv,np.float32); self.s=np.zeros(n,np.float32)
        self.ref=np.zeros(n,np.int32); self.history=np.zeros((self.p.delay_ticks,n),np.float32)

    def run(self, drive, record=False):
        drive=validate_drive(drive,self.g.n)
        counts=np.zeros(self.g.n,np.int32); spikes=[]; volts=[]
        decay=np.float32(math.exp(-self.p.dt_ms/self.p.tau_s_ms)); ratio=np.float32(self.p.dt_ms/self.p.tau_m_ms)
        for current in drive:
            delayed=self.history[self.tick % self.p.delay_ticks].copy()
            for i in range(self.g.n):
                total=np.float32(0)
                for e in range(self.g.indptr[i],self.g.indptr[i+1]):
                    total=np.float32(total+np.float32(self.g.weights[e]*delayed[self.g.indices[e]]))
                self.s[i]=np.float32(decay*self.s[i]+total)
            active=self.ref==0
            self.ref[~active]-=1
            self.v[active] += ratio*(np.float32(self.p.rest_mv)-self.v[active])+self.s[active]+current[active]
            z=(active & (self.v>=self.p.threshold_mv)).astype(np.float32)
            self.v[z>0]=self.p.rest_mv; self.ref[z>0]=self.p.refractory_ticks
            self.history[self.tick % self.p.delay_ticks]=z; self.tick+=1; counts+=z.astype(np.int32)
            if record: spikes.append(z.copy()); volts.append(self.v.copy())
        out={'counts': counts,'voltage': self.v.copy()}
        if record: out.update(spikes=np.asarray(spikes),voltages=np.asarray(volts))
        return out

def validate_drive(drive,n):
    a=np.ascontiguousarray(drive,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=n or a.shape[0]<1 or not np.isfinite(a).all():
        raise ValueError('Drive must be finite nonempty [ticks, neurons]')
    return a
