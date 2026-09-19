import ctypes as C
import math
from pathlib import Path
import numpy as np
from .model import Parameters,validate_drive
from .provenance import ROOT

class Cuda:
    def __init__(self,graph,parameters=Parameters(),library=None):
        self.g=graph.validate();self.p=parameters.validate();self.handle=None
        self.lib=C.CDLL(str(library or ROOT/'build/libffb.so'))
        ptr=C.c_void_p
        self.lib.ffb_error.restype=C.c_char_p
        self.lib.ffb_create.argtypes=[C.c_int,C.c_int64,ptr,ptr,ptr,C.c_float,C.c_float,C.c_float,C.c_float,C.c_int,C.c_int]
        self.lib.ffb_create.restype=ptr
        self.lib.ffb_run.argtypes=[ptr,C.c_int,ptr,ptr,ptr,ptr,ptr];self.lib.ffb_run.restype=C.c_int
        self.lib.ffb_reset.argtypes=[ptr];self.lib.ffb_reset.restype=C.c_int
        self.lib.ffb_destroy.argtypes=[ptr];self.lib.ffb_destroy.restype=None
        p=self.p;g=self.g
        self.handle=self.lib.ffb_create(g.n,g.m,g.indptr.ctypes.data,g.indices.ctypes.data,g.weights.ctypes.data,
              p.dt_ms/p.tau_m_ms,math.exp(-p.dt_ms/p.tau_s_ms),p.rest_mv,p.threshold_mv,p.delay_ticks,p.refractory_ticks)
        if not self.handle: raise RuntimeError(self.lib.ffb_error().decode())

    def reset(self):
        if self.lib.ffb_reset(self.handle):raise RuntimeError(self.lib.ffb_error().decode())

    def run(self,drive,record=False):
        a=validate_drive(drive,self.g.n)
        if record and a.size>5_000_000:raise ValueError('Full-state recording capped; use chunked population recording')
        counts=np.zeros(self.g.n,np.int32);v=np.empty(self.g.n,np.float32)
        z=np.empty_like(a) if record else None;vs=np.empty_like(a) if record else None
        err=self.lib.ffb_run(self.handle,len(a),a.ctypes.data,counts.ctypes.data,v.ctypes.data,
                             z.ctypes.data if record else None,vs.ctypes.data if record else None)
        if err:raise RuntimeError(self.lib.ffb_error().decode())
        if not np.isfinite(v).all():raise FloatingPointError('Nonfinite voltage')
        out={'counts':counts,'voltage':v}
        if record:out.update(spikes=z,voltages=vs)
        return out

    def close(self):
        if self.handle:self.lib.ffb_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
