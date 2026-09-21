"""Two-tick bounded recurrence and semi-gradient TD; CPU reference and CUDA wrapper."""
import ctypes as C
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from ..provenance import ROOT


def normalized_weights(graph):
    post=np.repeat(np.arange(graph.n,dtype=np.int32),np.diff(graph.indptr))
    sums=np.bincount(post,weights=np.abs(graph.weights),minlength=graph.n)
    w=(graph.weights*(.45/np.maximum(sums[post],1e-30))).astype(np.float32)
    return w,post


class CPU:
    def __init__(self,ptr,indices,base,channels,signs,seed,plastic=True):
        self.n=len(ptr)-1;self.ptr=ptr;self.indices=indices;self.base=base.copy();self.w=base.copy()
        self.post=np.repeat(np.arange(self.n),np.diff(ptr));self.channels=channels;self.signs=signs
        self.r=np.random.default_rng(seed).normal(0,.02/np.sqrt(self.n),(4,self.n)).astype(np.float32)
        self.b=np.zeros(4,np.float32);self.plastic=plastic;self.reset()
    def reset(self):self.h=np.zeros(self.n,np.float32);self.cache=None;self.seen=np.zeros(self.n,bool)
    def matrix(self):return csr_matrix((self.w,self.indices,self.ptr),shape=(self.n,self.n))
    def step(self,obs,preview=False):
        drive=np.where(self.channels>=0,np.asarray(obs,np.float32)[np.maximum(self.channels,0)]*self.signs,0)
        mat=self.matrix();hs=[self.h.copy()];ts=[]
        for _ in range(2):
            t=np.tanh(mat@hs[-1]+drive);ts.append(t);hs.append(.5*hs[-1]+.5*t)
        q=self.r@hs[-1]+self.b
        if not preview:
            self.h=hs[-1];self.cache=(hs,ts,q.copy());self.seen |= np.abs(self.h)>1e-6
        return q
    def gradients(self,action):
        hs,ts,_=self.cache;dh=self.r[action].copy();dw=np.zeros_like(self.w);mat=self.matrix()
        for j in (1,0):
            dz=.5*dh*(1-ts[j]*ts[j]);dw+=dz[self.post]*hs[j][self.indices]
            if j:dh=.5*dh+mat.T@dz
        return dw
    def learn(self,action,target):
        hs,ts,q=self.cache;delta=float(np.clip(target-q[action],-1,1));dw=self.gradients(action)
        if self.plastic:
            self.w[:]=np.clip(self.w+.01*delta*dw,np.minimum(0,2*self.base),np.maximum(0,2*self.base))
        rate=.25*delta/(1+float(np.dot(self.h,self.h)))
        self.r[action]+=rate*self.h;self.b[action]+=rate
        return delta
    def parameters(self):return self.w.copy(),self.r.copy(),self.b.copy()
    def set_parameters(self,w,r,b):self.w[:]=w;self.r[:]=r;self.b[:]=b
    def state(self):return self.h.copy(),self.seen.copy()
    def close(self):pass


class CUDA:
    def __init__(self,ptr,indices,base,channels,signs,seed,plastic=True,library=None,transpose=None):
        self.n=len(ptr)-1;self.m=len(base);self.plastic=plastic
        self.lib=C.CDLL(str(library or ROOT/'build/libffb_fullscale.so'))
        P=C.c_void_p;I=C.c_int;F=C.c_float
        signatures={'fs_create':([I,I]+[P]*11,P),'fs_reset':([P],I),'fs_step':([P,P,I,P],I),
          'fs_learn':([P,I,F,I,P],I),'fs_get':([P,P,P,P],I),'fs_set':([P,P,P,P],I),
          'fs_state':([P,P,P],I),'fs_destroy':([P],None),'fs_error':([],C.c_char_p)}
        for name,(args,result) in signatures.items():f=getattr(self.lib,name);f.argtypes=args;f.restype=result
        post=np.repeat(np.arange(self.n,dtype=np.int32),np.diff(ptr))
        if transpose is None:
            order=np.argsort(indices,kind='stable').astype(np.int32);tp=np.zeros(self.n+1,np.int32);tp[1:]=np.cumsum(np.bincount(indices,minlength=self.n))
        else:tp,order=transpose
        r=np.random.default_rng(seed).normal(0,.02/np.sqrt(self.n),(4,self.n)).astype(np.float32)
        b=np.zeros(4,np.float32)
        arrays=[np.asarray(ptr,np.int32),indices,post,tp,order,base,channels,signs,r,b,np.zeros(1,np.float32)]
        arrays=[np.ascontiguousarray(v) for v in arrays]
        self.handle=self.lib.fs_create(self.n,self.m,*[self.pointer(v) for v in arrays])
        if not self.handle:raise RuntimeError(self.lib.fs_error().decode())
    @staticmethod
    def pointer(a):return a.ctypes.data_as(C.c_void_p)
    def check(self,status):
        if status:raise RuntimeError(self.lib.fs_error().decode())
    def reset(self):self.check(self.lib.fs_reset(self.handle))
    def step(self,obs,preview=False):
        obs=np.ascontiguousarray(obs,np.float32);out=np.empty(4,np.float32)
        if obs.shape!=(24,) or not np.isfinite(obs).all():raise ValueError('Expected 24 finite observations')
        self.check(self.lib.fs_step(self.handle,self.pointer(obs),int(preview),self.pointer(out)))
        if not np.isfinite(out).all():raise FloatingPointError('Nonfinite Q output')
        return out
    def learn(self,action,target):
        if not 0<=action<4 or not np.isfinite(target):raise ValueError('Invalid TD target/action')
        out=np.empty(1,np.float32);self.check(self.lib.fs_learn(self.handle,action,float(target),int(self.plastic),self.pointer(out)));return float(out[0])
    def parameters(self):
        w=np.empty(self.m,np.float32);r=np.empty((4,self.n),np.float32);b=np.empty(4,np.float32)
        self.check(self.lib.fs_get(self.handle,*map(self.pointer,(w,r,b))));return w,r,b
    def set_parameters(self,w,r,b):
        a=[np.ascontiguousarray(v,np.float32) for v in (w,r,b)]
        if [v.shape for v in a]!=[(self.m,),(4,self.n),(4,)]:raise ValueError('Invalid checkpoint shapes')
        if not all(np.isfinite(v).all() for v in a):raise ValueError('Nonfinite checkpoint')
        self.check(self.lib.fs_set(self.handle,*map(self.pointer,a)))
    def state(self):
        h=np.empty(self.n,np.float32);seen=np.empty(self.n,np.uint8);self.check(self.lib.fs_state(self.handle,self.pointer(h),self.pointer(seen)));return h,seen.astype(bool)
    def close(self):
        if getattr(self,'handle',None):self.lib.fs_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
