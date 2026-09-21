"""Tabular and masked neural Q learners with identical observation information."""
import hashlib
import numpy as np

OBS_SIZE=24
STATE_COUNT=2*9*9*16


def observe(world):
    # Both identities are visible and always opposite; no rewarded label is read.
    assignment=0 if world.patterns==('A','B') else 1
    x,y=world.pos;legal=np.zeros(4,bool);legal[world.legal()]=True
    wall_code=sum(int(v)<<i for i,v in enumerate(legal))
    index=(((assignment*9+y)*9+x)*16+wall_code)
    obs=np.zeros(OBS_SIZE,np.float64);obs[x]=1;obs[9+y]=1
    obs[18+assignment]=1;obs[20:]=legal
    return obs,index,legal


def all_observations():
    out=np.zeros((STATE_COUNT,OBS_SIZE),np.float64)
    for assignment in range(2):
        for y in range(9):
            for x in range(9):
                for bits in range(16):
                    i=(((assignment*9+y)*9+x)*16+bits)
                    out[i,x]=out[i,9+y]=out[i,18+assignment]=1
                    out[i,20:]=[(bits>>j)&1 for j in range(4)]
    return out


def fingerprint(*arrays):
    h=hashlib.sha256()
    for value in arrays:h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


class Table:
    def __init__(self):self.q=np.zeros((STATE_COUNT,4),np.float64)
    def values(self,indices):return self.q[indices]
    def update(self,b):
        for s,a,r,ns,terminal,legal in zip(b['state'],b['action'],b['reward'],b['next_state'],b['terminal'],b['next_legal']):
            target=r if terminal else r+.95*self.q[ns,legal].max()
            self.q[s,a]+=.25*(target-self.q[s,a])
    def state_hash(self):return fingerprint(self.q)
    def save(self,path):np.savez_compressed(path,q=self.q)
    @classmethod
    def load(cls,path):
        result=cls()
        with np.load(path,allow_pickle=False) as z:result.q[:]=z['q']
        if not np.isfinite(result.q).all():raise ValueError('Nonfinite checkpoint')
        return result
    @property
    def parameters(self):return self.q.size


class Network:
    """Fixed engineered KC encoder; trainable masked KC->MBON and action readout."""
    def __init__(self,weights,seed,plastic=True):
        rng=np.random.default_rng(seed)
        self.projection=rng.normal(size=(OBS_SIZE,weights.shape[1]))
        inputs=all_observations()
        z=np.dot(inputs,self.projection)/np.sqrt(np.maximum(inputs.sum(1,keepdims=True),1))
        self.kc=np.maximum(z-1.,0.)
        self.w=weights.copy();self.initial_w=weights.copy();self.mask=weights>0;self.plastic=plastic
        self.readout=rng.normal(0,.02,(len(weights),4));self.bias=np.zeros(4)
        self.target_w=self.w.copy();self.target_readout=self.readout.copy();self.target_bias=self.bias.copy()
        self.m=[np.zeros_like(a) for a in (self.w,self.readout,self.bias)]
        self.v=[np.zeros_like(a) for a in (self.w,self.readout,self.bias)];self.updates=0
    def forward(self,indices,target=False):
        w,r,b=(self.target_w,self.target_readout,self.target_bias) if target else (self.w,self.readout,self.bias)
        h=np.tanh(np.dot(self.kc[indices],w.T))
        return np.dot(h,r)+b,h
    def values(self,indices):return self.forward(indices)[0]
    def gradients(self,indices,actions,targets):
        q,h=self.forward(indices);delta=q[np.arange(len(indices)),actions]-targets
        dq=np.zeros_like(q);dq[np.arange(len(indices)),actions]=np.clip(delta,-1,1)/len(indices)
        dr=np.dot(h.T,dq);db=dq.sum(0)
        dw=np.dot((np.dot(dq,self.readout.T)*(1-h*h)).T,self.kc[indices])
        dw*=self.mask
        if not self.plastic:dw.fill(0)
        loss=np.where(abs(delta)<=1,.5*delta**2,abs(delta)-.5).mean()
        return float(loss),(dw,dr,db)
    def update(self,b):
        nxt=self.forward(b['next_state'],target=True)[0]
        targets=b['reward']+.95*(~b['terminal'])*np.where(b['next_legal'],nxt,-1e9).max(1)
        loss,grads=self.gradients(b['state'],b['action'],targets)
        norm=np.sqrt(sum(np.sum(g*g) for g in grads));scale=min(1.,5/max(norm,1e-12));self.updates+=1
        for i,(parameter,gradient) in enumerate(zip((self.w,self.readout,self.bias),grads)):
            if i==0 and not self.plastic:continue
            gradient=gradient*scale;self.m[i]=.9*self.m[i]+.1*gradient;self.v[i]=.999*self.v[i]+.001*gradient**2
            parameter-=.001*(self.m[i]/(1-.9**self.updates))/(np.sqrt(self.v[i]/(1-.999**self.updates))+1e-8)
        self.w[:]=np.clip(self.w,0,3)*self.mask
        if not all(np.isfinite(a).all() for a in (self.w,self.readout,self.bias)):raise FloatingPointError('Nonfinite learned parameter')
        if self.updates%100==0:
            self.target_w[:]=self.w;self.target_readout[:]=self.readout;self.target_bias[:]=self.bias
        return loss
    def state_hash(self):
        return fingerprint(self.w,self.readout,self.bias,self.target_w,self.target_readout,self.target_bias,*self.m,*self.v,np.array([self.updates]))
    def save(self,path):
        np.savez_compressed(path,w=self.w,initial_w=self.initial_w,mask=self.mask,projection=self.projection,
            readout=self.readout,bias=self.bias,target_w=self.target_w,target_readout=self.target_readout,target_bias=self.target_bias,
            m_w=self.m[0],m_readout=self.m[1],m_bias=self.m[2],v_w=self.v[0],v_readout=self.v[1],v_bias=self.v[2],updates=self.updates,plastic=self.plastic)
    @classmethod
    def load(cls,path,plastic=None):
        with np.load(path,allow_pickle=False) as z:
            if plastic is None:
                if 'plastic' not in z:raise ValueError('Initial checkpoint format requires explicit plastic=True/False')
                plastic=bool(z['plastic'])
            result=cls(z['initial_w'],0,plastic)
            for key in ('w','mask','projection','readout','bias','target_w','target_readout','target_bias'):
                value=z[key]
                if value.shape!=getattr(result,key).shape or not np.isfinite(value).all():raise ValueError('Invalid checkpoint array '+key)
                setattr(result,key,value.copy())
            if not np.array_equal(result.mask,result.initial_w>0) or (result.w[~result.mask]!=0).any() or (result.w<0).any():raise ValueError('Invalid learned mask or strengths')
            result.m=[z['m_'+k].copy() for k in ('w','readout','bias')]
            result.v=[z['v_'+k].copy() for k in ('w','readout','bias')];result.updates=int(z['updates'])
        inputs=all_observations()
        result.kc=np.maximum(np.dot(inputs,result.projection)/np.sqrt(np.maximum(inputs.sum(1,keepdims=True),1))-1.,0.)
        return result
    @property
    def parameters(self):return int(self.mask.sum())*int(self.plastic)+self.readout.size+self.bias.size
