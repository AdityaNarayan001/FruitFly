import numpy as np

WIDTH,HEIGHT=32,24
DEFAULT={'kind':'blank','eye':'both','gain':0.6,'contrast':0.8,'frequency_hz':2.0,'cycles':4.0,'direction':1,'spot_x':0.5,'spot_y':0.5,'spot_radius':0.15,'paint':[0.0]*(WIDTH*HEIGHT)}

def validate_stimulus(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULT):raise ValueError('Unknown stimulus fields')
    v={**DEFAULT,**value}
    if v['kind'] not in ('blank','uniform','grating','spot','paint'):raise ValueError('Unknown stimulus')
    if v['eye'] not in ('both','L','R'):raise ValueError('Unknown eye')
    for key,lo,hi in [('gain',0,1.5),('contrast',0,1),('frequency_hz',0,8),('cycles',1,12),('spot_x',0,1),('spot_y',0,1),('spot_radius',.02,.5)]:
        x=float(v[key])
        if not np.isfinite(x) or not lo<=x<=hi:raise ValueError(f'{key} out of range')
        v[key]=x
    if v['direction'] not in (-1,1):raise ValueError('Direction must be -1 or 1')
    a=np.asarray(v['paint'],np.float32)
    if a.shape!=(WIDTH*HEIGHT,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Paint requires a 32x24 grid between 0 and 1')
    v['paint']=a.tolist();return v

def frame(stim,t_ms):
    y,x=np.mgrid[0:HEIGHT,0:WIDTH];x=x/(WIDTH-1);y=y/(HEIGHT-1)
    if stim['kind']=='blank':return np.zeros((HEIGHT,WIDTH),np.float32)
    if stim['kind']=='uniform':return np.ones((HEIGHT,WIDTH),np.float32)
    if stim['kind']=='paint':return np.asarray(stim['paint'],np.float32).reshape(HEIGHT,WIDTH)
    if stim['kind']=='spot':return np.exp(-((x-stim['spot_x'])**2+(y-stim['spot_y'])**2)/(2*stim['spot_radius']**2)).astype(np.float32)
    return (.5*(1+stim['contrast']*np.sin(2*np.pi*(stim['cycles']*x-stim['direction']*stim['frequency_hz']*t_ms/1000)))).astype(np.float32)

def sample_frame(image,uv):
    # The same discrete image displayed by the UI is sampled for neural drive.
    x=np.rint(uv[:,0]*(WIDTH-1)).astype(int);y=np.rint(uv[:,1]*(HEIGHT-1)).astype(int)
    return image[y,x]

def make_drive(atlas,stim,start_tick,steps,dt_ms,pulses):
    drive=np.zeros((steps,atlas.graph.n),np.float32)
    chosen=np.ones(len(atlas.input_ids),bool) if stim['eye']=='both' else atlas.input_sides==stim['eye']
    for t in range(steps):
        tick=start_tick+t;image=frame(stim,tick*dt_ms)
        drive[t,atlas.input_ids[chosen]]=sample_frame(image,atlas.uv[chosen])*stim['gain']
        for pulse in pulses:
            if pulse['start_tick']<=tick<pulse['end_tick']:drive[t,pulse['indices']]+=pulse['amplitude']
    return drive
