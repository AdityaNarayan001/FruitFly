"""Auditable finite maze task. Reward/path oracles never enter policy features."""
from collections import deque
import copy, hashlib, math
import numpy as np

SIZE=9
MOVES=((0,-1),(1,0),(0,1),(-1,0))
NAMES=('north','east','south','west')
DEFAULT_ROWS=['#########','#.......#','#.###.#.#','#...#.#.#','###.#.#.#','#...#...#','#.#####.#','#.......#','#########']

def default_config():
    return {'layout':{'walls':[[int(c=='#') for c in row] for row in DEFAULT_ROWS],
            'start':[1,7],'sites':[[1,1],[7,1]]},'rewarded_pattern':'A','seed':42,
            'alpha':.25,'gamma':.95,'max_steps':120}

def shortest(layout,goal):
    start=tuple(layout['start']);goal=tuple(goal)
    blocked={tuple(p) for p in layout['sites'] if tuple(p)!=goal}
    q=deque([(start,0)]);seen={start}
    while q:
        (x,y),d=q.popleft()
        if (x,y)==goal:return d
        for dx,dy in MOVES:
            p=(x+dx,y+dy)
            if 0<=p[0]<SIZE and 0<=p[1]<SIZE and not layout['walls'][p[1]][p[0]] and p not in seen and p not in blocked:
                seen.add(p);q.append((p,d+1))
    return None

def validate_config(value):
    if not isinstance(value,dict) or set(value)-set(default_config()):raise ValueError('Unknown experiment settings')
    c=copy.deepcopy({**default_config(),**value});l=c['layout']
    if not isinstance(l,dict) or set(l)!={'walls','start','sites'}:raise ValueError('Layout needs walls, start and two sites')
    a=np.asarray(l['walls'])
    if a.shape!=(SIZE,SIZE) or not np.isin(a,[0,1]).all():raise ValueError('Maze must be a 9 x 9 binary grid')
    if not (a[0].all() and a[-1].all() and a[:,0].all() and a[:,-1].all()):raise ValueError('Keep the outside boundary closed')
    if not isinstance(l['sites'],list) or len(l['sites'])!=2:raise ValueError('Place exactly two food sites')
    points=[l['start'],*l['sites']]
    for p in points:
        if not isinstance(p,list) or len(p)!=2 or any(type(v) is not int or not 0<v<SIZE-1 for v in p):raise ValueError('Markers must be interior grid cells')
        if a[p[1],p[0]]:raise ValueError('Start and food sites must be on open cells')
    if len({tuple(p) for p in points})!=3:raise ValueError('Start and food sites must be distinct')
    if any(shortest(l,p) is None for p in l['sites']):raise ValueError('Each food site needs a path that avoids the other food site')
    l['walls']=a.astype(int).tolist()
    if c['rewarded_pattern'] not in ('A','B'):raise ValueError('Rewarded pattern must be A or B')
    if type(c['seed']) is not int or not 0<=c['seed']<=1_000_000:raise ValueError('Seed must be an integer from 0 to 1000000')
    if type(c['max_steps']) is not int or not 20<=c['max_steps']<=500:raise ValueError('Step limit must be 20-500')
    for name,low,high in [('alpha',.001,1),('gamma',0,1)]:
        x=float(c[name])
        if not np.isfinite(x) or not low<=x<=high:raise ValueError(name+' out of range')
        c[name]=x
    return c

def make_maze(seed):
    rng=np.random.default_rng(seed)
    for _ in range(200):
        walls=np.ones((SIZE,SIZE),int);walls[1,1]=0;stack=[(1,1)]
        while stack:
            x,y=stack[-1];choices=[]
            for dx,dy in MOVES:
                xx,yy=x+2*dx,y+2*dy
                if 0<xx<SIZE-1 and 0<yy<SIZE-1 and walls[yy,xx]:choices.append((xx,yy,dx,dy))
            if not choices:stack.pop();continue
            xx,yy,dx,dy=choices[int(rng.integers(len(choices)))];walls[y+dy,x+dx]=walls[yy,xx]=0;stack.append((xx,yy))
        for y in range(1,SIZE-1):
            for x in range(1,SIZE-1):
                if (x+y)%2 and rng.random()<.28:walls[y,x]=0
        l={'walls':walls.tolist(),'start':[1,7],'sites':[[1,1],[7,1]]}
        if all(shortest(l,p) is not None for p in l['sites']):return l
    raise RuntimeError('Could not generate a valid maze')

def patterns(seed,episode):
    # Each pair has one of each assignment. No reward label participates here.
    flip=int(np.random.default_rng(seed+episode//2*7919).integers(2))^(episode%2)
    return ('A','B') if flip==0 else ('B','A')

def epsilon(episode):return max(.05,.8*.99**episode)

def digest(q):return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()

class World:
    def __init__(self,config,assignment):
        self.c=config;self.layout=config['layout'];self.patterns=tuple(assignment)
        self.pos=list(self.layout['start']);self.path=[self.pos.copy()];self.steps=0;self.total=0.;self.done=False;self.outcome=None;self.last_reward=0.
    def legal(self):
        x,y=self.pos
        return [i for i,(dx,dy) in enumerate(MOVES) if not self.layout['walls'][y+dy][x+dx]]
    def step(self,action):
        if self.done:raise ValueError('Trial ended; reset the trial before moving')
        if type(action) is not int or not 0<=action<4:raise ValueError('Action must be 0-3')
        reward=-.01
        if action not in self.legal():reward=-.04
        else:
            dx,dy=MOVES[action];self.pos=[self.pos[0]+dx,self.pos[1]+dy]
        self.steps+=1;self.path.append(self.pos.copy())
        for i,site in enumerate(self.layout['sites']):
            if self.pos==site:
                good=self.patterns[i]==self.c['rewarded_pattern'];reward=1. if good else -.25
                self.done=True;self.outcome='food' if good else 'wrong_pattern';break
        if not self.done and self.steps>=self.c['max_steps']:self.done=True;self.outcome='timeout'
        self.last_reward=reward;self.total+=reward
        return reward
    def summary(self):
        goal=self.layout['sites'][self.patterns.index(self.c['rewarded_pattern'])]
        optimal=shortest(self.layout,goal)
        return {'success':self.outcome=='food','outcome':self.outcome,'steps':self.steps,'return':self.total,
                'patterns':list(self.patterns),'optimal_steps':optimal,
                'efficiency':optimal/self.steps if self.outcome=='food' and self.steps else None}
    def public(self):
        return {'position':self.pos,'path':self.path,'steps':self.steps,'return':self.total,'last_reward':self.last_reward,
                'done':self.done,'outcome':self.outcome,'patterns':list(self.patterns),'legal_actions':self.legal()}

class Learner:
    def __init__(self,config,q=None):
        self.c=config;self.q=np.zeros((4,SIZE,SIZE,4),np.float64) if q is None else np.array(q,dtype=np.float64,copy=True)
        if self.q.shape!=(4,SIZE,SIZE,4) or not np.isfinite(self.q).all():raise ValueError('Invalid Q checkpoint')
    def state(self,position,tokens):return (int(tokens[0])*2+int(tokens[1]),position[1],position[0])
    def choose(self,state,legal,rng,exploration):
        if rng.random()<exploration:return int(rng.choice(legal))
        values=self.q[state][legal];best=np.flatnonzero(np.isclose(values,values.max(),rtol=0,atol=1e-12))
        return int(legal[int(rng.choice(best))])
    def update(self,state,action,reward,next_state,legal,terminal):
        target=reward if terminal else reward+self.c['gamma']*float(self.q[next_state][legal].max())
        td=target-self.q[state][action];self.q[state][action]+=self.c['alpha']*td;return float(td)

def rollout(config,assignment,learner,tokens,seed,exploration=0.,learn=False,random=False):
    w=World(config,assignment);rng=np.random.default_rng(seed)
    while not w.done:
        s=learner.state(w.pos,tokens);a=int(rng.choice(w.legal())) if random else learner.choose(s,w.legal(),rng,exploration)
        r=w.step(a)
        if learn:learner.update(s,a,r,learner.state(w.pos,tokens),w.legal(),w.done)
    return w.summary()

def summarize(records):
    n=len(records);k=sum(r['success'] for r in records);p=k/n if n else 0.;z=1.96;den=1+z*z/max(n,1)
    center=(p+z*z/(2*max(n,1)))/den;half=z*math.sqrt(p*(1-p)/max(n,1)+z*z/(4*max(n,1)**2))/den
    efficiency=[r['efficiency'] for r in records if r.get('efficiency') is not None]
    return {'n':n,'successes':k,'success_rate':p,'wilson95':[max(0,center-half),min(1,center+half)] if n else [0,1],
            'mean_return':float(np.mean([r['return'] for r in records])) if n else 0,
            'mean_steps':float(np.mean([r['steps'] for r in records])) if n else 0,
            'mean_success_efficiency':float(np.mean(efficiency)) if efficiency else None,
            'timeouts':sum(r['outcome']=='timeout' for r in records)}
