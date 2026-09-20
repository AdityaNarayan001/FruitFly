#!/usr/bin/env python3
"""Reuse the pinned installer without launching or modifying Experiment 1."""
import argparse,os,shlex,socket,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description='Install and launch Experiment 2 on its own port')
    p.add_argument('--backend',choices=['cpu','cuda','auto'],default='cpu');p.add_argument('--dataset',choices=['demo','male-cns'],default='demo')
    p.add_argument('--port',type=int,default=8767);p.add_argument('--no-start',action='store_true')
    p.add_argument('--venv',type=Path,default=ROOT/'.venv');p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--runs-dir',type=Path,default=ROOT/'runs/exp_2')
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error('Port must be 1-65535')
    if not a.no_start:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',a.port))
        except OSError:p.error(f'Port {a.port} is busy. Choose a different --port; existing services remain running.')
    for k in ('venv','data_dir','runs_dir'):setattr(a,k,getattr(a,k).expanduser().resolve())
    backend=a.backend
    if backend=='auto':
        ready=sys.platform.startswith('linux') and shutil.which('nvidia-smi') and (shutil.which('nvcc') or Path(os.environ.get('FFB_NVCC','/usr/local/cuda/bin/nvcc')).is_file())
        backend='cuda' if ready else 'cpu'
    print('Preparing the shared pinned runtime and selected dataset...',flush=True)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/setup.py'),'--no-start','--backend',backend,'--dataset',a.dataset,'--venv',str(a.venv),'--data-dir',str(a.data_dir),'--runs-dir',str(a.runs_dir),'--port',str(a.port)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout.split('Start later from this checkout:')[0],end='',flush=True)
    if result.returncode:raise subprocess.CalledProcessError(result.returncode,result.args)
    python=a.venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python');graph=a.data_dir/('demo-v1' if a.dataset=='demo' else 'malecns-graph-v1')
    command=[str(python),'-m','fruitflybrain.exp2','--graph',str(graph),'--runs',str(a.runs_dir),'--backend',backend,'--port',str(a.port)]
    print('\nEXPERIMENT 2: fixed neural probes + external Q-learning. Brain synapses do not learn.',flush=True)
    if a.no_start:print('Experiment 2 start command:\n'+shlex.join(command));return
    print(f'Open http://127.0.0.1:{a.port}/. Ctrl+C stops Experiment 2 only.',flush=True)
    os.execve(str(python),command,{**os.environ,'PYTHONPATH':str(ROOT/'src')})
if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as e:print('Setup failed; inspect the error above.',file=sys.stderr);sys.exit(e.returncode or 1)
