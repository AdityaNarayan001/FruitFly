#!/usr/bin/env python3
"""Repeatable local setup. No sudo, system package changes, SSH alias or user path."""
import argparse,hashlib,json,os,shutil,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def call(args,env=None):subprocess.run([str(a) for a in args],cwd=ROOT,env=env,check=True)
def main():
    parser=argparse.ArgumentParser(description='Install and start FruitFlyBrain. Default: real MaleCNS connectome on CPU; about 1.1 GB initial download.')
    parser.add_argument('--app',choices=['exp1','explorer'],default='exp1',help='Interface to launch; setup_explorer.sh selects explorer')
    parser.add_argument('--backend',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--dataset',choices=['demo','male-cns'],default='male-cns',help='Default: male-cns (real brain). Use demo only for the small synthetic teaching circuit.')
    parser.add_argument('--port',type=int,default=None)
    parser.add_argument('--no-start',action='store_true',help='Install, prepare data and verify; do not start the server')
    parser.add_argument('--venv',type=Path,default=ROOT/'.venv')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'runs/local')
    a=parser.parse_args()
    if a.port is None:a.port=8768 if a.app=='explorer' else 8765
    if not (3,11)<=sys.version_info[:2]<(3,14):parser.error('Use Python 3.11-3.13; dependency wheels are pinned for these versions')
    if not 1<=a.port<=65535:parser.error('Port must be 1-65535')
    if not a.no_start:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',a.port))
        except OSError:parser.error(f'Port {a.port} is in use. Choose --port 8876; no existing service was stopped.')
    a.venv=a.venv.expanduser().resolve();a.data_dir=a.data_dir.expanduser().resolve();a.runs_dir=a.runs_dir.expanduser().resolve()
    python=a.venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not a.venv.exists():
        print('Creating isolated environment:',a.venv,flush=True)
        call([sys.executable,'-m','venv',a.venv])
    if not python.exists():parser.error(f'{a.venv} exists but is not a Python environment; choose --venv with a new path')
    call([python,'-c','import sys; assert (3,11)<=sys.version_info[:2]<(3,14), "Existing venv requires Python 3.11-3.13"'])
    digest=hashlib.sha256((ROOT/'requirements.lock').read_bytes()+(ROOT/'pyproject.toml').read_bytes()).hexdigest()
    stamp=a.venv/'.fruitflybrain-install.json'
    prior=json.loads(stamp.read_text()) if stamp.exists() else {}
    pins=[line.strip().split('==') for line in (ROOT/'requirements.lock').read_text().splitlines() if '==' in line]
    check='import importlib.metadata as m; assert all(m.version(k)==v for k,v in '+repr(pins)+')'
    valid=subprocess.run([str(python),'-c',check],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    if not valid or prior.get('digest')!=digest or prior.get('checkout')!=str(ROOT):
        call([python,'-m','pip','install','--only-binary=:all:','-r',ROOT/'requirements.lock'])
        call([python,'-m','pip','install','--no-deps','-e',ROOT])
        stamp.write_text(json.dumps({'digest':digest,'checkout':str(ROOT)})+'\n')
    else:print('Pinned dependencies already installed.',flush=True)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    backend=a.backend
    cuda_available=sys.platform.startswith('linux') and shutil.which('nvidia-smi') and (shutil.which('nvcc') or (os.environ.get('FFB_NVCC') and Path(os.environ['FFB_NVCC']).is_file()) or Path('/usr/local/cuda/bin/nvcc').exists())
    if backend=='auto':backend='cuda' if cuda_available else 'cpu';print('Selected backend:',backend,flush=True)
    if backend=='cuda':
        if not cuda_available:parser.error('CUDA requires Linux, an NVIDIA driver/GPU and nvcc on PATH (or /usr/local/cuda). Use --backend cpu on this device.')
        call([python,ROOT/'scripts/build_cuda.py'])
    call([python,'-m','fruitflybrain.cli','doctor','--backend',backend],env)
    if a.dataset=='demo':
        graph=a.data_dir/'demo-v1'
        call([python,'-m','fruitflybrain.cli','demo','--output',graph],env)
        print('Dataset: SYNTHETIC 256-node teaching fixture. This is not the MaleCNS connectome.',flush=True)
        print('The demo viewer shows an invented circuit grid, not a brain shape. For actual anatomy, rerun with --dataset male-cns.',flush=True)
    else:
        graph=a.data_dir/'malecns-graph-v1'
        if not graph.exists():
            a.data_dir.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(a.data_dir).free<6*1024**3:parser.error('Full-data preparation needs at least 6 GiB free disk space')
            print('Preparing MaleCNS: about 1.1 GB download. Plan for >=16 GB RAM (32 GB recommended). CPU full-graph simulation can be slow.',flush=True)
            raw=a.data_dir/'malecns-v1'
            call([python,'-m','fruitflybrain.cli','download','--data',raw],env)
            call([python,'-m','fruitflybrain.cli','import','--data',raw,'--output',graph],env)
        call([python,'-c','from fruitflybrain.graph import Graph; import sys; g=Graph.load(sys.argv[1]); assert g.metadata.get("dataset")=="male-cns:v1.0"; print("Verified real graph:",g.n,"neurons,",g.m,"edges")',graph],env)
    if a.app=='explorer':
        command=[str(python),'-m','fruitflybrain.explorer','--graph',str(graph),'--port',str(a.port)]
        url=f'http://127.0.0.1:{a.port}/network'
        print(f'\nSetup verified. Read-only anatomy explorer; graph: {graph}\nNo neural simulation or training is started.',flush=True)
    else:
        a.runs_dir.mkdir(parents=True,exist_ok=True)
        command=[str(python),'-m','fruitflybrain.cli','serve','--graph',str(graph),'--runs',str(a.runs_dir),'--backend',backend,'--port',str(a.port)]
        url=f'http://127.0.0.1:{a.port}/'
        print(f'\nSetup verified. Backend: {backend}; graph: {graph}\nRun records: {a.runs_dir}',flush=True)
    if a.no_start:
        import shlex
        print('Start later from this checkout:\n'+shlex.join(command));return
    print(f'Open {url} in your browser. Leave this terminal open; Ctrl+C stops the service.',flush=True)
    os.execve(str(python),command,env)

if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as e:
        print(f'Setup stopped: command exited {e.returncode}. See the error above; existing data and runs were preserved.',file=sys.stderr);sys.exit(e.returncode or 1)
