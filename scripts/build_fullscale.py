#!/usr/bin/env python3
"""Build for the first visible NVIDIA GPU, or the explicit FFB_CUDA_ARCH."""
from pathlib import Path
import json,subprocess,hashlib,os,shutil,re,sys
root=Path(__file__).resolve().parents[1]
if not sys.platform.startswith('linux'):raise SystemExit('Native CUDA backend requires Linux. Use the CPU backend on this OS.')
compiler=os.environ.get('FFB_NVCC') or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
arch=os.environ.get('FFB_CUDA_ARCH')
if not arch:
    capability=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).splitlines()[0].strip()
    arch='sm_'+capability.replace('.','')
if not re.fullmatch(r'sm_\d{2,3}[a-z]?',arch):raise SystemExit('FFB_CUDA_ARCH must look like sm_89 or sm_121')
(root/'build').mkdir(exist_ok=True)
cmd=[compiler,'-O3','--fmad=false','-std=c++17','-arch='+arch,'-shared','-Xcompiler=-fPIC',str(root/'native/fullscale.cu'),'-o',str(root/'build/libffb_fullscale.so')]
subprocess.run(cmd,check=True)
meta={'architecture':arch,'command':cmd,'compiler':subprocess.check_output([compiler,'--version'],text=True),'sha256':hashlib.sha256((root/'build/libffb_fullscale.so').read_bytes()).hexdigest()}
(root/'build/fullscale_build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
