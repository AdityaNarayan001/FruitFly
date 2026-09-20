import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ('src', 'native', 'scripts', 'configs', 'tests', 'ui')
SOURCE_FILES = ('setup.sh', 'setup_exp_2.sh', 'setup_explorer.sh', 'pyproject.toml', 'requirements.lock', 'README.md', 'STATUS.md', 'AGENTS.md', '.gitignore')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def source_manifest(root=ROOT):
    paths = [root / n for n in SOURCE_FILES if (root/n).is_file()]
    for folder in SOURCE_DIRS:
        paths += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts) and p.suffix != '.pyc' and p.name != '.DS_Store']
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    return {'sha256': canonical_hash(files), 'files': files}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)

def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f'unavailable: {type(e).__name__}'

def environment():
    import importlib.metadata
    return {'host': platform.node(), 'platform': platform.platform(), 'python': platform.python_version(),
            'packages': {k: importlib.metadata.version(k) for k in ('numpy','pyarrow','scipy')},
            'gpu': command_output(['nvidia-smi','--query-gpu=name,driver_version,compute_cap','--format=csv,noheader']),
            'cuda_compiler': command_output(['/usr/local/cuda/bin/nvcc','--version'])}


def preserve_source(runs,manifest):
    """Keep one exact code archive per source identity, including local CPU runs."""
    import io,tarfile,tempfile
    deployed=ROOT.parent.parent/'snapshots'/f'{manifest["sha256"]}.tar.gz'
    if (ROOT/'SOURCE_MANIFEST.json').exists() and deployed.is_file():return str(deployed)
    dest=Path(runs)/'_sources'/f'{manifest["sha256"]}.tar.gz'
    if dest.exists():return str(dest.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='source-',suffix='.tmp',dir=dest.parent);os.close(fd)
    try:
        with tarfile.open(tmp,'w:gz') as archive:
            for name,digest in manifest['files'].items():
                payload=(ROOT/name).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Source changed while the service was running; restart before recording a new session')
                info=tarfile.TarInfo(name);info.size=len(payload);info.mode=0o755 if name in ('setup.sh','setup_exp_2.sh','setup_explorer.sh') else 0o644
                archive.addfile(info,io.BytesIO(payload))
            payload=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('SOURCE_MANIFEST.json');info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
        Path(tmp).replace(dest)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    return str(dest.resolve())
