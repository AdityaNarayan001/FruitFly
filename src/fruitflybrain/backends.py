"""Explicit compute selection; never silently substitute a graph or failed backend."""
def factory(name):
    if name=='cpu':
        from .cpu import Cpu
        return Cpu
    if name=='cuda':
        from .cuda import Cuda
        return Cuda
    raise ValueError('Backend must be cpu or cuda')

def label(cls):
    return {'Cpu':'CPU / SciPy float32','Cuda':'CUDA / native float32','Reference':'TEST_REFERENCE'}.get(cls.__name__,cls.__name__)
