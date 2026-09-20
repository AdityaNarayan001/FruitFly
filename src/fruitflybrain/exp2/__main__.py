import argparse
from .server import serve

def main():
    p=argparse.ArgumentParser(description='Experiment 2: fixed neural encoder + external reward learning')
    p.add_argument('--graph',required=True);p.add_argument('--runs',default='runs/exp_2');p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--port',type=int,default=8767)
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error('Invalid port')
    serve(a.graph,a.runs,a.backend,a.port)
if __name__=='__main__':main()
