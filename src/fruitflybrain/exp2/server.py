"""Independent loopback HTTP surface for Experiment 2."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit
import json,re,secrets,threading,signal
from ..provenance import ROOT
from .session import Session
from .core import make_maze
from ..explorer import NetworkRoutes,NetworkView

class Handler(BaseHTTPRequestHandler):
    def __init__(self,*args,session,token,network,**kwargs):self.session=session;self.token=token;self.network=network;super().__init__(*args,**kwargs)
    def send(self,payload,mime='application/json',status=200,attachment=None):
        if not isinstance(payload,bytes):payload=json.dumps(payload,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if attachment:self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
        self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            network=self.network.get(self.path)
            if network is not None:return self.send(network[0],network[1])
            if path in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if path=='/' else path[1:];mime={'html':'text/html; charset=utf-8','js':'text/javascript','css':'text/css'}[name.split('.')[-1]]
                return self.send((ROOT/'ui/exp2'/name).read_bytes(),mime)
            if path=='/api/session':return self.send({'token':self.token})
            if path=='/api/state':return self.send(self.session.snapshot())
            if path=='/api/runs':
                rows=[]
                for p in sorted(self.session.runs.glob('*/manifest.json'),reverse=True)[:100]:
                    m=json.loads(p.read_text());rows.append({k:m.get(k) for k in ('run_id','status','created_utc','dataset','encoder_sha256')})
                return self.send(rows)
            m=re.fullmatch(r'/api/maze/(\d{1,7})',path)
            if m:return self.send(make_maze(int(m[1])))
            m=re.fullmatch(r'/api/export/([A-Za-z0-9_-]+)/(checkpoint.npz|evaluations.json|manifest.json)',path)
            if m:
                file=self.session.runs/m[1]/m[2]
                if file.is_file():return self.send(file.read_bytes(),'application/octet-stream',attachment=m[1]+'-'+m[2])
            return self.send({'error':'Not found'},status=404)
        except (ValueError,OSError):return self.send({'error':'Record temporarily unavailable'},status=503)
    def do_POST(self):
        host=self.headers.get('Host','');origin=self.headers.get('Origin')
        if self.headers.get('X-FFB-Token')!=self.token or (origin and origin!=f'http://{host}'):
            return self.send({'error':'Invalid session or origin'},status=403)
        if urlsplit(self.path).path!='/api/control':return self.send({'error':'Not found'},status=404)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=10000 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected bounded JSON')
            return self.send(self.session.command(json.loads(self.rfile.read(size))))
        except (ValueError,KeyError,TypeError) as e:return self.send({'error':str(e)},status=400)
        except TimeoutError as e:return self.send({'error':str(e)},status=503)
    def log_message(self,*args):pass

def serve(graph,runs,backend='cpu',port=8767):
    session=Session(graph,runs,backend)
    def network_view():
        if session.encoder is None:raise ValueError('Neural dataset is still loading')
        return NetworkView(session.encoder.g,session.encoder.atlas,graph)
    network=NetworkRoutes(network_view)
    try:server=ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,session=session,token=secrets.token_urlsafe(32),network=network))
    except BaseException:session.close();raise
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    print(f'Experiment 2: http://127.0.0.1:{port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:session.close();server.server_close()
