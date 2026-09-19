"""Loopback experiment API: bounded manual control and archived run access."""
from functools import partial
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,re,secrets,signal,threading
from urllib.parse import urlsplit,parse_qs
from .provenance import ROOT

class Handler(BaseHTTPRequestHandler):
    def __init__(self,*args,runs,live,token,**kwargs):self.runs=Path(runs);self.live=live;self.token=token;super().__init__(*args,**kwargs)
    def do_GET(self):
        parsed=urlsplit(self.path);path=parsed.path
        try:
            if path in ('/','/index.html'):return self.send_content((ROOT/'ui/index.html').read_bytes(),'text/html; charset=utf-8')
            if path in ('/app.js','/style.css'):
                return self.send_content((ROOT/'ui'/path[1:]).read_bytes(),'text/javascript' if path.endswith('.js') else 'text/css')
            if path=='/api/session':return self.send_json({'token':self.token,'interactive':self.live is not None})
            if path=='/api/live':return self.send_json(self.live.snapshot() if self.live else {'status':'unavailable','message':'Start service with --graph'})
            if path=='/api/atlas':
                if not self.live or self.live.atlas is None:return self.send_json({'error':'Atlas loading'},503)
                return self.send_json(self.live.atlas.public())
            if path=='/api/search':
                if not self.live or self.live.atlas is None:return self.send_json([],503)
                return self.send_json(self.live.atlas.search(parse_qs(parsed.query).get('q',[''])[0]))
            if path=='/api/runs':
                runs=[]
                for f in sorted(self.runs.glob('*/manifest.json'),reverse=True)[:100]:
                    item=json.loads(f.read_text());runs.append({k:item.get(k) for k in ('run_id','status','mode','start_utc','n_neurons','n_edges','source_sha256','interpretation')})
                return self.send_json(runs)
            m=re.fullmatch(r'/api/runs/([A-Za-z0-9_-]+)/(manifest|telemetry|metrics)',path)
            if m:
                f=self.runs/m[1]/(m[2]+'.json')
                if f.exists():return self.send_json(json.loads(f.read_text()))
            self.send_error(404)
        except (OSError,ValueError):self.send_json({'error':'Run data temporarily unavailable'},503)
    def do_POST(self):
        host=self.headers.get('Host','');origin=self.headers.get('Origin')
        if self.headers.get('X-FFB-Token')!=self.token or (origin and origin not in (f'http://{host}',f'https://{host}')):
            return self.send_json({'error':'Invalid session or origin'},403)
        if urlsplit(self.path).path!='/api/control' or not self.live:return self.send_json({'error':'Control unavailable'},404)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=20000 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected bounded JSON request')
            data=json.loads(self.rfile.read(length));return self.send_json(self.live.command(data))
        except (ValueError,TypeError,KeyError) as e:return self.send_json({'error':str(e)},400)
        except TimeoutError:return self.send_json({'error':'Command timed out; check live state before retrying'},503)
    def send_json(self,data,status=200):self.send_content(json.dumps(data,allow_nan=False).encode(),'application/json',status)
    def send_content(self,data,mime,status=200):
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

def serve(runs,port=8765,graph=None,backend="cuda"):
    from .live import LiveSession
    live=LiveSession(graph,runs,backend=backend) if graph else None
    server=ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,runs=runs,live=live,token=secrets.token_urlsafe(32)))
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    print(f'FruitFlyBrain console: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if live:live.close()
        server.server_close()
