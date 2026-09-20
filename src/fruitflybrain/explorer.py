"""Read-only anatomy explorer. Landmarks and connections are not a functional decoder."""
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .atlas import Atlas
from .graph import Graph
from .provenance import ROOT, sha256


ROLES = [
    ('visual_input', 'Visual sensory neurons', '#59cbe8', ('ol_sensory',),
     'Annotated sensory input from the visual system. Our experiment bypasses these and stimulates L1 directly.'),
    ('other_input', 'Other sensory pathways', '#70d8a6', ('cb_sensory', 'vnc_sensory', 'sensory_ascending', 'sensory_descending'),
     'Annotated sensory pathways from the body and other senses. This is a broad anatomical grouping.'),
    ('visual', 'Visual processing circuits', '#8b9bff', ('ol_intrinsic', 'visual_projection', 'visual_centrifugal'),
     'Optic-lobe circuits, projections and feedback. L1 belongs here; these are not ANN hidden layers.'),
    ('central', 'Central brain circuits', '#c895f4', ('cb_intrinsic',),
     'Central-brain intrinsic neurons. Their annotation does not specify a single computation or task.'),
    ('descending', 'Descending pathways', '#f4bc6a', ('descending_neuron',),
     'Connections from brain toward nerve-cord motor circuits. These are not automatically motor neurons or turn commands.'),
    ('vnc', 'Nerve-cord circuits', '#6dabc9', ('vnc_intrinsic',),
     'Intrinsic ventral nerve cord (VNC) neurons. Processing also occurs here, close to motor circuits.'),
    ('ascending', 'Ascending pathways', '#d9a384', ('ascending_neuron',),
     'Connections carrying information from the nerve cord toward the brain; part of the feedback organization.'),
    ('motor', 'Motor neurons', '#fa8096', ('cb_motor', 'vnc_motor'),
     'Annotated motor neurons are an anatomical output category. No motor population controls the Exp 2 avatar.'),
    ('other', 'Other / uncertain annotations', '#7e8d9e', (),
     'Endocrine, efferent, enteric and uncertain categories remain separate. TBC labels are not promoted to confirmed roles.'),
]


class NetworkView:
    def __init__(self, graph, atlas, graph_path):
        self.g, self.atlas = graph, atlas
        self.role = np.full(graph.n, len(ROLES)-1, np.int32)
        for k, (_, _, _, classes, _) in enumerate(ROLES[:-1]):
            self.role[np.isin(atlas.classes, classes)] = k
        self.stimulated = np.zeros(graph.n, bool)
        self.stimulated[atlas.input_ids] = True
        self.xyz = np.full((graph.n, 3), np.nan)
        self.location_source = [None] * graph.n
        for i, row in enumerate(atlas.rows):
            for key in ('somaLocation', 'tosomaLocation'):
                point = row.get(key)
                if point is not None and len(point) == 3 and np.isfinite(point).all():
                    self.xyz[i] = point
                    self.location_source[i] = key
                    break
        valid = np.isfinite(self.xyz).all(axis=1)
        if valid.any():
            lo, hi = self.xyz[valid].min(axis=0), self.xyz[valid].max(axis=0)
            self.center, self.scale = (lo+hi)/2, max(float((hi-lo).max()/2), 1)
        else:
            self.center, self.scale = np.zeros(3), 1.
        self.valid = valid
        self.contacts = None
        path = Path(graph_path)/'anatomical_contacts.npy'
        expected = graph.metadata.get('supplemental_files', {}).get(path.name)
        if expected:
            if sha256(path) != expected:
                raise ValueError('Anatomical contact checksum mismatch')
            self.contacts = np.load(path, mmap_mode='r', allow_pickle=False)
            if self.contacts.shape != (graph.m,):
                raise ValueError('Anatomical contact alignment mismatch')
        rng = np.random.default_rng(22092026)
        shown = set(np.flatnonzero(valid & self.stimulated).tolist())
        self.groups = []
        for k, (key, name, color, classes, description) in enumerate(ROLES):
            mask = self.role == k
            pool = np.flatnonzero(mask & valid)
            shown.update(rng.choice(pool, min(600, len(pool)), replace=False).tolist())
            self.groups.append(dict(key=key, name=name, color=color, count=int(mask.sum()),
                                    located=len(pool), classes=list(classes), description=description))
        self.shown = np.array(sorted(shown), dtype=np.int32)

    def point(self, i):
        row = self.atlas.rows[i]
        return dict(id=str(self.g.ids[i]), type=str(self.atlas.types[i]),
                    side=str(self.atlas.sides[i]), superclass=str(self.atlas.classes[i]),
                    role=int(self.role[i]), experiment_input=bool(self.stimulated[i]),
                    experiment_readout=bool(self.stimulated[i]),
                    xyz=((self.xyz[i]-self.center)/self.scale).round(6).tolist() if self.valid[i] else None,
                    location_source=self.location_source[i], cell_class=row.get('class'))

    def overview(self):
        located_inputs = self.atlas.input_ids[self.valid[self.atlas.input_ids]]
        initial = int(located_inputs[0]) if len(located_inputs) else (int(self.shown[0]) if len(self.shown) else 0)
        return dict(dataset=self.g.metadata.get('dataset'), synthetic=self.atlas.synthetic,
                    neurons=self.g.n, directed_edges=self.g.m,
                    anatomical_contacts=self.g.metadata.get('retained_contact_count'),
                    graph_sha256=self.g.metadata.get('graph_sha256'),
                    input_count=len(self.atlas.input_ids), readout_count=len(self.atlas.input_ids),
                    features=32, motor_readout_count=0, groups=self.groups,
                    positioned=int(self.valid.sum()), displayed=len(self.shown),
                    missing_positions=int((~self.valid).sum()),
                    points=[self.point(int(i)) for i in self.shown],
                    initial_neuron=str(self.g.ids[initial]),
                    coordinate_note='Centered and uniformly scaled reconstruction coordinates; view axes are dataset X/Y/Z, not calibrated body directions.',
                    sample_note='All located experimental L1 neurons plus up to 600 located neurons per anatomical group. Missing positions are never invented.')

    def neuron(self, body_id):
        i = self.atlas.lookup.get(int(body_id))
        if i is None:
            raise ValueError('Neuron ID is not in the loaded graph')
        first, last = self.g.indptr[i:i+2]
        incoming = np.arange(first, last)
        outgoing = np.flatnonzero(self.g.indices == i)
        score = self.contacts if self.contacts is not None else np.abs(self.g.weights)
        neighbors, edges = {}, []
        for direction, candidates in (('incoming', incoming), ('outgoing', outgoing)):
            chosen = sorted(candidates.tolist(), key=lambda e: (-float(score[e]), e))[:20]
            for e in chosen:
                pre = int(self.g.indices[e])
                post = i if direction == 'incoming' else int(np.searchsorted(self.g.indptr, e, side='right')-1)
                neighbor = pre if direction == 'incoming' else post
                neighbors[str(self.g.ids[neighbor])] = self.point(neighbor)
                edges.append(dict(source=str(self.g.ids[pre]), target=str(self.g.ids[post]),
                                  direction=direction, neighbor=str(self.g.ids[neighbor]),
                                  contacts=int(self.contacts[e]) if self.contacts is not None else None,
                                  model_weight_mv=float(self.g.weights[e])))
        return dict(node=self.point(i), coordinates=self.xyz[i].tolist() if self.valid[i] else None,
                    incoming_count=len(incoming), outgoing_count=len(outgoing),
                    neighbors=list(neighbors.values()), edges=edges,
                    ranking='anatomical contact count' if self.contacts is not None else 'absolute assumed model weight',
                    note='At most 20 incoming and 20 outgoing connections. Lines join cell landmarks, not synapse positions or neuron arbors. Connection direction does not prove causal task involvement.')

    def search(self, query):
        query = str(query).strip().lower()[:80]
        if not query:
            return []
        if query.isdigit() and int(query) in self.atlas.lookup:
            return [self.point(self.atlas.lookup[int(query)])]
        matches = []
        for i in range(self.g.n):
            if query in str(self.atlas.types[i]).lower() or query in str(self.atlas.classes[i]).lower() or query in str(self.g.ids[i]):
                matches.append(self.point(i))
                if len(matches) == 30:
                    break
        return matches


class NetworkRoutes:
    def __init__(self, factory):
        self.factory, self.view, self.overview_bytes = factory, None, None
        self.lock = threading.Lock()

    def get(self, raw_path):
        parsed = urlsplit(raw_path)
        assets = {'/network': ('index.html', 'text/html; charset=utf-8'),
                  '/network/': ('index.html', 'text/html; charset=utf-8'),
                  '/network.js': ('app.js', 'text/javascript'), '/network.css': ('style.css', 'text/css')}
        if parsed.path in assets:
            name, mime = assets[parsed.path]
            return (ROOT/'ui/explorer'/name).read_bytes(), mime
        if not parsed.path.startswith('/api/network/'):
            return None
        with self.lock:
            if self.view is None:
                self.view = self.factory()
        if parsed.path == '/api/network/overview':
            with self.lock:
                if self.overview_bytes is None:
                    self.overview_bytes = json.dumps(self.view.overview(), allow_nan=False).encode()
            return self.overview_bytes, 'application/json'
        if parsed.path == '/api/network/search':
            data = self.view.search(parse_qs(parsed.query).get('q', [''])[0])
        else:
            match = re.fullmatch(r'/api/network/neuron/(\d{1,20})', parsed.path)
            if not match:
                return None
            data = self.view.neuron(match[1])
        return json.dumps(data, allow_nan=False).encode(), 'application/json'


def main():
    parser = argparse.ArgumentParser(description='Read-only connectome explorer; no simulation or learning')
    parser.add_argument('--graph', required=True)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be 1-65535')
    def make_view():
        graph = Graph.load(args.graph)
        return NetworkView(graph, Atlas(graph, Path(args.graph)/'neurons.feather'), args.graph)
    routes = NetworkRoutes(make_view)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                response = routes.get('/network' if self.path == '/' else self.path)
                if response is None:
                    self.send_error(404)
                    return
                body, mime = response
                self.send_response(200)
            except (OSError, ValueError) as exc:
                body, mime = json.dumps({'error': str(exc)}).encode(), 'application/json'
                self.send_response(400)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Read-only neural anatomy: http://127.0.0.1:{args.port}/network', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
