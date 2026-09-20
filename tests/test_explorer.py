import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from fruitflybrain.atlas import Atlas
from fruitflybrain.explorer import NetworkView, NetworkRoutes
from fruitflybrain.graph import Graph
from fruitflybrain.provenance import sha256


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        ids = np.array([9007199254740993+i for i in range(4)], dtype=np.int64)
        self.g = Graph.edges(ids, [1, 0, 0], [0, 2, 3], [-1, 0, 2], {'dataset_kind': 'synthetic'})
        rows = []
        for i, (typ, superclass) in enumerate([('L1', 'ol_intrinsic'), ('R1-R6', 'ol_sensory'), ('MN', 'vnc_motor'), ('DN', 'descending_neuron')]):
            rows.append(dict(bodyId=int(ids[i]), type=typ, superclass=superclass, somaSide='L',
                             assignedOlHex1=1. if i == 0 else None, assignedOlHex2=1. if i == 0 else None,
                             somaLocation=None if i == 2 else [i*10, 20, 30], tosomaLocation=None))
        feather.write_feather(pa.Table.from_pylist(rows), self.path/'neurons.feather')
        np.save(self.path/'anatomical_contacts.npy', np.array([10, 70, 5], np.int64))
        self.g.metadata['supplemental_files'] = {'anatomical_contacts.npy': sha256(self.path/'anatomical_contacts.npy')}
        self.atlas = Atlas(self.g, self.path/'neurons.feather')
        self.view = NetworkView(self.g, self.atlas, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles_coordinates_and_experiment_readout(self):
        overview = self.view.overview()
        self.assertEqual(sum(g['count'] for g in overview['groups']), 4)
        self.assertEqual(overview['missing_positions'], 1)
        self.assertEqual(overview['input_count'], 1)
        self.assertEqual(overview['motor_readout_count'], 0)
        l1 = self.view.neuron(str(self.g.ids[0]))['node']
        self.assertEqual(l1['id'], '9007199254740993')
        self.assertEqual(l1['superclass'], 'ol_intrinsic')
        self.assertTrue(l1['experiment_input'] and l1['experiment_readout'])
        motor = self.view.neuron(str(self.g.ids[2]))['node']
        self.assertIsNone(motor['xyz'])
        self.assertFalse(motor['experiment_readout'])

    def test_edges_direction_ranking_and_zero_model_weight(self):
        result = self.view.neuron(str(self.g.ids[0]))
        self.assertEqual(result['incoming_count'], 1)
        self.assertEqual(result['outgoing_count'], 2)
        incoming = [e for e in result['edges'] if e['direction'] == 'incoming']
        outgoing = [e for e in result['edges'] if e['direction'] == 'outgoing']
        self.assertEqual(incoming[0]['source'], str(self.g.ids[1]))
        self.assertEqual(incoming[0]['target'], str(self.g.ids[0]))
        self.assertEqual(outgoing[0]['target'], str(self.g.ids[2]))
        self.assertEqual(outgoing[0]['contacts'], 70)
        self.assertEqual(outgoing[0]['model_weight_mv'], 0)
        self.assertTrue(all(e['source'] == str(self.g.ids[0]) for e in outgoing))

    def test_lookup_routes_and_contact_integrity(self):
        self.assertEqual(self.view.search('L1')[0]['id'], str(self.g.ids[0]))
        self.assertEqual(self.view.search(str(self.g.ids[2]))[0]['type'], 'MN')
        routes = NetworkRoutes(lambda: self.view)
        payload, mime = routes.get('/api/network/overview')
        self.assertEqual(mime, 'application/json')
        self.assertEqual(json.loads(payload)['neurons'], 4)
        self.assertIsNone(routes.get('/network/../../AGENTS.md'))
        self.assertIsNone(routes.get('/api/network/control'))
        with self.assertRaises(ValueError):
            self.view.neuron('0')
        (self.path/'anatomical_contacts.npy').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            NetworkView(self.g, self.atlas, self.path)


if __name__ == '__main__':
    unittest.main()
