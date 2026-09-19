import json,tempfile,unittest
from pathlib import Path
import numpy as np
from fruitflybrain.cpu import Cpu
from fruitflybrain.model import Reference,Parameters
from fruitflybrain.graph import Graph,fixture
from fruitflybrain.demo import prepare
from fruitflybrain.atlas import Atlas
from fruitflybrain.provenance import source_manifest

class PortableTests(unittest.TestCase):
    def test_cpu_reference_and_chunk_continuity(self):
        rng=np.random.default_rng(22);g=fixture();drive=rng.uniform(-.2,1.1,(120,g.n)).astype(np.float32)
        for delay in (1,2,3):
            p=Parameters(delay_ms=delay)
            expected=Reference(g,p).run(drive,record=True)
            model=Cpu(g,p);a=model.run(drive[:37],record=True);b=model.run(drive[37:],record=True)
            np.testing.assert_array_equal(np.concatenate([a['spikes'],b['spikes']]),expected['spikes'])
            np.testing.assert_allclose(np.concatenate([a['voltages'],b['voltages']]),expected['voltages'],rtol=2e-5,atol=2e-4)
            model.reset();np.testing.assert_array_equal(model.run(np.zeros((10,g.n)))['counts'],0)
    def test_demo_is_labeled_and_reusable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data with spaces';prepare(p);before=(p/'graph.json').read_bytes();prepare(p)
            self.assertEqual(before,(p/'graph.json').read_bytes())
            g=Graph.load(p);atlas=Atlas(g,p/'neurons.feather')
            self.assertEqual(g.n,256);self.assertTrue(atlas.public()['synthetic'])
            self.assertEqual(atlas.mapping['status'],'synthetic_demo')
            self.assertEqual(atlas.mapping['mapped_neurons'],64)
    def test_setup_is_in_manifest_generated_metadata_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'src/example.egg-info').mkdir(parents=True)
            (root/'src/example.egg-info/PKG-INFO').write_text('machine-local artifact')
            (root/'setup.sh').write_text('#!/bin/sh\n')
            m=source_manifest(root);self.assertEqual(list(m['files']),['setup.sh'])
    def test_demo_refuses_other_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'graph';fixture().save(p)
            with self.assertRaises(ValueError):prepare(p)
