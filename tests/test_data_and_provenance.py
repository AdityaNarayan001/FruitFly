import json,tempfile,unittest
from pathlib import Path
import numpy as np
from fruitflybrain.graph import Graph
from fruitflybrain.provenance import source_manifest,sha256,atomic_json

class DataTests(unittest.TestCase):
    def test_exact_large_ids_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'graph';ids=np.array([2**53+1,2**53+2],np.int64)
            Graph.edges(ids,[0],[1],[1]).save(path)
            np.testing.assert_array_equal(Graph.load(path).ids,ids)
            with open(path/'weights.npy','ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):Graph.load(path)

    def test_manifest_identity_changes_with_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'src').mkdir();f=root/'src/x.py';f.write_text('a=1')
            first=source_manifest(root)['sha256'];f.write_text('a=2')
            self.assertNotEqual(first,source_manifest(root)['sha256'])

    def test_import_population_and_signs(self):
        try:import pyarrow as pa;import pyarrow.feather as f
        except ImportError:self.skipTest('pyarrow required on the data host')
        from fruitflybrain.importer import prepare
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);data=p/'data';data.mkdir();a,b,c=2**53+1,2**53+2,2**53+3
            tables={'annotations':pa.Table.from_pylist([{'bodyId':a,'superclass':'sensory','status':'Traced'}, {'bodyId':b,'superclass':'intrinsic','status':'Traced'}, {'bodyId':c,'superclass':'intrinsic','status':'Traced'}, {'bodyId':4,'superclass':None,'status':'Glia'}]),
            'neurotransmitters':pa.Table.from_pylist([{'body':a,'consensus_nt':'acetylcholine'},{'body':b,'consensus_nt':'gaba'}]),
            'connections':pa.Table.from_pylist([{'body_pre':a,'body_post':b,'weight':2},{'body_pre':b,'body_post':a,'weight':3},{'body_pre':4,'body_post':a,'weight':9}])}
            ordered=pa.array(['Traced','Traced','Traced','Glia']).dictionary_encode().cast(pa.dictionary(pa.int8(),pa.string(),ordered=True))
            tables['annotations']=tables['annotations'].append_column('statusLabel',ordered)
            lock={'dataset':'test','license':'test','files':{}}
            for key,t in tables.items():
                file=data/(key+'.feather');f.write_feather(t,file);lock['files'][key]={'name':file.name,'sha256':sha256(file)}
            atomic_json(data/'downloads.json',lock);prepare(data,p/'graph');g=Graph.load(p/'graph')
            self.assertEqual(g.n,3);self.assertEqual(g.m,2);self.assertEqual(g.metadata['isolated_retained_neurons'],1)
            self.assertEqual(g.metadata['retained_contact_count'],5)
            np.testing.assert_allclose(g.weights,[-.825,.55])

if __name__=='__main__':unittest.main()
