import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path
from forecast import graph,NODES,origins_for,EXPERIMENTS,inputs,network,HORIZONS,ingest
class ForecastTests(unittest.TestCase):
    def test_empty_duplicate_does_not_overwrite_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder,rows in [('1차',['2026-08-01 00:00,25,60']),('2차',['2026-08-01 00:00,25.001,60','2026-08-01 00:00,,',',,'])]:
                p=root/'rawdata'/folder;p.mkdir(parents=True);(p/'sensor(123).csv').write_text('time,temp,rh\n'+'\n'.join(rows),encoding='utf-8')
            clean,inventory,duplicates=ingest(root);self.assertEqual(len(clean),1);self.assertAlmostEqual(clean.temp.iloc[0],25.001);self.assertEqual(duplicates,1);self.assertEqual(sum(x['excluded'] for x in inventory),2)
    def test_graph_hierarchy(self):
        nodes=list(NODES);a=graph(nodes);np.testing.assert_allclose(a,a.T);np.testing.assert_array_equal(a.diagonal(),1)
        for i,n in enumerate(nodes):
            for j,m in enumerate(nodes):
                if i==j or a[i,j]==0:continue
                if 'N02' in (n,m):self.assertLessEqual(a[i,j],.020001)
                elif {'N15','N25'}&{n,m}:self.assertTrue(.15<=a[i,j]<=.25)
                else:self.assertGreaterEqual(a[i,j],.5)
    def test_split_no_leakage(self):
        for spec in EXPERIMENTS.values():
            ts=pd.date_range(spec['bounds'][0],pd.Timestamp(spec['bounds'][-1])-pd.Timedelta(minutes=10),freq='10min');splits=origins_for(ts,spec['bounds'])
            for (name,oo),start,end in zip(splits.items(),spec['bounds'][:-1],spec['bounds'][1:]):
                self.assertTrue((ts[oo-143]>=pd.Timestamp(start)).all());self.assertTrue((ts[oo+144]<pd.Timestamp(end)).all())
    def test_inputs_cannot_see_future(self):
        rng=np.random.default_rng(1);v=rng.normal(size=(400,8,2)).astype('float32');ts=pd.date_range('2026-08-01',periods=400,freq='10min');mu=np.zeros((8,2));sd=np.ones((8,2));o=np.array([143,149]);a=inputs(v,ts,o,mu,sd);v[150:]=999;b=inputs(v,ts,o,mu,sd);np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,24,8,8))
    def test_all_model_shapes_and_gradients(self):
        nodes=list(NODES)[:8];a=graph(nodes);x=torch.randn(2,24,8,8)
        for name in ['STGNN','GRU','LSTM']:
            m=network(name,8,a);p=m(x);self.assertEqual(p.shape,(2,5,8));self.assertTrue(torch.isfinite(p).all());p.square().mean().backward();self.assertIsNotNone(m.head.weight.grad)
if __name__=='__main__':unittest.main()
