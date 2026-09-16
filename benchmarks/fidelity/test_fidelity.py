"""Small CPU checks; full real-output replay receipts are in VALIDATION.json."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import fidelity

class Metrics(unittest.TestCase):
    def test_identity(self):
        x=np.array([[0.,1.,2.],[3.,3.,-4.]])
        m=fidelity.compare(x,x,np.array([2,0]))
        np.testing.assert_array_equal(m['kl'],[0,0])
        np.testing.assert_array_equal(m['top1'],[1,1])
    def test_exact_reference_tie(self):
        m=fidelity.compare(np.array([[0.,0.]]),np.array([[-1.,0.]]),np.array([0]))
        self.assertEqual(m['top1'][0],0)
        self.assertEqual(m['tie_aware'][0],1)
        self.assertAlmostEqual(m['kl'][0],np.log((np.exp(-1)+1)/2)+.5)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            fidelity.compare(np.array([[np.nan,0.]]),np.array([[0.,0.]]),np.array([0]))
    def test_corrupt_reference_rejected(self):
        first=fidelity.read(fidelity.DATA/'reference.json')['files'][0]['file']
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/first).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt cached reference'):
                fidelity.fetch_reference(Path(tmp))
    def test_partial_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            fidelity.save(p/'manifest.json',{'status':'PASS','output_shape':[1,1024,248320]})
            with self.assertRaisesRegex(ValueError,'Incomplete capture'):
                fidelity.score(p,p,'invalid partial',1)
    def test_frozen_dataset(self):
        self.assertEqual(fidelity.verify_data()['panel_sha256'],'d6f3f47327f7f6d2beb41878f59752dd98d6e3eed8a0bcf340e44242bb294f16')
if __name__=='__main__':unittest.main()
