import unittest,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from allocation_race import claim_allocation
class RaceTests(unittest.TestCase):
 def test_exactly_one_winner_and_same_job_retry_allowed(self):
  with tempfile.TemporaryDirectory() as d:
   with ThreadPoolExecutor(max_workers=2) as pool:r=list(pool.map(lambda job:claim_allocation(d,job),['a.gadi-pbs','b.gadi-pbs']))
   self.assertEqual(sum(won for won,_ in r),1)
   winner=next(receipt['job_id'] for won,receipt in r if won)
   self.assertTrue(claim_allocation(d,winner)[0]);self.assertFalse(claim_allocation(d,'other.gadi-pbs')[0])
 def test_incomplete_receipt_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'allocation_winner.json').write_text('{}')
   with self.assertRaises(KeyError):claim_allocation(d,'other')
if __name__=='__main__':unittest.main(verbosity=2)
