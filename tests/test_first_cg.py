import tempfile
import unittest
from pathlib import Path

import pandas as pd
from common.first_cg import read_first_cg_values, validate_first_cg_values


class FirstCgTests(unittest.TestCase):
    def test_monthly_projection_filter_completeness_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); directory = run/'first_cg/data'; directory.mkdir(parents=True)
            pd.DataFrame({'id':list(map(str, range(3, 11)))}).to_parquet(directory/'confirmed_anchors.parquet', index=False)
            frames = []
            for month in range(3, 11):
                frame = pd.DataFrame({'id':[str(month)]*9, 'kind':['compound']*9,
                                      'relative_hour':list(range(-12, 13, 3)), 'unused':list(range(9))})
                frames.append(frame)
                frame.to_parquet(directory/f'first_cg_values_m{month:02d}.parquet', index=False)
            self.assertEqual(validate_first_cg_values(run), 72)
            actual = read_first_cg_values(run, columns=['id','relative_hour'], filters=[('relative_hour','==',0)])
            self.assertEqual(list(actual.columns), ['id','relative_hour'])
            self.assertEqual(list(actual.id), list(map(str, range(3, 11))))
            pd.testing.assert_frame_equal(read_first_cg_values(run), pd.concat(frames, ignore_index=True))
            bad = frames[0].copy(); bad.loc[1,'relative_hour'] = bad.loc[0,'relative_hour']
            bad.to_parquet(directory/'first_cg_values_m03.parquet', index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_first_cg_values(run)
            (directory/'first_cg_values_m04.parquet').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Incomplete'):
                read_first_cg_values(run)


if __name__ == '__main__':
    unittest.main()
