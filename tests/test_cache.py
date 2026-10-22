import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from common import cache


class CacheTests(unittest.TestCase):
    def tearDown(self):
        cache.validate_cache.cache_clear()

    def test_changed_recipe_is_rejected_without_replacing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'metadata.json'
            recipe = {'years':[2006,2025], 'source_size':10}
            with patch.object(cache, 'metadata_path', return_value=path), \
                 patch.object(cache, 'expected_recipe', return_value=recipe.copy()), \
                 patch.object(cache, 'code_version', return_value={'git_commit':'test'}):
                cache.write_metadata('90p20', origin='adopted_existing')
                cache.validate_cache('90p20')
            original = path.read_bytes()
            recipe['source_size'] = 11
            cache.validate_cache.cache_clear()
            with patch.object(cache, 'metadata_path', return_value=path), \
                 patch.object(cache, 'expected_recipe', return_value=recipe):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    cache.validate_cache('90p20')
            self.assertEqual(path.read_bytes(), original)

    def test_failed_atomic_write_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'daily.npz'
            cache.save_array_cache(path, cg=np.array([1,2]))
            original = path.read_bytes()
            def fail(target, **kwargs):
                Path(target).write_bytes(b'incomplete')
                raise RuntimeError('simulated interruption')
            with patch.object(np, 'savez_compressed', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    cache.save_array_cache(path, cg=np.array([3]))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temp).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
