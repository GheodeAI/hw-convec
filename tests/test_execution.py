import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import execution
from scripts.pipeline import run_analysis


class ExecutionTests(unittest.TestCase):
    def test_completed_stage_adopts_without_launching_process(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'done.csv'; output.write_text('value\n1\n')
            state = {}
            with patch.object(execution, 'command', side_effect=AssertionError('unexpected command')):
                self.assertFalse(execution.run_stage('fit', ['python', 'fit'], [output], state))
            self.assertEqual(state['fit']['origin'], 'adopted_existing')
            state['fit']['recipe'] = 'different'
            with self.assertRaisesRegex(ValueError, 'different code/settings'):
                execution.run_stage('fit', ['python', 'fit'], [output], state)

    def test_missing_artifact_does_not_record_success(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {}
            with patch.object(execution, 'command'):
                with self.assertRaisesRegex(RuntimeError, 'expected outputs'):
                    execution.run_stage('fit', ['python', 'fit'], [Path(temp)/'missing'], state)
            self.assertEqual(state, {})

    def test_archive_preserves_files_and_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'outputs/run'; directory.mkdir(parents=True)
            source = directory/'result.csv'; source.write_bytes(b'original\x00bytes')
            with patch.object(execution, 'PROJECT_DIR', root):
                execution.archive_paths([directory, source], dry_run=True)
                self.assertEqual(list(root.iterdir()), [root/'outputs'])
                execution.archive_paths([directory, source])
            self.assertFalse(directory.exists())
            archived = list((root/'archive').rglob('result.csv'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), b'original\x00bytes')
            self.assertEqual(len((root/'logs/reruns.tsv').read_text().splitlines()), 1)

    def _pipeline(self, root):
        shutil.copytree(execution.PROJECT_DIR/'common', root/'common',
                        ignore=shutil.ignore_patterns('__pycache__'))
        run = root/'outputs/custom'
        specs = [('rebuild', 'checks/rebuild.json', []),
                 ('fit', 'fit/result.csv', ['fit']), ('report', 'report/result.csv', ['report'])]
        stages = [(name, ['python', name], [run/product], [run/p for p in cleanup])
                  for name, product, cleanup in specs]
        for _, _, products, _ in stages:
            products[0].parent.mkdir(parents=True, exist_ok=True); products[0].write_text('old')
        return run, stages

    def test_resume_partial_stage_archives_and_recomputes_downstream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, stages = self._pipeline(root)
            stages[1][2][0].unlink()
            (run/'fit/partial.csv').write_text('partial result')
            launched = []
            def launch(cmd, **kwargs):
                launched.append(cmd[-1])
                for name, _, products, _ in stages:
                    if name == cmd[-1]:
                        products[0].parent.mkdir(parents=True, exist_ok=True); products[0].write_text('new')
            with patch.object(run_analysis, 'main_stages', return_value=stages), \
                 patch.object(run_analysis, 'metadata_path', side_effect=lambda method:root/'cache'/method/'metadata.json'), \
                 patch.object(execution, 'PROJECT_DIR', root), \
                 patch.object(execution, 'command', side_effect=launch):
                run_analysis.run_main(run, root/'climatology')
            self.assertEqual(launched, ['fit', 'report'])
            self.assertEqual(stages[0][2][0].read_text(), 'old')
            self.assertEqual(stages[2][2][0].read_text(), 'new')
            self.assertTrue(list((root/'archive').rglob('partial.csv')))
            self.assertEqual(json.loads((run/'checks/workflow.json').read_text())['report']['origin'], 'generated')

    def test_interrupted_rebuild_restarts_without_requiring_complete_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, stages = self._pipeline(root)
            stages[0][2][0].unlink()
            (run/'checks/analysis_inputs.json').write_text('{"method":"90p20"}')
            launched = []
            def launch(cmd, **kwargs):
                launched.append(cmd[-1])
                for name, _, products, _ in stages:
                    if name == cmd[-1]:
                        products[0].parent.mkdir(parents=True, exist_ok=True)
                        products[0].write_text('new')
            with patch.object(run_analysis, 'main_stages', return_value=stages), \
                 patch.object(run_analysis, 'metadata_path', side_effect=lambda method:root/'cache'/method/'metadata.json'), \
                 patch.object(run_analysis, 'validate_run_inputs', side_effect=AssertionError('incomplete inputs are expected')), \
                 patch.object(execution, 'PROJECT_DIR', root), \
                 patch.object(execution, 'command', side_effect=launch):
                run_analysis.run_main(run, root/'climatology')
            self.assertEqual(launched, ['rebuild','fit','report'])
            self.assertTrue(list((root/'archive').rglob('analysis_inputs.json')))

    def test_force_preview_does_not_move_or_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, stages = self._pipeline(root)
            before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(run_analysis, 'main_stages', return_value=stages), \
                 patch.object(run_analysis, 'metadata_path', side_effect=lambda method:root/'cache'/method/'metadata.json'), \
                 patch.object(execution, 'PROJECT_DIR', root):
                run_analysis.run_main(run, root/'climatology', force=True, dry_run=True)
            after = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((root/'archive').exists())


if __name__ == '__main__':
    unittest.main()
