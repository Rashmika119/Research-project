from pathlib import Path
import sys
import tempfile
import unittest

from training.process import run_logged


class ProcessTests(unittest.TestCase):
    def test_failure_preserves_output_and_reports_exit_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'failure.log'
            with self.assertRaisesRegex(RuntimeError, 'exit 3') as error:
                run_logged([sys.executable, '-c', 'import sys; print("model file invalid", flush=True); sys.exit(3)'], log)
            self.assertIn(str(log), str(error.exception))
            self.assertIn('model file invalid', str(error.exception))
            self.assertIn('model file invalid', log.read_text())
