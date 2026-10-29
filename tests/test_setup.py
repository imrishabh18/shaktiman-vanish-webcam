import argparse
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shaktiman import app


class SetupTests(unittest.TestCase):
    def test_preview_is_default_and_meeting_camera_is_opt_in(self):
        with patch('sys.argv', ['shaktiman']):
            self.assertFalse(app.parse_args().virtual_camera)
        with patch('sys.argv', ['shaktiman', '--virtual-camera', '--virtual-backend', 'obs']):
            args = app.parse_args()
        self.assertTrue(args.virtual_camera)
        self.assertEqual(args.virtual_backend, 'obs')

    def test_invalid_frame_rates_are_rejected(self):
        for value in ('0', '121', 'nan', 'inf'):
            with self.assertRaises(argparse.ArgumentTypeError):
                app.positive_fps(value)

    def test_cached_models_do_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'hands.task'
            model.write_bytes(b'x' * 100_001)
            with patch.object(app, 'MODEL_DIR', Path(folder)), \
                 patch.object(app, 'MODEL_URLS', {model: 'https://example.com/model'}), \
                 patch.object(app.urllib.request, 'urlopen') as download:
                app.download_models()
            download.assert_not_called()

    def test_first_launch_downloads_model_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'hands.task'
            payload = b'x' * 100_001
            with patch.object(app, 'MODEL_DIR', Path(folder)), \
                 patch.object(app, 'MODEL_URLS', {model: 'https://example.com/model'}), \
                 patch.object(app.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                app.download_models()
            self.assertEqual(model.read_bytes(), payload)
            self.assertFalse(model.with_suffix('.task.part').exists())

    def test_interrupted_download_does_not_leave_a_partial_model(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'hands.task'
            with patch.object(app, 'MODEL_DIR', Path(folder)), \
                 patch.object(app, 'MODEL_URLS', {model: 'https://example.com/model'}), \
                 patch.object(app.urllib.request, 'urlopen', side_effect=OSError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'download failed'):
                    app.download_models()
            self.assertFalse(model.exists())
            self.assertFalse(model.with_suffix('.task.part').exists())
