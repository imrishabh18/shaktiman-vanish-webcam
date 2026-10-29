import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from shaktiman import app
from shaktiman.audio import EffectAudio, SAMPLE_RATE, create_audio, load_music, meeting_devices


def backend():
    result = Mock()
    result.query_devices.return_value = [
        dict(name="MacBook Microphone", max_input_channels=1, max_output_channels=0),
        dict(name="Speakers", max_input_channels=0, max_output_channels=2),
        dict(name="BlackHole 2ch", max_input_channels=2, max_output_channels=2),
    ]
    result.default.device = [0, 1]
    return result


class AudioPlaybackTests(unittest.TestCase):
    def test_local_stream_never_opens_the_microphone(self):
        device = backend()
        music = EffectAudio(np.full((10, 2), 0.2, np.float32), backend=device)
        device.Stream.assert_not_called()
        device.OutputStream.assert_called_once()
        output = np.empty((4, 2), np.float32)
        music._output_callback(output, 4, None, None)
        self.assertTrue(np.all(output == 0))
        music.play()
        music._output_callback(output, 4, None, None)
        np.testing.assert_allclose(output, 0.2)
        music.close()

    def test_cue_plays_once_then_outputs_silence(self):
        music = EffectAudio(np.full((3, 2), 0.25, np.float32), backend=backend())
        music.play()
        output = np.empty((5, 2), np.float32)
        music._output_callback(output, 5, None, None)
        np.testing.assert_allclose(output[:3], 0.25)
        np.testing.assert_allclose(output[3:], 0)
        music._output_callback(output, 5, None, None)
        np.testing.assert_allclose(output, 0)
        music.close()

    def test_retrigger_restarts_without_overlapping_and_stop_is_immediate(self):
        samples = np.repeat(np.arange(8, dtype=np.float32)[:, None] / 10, 2, axis=1)
        music = EffectAudio(samples, backend=backend())
        output = np.empty((3, 2), np.float32)
        music.play()
        music._output_callback(output, 3, None, None)
        music.play()
        music._output_callback(output, 3, None, None)
        np.testing.assert_allclose(output, samples[:3])
        music.stop()
        music._output_callback(output, 3, None, None)
        np.testing.assert_array_equal(output, 0)
        music.close()

    def test_meeting_stream_passes_voice_even_when_music_is_stopped(self):
        device = backend()
        music = EffectAudio(np.full((20, 2), 0.3, np.float32), meeting=True, backend=device)
        self.assertEqual(device.Stream.call_args.kwargs['device'], (0, 2))
        device.OutputStream.assert_not_called()
        voice = np.full((4, 1), 0.2, np.float32)
        output = np.empty((4, 2), np.float32)
        music._meeting_callback(voice, output, 4, None, None)
        np.testing.assert_allclose(output, 0.2)
        music.play()
        music._meeting_callback(voice, output, 4, None, None)
        np.testing.assert_allclose(output, 0.5)
        music.stop()
        music._meeting_callback(voice, output, 4, None, None)
        np.testing.assert_allclose(output, 0.2)
        music.close()

    def test_mixed_audio_is_limited_to_valid_sample_range(self):
        music = EffectAudio(np.full((10, 2), 0.8, np.float32), meeting=True, backend=backend())
        music.play()
        output = np.empty((4, 2), np.float32)
        music._meeting_callback(np.full((4, 1), 0.9, np.float32), output, 4, None, None)
        self.assertLessEqual(float(output.max()), 1)
        music.close()

    def test_virtual_device_is_not_used_as_its_own_microphone(self):
        device = backend()
        device.default.device = [2, 1]
        self.assertEqual(meeting_devices(device), (0, 2))
        with self.assertRaisesRegex(ValueError, 'physical microphone'):
            meeting_devices(device, microphone='BlackHole 2ch')

    def test_missing_virtual_device_has_an_actionable_error(self):
        device = backend()
        device.query_devices.return_value = device.query_devices.return_value[:2]
        with self.assertRaisesRegex(RuntimeError, 'Install BlackHole'):
            meeting_devices(device)

    def test_close_is_idempotent_and_releases_stream(self):
        device = backend()
        music = EffectAudio(np.zeros((2, 2), np.float32), backend=device)
        music.close()
        music.close()
        device.OutputStream.return_value.close.assert_called_once()


class MusicFileTests(unittest.TestCase):
    def test_offset_stereo_resampling_and_click_free_ends(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cue.wav'
            raw = np.concatenate((np.full(22_050, 0.1), np.full(22_050, 0.8)))
            sf.write(path, raw, 22_050, subtype='PCM_16')
            cue = load_music(path, start=1, volume=0.5)
            self.assertEqual(cue.shape, (SAMPLE_RATE, 2))
            np.testing.assert_allclose(cue[1000:-3000], 0.4, atol=0.001)
            np.testing.assert_array_equal(cue[0], 0)
            np.testing.assert_array_equal(cue[-1], 0)

    def test_long_files_are_limited_to_a_three_second_cue(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cue.wav'
            sf.write(path, np.full((SAMPLE_RATE * 5, 2), 0.25), SAMPLE_RATE)
            self.assertEqual(load_music(path).shape, (SAMPLE_RATE * 3, 2))
            with self.assertRaisesRegex(ValueError, 'past the end'):
                load_music(path, start=10)

    def test_no_clip_is_silent_and_muting_avoids_output_stream(self):
        with tempfile.TemporaryDirectory() as folder, patch('shaktiman.audio.EffectAudio') as player:
            self.assertIsNone(create_audio(SimpleNamespace(), Path(folder)))
            self.assertIsNone(create_audio(SimpleNamespace(no_music=True), Path(folder)))
            player.assert_not_called()

    def test_auto_discovered_clip_and_muted_meeting_voice(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'music.wav'
            sf.write(path, np.ones((1000, 2), np.float32) * 0.2, SAMPLE_RATE)
            with patch('shaktiman.audio.EffectAudio') as player:
                create_audio(SimpleNamespace(), Path(folder))
                self.assertGreater(len(player.call_args.args[0]), 0)
                create_audio(SimpleNamespace(no_music=True, meeting_audio=True), Path(folder))
                self.assertEqual(len(player.call_args.args[0]), 0)
                self.assertTrue(player.call_args.kwargs['meeting'])

    def test_audio_flags_and_invalid_values(self):
        with patch('sys.argv', ['shaktiman', '--meeting-audio', '--no-music', '--music-volume', '0.3']):
            args = app.parse_args()
        self.assertTrue(args.meeting_audio)
        self.assertTrue(args.no_music)
        self.assertEqual(args.music_volume, 0.3)
        for value in ('-1', '2', 'nan'):
            with self.assertRaises(argparse.ArgumentTypeError):
                app.music_volume(value)
