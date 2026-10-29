import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shaktiman import app
from shaktiman.cameras import CameraDevice, mac_camera_devices, preferred_webcam


class WebcamSelectionTests(unittest.TestCase):
    def setUp(self):
        self.phone = CameraDevice(0, "iPhone Camera", "1", False, True)
        self.webcam = CameraDevice(1, "MacBook Air Camera", "2", True, False)
        self.usb = CameraDevice(2, "USB Camera", "3", False, False)

    def test_builtin_webcam_wins_even_when_iphone_is_first(self):
        self.assertEqual(preferred_webcam([self.phone, self.usb, self.webcam]), self.webcam)

    def test_connected_webcam_is_used_when_no_builtin_exists(self):
        self.assertEqual(preferred_webcam([self.phone, self.usb]), self.usb)

    def test_phone_is_never_an_automatic_fallback(self):
        for devices in ([], [self.phone]):
            with self.assertRaisesRegex(RuntimeError, "No computer webcam"):
                preferred_webcam(devices)

    def test_camera_order_matches_opencv_unique_id_order(self):
        builtin = SimpleNamespace(uniqueID=lambda: "Z", localizedName=lambda: "MacBook Air Camera",
                                  deviceType=lambda: "AVCaptureDeviceTypeBuiltInWideAngleCamera")
        phone = SimpleNamespace(uniqueID=lambda: "A", localizedName=lambda: "My iPhone Camera",
                                deviceType=lambda: "AVCaptureDeviceTypeContinuityCamera")
        devices = Mock()
        devices.devicesWithMediaType_.side_effect = [[builtin, phone], []]
        module = SimpleNamespace(AVCaptureDevice=devices, AVMediaTypeVideo="video", AVMediaTypeMuxed="muxed")
        with patch.dict("sys.modules", AVFoundation=module):
            result = mac_camera_devices()
        self.assertEqual([device.name for device in result], ["My iPhone Camera", "MacBook Air Camera"])
        self.assertTrue(result[0].phone)
        self.assertTrue(result[1].built_in)
        self.assertEqual(preferred_webcam(result).index, 1)

    def test_auto_opens_webcam_with_native_backend(self):
        capture = Mock()
        with patch.object(app.platform, "system", return_value="Darwin"), \
             patch.object(app, "mac_camera_devices", return_value=[self.phone, self.webcam]), \
             patch.object(app.cv2, "VideoCapture", return_value=capture) as opening:
            self.assertIs(app.open_camera(None, 960, 540), capture)
        opening.assert_called_once_with(1, app.cv2.CAP_AVFOUNDATION)

    def test_explicit_index_is_respected(self):
        with patch.object(app.platform, "system", return_value="Darwin"), \
             patch.object(app, "mac_camera_devices") as devices, \
             patch.object(app.cv2, "VideoCapture") as opening:
            app.open_camera(3, 960, 540)
        opening.assert_called_once_with(3, app.cv2.CAP_AVFOUNDATION)
        devices.assert_not_called()

    def test_other_platforms_keep_default_camera_zero(self):
        with patch.object(app.platform, "system", return_value="Linux"), \
             patch.object(app.cv2, "VideoCapture") as opening:
            app.open_camera(None, 960, 540)
        opening.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
