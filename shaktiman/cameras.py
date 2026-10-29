"""Name macOS cameras in the same order used by OpenCV's AVFoundation backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    unique_id: str
    built_in: bool
    phone: bool


def mac_camera_devices() -> list[CameraDevice]:
    # Import only on macOS. This lists devices without opening any camera.
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo, AVMediaTypeMuxed
    except ImportError as error:
        raise RuntimeError(
            "Camera name detection needs the macOS dependencies. "
            "Run start.command again to install them."
        ) from error

    devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo))
    devices += list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))
    # OpenCV 4/5 sorts this same device list by uniqueID, not localized name.
    devices.sort(key=lambda device: str(device.uniqueID()))
    result = []
    for index, device in enumerate(devices):
        name = str(device.localizedName())
        kind = str(device.deviceType())
        phone = any(word in (name + " " + kind).lower()
                    for word in ("iphone", "continuity", "deskview"))
        built_in = kind == "AVCaptureDeviceTypeBuiltInWideAngleCamera" and not phone
        result.append(CameraDevice(index, name, str(device.uniqueID()), built_in, phone))
    return result


def preferred_webcam(devices: list[CameraDevice]) -> CameraDevice:
    """Prefer the built-in webcam; never silently fall back to a phone."""
    for device in devices:
        if device.built_in and not device.phone:
            return device
    for device in devices:
        if not device.phone:
            return device
    raise RuntimeError(
        "No computer webcam is available. Connect a webcam or use --camera INDEX "
        "to explicitly select another camera."
    )
