import pydicom
import numpy as np
from PIL import Image


def process_dicom(file):
    """Convert a DICOM file to a PIL image.

    Handles: constant pixel data (no division by zero), multi-frame
    (uses middle frame), and MONOCHROME1 (inverted) photometric
    interpretation.
    """
    try:
        dicom = pydicom.dcmread(file)
        pixel_array = dicom.pixel_array

        # Multi-frame (e.g. CT series): take the middle slice
        if pixel_array.ndim > 2:
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

        pixel_array = pixel_array.astype(np.float64)
        pmin, pmax = pixel_array.min(), pixel_array.max()

        # FIX: constant image (max == min) caused ZeroDivisionError
        if pmax == pmin:
            pixel_array = np.zeros_like(pixel_array, dtype="uint8")
        else:
            pixel_array = ((pixel_array - pmin) * (255 / (pmax - pmin))).astype("uint8")

        # MONOCHROME1 is inverted (0 = white); flip to standard MONOCHROME2
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = 255 - pixel_array

        img = Image.fromarray(pixel_array)
        return img.convert("L")
    except Exception as e:
        raise Exception(f"DICOM Processing Error: {e}")
