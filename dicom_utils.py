import pydicom
import numpy as np
from PIL import Image

def process_dicom(file):
    try:
        dicom = pydicom.dcmread(file)
        pixel_array = dicom.pixel_array
        # Normalize to 0-255
        pixel_array = ((pixel_array - pixel_array.min()) * (255 / (pixel_array.max() - pixel_array.min()))).astype('uint8')
        img = Image.fromarray(pixel_array)
        return img
    except Exception as e:
        raise Exception(f"DICOM Processing Error: {e}")
