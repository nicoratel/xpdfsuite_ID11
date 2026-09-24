
import numpy as np
from .eigerdata import EigerData
import fabio
import os
import sys
import shutil


def draw_mask(image_file):
    """
    Launch the pyFAI-drawmask GUI to interactively draw a pixel mask.

    The input DM4 image is temporarily exported as an EDF file, passed to
    the ``pyFAI-drawmask`` tool, then the EDF file is deleted. The mask
    produced by the GUI is saved alongside the image by pyFAI.

    Parameters
    ----------
    image_file : str
        Path to the image file.
    """
    # load data and metadata
    eiger = EigerData(image_file)
    if eiger.data.ndim == 3:
        eiger.data = np.mean(eiger.data, axis=0)  # average frames if multiple frames are present:
    # file extension detection
    extension = os.path.splitext(image_file)[1].lower()
        
    # Define output EDF file name
    edffile = image_file.replace(extension, '.edf')

    # Create EDF image and save
    edf_image = fabio.edfimage.EdfImage(data=eiger.data, header=eiger.detector_info)
    edf_image.write(edffile)
    # edit command to use the same python executable as the current environment (important for pyFAI-drawmask to find the right fabio installation)
    path = shutil.which("pyFAI-drawmask")
    os.system(f'"{sys.executable}" {path} {edffile}')
    os.remove(edffile)
