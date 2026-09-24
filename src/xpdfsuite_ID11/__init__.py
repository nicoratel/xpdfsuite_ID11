"""xpdfsuite package"""

# Imports absolus pour charger les modules
from .xpdfsuite import XRDProcessor
from .xpdfsuite import PDFResultsReference
from .xpdfsuite import extract_xpdf
#from .xpdfsuite import extract_xPDF_from_multiple_files
from .filereader import load_h5_data, load_image
from .calibration import perform_geometric_calibration
from .eigerdata import EigerData

#from .xpdf_pipeline import extract_xpdf_from_h5, extract_xpdf_from_image
# from .pdfanalysis import perform_automatic_pdf_analysis  # Module not found

__version__ = "0.1.0"

# Ce que les utilisateurs peuvent importer
__all__ = [
    'XRDProcessor',
    'PDFResultsReference',
    'extract_xpdf',
    'load_h5_data',
    'load_image',
    'EigerData',
    'extract_xPDF_from_multiple_files',
	#'perform_automatic_pdf_analysis',
    'perform_geometric_calibration',
    'extract_xpdf_from_h5',
    'extract_xpdf_from_image',
]
