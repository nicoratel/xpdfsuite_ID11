from matplotlib import pyplot as plt
from xpdfsuite_ID11 import extract_xpdf,XRDProcessor,EigerData
import os
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt

def extract_number_of_images(file):
    
    eiger = EigerData(file)
    return eiger.nb_frames


def process_lost_flow(file,
                      ref_file=None,
                      poni_file=None,
                      mask_file=None,
                      polarization = 0.99,
                      plot=True,
                      rmin = 0,
                      rmax = 50,
                      rstep = 0.01,
                      bgscale=1,
                      qmin=0.5,
                      qmax=24,
                      qmaxinst=None,
                      rpoly=0.9,
                      frame_binning=1):
    """
    Process a series of images from a lost-flow experiment, extracting the PDF for each frame.
    
    Parameters
    ----------
    file : str
        Path to the sample HDF5 file.
    ref_file : str
        Path to the reference HDF5 file.
    poni_file : str
        Path to the pyFAI calibration file.
    mask_file : str
        Path to the pixel mask file.
    polarization : float
        Polarization factor for the detector.  
    plot : bool
        Whether to plot the PDFs for each frame.
    rmin : float
        Minimum r value for PDF extraction.
    rmax : float
        Maximum r value for PDF extraction.
    rstep : float
        Step size for r values in PDF extraction.
    bgscale : float
        Background scaling factor for PDF extraction.
    qmin : float
        Minimum q value for PDF extraction.
    qmax : float
        Maximum q value for PDF extraction.
    qmaxinst : float
        Maximum q value for polynomial fitting.
    rpoly : float
        Radius for polynomial fitting.
    frame_binning : int
        Number of consecutive frames to average together for PDF extraction.

    Returns
    -------
    pdf_files : list of str
        List of paths to the output PDF files for each frame.
    res : dict
        Dictionary containing the extracted r and G values for each frame, along with the scanned motor positions
        res[index]['positions'] = {motor_name: motor_value}
        res[index]['r'] = r_values
        res[index]['G'] = G_values
    """
    nb_images = extract_number_of_images(file)
    if ref_file is not None:
        nb_images_ref = extract_number_of_images(ref_file)
    else:
        nb_images_ref = nb_images
    pdf_files = []
    if nb_images != nb_images_ref:
        raise ValueError("The number of images in the sample and reference files do not match.")

    # Group consecutive frames by bins of size `frame_binning` (averaged together)
    nb_bins = nb_images // frame_binning
    if nb_bins == 0:
        raise ValueError(f"frame_binning={frame_binning} is larger than the number of images ({nb_images}).")
    if nb_images % frame_binning != 0:
        print(f"Warning: {nb_images} images is not a multiple of frame_binning={frame_binning}; "
              f"the last {nb_images % frame_binning} image(s) will be discarded.")

    if plot:
        plt.figure()

    res = {}
    for bin_idx in range(nb_bins):
        
        frame_group = list(range(bin_idx * frame_binning, (bin_idx + 1) * frame_binning))
        frame_arg = frame_group[0] if frame_binning == 1 else frame_group
        res[str(frame_group)] = {}
        # Initialize XRDProcessor for sample and reference data
        print(f"Processing frame group: {frame_group}")
        sample_processor = XRDProcessor(file, frame=frame_arg, poni_file=poni_file, mask=mask_file, polarization_factor=polarization)
        if ref_file is not None:
            ref_processor = XRDProcessor(ref_file, frame=frame_arg, poni_file=poni_file, mask=mask_file, polarization_factor=polarization)
        else:
            ref_processor = None

        # Extract PDF from sample and reference data
        if len(frame_group) == 1:
            frame_str = str(frame_group[0])
        else:
            frame_str = f"{frame_group[0]}-{frame_group[-1]}"
        outputfile = f'{os.path.dirname(file)}/{os.path.basename(file).split(".")[0]}_frames={frame_str}.gr'
        position ={}
        for motor, value in sample_processor.scanned_motors.items():            
            position[motor] = value[bin_idx] if frame_binning == 1 else value[frame_group][0] #assuming the motor position is constant over the binned frames, we take the first value

        res[str(frame_group)]['position'] = position
        
        r, G = extract_xpdf(sample_processor,
                            ref_processor=ref_processor,
                            composition='Au',
                            rmin=rmin,
                            rmax=rmax,
                            rstep=rstep,
                            outputfile=outputfile,
                            interactive=False,
                            plot=False,
                            bgscale=bgscale,
                            qmin=qmin,
                            qmax=qmax,
                            qmaxinst=qmaxinst,
                            rpoly=rpoly)
        res[str(frame_group)]['r'] = r
        res[str(frame_group)]['G'] = G
        pdf_files.append(outputfile)
        if plot:
            plt.plot(r, G, label=f'Frame {",".join(map(str, frame_str))}')
    if plot:
        plt.xlabel('r (Å)')
        plt.ylabel('G(r)')
        plt.legend()
    return pdf_files, res


