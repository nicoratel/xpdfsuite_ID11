from matplotlib import pyplot as plt
from xpdfsuite_ID11 import extract_xpdf,XRDProcessor,ID11Data
import os
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt

def extract_number_of_images(file,entry):
    
    eiger = ID11Data(file)
    return eiger.nb_frames[entry]


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
                      frame_step=None,
                      nb_repeats=None,
                      frame_binning=1,
                      entry_sample=None,
                      entry_ref=None):
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
    frame_step : int
        Step size between consecutive frames in a series. 
        e.g., if frame_binning=2 and frame_step=4, the first series will be frames 0-1, the second series will be frames 4-5, etc.
    nb_repeats : int
        Number of repeats to average together. (None means all repeats that fit in the number of images will be used.)

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
    if entry_sample is None:
        eiger = ID11Data(file)
        entry_sample = eiger.entries[0]
    
    nb_images = extract_number_of_images(file, entry_sample)
    nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
    pdf_files = []
    if nb_images != nb_images_ref:
        raise ValueError("The number of images in the sample and reference files do not match.")

    # Paramètres (à ajouter à la signature de la fonction)
    # frame_binning : nb de frames consécutives moyennées dans une série
    # frame_step    : décalage (en frames) entre deux répétitions de la série (None = pas de moyenne inter-séries)
    # nb_repeats    : nb de séries à moyenner (None = toutes celles qui rentrent dans nb_images)

    if frame_step is None:
        # Comportement actuel : groupes consécutifs uniquement
        nb_bins = nb_images // frame_binning
        if nb_bins == 0:
            raise ValueError(f"frame_binning={frame_binning} is larger than the number of images ({nb_images}).")
        if nb_images % frame_binning != 0:
            print(f"Warning: {nb_images} images is not a multiple of frame_binning={frame_binning}; "
                f"the last {nb_images % frame_binning} image(s) will be discarded.")
        nb_repeats_eff = 1
        series_len = nb_images
    else:
        if frame_step < frame_binning:
            raise ValueError(f"frame_step={frame_step} must be >= frame_binning={frame_binning}.")
        series_len = frame_step
        max_repeats = nb_images // frame_step
        nb_repeats_eff = max_repeats if nb_repeats is None else nb_repeats
        if nb_repeats_eff < 1 or nb_repeats_eff > max_repeats:
            raise ValueError(f"nb_repeats={nb_repeats} invalid: {nb_images} images with "
                            f"frame_step={frame_step} allow at most {max_repeats} repeats.")
        if frame_step % frame_binning != 0:
            print(f"Warning: frame_step={frame_step} is not a multiple of frame_binning={frame_binning}; "
                f"the last {frame_step % frame_binning} frame(s) of each series will be discarded.")
        if nb_images % frame_step != 0 and nb_repeats is None:
            print(f"Warning: {nb_images % frame_step} trailing image(s) do not form a complete series and will be discarded.")
        nb_bins = frame_step // frame_binning

    if plot:
        plt.figure()

    res = {}
    for bin_idx in range(nb_bins):
        base = bin_idx * frame_binning
        step = frame_step if frame_step is not None else 0
        frame_group = [
            base + k + r * step
            for r in range(nb_repeats_eff)
            for k in range(frame_binning)
        ]
        frame_arg = frame_group[0] if len(frame_group) == 1 else frame_group
        res[str(frame_group)] = {}
        print(f"Processing frame group: {frame_group}")
        sample_processor = XRDProcessor(file, frame=frame_arg, poni_file=poni_file,
                                        mask=mask_file, polarization_factor=polarization, entry=entry_sample)
        if ref_file is not None:
            ref_processor = XRDProcessor(ref_file, frame=frame_arg, poni_file=poni_file, mask=mask_file, polarization_factor=polarization, entry=entry_ref)
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
            plt.plot(r, G, label=f'Frame {frame_str}')
    if plot:
        plt.xlabel('r (Å)')
        plt.ylabel('G(r)')
        plt.legend()
    return pdf_files, res

def convert_position_to_time(res, scanned_motors, débit=8):  # débit en µL/min
    """
    Convert the scanned motor positions to reaction time based on the experimental setup.

    Parameters
    ----------
    res : dict
        Dictionary containing the extracted r and G values for each frame, along with the scanned motor positions.
    scanned_motors : dict
        Dictionary containing the scanned motor names and their corresponding values for each frame group.
        

    Returns
    -------
    reaction_time : dict
        Dictionary containing the reaction time for each frame group.
        reaction_time[frame_group] = time_value
    """
    # Implement this function based on your experimental setup.
    # For example, if you have a linear relationship between a motor position and time, you can use that here.
    # This is a placeholder implementation and should be replaced with actual logic.
    
    reaction_time = {}
    i=0
    for frame_group, data in res.items():
        
        #placeholder logic: simply using the index as time value. Replace this with actual conversion logic.
        reaction_time[frame_group] = i  # Replace this with actual conversion logic based on your experimental setup.
        i+=1
    
    return reaction_time

def monitor_pdf_peaks_lost_flow(file,
                      ref_file=None,
                      poni_file=None,
                      mask_file=None,
                      polarization = 0.99,
                      rmin = 0,
                      rmax = 50,
                      rstep = 0.01,
                      bgscale=1,
                      qmin=0.5,
                      qmax=24,
                      qmaxinst=None,
                      rpoly=0.9,
                      frame_step=None,
                      nb_repeats=None,
                      frame_binning=1,
                      entry_sample=None,
                      entry_ref=None,
                      peak_positions=[2.3,2.9],
                      fit_window=0.4,
                      plot=True):

    """
    Monitor the evolution of specific PDF peaks over a series of images from a lost-flow experiment.

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
        Whether to plot the peak evolution for each frame.
    rmin : float
        Minimum r value for PDF extraction.
    rmax : float
        Maximum r value for PDF extraction.
    rstep : float
        Step size for the r values in the PDF.
    bgscale : float
        Scale factor for the background.
    qmin : float
        Minimum q value for PDF extraction.
    qmax : float
        Maximum q value for PDF extraction.
    qmaxinst : float
        Maximum instrumental q value for PDF extraction.
    rpoly : float
        Polynomial degree for the background fitting.
    frame_binning : int
        Number of consecutive frames to average together for PDF extraction.
    frame_step : int
        Step size between consecutive frames in a series. 
        e.g., if frame_binning=2 and frame_step=4, the first series will be frames 0-1, the second series will be frames 4-5, etc.
    nb_repeats : int
        Number of repeats to average together. (None means all repeats that fit in the number of images will be used.)
    entry_sample : str
        The entry in the sample HDF5 file to process. If None, the first entry will be used.
    entry_ref : str
        The entry in the reference HDF5 file to process. If None, the first entry will be used.
    peak_positions : list of float  
        List of approximate peak positions (in r) to monitor. 
    
    Returns
    -------
    peak_evolution : dict
        Dictionary containing the fitted peak parameters for each frame and each monitored peak.
        peak_evolution[frame_group][peak_position] = {'amplitude': ..., 'center': ..., 'fwhm': ..., 'eta': ..., 'bkg': ..., 'area': ...}
    """

    pdf_files, res = process_lost_flow(file,
                                       ref_file=ref_file,
                                       poni_file=poni_file,
                                       mask_file=mask_file,
                                       polarization=polarization,
                                       rmin=rmin,
                                       rmax=rmax,
                                       rstep=rstep,
                                       bgscale=bgscale,
                                       qmin=qmin,
                                       qmax=qmax,
                                       qmaxinst=qmaxinst,
                                       rpoly=rpoly,
                                       frame_step=frame_step,
                                       nb_repeats=nb_repeats,
                                       frame_binning=frame_binning,
                                       entry_sample=entry_sample,
                                       entry_ref=entry_ref,
                                       plot=False)

    scanned_motors = res[list(res.keys())[0]]['position']  # Assuming all frame groups have the same scanned motors
    print(f"Scanned motors: {scanned_motors}")
    # il va falloir relier les positions (x,z) au temps de réaction, d'une façon ou d'une autre...
    reaction_time = convert_position_to_time(res, scanned_motors)  # Implement this function based on your experimental setup

    peak_evolution = {}
    for frame_group, data in res.items():
        r = res[frame_group]['r']
        G = res[frame_group]['G']
        # reaction_time has the same length as the list of frame_groups, so we can use it to label the x-axis in the plot
        time_value = reaction_time[frame_group]
        peak_evolution[time_value] = {}
        for peak_pos in peak_positions:
            print(f"Fitting peak at {peak_pos} Å for frame group {frame_group} (time={time_value})")
            try:
                popt, pcov, area = fit_peak_pseudovoigt(r, G, position=peak_pos, window=fit_window, plot=False)
                peak_evolution[time_value][peak_pos] = {
                    'amplitude': popt[0],
                    'center': popt[1],
                    'fwhm': popt[2],
                    'eta': popt[3],
                    'bkg': popt[4],
                    'area': area
                }
            except Exception as e:
                print(f"Peak fitting failed for frame group {frame_group} at peak position {peak_pos}: {e}")
                peak_evolution[time_value][peak_pos] = None

    if plot:
        fig, ax = plt.subplots(2,2, figsize=(12, 8),dpi=200)
        for peak_pos in peak_positions:
            times = []
            amplitudes = []
            area = []
            fwhm = []
            center = []
            for time_value in sorted(peak_evolution.keys()):
                if peak_evolution[time_value][peak_pos] is not None:
                    times.append(time_value)
                    amplitudes.append(peak_evolution[time_value][peak_pos]['amplitude'])
                    area.append(peak_evolution[time_value][peak_pos]['area'])
                    fwhm.append(peak_evolution[time_value][peak_pos]['fwhm'])
                    center.append(peak_evolution[time_value][peak_pos]['center'])

            ax[0,0].plot(times, amplitudes, marker='o', label=f'Peak at {peak_pos} Å')
            ax[0,1].plot(times, area, marker='o', label=f'Peak at {peak_pos} Å')
            ax[1,0].plot(times, fwhm, marker='o', label=f'Peak at {peak_pos} Å')
            ax[1,1].plot(times, center, marker='o', label=f'Peak at {peak_pos} Å')
        ax[0,0].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[0,0].set_ylabel('Peak Amplitude')
        ax[0,0].set_title('Evolution of PDF Peaks Over Time')
        ax[0,0].legend()
        ax[0,0].grid()
        ax[0,1].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[0,1].set_ylabel('Peak Area')
        ax[0,1].set_title('Evolution of PDF Peaks Over Time')
        ax[0,1].legend()
        ax[0,1].grid()
        ax[1,0].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[1,0].set_ylabel('Peak FWHM')
        ax[1,0].set_title('Evolution of PDF Peaks Over Time')
        ax[1,0].legend()
        ax[1,0].grid()
        ax[1,1].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[1,1].set_ylabel('Peak Center')
        ax[1,1].set_title('Evolution of PDF Peaks Over Time')
        ax[1,1].legend()
        ax[1,1].grid()
        plt.show()

    return peak_evolution



def process_stop_flow(file,
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
                      entry_sample=None,
                      entry_ref=None
                      ):
    """
    Process a single image from a stop-flow experiment, extracting the PDF.
    
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
        Whether to plot the PDF.
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
    entry_sample : str
        The entry in the sample HDF5 file to process. If None, the first entry will be used.
    entry_ref : str
        The entry in the reference HDF5 file to process. If None, the first entry will be used.
    
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
    if entry_sample is None:
        eiger = ID11Data(file)
        entry_sample = eiger.entries[0]
    
    nb_images = extract_number_of_images(file, entry_sample)
    nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
    pdf_files = []
    if nb_images != nb_images_ref:
        raise ValueError("The number of images in the sample and reference files do not match.")

    # Paramètres (à ajouter à la signature de la fonction)
    # frame_binning : nb de frames consécutives moyennées dans une série
    # frame_step    : décalage (en frames) entre deux répétitions de la série (None = pas de moyenne inter-séries)
    # nb_repeats    : nb de séries à moyenner (None = toutes celles qui rentrent dans nb_images)

    

    if plot:
        plt.figure()

    res = {}
    for bin_idx in range(nb_images):
        
        res[str(bin_idx)] = {}
        print(f"Processing frame group: {bin_idx}")
        frame_arg = bin_idx
        sample_processor = XRDProcessor(file, frame=frame_arg, poni_file=poni_file,
                                        mask=mask_file, polarization_factor=polarization, entry=entry_sample)
        
        if ref_file is not None:
            ref_processor = XRDProcessor(ref_file, frame=frame_arg, poni_file=poni_file, mask=mask_file, polarization_factor=polarization, entry=entry_ref)
        else:
            ref_processor = None
        frame_group = [bin_idx] 
        # Extract PDF from sample and reference data
        if len(frame_group) == 1:
            frame_str = str(frame_group[0])
        else:
            frame_str = f"{frame_group[0]}-{frame_group[-1]}"
        outputfile = f'{os.path.dirname(file)}/{os.path.basename(file).split(".")[0]}_frames={frame_str}.gr'
        timedict={}

        reaction_time =compute_reaction_time_stop_flow(sample_processor)
        for i, frame in enumerate(frame_group):
            timedict[str(frame)] = reaction_time[i]

        res[str(bin_idx)]['reaction_time'] = timedict
        
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
        res[str(bin_idx)]['r'] = r
        res[str(bin_idx)]['G'] = G
        pdf_files.append(outputfile)
        if plot:
            plt.plot(r, G, label=f'Frame {frame_str}')
    if plot:
        plt.xlabel('r (Å)')
        plt.ylabel('G(r)')
        plt.legend()
    return pdf_files, res

def compute_reaction_time_stop_flow(processor):
    """
    Compute the reaction time for a stop-flow experiment based on the scanned motor positions.

    Parameters
    ----------
    processor : XRDProcessor
        The XRDProcessor object containing the scanned motor positions.

    Returns
    -------
    reaction_time : dict
        Dictionary containing the reaction time for each frame group.
        reaction_time[frame_group] = time_value
    """
    epoch_trig_array = processor.times.get('epoch_trig')
    if epoch_trig_array is None:
        raise ValueError("The 'epoch_trig' motor is not found in the processor's times.")
    else: 
        reaction_time = [0] * len(epoch_trig_array)
        for i, epoch_trig in enumerate(epoch_trig_array):
            # Assuming the first frame corresponds to time zero
            time_value = epoch_trig - epoch_trig_array[0]
            reaction_time[i] = time_value    
    return reaction_time


def monitor_pdf_peaks_stop_flow(file,
                      ref_file=None,
                      poni_file=None,
                      mask_file=None,
                      polarization = 0.99,
                      rmin = 0,
                      rmax = 50,
                      rstep = 0.01,
                      bgscale=1,
                      qmin=0.5,
                      qmax=24,
                      qmaxinst=None,
                      rpoly=0.9,                      
                      entry_sample=None,
                      entry_ref=None,
                      peak_positions=[2.3,2.9],
                      fit_window=0.4,
                      plot=True):

    """
    Monitor the evolution of specific PDF peaks over a series of images from a lost-flow experiment.

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
        Whether to plot the peak evolution for each frame.
    rmin : float
        Minimum r value for PDF extraction.
    rmax : float
        Maximum r value for PDF extraction.
    rstep : float
        Step size for the r values in the PDF.
    bgscale : float
        Scale factor for the background.
    qmin : float
        Minimum q value for PDF extraction.
    qmax : float
        Maximum q value for PDF extraction.
    qmaxinst : float
        Maximum instrumental q value for PDF extraction.
    rpoly : float
        Polynomial degree for the background fitting.
    entry_sample : str
        The entry in the sample HDF5 file to process. If None, the first entry will be used.
    entry_ref : str
        The entry in the reference HDF5 file to process. If None, the first entry will be used.
    peak_positions : list of float  
        List of approximate peak positions (in r) to monitor. 
    
    Returns
    -------
    peak_evolution : dict
        Dictionary containing the fitted peak parameters for each frame and each monitored peak.
        peak_evolution[frame_group][peak_position] = {'amplitude': ..., 'center': ..., 'fwhm': ..., 'eta': ..., 'bkg': ..., 'area': ...}
    """

    pdf_files, res = process_stop_flow(file,
                                       ref_file=ref_file,
                                       poni_file=poni_file,
                                       mask_file=mask_file,
                                       polarization=polarization,
                                       rmin=rmin,
                                       rmax=rmax,
                                       rstep=rstep,
                                       bgscale=bgscale,
                                       qmin=qmin,
                                       qmax=qmax,
                                       qmaxinst=qmaxinst,
                                       rpoly=rpoly,                                       
                                       entry_sample=entry_sample,
                                       entry_ref=entry_ref,
                                       plot=False)

    processor = XRDProcessor(file, frame=0, poni_file=poni_file,
                                        mask=mask_file, polarization_factor=polarization, entry=entry_sample)
    
    reaction_time = compute_reaction_time_stop_flow(processor) # ndarray of corresponding reaction time for each frame

    peak_evolution = {}
    i=0
    for frame_group, data in res.items():
        r = res[frame_group]['r']
        G = res[frame_group]['G']
        # reaction_time has the same length as the list of frame_groups, so we can use it to label the x-axis in the plot
        time_value = reaction_time[i]
        i+=1
        peak_evolution[time_value] = {}
        for peak_pos in peak_positions:
            print(f"Fitting peak at {peak_pos} Å for frame group {frame_group} (time={time_value})")
            try:
                popt, pcov, area = fit_peak_pseudovoigt(r, G, position=peak_pos, window=fit_window, plot=False)
                peak_evolution[time_value][peak_pos] = {
                    'amplitude': popt[0],
                    'center': popt[1],
                    'fwhm': popt[2],
                    'eta': popt[3],
                    'bkg': popt[4],
                    'area': area
                }
            except Exception as e:
                print(f"Peak fitting failed for frame group {frame_group} at peak position {peak_pos}: {e}")
                peak_evolution[time_value][peak_pos] = None

    if plot:
        fig, ax = plt.subplots(2,2, figsize=(12, 8),dpi=200)
        for peak_pos in peak_positions:
            times = []
            amplitudes = []
            area = []
            fwhm = []
            center = []
            for time_value in sorted(peak_evolution.keys()):
                if peak_evolution[time_value][peak_pos] is not None:
                    times.append(time_value)
                    amplitudes.append(peak_evolution[time_value][peak_pos]['amplitude'])
                    area.append(peak_evolution[time_value][peak_pos]['area'])
                    fwhm.append(peak_evolution[time_value][peak_pos]['fwhm'])
                    center.append(peak_evolution[time_value][peak_pos]['center'])

            ax[0,0].plot(times, amplitudes, marker='o', label=f'Peak at {peak_pos} Å')
            ax[0,1].plot(times, area, marker='o', label=f'Peak at {peak_pos} Å')
            ax[1,0].plot(times, fwhm, marker='o', label=f'Peak at {peak_pos} Å')
            ax[1,1].plot(times, center, marker='o', label=f'Peak at {peak_pos} Å')
        ax[0,0].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[0,0].set_ylabel('Peak Amplitude')
        ax[0,0].set_title('Evolution of PDF Peaks Over Time')
        ax[0,0].legend()
        ax[0,0].grid()
        ax[0,1].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[0,1].set_ylabel('Peak Area')
        ax[0,1].set_title('Evolution of PDF Peaks Over Time')
        ax[0,1].legend()
        ax[0,1].grid()
        ax[1,0].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[1,0].set_ylabel('Peak FWHM')
        ax[1,0].set_title('Evolution of PDF Peaks Over Time')
        ax[1,0].legend()
        ax[1,0].grid()
        ax[1,1].set_xlabel('Reaction Time (s)')  # Adjust the unit based on your experimental setup
        ax[1,1].set_ylabel('Peak Center')
        ax[1,1].set_title('Evolution of PDF Peaks Over Time')
        ax[1,1].legend()
        ax[1,1].grid()
        plt.show()

    return peak_evolution