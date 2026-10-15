from matplotlib import pyplot as plt
from xpdfsuite_ID11 import extract_xpdf,XRDProcessor,ID11Data
import os
import numpy as np
from scipy.spatial import cKDTree
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt


def extract_number_of_images(file,entry,poni_file):
    
    eiger = XRDProcessor(file, entry=entry, poni_file=poni_file)
    
    return eiger.nb_frames


def _resolve_entry_name(data, entry):
    """
    Convertit un identifiant d'entry (None, int ou str) vers la clé d'entry réelle
    utilisée par `ID11Data.entries`, selon la même convention que `XRDProcessor`
    (un entier `i` correspond à l'entry `'{i+1}.1'`).

    Renvoie None si `entry` est None (pas de restriction d'entry).
    """
    if entry is None:
        return None
    if isinstance(entry, (int, np.integer)):
        entry_name = f'{int(entry) + 1}.1'
    else:
        entry_name = entry
    if entry_name not in data.entries:
        raise ValueError(f"Entrée '{entry_name}' inconnue. Entrées disponibles : {data.entries}")
    return entry_name


def _get_motor_position_table(data, motor_names):
    """
    Construit, pour chaque frame de chaque entry d'un objet ID11Data, le vecteur de
    position des moteurs demandés (moteur scanné -> une valeur par frame, moteur fixe
    -> valeur de positionneur répétée sur toutes les frames de l'entry).

    Parameters
    ----------
    data : ID11Data
        Objet déjà ouvert sur le fichier HDF5 à analyser.
    motor_names : sequence of str
        Noms des moteurs à récupérer (ex: 'difftz', 'samty', 'samtz').

    Returns
    -------
    positions : ndarray, shape (N, len(motor_names))
        Position des moteurs pour chaque frame (toutes entries concaténées).
    frame_info : list of tuple (entry, frame_index_in_entry, global_index)
        Permet de retrouver, pour chaque ligne de `positions`, l'entry et l'indice
        de frame d'origine.
    """
    rows = []
    frame_info = []
    global_idx = 0
    for entry in data.entries:
        nb = data.nb_frames[entry]
        meas = data.positions[entry]['measurement']
        pos = data.positions[entry]['positioners']
        cols = []
        for name in motor_names:
            if name in meas:
                values = np.asarray(meas[name], dtype=float)
                if values.shape[0] != nb:
                    raise ValueError(
                        f"Le moteur '{name}' de l'entry {entry} a {values.shape[0]} points, "
                        f"alors que {nb} frames sont attendues.")
            elif name in pos:
                values = np.full(nb, float(pos[name]))
            else:
                raise KeyError(f"Moteur '{name}' introuvable (measurement/positioners) dans l'entry {entry}.")
            cols.append(values)
        rows.append(np.column_stack(cols))
        for i in range(nb):
            frame_info.append((entry, i, global_idx))
            global_idx += 1
    positions = np.vstack(rows)
    return positions, frame_info


def match_frames_by_motor_positions(file, ref_file, motor_names=('difftz', 'samty', 'samtz'), tol=0.05,
                                     entry_sample=None, entry_ref=None):
    """
    Apparie les frames du fichier échantillon `file` avec celles du fichier de référence
    `ref_file` en comparant les positions des moteurs `motor_names` lues dans les
    métadonnées HDF5 (measurement si le moteur est scanné, positioners s'il est fixe).

    Deux frames sont considérées comme correspondantes si l'écart sur CHAQUE moteur est
    inférieur ou égal à `tol`.

    Parameters
    ----------
    file : str
        Chemin du fichier HDF5 échantillon.
    ref_file : str
        Chemin du fichier HDF5 de référence.
    motor_names : sequence of str
        Noms des moteurs à comparer (défaut : difftz, samty, samtz).
    tol : float
        Tolérance (mm) sur chaque moteur pour considérer deux positions comme identiques
        (défaut 0.05 mm = 50 µm).
    entry_sample : str, optional
        Limiter la recherche à une seule entry du fichier échantillon. Par défaut, toutes
        les entries du fichier sont utilisées. Accepte aussi un entier (index 0-based,
        converti en entry '{i+1}.1', comme pour `XRDProcessor`).
    entry_ref : str, optional
        Limiter la recherche à une seule entry du fichier de référence (idem `entry_sample`).

    Returns
    -------
    matches : list of dict
        Une entrée par frame de `file` ayant trouvé une correspondance dans `ref_file`,
        avec les clés : 'sample_entry', 'sample_frame', 'sample_global_index',
        'ref_entry', 'ref_frame', 'ref_global_index', 'distance' (écart euclidien, mm).
    unmatched : list of dict
        Frames de `file` n'ayant trouvé aucune correspondance dans `ref_file` à la
        tolérance donnée (clés : 'sample_entry', 'sample_frame', 'sample_global_index').
    """
    sample_data = ID11Data(file)
    ref_data = ID11Data(ref_file)
    try:
        entry_sample_name = _resolve_entry_name(sample_data, entry_sample)
        entry_ref_name = _resolve_entry_name(ref_data, entry_ref)
        if entry_sample_name is not None:
            sample_data.entries = [entry_sample_name]
        if entry_ref_name is not None:
            ref_data.entries = [entry_ref_name]
        sample_positions, sample_info = _get_motor_position_table(sample_data, motor_names)
        ref_positions, ref_info = _get_motor_position_table(ref_data, motor_names)
    finally:
        sample_data.close()
        ref_data.close()

    tree = cKDTree(ref_positions)
    # borne large (sphère englobant le pavé de tolérance par axe), puis vérification stricte par axe
    max_dist = tol * np.sqrt(len(motor_names))
    distances, indices = tree.query(sample_positions, distance_upper_bound=max_dist)

    matches = []
    unmatched = []
    for (sample_entry, sample_frame, sample_global), dist, ref_idx in zip(sample_info, distances, indices):
        if np.isfinite(dist) and ref_idx < len(ref_info):
            ref_entry, ref_frame, ref_global = ref_info[ref_idx]
            delta = np.abs(sample_positions[sample_global] - ref_positions[ref_global])
            if np.all(delta <= tol):
                matches.append({
                    'sample_entry': sample_entry,
                    'sample_frame': sample_frame,
                    'sample_global_index': sample_global,
                    'ref_entry': ref_entry,
                    'ref_frame': ref_frame,
                    'ref_global_index': ref_global,
                    'distance': float(dist),
                })
                continue
        unmatched.append({
            'sample_entry': sample_entry,
            'sample_frame': sample_frame,
            'sample_global_index': sample_global,
        })

    return matches, unmatched





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
                      entry_ref=None,
                      motor_names=('difftz', 'samty', 'samtz'),
                      position_tol=0.05):
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
    motor_names : sequence of str
        Noms des moteurs utilisés pour apparier les frames de `file` avec celles de `ref_file`
        (défaut : difftz, samty, samtz). Ignoré si `ref_file` est None.
    position_tol : float
        Tolérance (mm) sur chaque moteur pour considérer deux positions comme identiques
        lors de l'appariement sample/ref (défaut 0.05 mm = 50 µm).

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

    nb_images = extract_number_of_images(file, entry_sample,poni_file)
    nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
    pdf_files = []
    if nb_images != nb_images_ref:
        raise ValueError("The number of images in the sample and reference files do not match.")

    # Appariement des frames de `file` avec celles de `ref_file` par position moteur
    # (difftz/samty/samtz), à la place de l'hypothèse "même indice de frame".
    ref_frame_map = {}
    if ref_file is not None:
        matches, unmatched = match_frames_by_motor_positions(
            file, ref_file, motor_names=motor_names, tol=position_tol,
            entry_sample=entry_sample, entry_ref=entry_ref)
        ref_frame_map = {m['sample_frame']: m['ref_frame'] for m in matches}
        if unmatched:
            missing = [u['sample_frame'] for u in unmatched]
            print(f"Warning: no reference frame found within position_tol={position_tol} mm for sample frames "
                  f"{missing}; falling back to the same frame index in ref_file for those.")

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
            frame_group_ref = [ref_frame_map.get(f, f) for f in frame_group]
            frame_arg_ref = frame_group_ref[0] if len(frame_group_ref) == 1 else frame_group_ref
            ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_file, polarization_factor=polarization, entry=entry_ref)
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
            plt.plot(r, G/G.max(), label=f'Frame {frame_str}')
    if plot:
        plt.xlabel('r (Å)')
        plt.ylabel('G(r)')
        plt.legend()
    return pdf_files, res

def convert_position_to_time(position, position0=[0,0], x_coude=31, z_retour=1, debit=8.0,
                   section=2.0, long_tri=0.0, section_tri=0.5*5**(1/2), tol=1e-1):
    """
    Convertit la position dans la puce en temps de réaction.
    Temps de réaction (s) au point `position`, depuis la fin du mélangeur.

    t= volume/debit, avec volume cumulé depuis la fin du mélangeur jusqu'à la position `position`.
    l'utilisation du volume plutot que de la surface permet de prendre en compte les variations de section du canal.

    position, position0 : (x, z) en mm (position0 = fin du mélangeur)
    x_coude  : x du palier vertical (mm)
    z_retour : offset en z entre les 2 paliers (mm)
    debit    : débit total en µL/s (défaut 8)
    section  : section du canal en mm² (défaut 2 x 2 mm)
    long_tri, section_tri : zone triangulaire au début (ignorée si long_tri=0, section triangle équilatéral de 2mm de côté par défaut)
    tol      : tolérance (mm) pour décider sur quel palier on se trouve
    """
    x, z = position
    x0, z0 = position0
    

    L1 = abs(x_coude - x0)      # 1er palier horizontal
    L2 = abs(z_retour)     # palier vertical

    if abs(z - (z0 + z_retour)) < tol:          # 3e palier : on revient en sens inverse
        d = L1 + L2 + abs(x - x_coude)
    elif abs(x - x_coude) < tol:         # 2e palier (vertical)
        d = L1 + abs(z - z0)
    else:                                # 1er palier
        d = abs(x - x0)

    # volume cumulé (1 µL = 1 mm³) : zone triangulaire puis zone carrée
    d_tri = min(d, long_tri)
    volume = d_tri * section_tri + (d - d_tri) * section
    #print(f"Position:{position}, time={volume/debit:.2f}s, volume={volume:.2f}mm³, debit={debit:.2f}µL/s")
    return volume / debit

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
                      plot=True,
                      position0=[0,0], x_coude=31, z_retour=1, debit=8.0,section=2.0, long_tri=0.0, section_tri=0.5*5**(1/2), tol=1e-1):

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
                                       plot=False,
                                       )

    scanned_motors = res[list(res.keys())[0]]['position']  # Assuming all frame groups have the same scanned motors
    print(f"Scanned motors: {scanned_motors}")
    # il va falloir relier les positions (x,z) au temps de réaction, d'une façon ou d'une autre...
    # convert_position_to_time() est une fonction qui fait ça, mais il faut lui passer les positions (x,z) et les paramètres du canal microfluidique.
    reaction_time = {}
    for frame_group, data in res.items():
        scanned_motors = res[frame_group]['position'] 
        for key, value in scanned_motors.items():
            if 'y' in key.lower():
                y_motor = value
            else:
                y_motor = position0[1]
            if 'z' in key.lower():
                z_motor = value
            else:
                z_motor = position0[1] 
        position = [y_motor, z_motor]

        reaction_time[frame_group] = convert_position_to_time(position, position0=position0, x_coude=x_coude, z_retour=z_retour, debit=debit, section=section, long_tri=long_tri, section_tri=section_tri, tol=tol)  # Implement this function based on your experimental setup

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
                      entry_ref=None,
                      motor_names=('difftz', 'samty', 'samtz'),
                      position_tol=0.05
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
    motor_names : sequence of str
        Noms des moteurs utilisés pour apparier les frames de `file` avec celles de `ref_file`
        (défaut : difftz, samty, samtz). Ignoré si `ref_file` est None.
    position_tol : float
        Tolérance (mm) sur chaque moteur pour considérer deux positions comme identiques
        lors de l'appariement sample/ref (défaut 0.05 mm = 50 µm).
    
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
    
 
    nb_images = extract_number_of_images(file, entry_sample,poni_file)
    nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
    pdf_files = []
    if nb_images != nb_images_ref:
        raise ValueError("The number of images in the sample and reference files do not match.")

    # Appariement des frames de `file` avec celles de `ref_file` par position moteur
    # (difftz/samty/samtz), à la place de l'hypothèse "même indice de frame".
    ref_frame_map = {}
    if ref_file is not None:
        matches, unmatched = match_frames_by_motor_positions(
            file, ref_file, motor_names=motor_names, tol=position_tol,
            entry_sample=entry_sample, entry_ref=entry_ref)
        ref_frame_map = {m['sample_frame']: m['ref_frame'] for m in matches}
        if unmatched:
            missing = [u['sample_frame'] for u in unmatched]
            print(f"Warning: no reference frame found within position_tol={position_tol} mm for sample frames "
                  f"{missing}; falling back to the same frame index in ref_file for those.")

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
            frame_arg_ref = ref_frame_map.get(bin_idx, bin_idx)
            ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_file, polarization_factor=polarization, entry=entry_ref)
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