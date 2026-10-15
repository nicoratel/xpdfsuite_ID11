import re
import os
import datetime
import numpy as np
import h5py

# Please change  _MOTOR_KEY_ALIASES to match as desired for your specific beamline or detector. The current aliases are generic and may not match your data files.
_MOTOR_KEY_ALIASES = {
    'motor_x': ('samx', 'sample_x', 'pos_x', 'motor_x','diffrx'),
    'motor_y': ('samy', 'sample_y', 'pos_y', 'motor_y','diffry'),
    'motor_z': ('samz', 'sample_z', 'pos_z', 'motor_z','diffrz'),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EMPTY_DETECTOR_INFO = {
    'camera_type': None,
    'camera_title': None,
    'pixel_size': None,
    'image_width': None,
    'image_height': None,
    'binning': 1,
    'description': None,
    'wavelength': None,
    'nb_frames': None,
    'exposure_time': None,
    'latency_time': None,
    'start_time': None,
    'start_time_str': None,
    'end_time': None,
    'end_time_str': None,
    'motor_x': None,
    'motor_y': None,
    'motor_z': None,
}


def _parse_h5_timestamp(raw_value):
    """
    Parse a raw h5py scalar that represents a timestamp.

    Accepts either a numeric value (already an epoch, in seconds) or a
    bytes/str ISO-8601 date-time string (e.g. ``b'2024-01-15T14:32:07.123'``).

    Returns
    -------
    epoch : float or None
        Unix epoch in seconds, or ``None`` if it could not be determined.
    iso_str : str or None
        The original string representation, or ``None`` if the raw value was
        purely numeric.
    """
    val = raw_value
    if isinstance(val, np.ndarray):
        val = val.item() if val.size == 1 else None
    if isinstance(val, bytes):
        val = val.decode('utf-8', errors='ignore')
    if isinstance(val, str):
        val = val.strip()
        try:
            return float(val), None
        except ValueError:
            pass
        try:
            dt = datetime.datetime.fromisoformat(val.replace('Z', '+00:00'))
            return dt.timestamp(), val
        except ValueError:
            return None, val
    try:
        return float(val), None
    except (TypeError, ValueError):
        return None, None


def _finalize_image(raw_image, detector_info, normalize, verbose, filepath):
    """Shared post-processing: shape check, metadata fill, optional normalisation."""
    if raw_image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {raw_image.shape} from {filepath}")
    detector_info['image_height'], detector_info['image_width'] = raw_image.shape
    if verbose:
        print(f"Loaded: {filepath}")
        for k, v in detector_info.items():
            if v is not None:
                print(f"  {k}: {v}")
    if normalize and raw_image.max() > 0:
        raw_image = raw_image / raw_image.max()
    return detector_info, raw_image


def _load_edf_image(file, normalize=False, verbose=False):
    """
    Load a FreLON (or any EDF-formatted) image using *fabio*.

    Attempts to read common FreLON/ESRF header keys if present:

    * Pixel size: ``PixelSize``, ``PSize_1`` (µm or mm), ``pixel_size`` (m)
    * Wavelength: ``WaveLength``, ``wavelength`` (m or Å auto-detected)
    * Frame count: ``nframes`` (common in averaged EDF files)
    * Title/instrument: ``title``, ``Instrument``, ``detector``

    Keys absent from the header are silently ignored (field stays ``None``).

    Parameters
    ----------
    file : str
        Path to the ``.edf`` (or ``.edf.gz``) file.

    Returns
    -------
    detector_info : dict
        Same keys as :func:`load_h5_data`.
    raw_image : ndarray
        2-D float array.
    """
    import fabio
    obj = fabio.open(file)
    raw_image = obj.data.astype(float)
    h = obj.header  # dict-like, values are strings

    info = dict(_EMPTY_DETECTOR_INFO)
    info['description'] = 'EDF (FreLON / ESRF)'
    info['camera_type'] = 'FreLON'
    info['nb_frames'] = raw_image.shape[0] if raw_image.ndim == 3 else 1

    # Camera title / instrument name
    for key in ('title', 'Title', 'instrument', 'Instrument', 'detector', 'Detector'):
        if key in h and str(h[key]).strip():
            info['camera_title'] = str(h[key]).strip()
            break

    # Number of frames (present when the EDF is a mean/sum of a stack)
    for key in ('nframes', 'NFrames', 'nb_frames'):
        if key in h:
            try:
                info['nb_frames'] = int(h[key])
                break
            except (ValueError, TypeError):
                pass

    # Pixel size — try several key names and unit conventions
    # µm: PixelSize, pixelsize  |  mm: PSize_1, psize_1  |  m: pixel_size

    for key in ('PixelSize','pixelsize','pixel_size','Pixel_size','PSize_1','psize_1'):
        if key in h:
            try:
                info['pixel_size'] = float(h[key])
                break
            except (ValueError, TypeError):
                pass
    # Wavelength — may be stored in Å or m depending on beamline software
    for key in ('WaveLength', 'wavelength', 'Wavelength', 'wave_length'):
        if key in h:
            try:
                wl = float(h[key])
                # Convert from m to Å if value looks like it is in metres (< 1e-6)
                if wl < 1e-6:
                    wl *= 1e10
                info['wavelength'] = wl
                break
            except (ValueError, TypeError):
                pass

    return _finalize_image(raw_image, info, normalize, verbose, file)


def _load_h5_image(file, frame='mean', normalize=False, verbose=False):
    """
    Load an image from an HDF5/NXS file.

    Tries in order:
    1. EigerII / NeXus structure (``/entry/data/data``, pixel size under
       ``/entry/instrument/detector/``).
    2. ID15 / Pilatus structure (``<group>/measurement/data``).
    3. Generic fallback: largest 2-D or 3-D numeric dataset.

    Parameters
    ----------
    file : str
        Path to the ``.h5``, ``.hdf5``, or ``.nxs`` file.
    frame : int, 'mean', or list/tuple/ndarray of int, optional
        Frame to return when the dataset is 3-D. If a list/tuple/ndarray of
        indices is given, those frames are averaged together (useful for
        frame binning).

    Returns
    -------
    detector_info : dict
        Same keys as :func:`load_h5_data`.
    raw_image : ndarray
        2-D float array.
    """
    info = dict(_EMPTY_DETECTOR_INFO)
    info['description'] = 'HDF5/NXS file'
    data = None

    with h5py.File(file, 'r') as f:

        # ---- 1. EigerII / NeXus structure ----
        try:
            # Data may be split (data_000001, data_000002, …) or a single dataset
            entry = f.get('entry') or f.get(list(f.keys())[0])
            data_grp = entry['data']
            # Collect all numeric datasets in the data group
            frames = []
            for k in sorted(data_grp.keys()):
                ds = data_grp[k]
                if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.number):
                    frames.append(np.array(ds))
            if frames:
                data = np.concatenate(frames, axis=0) if frames[0].ndim == 3 else np.stack(frames)
                info['description'] = 'EigerII / NeXus HDF5'
                info['camera_type'] = 'EigerII'

                det = entry.get('instrument', {}).get('detector', {})
                for px_key in ('x_pixel_size', 'pixel_size_x', 'x_pixel_length','x_size'):
                    if px_key in det:
                        info['pixel_size'] = float(det[px_key][()])
                        break

                beam = entry.get('instrument', {}).get('beam', {})
                for wl_key in ('incident_wavelength', 'wavelength'):
                    if wl_key in beam:
                        wl = float(beam[wl_key][()])
                        if wl < 1e-6:
                            wl *= 1e10   # m → Å
                        info['wavelength'] = wl
                        break

                if 'description' in det:
                    try:
                        info['camera_title'] = str(det['description'][()], 'utf-8').strip()
                    except Exception:
                        info['camera_title'] = str(det['description'][()])

                if verbose:
                    print(f"  ✓ EigerII/NeXus structure detected in '{file}'")
        except Exception:
            data = None

        # ---- 2. ID15 / Pilatus structure ----
        if data is None:
            try:
                group = list(f.keys())[0]
                info['camera_title'] = str(group)
                data = np.array(f[group + '/measurement/data'])

                try:
                    ps_path = group + '/instrument/pilatus/detector_information/pixel_size/'
                    info['pixel_size'] = float(f[ps_path + 'xsize'][()])
                except Exception:
                    pass
                try:
                    info['nb_frames'] = int(
                        f[group + '/instrument/pilatus/acquisition/nb_frames'][()]
                    )
                except Exception:
                    info['nb_frames'] = data.shape[0] if data.ndim == 3 else 1

                if verbose:
                    print(f"  ✓ ID15/Pilatus structure detected in '{file}'")
            except Exception:
                data = None

        # ---- 3. Generic fallback ----
        if data is None:
            best_key, best_size = None, 0

            def _find(name, obj):
                nonlocal best_key, best_size
                if isinstance(obj, h5py.Dataset):
                    if obj.ndim in (2, 3) and np.issubdtype(obj.dtype, np.number):
                        if obj.size > best_size:
                            best_size, best_key = obj.size, name

            f.visititems(_find)
            if best_key is None:
                raise ValueError(f"No suitable image dataset found in {file}")
            data = np.array(f[best_key])
            if verbose:
                print(f"  ✓ Generic HDF5: using dataset '{best_key}' (shape {data.shape})")

        # ---- Fallback générique : recherche de pixel_size n'importe où dans le fichier ----
        if info.get('pixel_size') is None:
            candidates = []

            def _find_pixel_size(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    parent = parts[-2] if len(parts) > 1 else ''
                    # on exige que le parent contienne "pixel" (pixel_size, pixelsize, ...)
                    if 'pixel' in parent and leaf in (
                        'xsize', 'x_pixel_size', 'pixel_size_x', 'x_pixel_length', 'x_size'
                    ):
                        candidates.append(name)

            f.visititems(_find_pixel_size)
            if candidates:
                path = sorted(candidates, key=len)[0]
                try:
                    val = float(f[path][()])
                    info['pixel_size'] = val
                    if verbose:
                        print(f"  ✓ Pixel size trouvée via recherche générique : '{path}' = {val}")
                except Exception:
                    pass
        # ---- Fallback générique : wavelength ou energy n'importe où dans le fichier ----
        if info.get('wavelength') is None:
            wl_candidates = []
            en_candidates = []

            def _find_wl_en(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    if leaf in ('wavelength', 'incident_wavelength', 'lambda'):
                        wl_candidates.append(name)
                    elif leaf in ('energy', 'incident_energy', 'photon_energy', 'monochromator_energy'):
                        en_candidates.append(name)

            f.visititems(_find_wl_en)

            # 1) wavelength directe
            if wl_candidates:
                path = sorted(wl_candidates, key=len)[0]
                try:
                    wl = float(f[path][()])
                    if wl < 1e-6:      # m -> Å
                        wl *= 1e10
                    elif wl > 1e3:     # pm -> Å (peu probable mais prudence)
                        wl /= 100
                    info['wavelength'] = wl
                    if verbose:
                        print(f"  ✓ Wavelength trouvée via recherche générique : '{path}' = {wl} Å")
                except Exception:
                    pass

            # 2) sinon, energie -> conversion en longueur d'onde
            if info.get('wavelength') is None and en_candidates:
                path = sorted(en_candidates, key=len)[0]
                try:
                    energy = float(f[path][()])
                    # heuristique unité : eV si grand nombre, keV si petit
                    if energy > 1000:
                        energy /= 1000.0   # eV -> keV
                    wl = 12.398 / energy   # Å
                    info['wavelength'] = wl
                    if verbose:
                        print(f"  ✓ Wavelength dérivée de l'énergie : '{path}' = {energy} keV -> {wl:.5f} Å")
                except Exception:
                    pass
        # --- Generci case exposure time ---
        if info.get('exposure_time') is None:
            exp_candidates = []

            def _find_exposure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    if leaf in ('exposure_time', 'exposure', 'acquisition_time'):
                        exp_candidates.append(name)

            f.visititems(_find_exposure)
            if exp_candidates:
                path = sorted(exp_candidates, key=len)[0]
                try:
                    exp_time = float(f[path][()])
                    info['exposure_time'] = exp_time
                    if verbose:
                        print(f"  ✓ Exposure time trouvée via recherche générique : '{path}' = {exp_time} s")
                except Exception:
                    pass

        # Generic case : latency time
        if info.get('latency_time') is None:
            lat_candidates = []

            def _find_latency(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    if leaf in ('latency_time', 'latency', 'dead_time'):
                        lat_candidates.append(name)

            f.visititems(_find_latency)
            if lat_candidates:
                path = sorted(lat_candidates, key=len)[0]
                try:
                    latency_time = float(f[path][()])
                    info['latency_time'] = latency_time
                    if verbose:
                        print(f"  ✓ Latency time trouvée via recherche générique : '{path}' = {latency_time} s")
                except Exception:
                    pass    
        # generic case : start time
        if info.get('start_time') is None:
            start_candidates = []

            def _find_start_time(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    if leaf in ('start_time', 'start', 'timestamp'):
                        start_candidates.append(name)

            f.visititems(_find_start_time)
            if start_candidates:
                path = sorted(start_candidates, key=len)[0]
                epoch, iso_str = _parse_h5_timestamp(f[path][()])
                if epoch is not None:
                    info['start_time'] = epoch
                if iso_str is not None:
                    info['start_time_str'] = iso_str
                if verbose:
                    print(f"  ✓ Start time trouvée via recherche générique : '{path}' = {info.get('start_time')} ({info.get('start_time_str')})")

        # generic case : end time
        if info.get('end_time') is None:
            end_candidates = []

            def _find_end_time(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.lower().split('/')
                    leaf = parts[-1]
                    if leaf in ('end_time', 'end', 'timestamp'):
                        end_candidates.append(name)

            f.visititems(_find_end_time)
            if end_candidates:
                path = sorted(end_candidates, key=len)[0]
                epoch, iso_str = _parse_h5_timestamp(f[path][()])
                if epoch is not None:
                    info['end_time'] = epoch
                if iso_str is not None:
                    info['end_time_str'] = iso_str
                if verbose:
                    print(f"  ✓ End time trouvée via recherche générique : '{path}' = {info.get('end_time')} ({info.get('end_time_str')})")

        # generic case : motor positions (X, Y, Z)
       # _MOTOR_KEY_ALIASES = {
       #     'motor_x': ('samx', 'sample_x', 'pos_x', 'motor_x','diffrx'), # add alias for mator x if necessary
       #     'motor_y': ('samy', 'sample_y', 'pos_y', 'motor_y','diffry'),
       #     'motor_z': ('samz', 'sample_z', 'pos_z', 'motor_z','diffrz'),
    #}
        for info_key, leaf_names in _MOTOR_KEY_ALIASES.items():
            if info.get(info_key) is None:
                motor_candidates = []

                def _find_motor(name, obj, leaf_names=leaf_names):
                    if isinstance(obj, h5py.Dataset):
                        parts = name.lower().split('/')
                        leaf = parts[-1]
                        if leaf in leaf_names and obj.size == 1:
                            motor_candidates.append(name)

                f.visititems(_find_motor)
                if motor_candidates:
                    path = sorted(motor_candidates, key=len)[0]
                    try:
                        value = float(f[path][()])
                        info[info_key] = value
                        if verbose:
                            print(f"  ✓ Position moteur trouvée via recherche générique : '{path}' = {value}")
                    except Exception:
                        pass

    # ---- Collapse 3-D stack to 2-D ----
    if data.ndim == 3:
        nb = data.shape[0]
        if info['nb_frames'] is None:
            info['nb_frames'] = nb
        if frame == 'mean':
            raw_image = np.mean(data, axis=0)
        elif isinstance(frame, (list, tuple, np.ndarray)):
            # Average an arbitrary subset of frames (e.g. for frame binning)
            indices = [i for i in frame if 0 <= i < nb]
            if not indices:
                raise ValueError(f"No valid frame indices in {frame} for stack of {nb} frames ({file})")
            raw_image = np.mean(data[indices], axis=0)
        elif isinstance(frame, int) and 0 <= frame < nb:
            raw_image = data[frame]
        else:
            raw_image = np.mean(data, axis=0)
        raw_image = raw_image.astype(float)
        print(f"  ✓ Collapsed 3-D stack to 2-D image (frame={frame})")
    elif data.ndim == 2:
        raw_image = data.astype(float)
        info['nb_frames'] = 1
    else:
        raise ValueError(f"Unexpected dataset shape {data.shape} in {file}")

    return _finalize_image(raw_image, info, normalize, verbose, file)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Extensions routed to the HDF5 loader.
_H5_EXTS = {'.h5', '.hdf5', '.nxs'}
#: Extensions routed to the EDF (fabio) loader.
_EDF_EXTS = {'.edf'}


def load_image(file, frame='mean', normalize=False, verbose=True):
    """
    Unified image loader for HDF5/NeXus (EigerII) and EDF (FreLON) files.

    Dispatches to the appropriate reader based on the file extension:

    * ``.h5 / .hdf5 / .nxs`` → HDF5 reader (EigerII NeXus, then ID15/Pilatus,
      then generic fallback).
    * ``.edf / .edf.gz`` → EDF reader via *fabio* (FreLON and any EDF-formatted
      detector).
    * ``.tif / .tiff`` → fabio fallback (treated like EDF with minimal metadata).

    All paths return the **same** ``(detector_info, raw_image)`` tuple so the
    function is a drop-in replacement for :func:`load_h5_data`.

    Parameters
    ----------
    file : str
        Path to the diffraction image file.
    frame : int, 'mean', or list/tuple/ndarray of int, optional
        Relevant for HDF5 stacks only: frame index, ``'mean'`` (default) to
        average all frames, or a list/tuple/ndarray of indices to average
        that subset of frames (useful for frame binning).
    normalize : bool, optional
        If ``True``, divide the image by its maximum value. Default is ``False``.
    verbose : bool, optional
        Print a summary of detected metadata. Default is ``True``.

    Returns
    -------
    detector_info : dict
        Keys: ``camera_type``, ``camera_title``, ``pixel_size`` (m),
        ``image_width`` (px), ``image_height`` (px), ``binning``,
        ``description``, ``wavelength``, ``nb_frames``.
        Fields that cannot be extracted from the file are ``None``.
    raw_image : ndarray
        2-D float array (height × width).

    Raises
    ------
    ValueError
        If no suitable image dataset is found.
    """
    lower = file.lower()

    # Handle double extension .edf.gz
    if lower.endswith('.edf.gz') or lower.endswith('.edf'):
        return _load_edf_image(file, normalize=normalize, verbose=verbose)

    ext = os.path.splitext(lower)[1]
    if ext in _H5_EXTS:
        return _load_h5_image(file, frame=frame, normalize=normalize, verbose=verbose)

    # TIFF and unknown formats: try fabio (behaves like EDF loader)
    return _load_edf_image(file, normalize=normalize, verbose=verbose)


   
def load_h5_data(file, frame='mean', normalize=False, verbose=True):
    """
    Load image data from an HDF5/NXS file using h5py.

    .. deprecated::
        Use :func:`load_image` instead, which also supports EDF (FreLON)
        and auto-detects EigerII NeXus files.

    Tries the ID15/ESRF Pilatus structure first:

    * ``<group>/measurement/data``  → image stack (frames × height × width)
    * ``<group>/instrument/pilatus/acquisition/nb_frames``
    * ``<group>/instrument/pilatus/detector_information/pixel_size/{xsize,ysize}``

    Falls back to a generic search: the largest 2-D or 3-D numeric dataset in
    the file is used as the image stack.

    Parameters
    ----------
    file : str
        Path to the ``.h5``, ``.hdf5``, or ``.nxs`` file.
    frame : int or 'mean', optional
        Frame index to return, or ``'mean'`` (default) to average all frames.
    verbose : bool, optional
        Print a summary of detected metadata when ``True``.

    Returns
    -------
    detector_info : dict
        Keys: ``camera_type``, ``camera_title``, ``pixel_size`` (m),
        ``image_width`` (px), ``image_height`` (px), ``binning``,
        ``description``, ``wavelength``, ``nb_frames``.
    raw_image : ndarray
        2-D float array (the selected or averaged frame).

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If no suitable image dataset is found in the file.
    """
    return _load_h5_image(file, frame=frame, normalize=normalize, verbose=verbose)


def average_h5files_2_edf(h5filelist, output_edf=None,cps=False):
    """
    Average a list of HDF5 files and save the result as an EDF file.

    Parameters
    ----------
    h5filelist : list of str
        List of paths to HDF5 files to average.
    output_edf : str, optional
        Path to the output EDF file. If not provided, a default name will be used.


    Returns
    -------
    edffile : str
        Path to the output EDF file containing the averaged image.
    """
    if not h5filelist:
        raise ValueError("No HDF5 files provided for averaging.")

    # Load the first file to get the shape and metadata
    detector_info, raw_image = load_h5_data(h5filelist[0], frame='mean', verbose=False)
    sum_image = np.zeros_like(raw_image)

    # Accumulate images from all files
    total_frames = 0
    for h5file in h5filelist:
        detector_info, img = load_h5_data(h5file, frame='mean', verbose=False)
        total_frames += detector_info.get('nb_frames', 1)
        if cps:
            exposure_time = detector_info.get('exposure_time', 1.0)
            sum_image += img / exposure_time  # Sum weighted by exposure time (in cps!)
        else:
            sum_image += img  # Simple sum without exposure time weighting
       
       

    # Compute the average
    avg_image = sum_image  / len(h5filelist) # compute average
    detector_info['nframes'] = total_frames
    detector_info['method'] = 'mean'
    # Define output EDF file name
    if output_edf is None:
        edffile = os.path.splitext(h5filelist[0])[0] + '_average.edf'
    else:
        edffile = output_edf
    

    # Create EDF image and save
    import fabio
    edf_image = fabio.edfimage.EdfImage(data=avg_image, header=detector_info)
    edf_image.write(edffile)

    return edffile


# ID11: we are given h5 files containing the diffraction images, we need to average them and save as EDF for pyFAI-calib2.
def average_singleh5_2_edf(h5file, output_edf=None):
    detector_info, raw_image = load_h5_data(h5file, frame='mean', verbose=False)
    # Define output EDF file name
    if output_edf is None:
        edffile = os.path.splitext(h5file)[0] + '_average.edf'
    else:
        edffile = output_edf    
    # Create EDF image and save
    import fabio
    edf_image = fabio.edfimage.EdfImage(data=raw_image, header=detector_info)
    edf_image.write(edffile)

    return edffile

