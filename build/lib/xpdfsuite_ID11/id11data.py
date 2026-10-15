import h5py
import os
import hdf5plugin  # nécessaire pour décompresser les images eiger (filtre bitshuffle/lz4)
import numpy as np
import re

os.environ['HDF5_PLUGIN_PATH'] = hdf5plugin.PLUGIN_PATH


# Moteurs candidats à rechercher (fixes -> positioners, mobiles -> measurement).
# A adapter selon les moteurs réellement scannés (ex: samtx/samtz pour un scan en x/z).
MOTOR_NAMES = ['samtx', 'samty', 'samtz', 'difftz', 'diffty', 'diffrz']

# Champs NXdetector usuels (convention NeXus/bliss) pouvant être présents
# directement sous 'instrument/eiger'. Les champs absents restent à None.
DETECTOR_INFO_KEYS = [
    'x_pixel_size', 'y_pixel_size', 'pixel_size',
    'distance', 'beam_center_x', 'beam_center_y',
    'description', 'type', 'serial_number',
    'saturation_value', 'bit_depth_image', 'bit_depth_readout',
    'threshold_energy', 'countrate_correction_count_cutoff',
    'sensor_material', 'sensor_thickness', 'layout', 'wavelength', 'image_width', 'image_height'
]


class ID11Data:
    """Lecteur générique de fichiers ID11 (bliss/NeXus), avec support de 1 à N entries.

    Que le fichier ne contienne qu'un seul scan ou plusieurs, l'API est la même :
    tous les attributs de métadonnées sont des dictionnaires indexés par entry
    (ex: 'eiger', 'frelon3'), ex : ``self.data['2.1']``, ``self.nb_frames['2.1']``.

    Pour compatibilité et confort quand il n'y a qu'une entry (ou qu'on veut une
    entry "par défaut"), ``self.entry`` pointe vers la première entry trouvée.
    """

    DETECTORS = ('eiger', 'frelon3')

    def __init__(self, file_path, verbose=False):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')

        # --- entrées et datasets (lazy, h5py.Dataset, pas encore chargés en mémoire) ---
        self.detector_names = {}   # {entry: 'eiger' | 'frelon3'}
        self.data = {}             # {entry: h5py.Dataset}
        self.entries = self.list_entries()
        if not self.entries:
            raise KeyError("Aucune entrée avec 'measurement/eiger' ou 'measurement/frelon3'.")
        for entry in self.entries:
            name, ds = self._find_detector(entry)
            self.detector_names[entry] = name
            self.data[entry] = ds

        self.nb_frames = {e: ds.shape[0] for e, ds in self.data.items()}
        self.total_frames = sum(self.nb_frames.values())
        # offsets[i] = indice global de la 1re frame de l'entrée i (utile pour get_frame global)
        self._offsets = np.cumsum([0] + [self.nb_frames[e] for e in self.entries])

        # --- métadonnées par entrée ---
        self.eiger_source = {e: self.get_eiger_source(e) for e in self.entries}
        self.positions = {e: self.get_motor_positions(e) for e in self.entries}
        self.scanned_motors = {e: p['measurement'] for e, p in self.positions.items()}
        self.times = {e: self.get_time_metadata(e) for e in self.entries}
        self.epoch = {e: t.get('epoch_trig') for e, t in self.times.items()}
        self.samplename = {e: t.get('sample_name') for e, t in self.times.items()}
        self.command = {e: t.get('title') for e, t in self.times.items()}
        self.detector_info = {e: self.get_detector_info(e) for e in self.entries}

        # entry "par défaut" : la première trouvée (comportement rétro-compatible
        # pour du code qui suppose une seule entry, ex: self.entry, self.epoch[self.entry]...)
        self.entry = self.entries[0]

        if verbose:
            self._check_scan_consistency()

    # ------------------------------------------------------------------
    def _find_detector(self, entry):
        for name in self.DETECTORS:
            path = f'{entry}/measurement/{name}'
            if path in self.h5_file:
                return name, self.h5_file[path]
        raise KeyError(f"Neither 'measurement/eiger' nor 'measurement/frelon3' found in entry {entry}.")

    def list_entries(self):
        """Liste les entrées (scans) contenant un détecteur, triées numériquement
        ('2.1' avant '10.1')."""
        def key(k):
            return tuple(int(x) for x in re.findall(r'\d+', k))
        entries = []
        for k in sorted(self.h5_file.keys(), key=key):
            if any(f'{k}/measurement/{d}' in self.h5_file for d in self.DETECTORS):
                entries.append(k)
        return entries

    # ------------------------------------------------------------------
    def get_frame(self, index, entry=None):
        """Si entry est donnée : frame `index` de cette entrée.
        Sinon : `index` est un indice global sur toutes les entrées concaténées."""
        if entry is not None:
            return self.data[entry][index]
        if not 0 <= index < self.total_frames:
            raise IndexError(index)
        i = np.searchsorted(self._offsets, index, side='right') - 1
        return self.data[self.entries[i]][index - self._offsets[i]]

    def iter_frames(self):
        for entry in self.entries:
            for i in range(self.nb_frames[entry]):
                yield entry, i, self.data[entry][i]

    # ------------------------------------------------------------------
    def get_eiger_source(self, entry):
        """Résout le softlink/Virtual Dataset des images eiger d'une entrée."""
        ds = self.data[entry]
        info = {
            'vds_path': ds.name,
            'shape': ds.shape,
            'dtype': ds.dtype,
            'real_file': None,
            'real_dataset': None,
        }
        if ds.is_virtual:
            vs = ds.virtual_sources()[0]
            info['real_file'] = vs.file_name
            info['real_dataset'] = vs.dset_name
        else:
            info['real_file'] = ds.file.filename
            info['real_dataset'] = ds.name
        return info

    def get_motor_positions(self, entry, motor_names=MOTOR_NAMES):
        """Renvoie la position de chaque moteur, catégorisée par source.

        - 'measurement' : moteurs qui bougent pendant le scan -> tableau (1 valeur/point)
        - 'positioners' : moteurs fixes pendant le scan -> valeur scalaire
        """
        positions = {'measurement': {}, 'positioners': {}}
        meas = self.h5_file.get(f'{entry}/measurement')
        pos = self.h5_file.get(f'{entry}/instrument/positioners')
        for name in motor_names:
            if meas is not None and name in meas:
                positions['measurement'][name] = meas[name][()]
            elif pos is not None and name in pos:
                positions['positioners'][name] = pos[name][()]
        return positions

    def get_time_metadata(self, entry):
        """Récupère les métadonnées temporelles (scan + par frame) d'une entrée."""
        g = self.h5_file[entry]
        inst = g['instrument']

        def _decode(ds):
            val = ds[()]
            return val.decode() if isinstance(val, bytes) else val

        meta = {
            'start_time': _decode(g['start_time']),
            'end_time': _decode(g['end_time']),
            'end_reason': _decode(g['end_reason']) if 'end_reason' in g else None,
            'title': _decode(g['title']) if 'title' in g else None,
            'sample_name': _decode(g['sample/name']) if 'sample' in g and 'name' in g['sample'] else None,
        }

        for key in ['epoch_trig', 'sec', 'timer_raw', 'timer_delta', 'timer_period', 'timer_trig']:
            if key in inst and 'data' in inst[key]:
                meta[key] = inst[key]['data'][()]

        acq = inst.get('eiger/acq_parameters')
        if acq is not None:
            meta['acq_expo_time'] = acq['acq_expo_time'][()]
            meta['latency_time'] = acq['latency_time'][()]
            meta['acq_nb_frames'] = acq['acq_nb_frames'][()]

        return meta

    def get_detector_info(self, entry):
        """Récupère les métadonnées géométriques/techniques du détecteur eiger
        pour une entrée donnée. Les champs absents restent à None."""
        det = self.h5_file.get(f'{entry}/instrument/eiger')
        info = {key: None for key in DETECTOR_INFO_KEYS}
        if det is None:
            return info

        def _decode(val):
            if isinstance(val, bytes):
                return val.decode()
            if isinstance(val, np.ndarray) and val.dtype.kind == 'S':
                return val.astype(str)
            return val

        for key in DETECTOR_INFO_KEYS:
            if key in det:
                info[key] = _decode(det[key][()])
        data = self.data[entry]
        info['image_width'] = data.shape[2] if data.ndim == 3 else data.shape[1]
        info['image_height'] = data.shape[1] if data.ndim == 3 else data.shape[0]
        return info

    def list_detector_keys(self, entry=None):
        """Liste tous les champs disponibles sous 'instrument/eiger' (1re entry par défaut)."""
        entry = entry or self.entries[0]
        det = self.h5_file.get(f'{entry}/instrument/eiger')
        if det is None:
            return []
        keys = []
        det.visit(keys.append)
        return keys

    # ------------------------------------------------------------------
    def _check_scan_consistency(self):
        for entry in self.entries:
            nb = self.nb_frames[entry]
            motors = self.scanned_motors[entry]
            print(f"[{entry}] {nb} frames")
            for motor, value in motors.items():
                if value.shape[0] == nb:
                    print(f"  {motor}: {value.shape[0]} points = nb de frames")
            if len(motors) > 1:
                print(f"  Moteurs scannés : {', '.join(motors)}")
                product = int(np.prod([v.shape[0] for v in motors.values()]))
                if product == nb:
                    print(f"  Le produit des points ({product}) correspond au nb de frames.")

    def close(self):
        """Ferme le fichier HDF5."""
        self.h5_file.close()