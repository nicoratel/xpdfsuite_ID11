import h5py
import os
import hdf5plugin  # nécessaire pour décompresser les images eiger (filtre bitshuffle/lz4)
import numpy as np
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
    'sensor_material', 'sensor_thickness', 'layout','wavelength','image_width','image_height'
]

class EigerData:
    def __init__(self, file_path,verbose=False):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.entry = self.list_entries()[0]  # on prend la première entrée pour l'exemple
        self.eiger_source = self.get_eiger_source()

        try:
            self.data = self.h5_file[f'{self.entry}/measurement/eiger']
        except KeyError:
            try:
                self.data = self.h5_file[f'{self.entry}/measurement/frelon3']
            except KeyError:
                raise KeyError(f"Neither 'measurement/eiger' nor 'measurement/frelon3' found in entry {self.entry}.")
    
        self.nb_frames = self.data.shape[0]

        self.positions = self.get_motor_positions()
        self.scanned_motors = self.positions['measurement'] # {name: value}

        # vérification que le nombre de poinst scannés sur au moins 1 moteur correspond au nombre de frames eiger
        if verbose:
            for motor, value in self.positions['measurement'].items():
                if value.shape[0] == self.nb_frames:
                    print(f"Le nombre de points scannés sur le moteur {motor} correspond au nombre de frames eiger.")
                if len(self.positions['measurement']) > 1:
                    print(f"Les moteurs {', '.join(self.positions['measurement'].keys())} ont été scannés.")
                    # Calcul du nobmre de points scannés sur chaque moteur
                    for motor, value in self.positions['measurement'].items():
                        print(f"Nombre de points scannés sur le moteur {motor}: {value.shape[0]}")
                    # le produit doit être égale au nombre de frames eiger
                    product = 1
                    for motor, value in self.positions['measurement'].items():
                        product *= value.shape[0]
                    if product == self.nb_frames:
                        print(f"Le produit du nombre de points scannés sur chaque moteur ({product}) correspond au nombre de frames eiger ({self.nb_frames}).")
        self.times = self.get_time_metadata()
        self.epoch = self.times['epoch_trig'] if 'epoch_trig' in self.times else None
        self.samplename = self.times['sample_name'] if 'sample_name' in self.times else None
        self.command = self.times['title'] if 'title' in self.times else None

        self.detector_info = self.get_detector_info()


    def list_entries(self):
        """Liste les numéros d'entrée (scans) présents dans le fichier, ex: ['1.1', '2.1', ...]."""
       
        return list(self.h5_file.keys())



    def get_eiger_source(self):
        """Résout le softlink/Virtual Dataset des images eiger d'une entrée.

        Retourne un dict avec:
        - 'vds_path': chemin interne dans le fichier de métadonnées (measurement/eiger)
        - 'real_file': chemin (relatif au dossier du .h5) du fichier physique contenant les images
        - 'real_dataset': chemin du dataset dans le fichier physique
        - 'shape', 'dtype': du dataset vu depuis le fichier de métadonnées
        """
        try:
            ds = self.h5_file[f'{self.entry}/measurement/eiger']
        except KeyError:
            try:
                ds = self.h5_file[f'{self.entry}/measurement/frelon3']
            except KeyError:
                raise KeyError(f"Neither 'measurement/eiger' nor 'measurement/frelon3' found in entry {self.entry}.")
        #ds = self.h5_file[f'{self.entry}/instrument/eiger/image']
        info = {
            'vds_path': f'{self.entry}/measurement/eiger',
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


    def get_motor_positions(self, motor_names=MOTOR_NAMES):
        """Renvoie la position de chaque moteur, catégorisée par source.

        - 'measurement' : moteurs qui bougent pendant le scan -> tableau (1 valeur/point)
        - 'positioners' : moteurs fixes pendant le scan -> valeur scalaire
        """
        positions = {'measurement': {}, 'positioners': {}}
        meas = self.h5_file.get(f'{self.entry}/measurement')
        pos = self.h5_file.get(f'{self.entry}/instrument/positioners')
        for name in motor_names:
            if meas is not None and name in meas:
                positions['measurement'][name] = meas[name][()]
            elif pos is not None and name in pos:
                positions['positioners'][name] = pos[name][()]
        return positions



    def get_time_metadata(self):
        """Récupère les métadonnées temporelles (scan + par frame) d'une entrée."""
        g = self.h5_file[self.entry]
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


    def get_detector_info(self):
        """Récupère les métadonnées géométriques/techniques du détecteur eiger.

        Lit les champs NXdetector standards (taille de pixel, distance,
        centre du faisceau, ...) directement sous 'instrument/eiger'.
        Les champs absents du fichier restent à None (dépend de la version
        de bliss/du firmware Eiger utilisés lors de l'acquisition).
        """
        det = self.h5_file.get(f'{self.entry}/instrument/eiger')
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
        image_width = self.data.shape[2] if self.data.ndim == 3 else self.data.shape[1]
        image_height = self.data.shape[1] if self.data.ndim == 3 else self.data.shape[0]
        info['image_width'] = image_width
        info['image_height'] = image_height
        return info


    def list_detector_keys(self):
        """Liste tous les champs disponibles sous 'instrument/eiger'.

        Utile pour explorer les métadonnées réellement présentes dans un
        fichier donné (le contenu exact varie selon la version de bliss).
        """
        det = self.h5_file.get(f'{self.entry}/instrument/eiger')
        if det is None:
            return []
        keys = []
        det.visit(keys.append)
        return keys


    def close(self):
        """Ferme le fichier HDF5."""
        self.h5_file.close()