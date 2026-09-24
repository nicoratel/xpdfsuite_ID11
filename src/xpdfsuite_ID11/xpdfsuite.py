from .id11data import ID11Data
from .pdf_extraction import compute_xPDF
from pyFAI import load
import fabio
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def _mask_as_array(mask):
    """Return a boolean ndarray from a mask that may be a file path or already an ndarray."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return mask.astype(bool)
    
    return fabio.open(mask).data.astype(bool)



class XRDProcessor(ID11Data):
    def __init__(self,
                image_file,
                frame='mean',
                poni_file=None,
                mask=None,
                dark=None,
                flat=None,
                spline=None,
                polarization_factor=0.99,
                verbose=False):
        """
        Initialise an X-ray diffraction data processor.

        Loads the image, identifies the detector geometry from the PONI file,
        optionally applies dark current subtraction and flat-field correction
        during azimuthal integration, and auto-detects the beam centre via
        iso-intensity contours.

        Parameters
        ----------
        image_file : str
            Path to the diffraction data file (h5, hdf5, nxs, tif, tiff).
        poni_file : str, optional
            Path to the pyFAI geometric calibration file (.poni).
            Required for X-ray data.
        mask : str, optional
            Path to a fabio mask file (EDF or similar). Convention:
            0 = valid pixel, 1 = masked pixel.
        dark : str or ndarray, optional
            Dark current image, either as a file path (loaded with
            :func:`~xpdfsuite.filereader.load_image`) or a numpy array.
            Passed to ``ai.integrate1d`` as ``dark``. Default is ``None``.
        flat : str or ndarray, optional
            Flat-field image, either as a file path or a numpy array.
            Passed to ``ai.integrate1d`` as ``flat``. Default is ``None``.
        spline : str, optional
            Path to a spline file for spline correction.
        verbose : bool, optional
            If ``True``, print metadata and detector info. Default is ``False``.
        """
        super().__init__(image_file)
        # héritage de EigerData pour charger data dans self.data (3d array), 
        # et metadata dans self.positions, self.scanned_motors, self.times, self.epoch, self.nb_frames, self.h5_files
        
        self.filename = image_file

        # Select data using frame keyword: 'mean' for average, or an integer index for a specific frame.
        if isinstance(frame, str) and frame == 'mean':
            self.img = np.mean(self.data, axis=0) if self.nb_frames > 1 else self.data[0]
        elif isinstance(frame, (int, np.integer)):
            if frame < 0 or frame >= self.nb_frames:
                raise ValueError(f"Frame index {frame} is out of bounds for {self.nb_frames} frames.")
            self.img = self.data[frame]
        elif isinstance(frame, (list, tuple, np.ndarray)):
            frames = np.asarray(frame, dtype=int)
            if frames.size == 0:
                raise ValueError("Frame list is empty.")
            if frames.min() < 0 or frames.max() >= self.nb_frames:
                raise ValueError(f"Frame indices {frames.tolist()} out of bounds for {self.nb_frames} frames.")
            self.img = np.mean(self.data[frames], axis=0)
        else:
            raise ValueError(f"Invalid frame specification: {frame}. Expected 'mean', an integer or a list of integers.")

        self.poni_file = poni_file
        # load mask if provided, otherwise create an empty mask
        if mask is not None:
            mask_img = fabio.open(mask)
            self.mask = mask_img.data
        else:
            self.mask = np.zeros_like(self.img)

        # load poni file if provided, and prepare pyFAI integrator
        if poni_file is not None:
            self.ai = load(poni_file)
            self.use_pyfai = True

            # ---- Correction de spline (distorsion géométrique) ----
            self.spline = spline
            if self.spline is not None:
                self.ai.detector.splinefile = self.spline
                if verbose:
                    print(f"  ✓ Spline appliquée au détecteur : '{self.spline}'")

            if mask is not None:
                mask_img = fabio.open(mask)
                self.mask = mask_img.data.astype(bool)
            else:
                self.mask = np.zeros(self.img.shape, dtype=bool)
        else:
            raise ValueError(
                "A PONI file is required for X-ray data. "
                "Please provide poni_file= when creating the processor."
            )

        # Load dark and flat correction images
        if dark is not None:
            self.dark_img = fabio.open(dark)
            self.dark = self.dark_img.data
            nframes_dark = int(self.dark_img.header.get('nframes', 1))
            print(f"Dark image loaded: {dark} with {nframes_dark} frame(s).")
            diff = self.img - self.dark/nframes_dark
            print("Dark subtraction: min:", diff.min(), "  pixels négatifs:", (diff < 0).sum(), "/", diff.size)   
        else:
            self.dark = None
        if flat is not None:
            self.flat_img = fabio.open(flat)
            self.flat = self.flat_img.data
        else:
            self.flat = None
        self.polarization_factor = polarization_factor


                   


    def integrate(self, npt=2500, plot=False):
        """
        Azimuthally integrate the SAED pattern to a 1D I(q) profile.

        Beam centre recalibration is always performed with
        :func:`recalibrate_from_isocurve` using ``self.center`` as the
        initial estimate.  If *center* is provided it overwrites
        ``self.center`` and is used directly without re-running isocurve.

        Parameters
        ----------
        npt : int, optional
            Number of points in the output q profile. Default is 2500.
        polarization_factor : float, optional
            Polarization factor for the integration. Default is 0.99.
        plot : bool, optional
            If ``True``, display the integrated I(q) pattern. Default is ``False``.

        Returns
        -------
        q : ndarray
            Scattering vector in Å⁻¹.
        I : ndarray
            Azimuthally averaged intensity.
        """
        
        if self.use_pyfai:
            q, I = self.ai.integrate1d(
                self.img, npt, mask=self.mask, unit="q_A^-1",
                polarization_factor=self.polarization_factor,
                dark=self.dark, flat=self.flat,
            )
        else:
            pass

        if plot:
            plt.figure()
            plt.plot(q, I)
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            #plt.title('Azimuthally Integrated SAED Pattern')
            plt.grid()
            plt.show()
        
        return q, I
    
    def plot(self,vmin=-4, vmax=0,cmap='jet',display_mask=False):
        plt.figure()
        if display_mask:
            
            # Create a copy of the image for display
            img_display = self.img.copy() / np.max(self.img)
            # Set masked pixels to NaN to display them in white
            img_display[self.mask.astype(bool)] = np.nan
            plt.imshow(img_display, cmap=cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
            # Set NaN color to white
            current_cmap = plt.get_cmap(cmap).copy()
            current_cmap.set_bad(color='white')
            plt.imshow(img_display, cmap=current_cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
            plt.plot(self.center[0], self.center[1], 'k+', markersize=8)
        else:            
            plt.imshow(self.img/np.max(self.img), cmap=cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
        

    def extract_xpdf(self,
                     ref_diffraction_image=None,
                     ref_poni_file=None,
                     composition='Au',
                     rmin=0.1,
                     rmax=50.0,
                     rstep=0.01,
                     outputfile=None,
                     interactive=True,
                     plot=False,
                     bgscale=1,
                     qmin=1.5,
                     qmax=24,
                     qmaxinst=24,
                     rpoly=1.4):
        """
        Extract the xPDF from the diffraction data (convenience wrapper).

        Creates a :class:`XRDProcessor` for the reference image if provided
        (assumed in identical experimetnal conditions), then calls :func:`extract_xpdf` with the appropriate arguments.

        Parameters
        ----------
        ref_diffraction_image : str, optional
            Path to the reference (background) diffraction image.
        ref_poni_file : str, optional
            Path to a PONI file for the reference image if it differs from
            the sample.
        composition : str, optional
            Chemical formula of the sample (e.g. ``'Au'``, ``'SiO2'``).
            Default is ``'Au'``.
        rmin, rmax, rstep : float, optional
            Real-space range and step for G(r) in Å.
        outputfile : str, optional
            Path for the output ``.gr`` file. Auto-generated if ``None``.
        interactive : bool, optional
            If ``True`` (default), open the interactive slider GUI.
        plot : bool, optional
            If ``True``, display plots in non-interactive mode.
        bgscale : float, optional
            Background scaling factor. Default is 1.
        qmin, qmax, qmaxinst : float, optional
            Q-range limits in Å⁻¹ for PDF computation.
        rpoly : float, optional
            Polynomial background degree control (PDFgetX3 convention).

        Returns
        -------
        results : PDFResultsReference or tuple
            Interactive mode: :class:`PDFResultsReference` supporting
            ``r, g = results`` unpacking.
            Non-interactive mode: ``(r, G)`` tuple of ndarrays.
        """
        ref_processor = None
        if ref_diffraction_image is not None:
            ref_processor = XRDProcessor(
                ref_diffraction_image,
                poni_file=ref_poni_file if ref_poni_file is not None else self.poni_file,
                mask=self.mask,
                dark=self.dark,
                flat=self.flat,
                spline=self.spline,
                polarization_factor=self.polarization_factor,

                verbose=False,
            )

        return extract_xpdf(
            sample_processor=self,
            ref_processor=ref_processor,
            composition=composition,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            outputfile=outputfile,
            interactive=interactive,
            plot=plot,
            bgscale=bgscale,
            qmin=qmin,
            qmax=qmax,
            qmaxinst=qmaxinst,
            rpoly=rpoly,
        )



# ------------------
# Standalone xPDF extraction function
# ------------------
def extract_xpdf(sample_processor,
                 ref_processor=None,
                 composition='Au',
                 rmin=0.1,
                 rmax=50.0,
                 rstep=0.01,
                 outputfile=None,
                 interactive=True,
                 plot=False,
                 bgscale=1,
                 qmin=1.5,
                 qmax=24,
                 qmaxinst=24,
                 rpoly=1.4):
    """
    Extract the X-ray Pair Distribution Function (xPDF) from diffraction data.

    Integrates each :class:`XRDProcessor` to a 1D I(q) profile, then calls
    :func:`~xpdfsuite.pdf_extraction.compute_xPDF`. In interactive mode an
    ipywidgets GUI is shown; in non-interactive mode G(r) is computed once
    and saved to a ``.gr`` file.

    Parameters
    ----------
    sample_processor : XRDProcessor
        Processor loaded with the sample diffraction image.
        Set ``sample_processor.initial_center`` before calling if needed.
    ref_processor : XRDProcessor, optional
        Processor loaded with the background/reference image.
        If ``None``, no background subtraction is performed.
    composition : str, optional
        Chemical formula of the sample (e.g. ``'Au'``, ``'Fe2O3'``).
        Default is ``'Au'``.
    rmin, rmax, rstep : float, optional
        Real-space range and step for G(r) in Å.
        Defaults: 0.1, 50.0, 0.01.
    outputfile : str, optional
        Path for the output ``.gr`` file.
        Auto-generated from the sample filename if ``None``.
    interactive : bool, optional
        If ``True`` (default), open the interactive slider GUI via
        :class:`PDFInteractive`.
    plot : bool, optional
        If ``True``, display G(r) and F(Q) in non-interactive mode.
    bgscale : float, optional
        Background scaling factor applied to the reference. Default is 1.
    qmin, qmax, qmaxinst : float, optional
        Q-range limits in Å⁻¹ used for the Fourier transform and polynomial
        background fitting.
    rpoly : float, optional
        Polynomial degree control (PDFgetX3 convention). Default is 1.4.

    Returns
    -------
    results : PDFResultsReference or tuple
        Interactive mode: :class:`PDFResultsReference` — supports
        ``r, g = results`` unpacking after slider adjustment.
        Non-interactive mode: ``(r, G)`` tuple of ndarrays.

    Examples
    --------
    >>> sample = XRDProcessor('sample.h5', poni_file='calib.poni')
    >>> sample.initial_center = (335, 275)
    >>> ref = XRDProcessor('ref.h5', poni_file='calib.poni')
    >>> results = extract_xpdf(sample, ref, composition='Au', interactive=False)
    >>> r, G = results
    """
    # Integrate sample
    q_sample, intensity_sample = sample_processor.integrate(plot=False)
    if qmaxinst is None:
        qmaxinst = q_sample.max()
    # Integrate reference if provided
    if ref_processor is not None:
        q_ref, intensity_ref = ref_processor.integrate(plot=False)
    else:
        q_ref, intensity_ref = None, None


    # Generate output filename if not provided
    if outputfile is None:
        outputfile = sample_processor.filename.split('.')[0] + '_pdf.gr'
    
    if interactive:
        # Création de l'objet PDFInteractive avec la nouvelle interface
        pdf_interactive = PDFInteractive(
            sample_processor,
            ref_processor=ref_processor,
            composition=composition,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            xray=False,
            outputfile=outputfile
        )
        # Si une méthode d'export existe, l'appeler ici
        if hasattr(pdf_interactive, 'save_results'):
            pdf_interactive.save_results(outputfile)
        pdf_interactive.show()
        # Store the interactive object for access to results
        sample_processor.pdf_interactive = pdf_interactive
        # Return a reference to the results that will be updated by sliders
        return PDFResultsReference(pdf_interactive)
    else:
        print('Compute PDF with given parameters')
        r, G = compute_xPDF(
            q_sample,
            intensity_sample,
            composition,
            Iref=intensity_ref if ref_processor is not None else None,
            bgscale=bgscale,
            qmin=qmin,
            qmax=qmax,
            qmaxinst=qmaxinst,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            rpoly=rpoly,
            Lorch=True,
            plot=plot)
        
        # Generate header for .gr file
        header = '[DEFAULT]\n\nversion = xpdfsuite 1.0\n\n'
        header += '#input and output specifications\n'
        header += 'dataformat = q_A \n'
        header += f'inputfile = {sample_processor.filename}\n'
        header += f'backgroundfile = {ref_processor.filename if ref_processor is not None else "None"}\n'
        header += 'outputtype = gr\n\n'
        header += '#PDF calculation setup\n'
        header += 'mode = xrays\n'
        try:
            header += f'wavelength = {sample_processor.ai.wavelength:.4f}\n'
        except:
            header += 'wavelength = unknown\n'
        header += 'twothetazero = 0\n'
        header += f'composition={composition} \n'
        header += f'bgscale = {bgscale:.2f} \n'
        header += f'rpoly = {rpoly:.2f} \n'
        header += f'qmaxinst = {qmaxinst:.2f}\n'
        header += f'qmin = {qmin:.2f} \n'
        header += f'qmax = {qmax:.2f}  \n'
        header += f'rmin = {rmin:.2f} \n'
        header += f'rmax = {rmax:.2f} \n'
        header += f'rstep = {rstep:.2f}\n\n'
        header += '# End of config --------------------------------------------------------------\n'
        header += '#### start data\n\n'
        header += '#S 1 \n'
        header += '#L r(Å)  G(Å$^{-2}$)\n'
        
        # Write output file
        with open(outputfile, 'w') as f:
            f.write(header)
            for ri, Gi in zip(r, G):
                f.write(f'{ri:.4f}  {Gi:.6f}\n')
        
        print(f'PDF saved to {outputfile}')
        sample_processor.close()
        if ref_processor is not None:
            ref_processor.close()
        return r, G


# ------------------
# Results Reference Class
# ------------------
class PDFResultsReference:
    """
    Proxy object providing access to the most recent PDF results from interactive mode.

    Wraps a :class:`PDFInteractive` instance and supports tuple unpacking
    (``r, g = reference``) so that the same syntax works whether the
    extraction is interactive or not. Values reflect the **last slider
    state** — call ``r, g = results`` after adjusting the sliders.
    """
    
    def __init__(self, pdf_interactive):
        """
        Parameters
        ----------
        pdf_interactive : PDFInteractive
            The interactive GUI object holding the computed results.
        """
        self._pdf_interactive = pdf_interactive
    
    def __iter__(self):
        """
        Support tuple unpacking: ``r, g = reference``.

        Returns the latest r and G arrays computed by the interactive GUI.
        Prints a warning if no computation has been performed yet.
        """
        if self._pdf_interactive.last_r is None or self._pdf_interactive.last_G is None:
            print("⚠️ Aucune valeur disponible. Ajustez les paramètres avec les sliders pour générer r et G.")
            return iter([None, None])
        return iter([self._pdf_interactive.last_r, self._pdf_interactive.last_G])
    
    def __repr__(self):
        """String representation showing r-range and number of points."""
        if self._pdf_interactive.last_r is None:
            return "PDFResultsReference(no data yet - adjust sliders to compute)"
        return f"PDFResultsReference(r: {len(self._pdf_interactive.last_r)} points, " \
               f"r_range=[{self._pdf_interactive.last_r.min():.2f}, {self._pdf_interactive.last_r.max():.2f}] Å)"
    
    @property
    def r(self):
        """ndarray : Real-space distance axis in Å from the last computation."""
        return self._pdf_interactive.last_r
    
    @property
    def g(self):
        """ndarray : G(r) values in Å⁻² from the last computation."""
        return self._pdf_interactive.last_G


# ------------------
# Interactive GUI Class
# ------------------
class PDFInteractive:
    """
    Jupyter widget GUI for interactive xPDF parameter optimisation.

    Displays ipywidgets sliders for ``bgscale``, ``qmin``, ``qmax``,
    ``qmaxinst``, and ``rpoly``, recomputing G(r) in real time.
    Results can be exported to a ``.gr`` file via the Save button.

    Intended to be created by :func:`extract_xpdf` — not directly by users.
    """

    def __init__(self,
                 sample_processor,
                 ref_processor=None,
                 composition='Au',
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 xray: bool = False,
                 outputfile: str = './pdf_results.csv'):
        """
        Parameters
        ----------
        sample_processor : XRDProcessor
            Processor for the sample diffraction data.
        ref_processor : XRDProcessor, optional
            Processor for the background/reference image. Default is ``None``.
        composition : str, optional
            Chemical formula of the sample. Default is ``'Au'``.
        rmin, rmax, rstep : float, optional
            Real-space range and step for G(r) in Å.
        xray : bool, optional
            If ``True``, use X-ray scattering factors. Default is ``False``.
        outputfile : str, optional
            Default path for the Save button output. Default is
            ``'./pdf_results.csv'``.
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.widgets = widgets
        self.display = display

        print('Slide cursors to ajdust parameters values. Click "Save" to export results.')

        # Stocker les processeurs pour accès ultérieur
        self.sample_processor = sample_processor
        self.ref_processor = ref_processor
        self.composition = composition

        # Intégration des données (sample et ref)
        q, Iexp = sample_processor.integrate(plot=False)
        if ref_processor is not None:
            _, Iref = ref_processor.integrate(plot=False)
        else:
            Iref = None

        # Métadonnées utiles
        self.wavelength = getattr(sample_processor, 'wavelength', None)
        self.camera = getattr(sample_processor, 'camera_title', None)
        self.sample_diffraction_image = getattr(sample_processor, 'dm4_file', None)
        self.ref_diffraction_image = getattr(ref_processor, 'dm4_file', None) if ref_processor is not None else None

        # PDF config
        self.xray = xray
        self.pdf_config = dict(
            q=q, Iexp=Iexp, Iref=Iref, composition=composition,
            rmin=rmin, rmax=rmax, rstep=rstep,
        )

        self.last_r = None
        self.last_G = None

        # Create parameter control sliders
        self.bgscale_slider = self.widgets.FloatSlider(
            value=1, min=0, max=2, step=0.01, 
            description="bgscale", readout_format=".2f"
        )
        self.qmin_slider = self.widgets.FloatSlider(
            value=1.5, min=np.min(q), max=min(24,np.max(q)), step=0.01,
            description="qmin", readout_format=".2f"
        )
        self.qmax_slider = self.widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmax", readout_format=".2f"
        )
        self.qmaxinst_slider = self.widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmaxinst", readout_format=".2f"
        )
        self.rpoly_slider = self.widgets.FloatSlider(
            value=1.4, min=0.1, max=2.5, step=0.01,
            description="rpoly", readout_format=".2f"
        )
        
        self.lorch_checkbox = self.widgets.Checkbox(
            value=True,
            description="apply Lorch window correction to eliminate termination ripples",
            indent=False)

        # Save button for exporting results
        self.save_button = self.widgets.Button(description="💾 Save", button_style="success")
        self.save_button.on_click(lambda b: self.save_results(b, outputfile))

        # Organize widgets in vje veux quelque xhiose de plus simple. Je vais me débrouillerertical layout
        self.sliders = self.widgets.VBox([
            self.bgscale_slider,
            self.qmin_slider,
            self.qmax_slider,
            self.qmaxinst_slider,
            self.rpoly_slider,
            self.lorch_checkbox,
            self.save_button])

        # Output area for plots
        self.plot_output = self.widgets.Output()

        # Link sliders to update function for real-time feedback
        self.widgets.interactive_output(self.update_plot, {
            "bgscale": self.bgscale_slider,
            "qmin": self.qmin_slider,
            "qmax": self.qmax_slider,
            "qmaxinst": self.qmaxinst_slider,
            "rpoly": self.rpoly_slider,
            "lorch": self.lorch_checkbox})

    def update_plot(self, bgscale, qmin, qmax, qmaxinst, rpoly, lorch):
        """
        Recompute G(r) and refresh the output plot.

        Called automatically by ``ipywidgets.interactive_output`` whenever
        a slider value changes. Stores the result in ``self.last_r`` and
        ``self.last_G``.

        Parameters
        ----------
        bgscale : float
            Background scaling factor.
        qmin, qmax, qmaxinst : float
            Q-range limits in Å⁻¹.
        rpoly : float
            Polynomial degree control parameter.
        lorch : bool
            Whether to apply the Lorch modification function.
        """
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            # Recompute PDF with new parameters
            r, G = compute_xPDF(
                **self.pdf_config,
                bgscale=bgscale, qmin=qmin, qmax=qmax,
                qmaxinst=qmaxinst, rpoly=rpoly, plot=True, Lorch=lorch)
            # Store results for potential saving
            self.last_r, self.last_G = r, G

    def save_results(self, b, outputfile='./pdf_results.gr'):
        """
        Save the last computed G(r) to a ``.gr`` text file.

        The file format is compatible with PDFgetX3 / PDFBatchAnalysis,
        with a structured header containing all computation parameters.

        Parameters
        ----------
        b : widget button event
            Unused; required by the ipywidgets callback signature.
        outputfile : str, optional
            Output file path. Default is ``'./pdf_results.gr'``.
        """
        if self.last_r is None or self.last_G is None:
            print("⚠️ Aucun résultat à sauvegarder (génère d'abord un plot).")
            return

        # make header similar to pdfgetx3 for further compatibility with PDFBatchANalayis
        # header should have same architecture as .gr files from pdfgetx3 for compatibility with PDFBatchAnalysis
        header  = '[DEFAULT]\n\nversion = xpdfsuite 1.0\n\n'
        header += '# input and output specifications\n'
        header +=f'camera = {self.camera} \n'
        header +=f'inputfile = {self.sample_diffraction_image}\n'
        header +=f'backgroundfile = {self.ref_diffraction_image}\n'
        header += 'outputtype = gr\n\n'
        header += '#PDF calculation setup\n'
        header += 'mode = xrays\n'        
        header +=f'wavelength = {self.sample_processor.ai.wavelength:.4f}\n'
        header += 'twothetazero = 0\n'        
        header +=f'composition={self.composition} \n'
        header +=f'bgscale = {1:.2f} \n'
        header +=f'rpoly = {1.4} \n'
        header +=f'qmaxinst = {self.qmaxinst_slider.value:.2f}\n'
        header +=f'qmin = {self.qmin_slider.value:.2f} \n'
        header +=f'qmax = {self.qmax_slider.value:.2f}  \n'
        header +=f'rmin = {0:.2f} \n'
        header +=f'rmax = {50:.2f} \n'
        header +=f'rstep = {0.01:.2f}\n\n'
        header += '# End of config --------------------------------------------------------------\n#### start data\n\n'
        header += '#S 1 \n'
        header += '#L r(Å)  G(Å$^{-2}$)'

        np.savetxt(outputfile, np.column_stack((self.last_r, self.last_G)),header=header,delimiter=' ',comments='')
        print(f'PDF saved to {outputfile}')
        

    def show(self):
        """
        Render and display the interactive GUI in a Jupyter notebook.

        Computes G(r) with the current slider values before displaying
        the interface, so that ``last_r`` and ``last_G`` are immediately
        available for tuple unpacking.
        """
        # Generate initial plot with default parameter values BEFORE displaying UI
        # This ensures last_r and last_G are immediately available for unpacking
        self.update_plot(
            self.bgscale_slider.value, self.qmin_slider.value,
            self.qmax_slider.value, self.qmaxinst_slider.value,
            self.rpoly_slider.value, self.lorch_checkbox.value
        )
        
        ui = self.widgets.HBox([self.sliders, self.plot_output])
        self.display(ui)



       
