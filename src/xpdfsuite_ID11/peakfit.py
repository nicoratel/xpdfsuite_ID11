from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def pseudo_voigt(x, amplitude, center, fwhm, eta, bkg):
    """
    Pseudo-Voigt profile = linear combination of a Gaussian and a Lorentzian
    sharing the same center and FWHM.

    eta = 0 -> pure Gaussian, eta = 1 -> pure Lorentzian
    """
    sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-(x - center) ** 2 / (2 * sigma_g ** 2))
    lorentz = 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)
    return amplitude * (eta * lorentz + (1 - eta) * gauss) + bkg


def pseudo_voigt_area(amplitude, fwhm, eta):
    """
    Analytical integrated area (background excluded) of a pseudo-Voigt peak,
    consistent with the `pseudo_voigt` parameterization above.

    Gaussian component area  : amplitude * sigma_g * sqrt(2*pi)
    Lorentzian component area: amplitude * (fwhm/2) * pi
    """
    sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
    area_gauss = sigma_g * np.sqrt(2 * np.pi)
    area_lorentz = (fwhm / 2) * np.pi
    return amplitude * (eta * area_lorentz + (1 - eta) * area_gauss)


def fit_peak_pseudovoigt(r,G, position=2.9, window=0.2, plot=True):
    """
    Refine a single peak located around `position` (+/- window) with a
    pseudo-Voigt profile using scipy.optimize.curve_fit.

    Parameters
    ----------
    x, y : array-like
        Full data arrays (e.g. tth2, I2n).
    position : float
        Approximate peak center to refine.
    window : float, default 0.2
        Half-width of the fitting window around `position`.
    plot : bool, default True
        If True, plot the data window together with the fitted profile.

    Returns
    -------
    popt : ndarray
        Fitted parameters [amplitude, center, fwhm, eta, bkg].
    pcov : ndarray
        Covariance matrix of the fit (use np.sqrt(np.diag(pcov)) for 1-sigma errors).
    area : float
        Analytical integrated area of the peak (background excluded), computed
        from the fitted amplitude, fwhm and eta with :func:`pseudo_voigt_area`.
    """
   
    x = np.asarray(r)
    y = np.asarray(G)

    mask = (x >= position - window) & (x <= position + window)
    xw, yw = x[mask], y[mask]

    if len(xw) < 5:
        raise ValueError("Not enough points in the selected window to fit a peak.")

    # initial guesses
    amp0 = yw.max() - yw.min()
    bkg0 = yw.min()
    fwhm0 = window

    p0 = [amp0, position, fwhm0, 0.5, bkg0]
    bounds = (
        [0, position - window, 1e-4, 0, -np.inf],
        [np.inf, position + window, 2 * window, 1, np.inf],
    )

    popt, pcov = curve_fit(pseudo_voigt, xw, yw, p0=p0, bounds=bounds)          
    area = pseudo_voigt_area(popt[0], popt[2], popt[3])

    if plot:
        xfine = np.linspace(xw.min(), xw.max(), 500)
        plt.figure()
        plt.plot(xw, yw, 'o', ms=3, label='data')
        plt.plot(xfine, pseudo_voigt(xfine, *popt), '-', label='pseudo-Voigt fit')
        plt.axvline(popt[1], color='k', linestyle='--', label=f"center = {popt[1]:.4f}")
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Normalized Intensity')
        plt.title(f"Peak fit around {position}° (refined center = {popt[1]:.4f}°)")
        plt.legend()
    print(f"Fitted parameters: amplitude={popt[0]:.4f}, center={popt[1]:.4f}, fwhm={popt[2]:.4f}, eta={popt[3]:.4f}, bkg={popt[4]:.4f}")
    print(f"Integrated area: {area:.4f}")
    return popt, pcov, area