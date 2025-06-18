import numpy as np


# from spectral_denoising repository, source: https://github.com/FanzhouKong/spectral_denoising/blob/main/spectral_denoising/spectral_denoising.py
def electronic_denoising(msms):
    """
    Perform electronic denoising on a given mass spectrometry (MS/MS) spectrum.
    This function processes the input MS/MS spectrum by sorting the peaks based on their intensity,
    and then iteratively selects and confirms peaks based on a specified intensity threshold.
    The confirmed peaks are then packed and sorted before being returned.

    Parameters:
        msms (np.ndarray): The first item is always m/z and the second item is intensity. [i.e., not binned]

    Returns:
        np.ndarray: The cleaned spectrum with electronic noises removed. If no ion presents, will return np.nan.
    """
    if isinstance(msms, float):
        return np.nan
    mass, intensity = msms.T[0], msms.T[1]
    order = np.argsort(intensity)
    mass = mass[order]
    intensity = intensity[order]
    mass_confirmed = np.array([])
    intensity_confirmed = np.array([])
    while len(intensity) > 0:
        seed_intensity = np.max(intensity)
        idx_left = np.searchsorted(intensity, seed_intensity * 0.999, side='left')
        mass_temp = mass[idx_left:]
        intensity_temp = intensity[idx_left:]
        if len(mass_temp) <= 3:
            mass_confirmed = np.concatenate((mass_confirmed, mass_temp))
            intensity_confirmed = np.concatenate((intensity_confirmed, intensity_temp))
        intensity = intensity[0:idx_left]
        mass = mass[0:idx_left]
    if len(mass_confirmed) == 0:
        return np.nan
    return (sort_spectrum(pack_spectrum(mass_confirmed, intensity_confirmed)))


def sort_spectrum(msms):
    """
    Sorts the spectrum data based on m/z values.

    Parameters:
        msms (numpy.ndarray): A 2D numpy array.
    Returns:
        numpy.ndarray: A 2D numpy array with the same shape as the input, but sorted by the m/z values in ascending order.
    """
    if isinstance(msms, float) or len(msms) == 0:
        return np.nan
    msms_T = msms.T
    order = np.argsort(msms_T[0])
    msms_T[0] = msms_T[0][order]
    msms_T[1] = msms_T[1][order]

    return msms_T.T


def pack_spectrum(mass, intensity):
    """
    Inverse of break_spectrum. Packs mass and intensity arrays into a single 2D array, which is standardized MS/MS spectrum data format in this project.
    This function takes two arrays, `mass` and `intensity`, and combines them into a single 2D array where each row
    corresponds to a pair of mass and intensity values. If either of the input arrays is empty, the function returns NaN.

    Parameters:
        mass (numpy.ndarray): An array of mass values.
        intensity (numpy.ndarray): An array of intensity values.
    Returns:
        numpy.ndarray: A 2D array with mass and intensity pairs if both input arrays are non-empty, otherwise NaN.
    """
    if len(mass) > 0 and len(intensity) > 0:
        return (np.array([mass, intensity]).T)
    else:
        return (np.nan)
