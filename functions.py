import numpy as np
import math
import pandas as pd
from astroquery.jplhorizons import Horizons
import matplotlib.pyplot as plt


# from jax import numpy as np


def get_object_orbit(name: str, epochs: iter, location: str = '@10', refplane='ecliptic',
                     data_type='elements', n_epochs=90) -> pd.DataFrame:
    """Queries JPL Horizons system to get the orbit or the object.

    Args:
        name (str): The id of the object (name, provisional designation, number) .
        epochs (iter): The epochs to query for in JD format.
        location (str, optional): The location of the observer.
        Defaults to '@10' - heliocentric
        '@0' - Solar system barycenter
        refplane (str, optional): The reference plane to use. Defaults to 'ecliptic'.
        data_type (str, optional): The type of data to retrieve. Can be 'elements' or 'vectors'. Defaults to 'elements'.

    Returns:
        pd.DataFrame: [name, a, e, i, om, w, ma, epoch] if type == 'elements'
        a in AU, e in [-], i in [red], om in [rad], w in [rad], ma in [rad], epoch in [JD]

        pd.DataFrame: [name, x, y, z, vx, vy, vz, epoch] if type == 'vectors'
        x in AU, y in AU, z in AU, vx in AU/day, vy in AU/day, vz in AU/day, epoch in [JD]
    Raises:
        ValueError: If the type is not 'elements' or 'vectors'.
    References:
        For more information see:
        https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
    """

    def _name_handler(name: str):
        """Helper function to handle object names."""
        dict_names = {
            'Earth': '399',
            'Moon': '301',
            'Venus': '299',
            'Mercury': '199',
            'Mars': '499',
            'Jupiter': '599',
            'Saturn': '699',
            'Neptune': '899',
            'Uranus': '799'
        }
        return dict_names.get(name, name)

    def _divide_chunks(epochs: iter, n: int = 90):
        """divide list into chunks of size n"""
        for i in range(0, len(epochs), n):
            yield epochs[i:i + n]

    data_all = []
    # Query Horizons system
    for epoch_chunk in _divide_chunks(epochs=epochs, n=n_epochs):
        obj = Horizons(id=_name_handler(name), location=location, epochs=epoch_chunk)

        if data_type == 'elements':
            # Get necessary columns and transform to pandas.DataFrame
            orb_elems = obj.elements(refplane=refplane)[['targetname', 'a', 'e',
                                                         'incl', 'Omega', 'w', 'M', 'datetime_jd']].to_pandas()
            # Rename columns for consistency with other query functions
            orb_elems = orb_elems.rename(columns={'incl': 'i', 'Omega': 'om', 'M': 'ma',
                                                  'datetime_jd': 'epoch', 'targetname': 'name'})
            # Convert degrees to radians
            orb_elems['i'] = np.radians(orb_elems['i'])
            orb_elems['om'] = np.radians(orb_elems['om'])
            orb_elems['w'] = np.radians(orb_elems['w'])
            orb_elems['ma'] = np.radians(orb_elems['ma'])

            data_chunk = orb_elems

        elif data_type == 'vectors':
            # Units are AU for distances and AU/d for velocities
            vec_elems = obj.vectors(refplane=refplane)[['targetname', 'x', 'y', 'z',
                                                        'vx', 'vy', 'vz', 'datetime_jd']].to_pandas()
            # Rename columns for consistency with other query functions
            vec_elems = vec_elems.rename(columns={'datetime_jd': 'epoch', 'targetname': 'name'})
            data_chunk = vec_elems

        else:
            raise ValueError('type must be either "elements" or "vectors"')

        data_all.append(data_chunk)

    return pd.DataFrame(pd.concat(data_all))


def center_of_mass(image: np.ndarray) -> tuple:
    """Compute the center of mass of an image"""
    # Ensure the image is a numpy array
    image = np.array(image)
    # Compute the total mass
    total_mass = np.sum(image)
    # Compute the coordinates of the center of mass
    x_indices, y_indices = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x_center_of_mass = np.sum(x_indices * image) / total_mass
    y_center_of_mass = np.sum(y_indices * image) / total_mass
    return (y_center_of_mass, x_center_of_mass)


def estimate_density(a1, b1, c1, a2, b2, c2, rot_per) -> np.ndarray:
    """
    Estimates the density of a binary asteroid based on the ratio of ellipsis parameters
    Args:
        a1, b1, c1, a2, b2, c2 (float) - semi-major axes of the two ellipsoid [any units]
        rot_per (float) - rotational period of the binary [days]
    Returns:
        density (float): Estimated density of the binary asteroid [kg/m^3].
    """
    G = 6.67408e-11
    density = 3 * np.pi * (a1 + a2) ** 3 / (G * (rot_per * 60 * 60 * 24) ** 2 * (a1 * b1 * c1 + a2 * b2 * c2))
    return density


def build_lc_plots(lightcurve, sun_data, lightcurve_theor, num_rows: int, num_cols: int, figsize: tuple = (9, 4.5),
                   dpi: int = 100, show_lc_names=False):
    """Builds the plots for the lightcurves"""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 15
    fig, axarr = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows), dpi=dpi)
    for idx in range(0, len(lightcurve.lightcurves)):
        lc = lightcurve.lightcurves[idx].data
        idx_row, idx_col = idx // num_cols, idx % num_cols
        axarr[idx_row][idx_col].scatter(sun_data[idx]['reduc_epoch'], -lightcurve_theor[idx], s=8,
                                                       label='model')
        axarr[idx_row][idx_col].scatter(sun_data[idx]['reduc_epoch'],
                                                   -lightcurve.lightcurves[idx].data['mag_shifted'], s=8, label='obs')

        axarr[idx_row][idx_col].set_xlabel('reduc_epoch [days]')
        axarr[idx_row][idx_col].set_ylabel('H [mag]')
        axarr[idx_row][idx_col].legend(loc='upper left', prop={"size": 15})
    return fig, axarr


def plot_partial_lightcurve(curve_idx, lightcurve, lightcurve_theor, sun_data):
    """
    Plot lightcurve with theoretical and observed data.

    Parameters:
    curve)idx (int): Index of the lightcurve to plot
    lightcurve (object): Lightcurve object
    lightcurve_theor (array): Theoretical lightcurve data
    sun_data (dict): Sun data dictionary
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 15
    plt.figure(figsize=(15, 8))

    lc = lightcurve.lightcurves[curve_idx].data
    plt.scatter(sun_data[curve_idx]['reduc_epoch'], -lightcurve_theor[curve_idx], s=8, label='theor')
    plt.scatter(sun_data[curve_idx]['reduc_epoch'], -lc['mag_shifted'], s=8, label='obs')

    plt.xlabel('reduc_epoch [days]')
    plt.ylabel('H [mag]')
    plt.legend(loc='upper left', prop={"size": 15})
    plt.grid()

    # calculate chi-squared
    chi2_part = np.sum((lc['mag_shifted'] - lightcurve_theor[curve_idx]) ** 2 / lc['mag_err'] ** 2)
    chi2 = np.sum(
        (lightcurve.joined['mag_shifted'] - lightcurve_theor.joined) ** 2 / lightcurve.joined['mag_err'] ** 2 *
        lightcurve.joined['weight'])
    print(f"\u03C7^2_part{curve_idx} = {chi2_part:.1f},\n\u03C7^2 total = {chi2:.1f}")


def derive_err_ellipse(xs, ys, n_points=50):
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    r_xy = np.sum((xs - x_mean) * (ys - y_mean)) / (len(xs) - 1)
    disp_x = np.var(xs, ddof=1)
    disp_y = np.var(ys, ddof=1)
    alpha = 1 / 2 * np.arctan2(2 * r_xy, disp_y - disp_x)
    sigma_zeta = (disp_x * np.cos(alpha) ** 2 + disp_y * np.sin(alpha) ** 2 + 2 * r_xy * np.sin(alpha) * np.cos(
        alpha)) ** 0.5
    sigma_eta = (disp_x * np.sin(alpha) ** 2 + disp_y * np.cos(alpha) ** 2 - 2 * r_xy * np.sin(alpha) * np.cos(
        alpha)) ** 0.5
    phi = np.linspace(0, 2 * np.pi, n_points)
    x_el = x_mean + sigma_zeta * np.cos(alpha) * np.cos(phi) - sigma_eta * np.sin(alpha) * np.sin(phi)
    y_el = y_mean + sigma_zeta * np.sin(alpha) * np.cos(phi) + sigma_eta * np.sin(alpha) * np.cos(phi)
    return x_el, y_el


def calc_ellips(points):
    # Calculate the mean of the points
    mean = np.mean(points, axis=0)
    # Calculate the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    # Calculate the width and height of the ellipse
    width, height = 2 * np.sqrt(eigenvalues)
    return mean, width, height, angle


def estimate_density_new(a1, b1, c1, a2, b2, c2, rot_per) -> np.ndarray:
    G = 6.67408e-11
    density = (3 / 4 * (2 * np.pi / (rot_per * 60 * 60 * 24)) ** 2 / (np.pi * G)
               * (a1 ** 2 * b1 * c1 - a2 ** 2 * b2 * c2)
               / ((a1 * b1 * c1 + a2 * b2 * c2) * (b1 * c1 / a1 - b2 * c2 / a2)))
    return density


def adjust_angle(params):
    # adjust lambdas
    lambdas = params[:, 6]
    betas = params[:, 7]
    lambdas = np.where(betas < -np.pi/2, lambdas + np.pi, lambdas)
    lambdas = np.where(lambdas > 2*np.pi, lambdas - 2*np.pi, lambdas)
    # adjust_beta
    betas = np.where(betas < -np.pi / 2, -np.pi - betas, betas)
    # betas = np.where(betas > -1.05, -1, betas)
    return np.array([lambdas, betas]).T

def adjust_angle_single(data):
    if data[7] < -np.pi/2:
        data[6] = data[6] + np.pi
    if data[6] > 2*np.pi:
        data[6] = data[6] - 2*np.pi
    if data[7] < -np.pi/2:
        data[7] = -np.pi - data[7]
    if data[7] > -1.05:
        data[7] = -1
    return data


# write list of lists to file
def write_bootstrap_data_to_file(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(' '.join(map(str, item)) + '\n')

# read bootstrap result data
def read_bootstrap_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    return [list(map(float, line.strip().split())) for line in lines]

# jump an angle if diff is larger than pi
def angle_diff(angle):
    angle = np.where(abs(angle) > np.pi, 2 * np.pi - angle, angle)
    return angle
