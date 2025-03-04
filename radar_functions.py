import math
import numpy as np
import matplotlib.image as image
import cv2
from dataclasses import dataclass
import functions as func
import lc_functions as lc_func
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


# constants
NU0 = 8560000000.
C_MS = 299792458.

class Parameters:
    def __init__(self, a1=1, b1=1, c1=1, a2=1, b2=1, c2=1, rot_per=5.022/24, l=0, b=0, init_phase=0, radar_sat=500, pow_coeff=2):
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.a2 = a2
        self.b2 = b2
        self.c2 = c2
        self.rot_per = rot_per  # rotational period [h]
        self.ecl_longitude = l  # longitude of the pole
        self.ecl_latitude = b  # latitude of the pole
        self.init_phase = init_phase
        self.radar_saturation = radar_sat
        self.radar_background = 0.1
        self.radar_shift1 = 13
        self.radar_shift2 = 2
        self.pow_coeff = pow_coeff



@dataclass
class ImageParameters:
    N_points: int
    filename: str
    jd: float
    dr: float
    dnu: float
    presence: int
    size_px: int

@dataclass
class ListOfImageParameters:
    N_points: int
    filename: list
    jd: list
    dr: list
    dnu: list
    presence: list
    size_px: list

class RadarImageParameters:
    """Class to store radar image parameters"""
    def __init__(self, file_name):
        radar_file = open(file_name, "rt")
        radar_string = radar_file.read()
        radar_list = radar_string.split()

        self.N_points = int(len(radar_list) / 9 - 6)

        self.file_name = [(radar_list[0 + 9 * (i + 1)]) for i in range(self.N_points)]
        day = [float(radar_list[1 + 9 * (i + 1)]) for i in range(self.N_points)]
        hour = [float(radar_list[2 + 9 * (i + 1)]) for i in range(self.N_points)]
        minute = [float(radar_list[3 + 9 * (i + 1)]) for i in range(self.N_points)]
        second = [float(radar_list[4 + 9 * (i + 1)]) for i in range(self.N_points)]
        self.jd = [(59153.5 + ((second[i] / 60 + minute[i]) / 60 + hour[i]) / 24 + day[i]) for i in
                   range(self.N_points)]
        self.dr = [(float(radar_list[5 + 9 * (i + 1)])) for i in range(self.N_points)]
        self.dnu = [(float(radar_list[6 + 9 * (i + 1)])) for i in range(self.N_points)]
        self.presence = [(int(radar_list[7 + 9 * (i + 1)])) for i in range(self.N_points)]
        self.size_px = [int(radar_list[8 + 9 * (i + 1)]) for i in range(self.N_points)]
        self._img_params = [ImageParameters(self.N_points, self.file_name[i], self.jd[i], self.dr[i], self.dnu[i],
                                            self.presence[i], self.size_px[i]) for i in range(self.N_points)]
        self._list_params = ListOfImageParameters(self.N_points, self.file_name, self.jd, self.dr, self.dnu,
                                                   self.presence, self.size_px)
        radar_file.close()

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            # form a new instance if the index is a list of array of indices
            return ListOfImageParameters(len(index), [self.file_name[i] for i in index],
                                         [self.jd[i] for i in index],
                                         [self.dr[i] for i in index], [self.dnu[i] for i in index],
                                         [self.presence[i] for i in index], [self.size_px[i] for i in index])
        elif isinstance(index, slice):
            # form a new instance if the index is a slice
            return ListOfImageParameters(len(self.file_name[index]), self.file_name[index], self.jd[index],
                                         self.dr[index], self.dnu[index],
                                         self.presence[index], self.size_px[index])
        return self._img_params[index]

    def __len__(self):
        return len(self.file_name)

    def __iter__(self):
        return iter(self._img_params)


class EllipsoidRad(lc_func.Ellipsoid):
    dalphadbeta = math.pi / lc_func.Ellipsoid.N_ELLIPSOID_POINTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = 0

    @property
    def normals(self):
        if self._normals is None:
            self.calc_normals()
        return self._normals / EllipsoidRad.dalphadbeta

    @property
    def norm_areas(self):
        if self._norm_areas is None:
            self.calc_normals()
        return self._norm_areas * EllipsoidRad.dalphadbeta ** 2


def rotate(x, y, z, sinl, cosl, sinb, cosb, sinphi, cosphi):
    x1 = cosl * x + sinl * y
    y1 = -sinl * x + cosl * y
    z1 = z

    x2 = sinb * x1 - cosb * z1
    y2 = y1
    z2 = cosb * x1 + sinb * z1

    x3 = cosphi * x2 + sinphi * y2
    y3 = -sinphi * x2 + cosphi * y2
    z3 = z2

    return x3, y3, z3

def read_observed_radar_image(radar_image_list, number):
    img = image.imread("radar_data/" + radar_image_list.file_name[number])

    N_radar_pixels = img.shape[0]
    brightness_array = [[(1. * img[i][j][0] + 1. * img[i][j][1] + 1. * img[i][j][2]) / 3. / img[i][j][3] for j in
                         range(N_radar_pixels)] for i in range(N_radar_pixels)]

    return brightness_array

def visible(positions: np.array, direction: np.array, ellips, xc):
    """ Checks the condition if the ray intersects the ellipsoid.
     Ray has coordinates (x0+px*tau, y0+py*tau, z0+pz*tau), tau>0.
     Ellipsoid has the equation x^2/a^2+y^2/b^2+z^2/c^2 = 0.
     Assumes that the ray starts outside the ellipsoid (when tau=0).
     direction (np.array): The direction vector of the ray as a 3d vector
     position (np.array): The initial origin point of the ray as a 3d vector
     xc (float):
     """
    position = positions[0]/ ellips.a, positions[1] / ellips.b, positions[2] / ellips.c
    direction = direction[0] / ellips.a, direction[1] / ellips.b, direction[2] / ellips.c
    tau = ((direction[0] * xc
           - direction[0] * position[0]
           - direction[1] * position[1]
           - direction[2] * position[2]) /
           (direction[0] ** 2
            + direction[1] ** 2
            + direction[2] ** 2))

    dist_square = ((position[0] + direction[0] * tau - xc) ** 2
                   + (position[1] + direction[1] * tau) ** 2
                   + (position[2] + direction[2] * tau) ** 2)

    dist_zero = ((position[0] - xc) ** 2 + position[1] ** 2 + position[2] ** 2)
    results = np.zeros(np.shape(tau), dtype=bool)
    results[(dist_square > 1) | ((dist_zero > 1) & (tau < 0))] = True
    return results

def pixel_brightness_vec(body: lc_func.Body, i_vecs, pow_coeff: float=2, switch_places=False):
    """calculates pixel brightness"""
    if switch_places:
        reflecter = body.component2
        shader = body.component1
        distance = body.component2.distance
        shader_distance = body.component1.distance
    else:
        reflecter = body.component1
        shader = body.component2
        distance = body.component1.distance
        shader_distance = body.component2.distance

    # get surface coords, normal vectors and areas from the ellipsoid
    positions_nonshift = reflecter.surface
    # shift surface coords by distance between components (along x-axis)
    positions = np.array(positions_nonshift, copy=True)
    positions[0] += distance
    norm_vecs = reflecter.normals #/ dalphadbeta
    norm_areas = reflecter.norm_areas #* dalphadbeta ** 2
    mu_i = np.sum(norm_vecs * i_vecs[:, np.newaxis, np.newaxis], axis=0)
    mu_i = np.where(mu_i < 0, 0, mu_i)

    # mu_e = np.sum(norm_vecs * e_vecs[:, np.newaxis, np.newaxis], axis=0)
    # mu_e = np.where(mu_e < 0, 0, mu_e)

    view_i = visible(positions, i_vecs, shader, shader_distance / shader.a)
    # view_e = visible(positions, e_vecs, shader, shader_distance / shader.a)
    return norm_areas * mu_i**pow_coeff * view_i


def component_brightness(body: lc_func.Body, e_vecs, P, N_radar_pixels, dr, dnu, pow_coeff: float=2, switch_places=False):
    ## array to store brightness for all pixels
    brightness_arr = np.zeros((N_radar_pixels, N_radar_pixels))

    component = body.component2 if switch_places else body.component1
    pos = np.array(component.surface, copy=True)
    pos[0] += component.distance

    Ny = np.int_(0.5 + 0.5 * N_radar_pixels
                 - (pos[0] * e_vecs[0] + pos[1] * e_vecs[1] + pos[2] * e_vecs[2]) / dr / 2)
    Nx = np.int_(0.5 + 0.5 * N_radar_pixels
                 + 2 * NU0 / C_MS / dnu * 2 * np.pi / P / 84600 * (pos[0] * e_vecs[1] - pos[1] * e_vecs[0] ))
    additional_brightness_primary = pixel_brightness_vec(body, e_vecs, pow_coeff=pow_coeff, switch_places=switch_places)

    mask = (additional_brightness_primary > 0) & (Nx >= 0) & (Nx < N_radar_pixels) & (Ny >= 0) & (Ny < N_radar_pixels)
    np.add.at(brightness_arr, (Ny[mask], Nx[mask]), additional_brightness_primary[mask])
    return brightness_arr


def align_images(observed_img, theor_img) -> np.array:
    # shift theoretical image
    x_center_theor, y_center_theor = func.center_of_mass(theor_img)
    x_center_obs, y_center_obs = func.center_of_mass(observed_img - np.median(observed_img))
    shift_x = int(x_center_obs - x_center_theor)
    shift_y = int(y_center_obs - y_center_theor)
    theor_img_roll = np.roll(theor_img, (shift_x, shift_y), axis=(0, 1))
    return theor_img_roll


def create_body(parameters, n_points: int = 15) -> lc_func.Body:
    ps = parameters
    # create two components of the binary
    ellipsoid_R = EllipsoidRad
    ellipsoid_R.N_ELLIPSOID_POINTS = n_points
    ellipsoid_R.sinbeta, ellipsoid_R.cosbeta, ellipsoid_R.sinalpha, ellipsoid_R.cosalpha = ellipsoid_R.create_meshgrid(
        n_points)
    primary = ellipsoid_R(ps.a1, ps.b1, ps.c1)
    secondary = ellipsoid_R(ps.a2, ps.b2, ps.c2)
    primary.distance = -(ps.a1 + ps.a2) * ps.a2 * ps.b2 * ps.c2 / (ps.a1 * ps.b1 * ps.c1 + ps.a2 * ps.b2 * ps.c2)
    secondary.distance = (ps.a1 + ps.a2) * ps.a1 * ps.b1 * ps.c1 / (ps.a1 * ps.b1 * ps.c1 + ps.a2 * ps.b2 * ps.c2)
    body = lc_func.Body(component1=primary, component2=secondary,
                     eq_latitude=ps.ecl_latitude, eq_longitude=ps.ecl_longitude,
                     init_phase=ps.init_phase, distance=0, rotation_period=ps.rot_per)
    return body


def theor_radar_image_vec(body: lc_func.Body, e_vecs, P, N_radar_pixels, dr, dnu, pow_coeff):
    # array to store brightness for all pixels
    brightness_arr = component_brightness(body, e_vecs, P, N_radar_pixels, dr, dnu, pow_coeff)
    brightness_arr += component_brightness(body, e_vecs, P, N_radar_pixels, dr, dnu, pow_coeff, switch_places=True)
    return brightness_arr

def denoise_img(img: np.array) -> np.array:
    """denoises image"""
    img = (img * 255).astype(np.uint8)
    dst = cv2.fastNlMeansDenoising(img, 20,20,5,21) / 255
    dst = dst - np.quantile(dst, 0.1)
    dst[dst < 0] = 0
    dst = dst / np.max(dst)
    return dst

def saturate_img(radar_img, parameters):
    k = (1 - parameters.radar_background) / parameters.radar_saturation
    b = parameters.radar_background
    radar_img_sat = np.minimum(radar_img, parameters.radar_saturation)
    return k * radar_img_sat + b

def img_difference(observed, theoretical):
    return np.sum((observed - theoretical)**2, axis=None)

def radar_images_diff_rgb(params: tuple, radar_image_params: RadarImageParameters,
                radar_images: list, position_vecs, mult=400, lt_num=30, pow_coeff: float = 2):
    """
    calculates the theoretical radar images, compares them with obsertational data, returns the list of
    red/green overlay of observational and theoretical data, and a pixelwise difference between images
    """
    ps = Parameters(a1=params[0] * mult, b1=params[1] * mult, c1=params[2] * mult,
                    a2=params[3] * mult, b2=params[4] * mult, c2=params[5] * mult,
                    l=params[6], b=params[7], init_phase=params[8], rot_per=params[9], radar_sat=params[10])
    # number of points in latitude and longitude used in ray tracing
    cosl = math.cos(ps.ecl_longitude)
    sinl = math.sin(ps.ecl_longitude)
    cosb = math.cos(ps.ecl_latitude)
    sinb = math.sin(ps.ecl_latitude)

    body = create_body(parameters=ps, n_points=int(lt_num / 2))
    diff_total = []
    imgs = []
    # subplot(r,c) provide the no. of rows and columns
    for idx_img in range(radar_image_params.N_points):
        if radar_image_params.presence[idx_img] == 1:
            phi = 2 * math.pi * (radar_image_params.jd[idx_img] - 59185.77807) / ps.rot_per + ps.init_phase
            sinphi = math.sin(phi)
            cosphi = math.cos(phi)
            ex0 = position_vecs['x'].iloc[idx_img]
            ey0 = position_vecs['y'].iloc[idx_img]
            ez0 = position_vecs['z'].iloc[idx_img]
            e_vecs = np.array(rotate(ex0, ey0, ez0,
                                     sinl, cosl, sinb, cosb,
                                     sinphi, cosphi))
            observed_radar_brightness = radar_images[idx_img]
            # pad the observed radar image with 15 pixels on each side
            observed_radar_brightness = np.pad(observed_radar_brightness, 10, 'constant', constant_values=0)

            theor_radar_img = theor_radar_image_vec(body=body,
                                                    e_vecs=e_vecs,
                                                    P=ps.rot_per,
                                                    N_radar_pixels=observed_radar_brightness.shape[0],
                                                    dr=radar_image_params.dr[idx_img],
                                                    dnu=0.96,
                                                    pow_coeff=pow_coeff)
            theor_radar_img = saturate_img(theor_radar_img, ps) - 0.1

            # shift theoretical image
            theor_radar_img = align_images(observed_img=observed_radar_brightness,
                                           theor_img=theor_radar_img)
            # # transform grayscale theoretical image to rgb red filter
            # calculate difference between theoretical and observed images
            diff = img_difference(observed_radar_brightness, theor_radar_img)
            color_image = np.zeros((observed_radar_brightness.shape[0], observed_radar_brightness.shape[1], 3),
                                   dtype=np.uint8)
            color_image[:, :, 0] = theor_radar_img * 255
            color_image[:, :, 1] = observed_radar_brightness * 255
            color_image[:, :, 2] = observed_radar_brightness * 255
            imgs.append(color_image)
            diff_total.append(diff)

    return diff_total, imgs


def radar_images_diff(params: tuple, radar_image_params: RadarImageParameters,
                radar_images: list, position_vecs, mult=400, lt_num=30, pow_coeff: float = 2):
    """
    calculates the theoretical radar images, compares them with obsertational data, returns the list of
    red/green overlay of observational and theoretical data, and a pixelwise difference between images
    """
    ps = Parameters(a1=params[0] * mult, b1=params[1] * mult, c1=params[2] * mult,
                    a2=params[3] * mult, b2=params[4] * mult, c2=params[5] * mult,
                    l=params[6], b=params[7], init_phase=params[8], rot_per=params[9], radar_sat=params[10])
    # number of points in latitude and longitude used in ray tracing
    cosl = math.cos(ps.ecl_longitude)
    sinl = math.sin(ps.ecl_longitude)
    cosb = math.cos(ps.ecl_latitude)
    sinb = math.sin(ps.ecl_latitude)

    body = create_body(parameters=ps, n_points=int(lt_num / 2))
    diff_total = []
    imgs = []
    # subplot(r,c) provide the no. of rows and columns
    for idx_img in range(radar_image_params.N_points):
        if radar_image_params.presence[idx_img] == 1:
            phi = 2 * math.pi * (radar_image_params.jd[idx_img] - 59185.77807) / ps.rot_per + ps.init_phase
            sinphi = math.sin(phi)
            cosphi = math.cos(phi)
            ex0 = position_vecs['x'].iloc[idx_img]
            ey0 = position_vecs['y'].iloc[idx_img]
            ez0 = position_vecs['z'].iloc[idx_img]
            e_vecs = np.array(rotate(ex0, ey0, ez0,
                                     sinl, cosl, sinb, cosb,
                                     sinphi, cosphi))
            observed_radar_brightness = radar_images[idx_img]
            # pad the observed radar image with 15 pixels on each side
            observed_radar_brightness = np.pad(observed_radar_brightness, 10, 'constant', constant_values=0)

            theor_radar_img = theor_radar_image_vec(body=body,
                                                    e_vecs=e_vecs,
                                                    P=ps.rot_per,
                                                    N_radar_pixels=observed_radar_brightness.shape[0],
                                                    dr=radar_image_params.dr[idx_img],
                                                    dnu=0.96,
                                                    pow_coeff=pow_coeff)
            theor_radar_img = saturate_img(theor_radar_img, ps) - 0.1

            # shift theoretical image
            theor_radar_img = align_images(observed_img=observed_radar_brightness,
                                           theor_img=theor_radar_img)
            # # transform grayscale theoretical image to rgb red filter
            # calculate difference between theoretical and observed article_images
            diff = img_difference(observed_radar_brightness, theor_radar_img)
            imgs.append([theor_radar_img, observed_radar_brightness])
            diff_total.append(diff)

    return diff_total, imgs


def build_cornerplot(data: np.ndarray, best_params: list, labels: list, bins: int=20, figsize: tuple=(20, 20), dpi: int=200):
    """Builds a corner plot for the MCMC samples"""
    num_params = data.shape[1]
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 15
    # Create a figure with subplots
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize, dpi=dpi)
    bins_heat = bins
    bins_hist = bins
    # Variable to store the mappable object for the colorbar
    mappable = None
    # Plot the histograms and scatter plots
    for i in range(num_params):
        for j in range(num_params):
            if i < j:
                # Skip the upper triangle and diagonal
                axes[i, j].axis('off')
            elif i == j:
                # get the desired percentiles
                data_axis = data[:, i]
                p16 = np.percentile(data_axis, q=16)
                p84 = np.percentile(data_axis, q=84)
                median = np.median(data_axis)
                # data_axis = data_axis[(data_axis >= p16) & (data_axis <= p84)]
                # Diagonal: plot histograms
                axes[i, j].hist(data_axis, bins=bins_hist, color='gray', density=True)

                axes[i, j].axvline(median, color='black')
                axes[i, j].axvline(p16, color='black', linestyle='--')
                axes[i, j].axvline(p84, color='black', linestyle='--')
                # Add title with median and percentiles
                axes[i, j].title.set_text(r'%s=${%.2f}_{-%.2f}^{+%.2f}$' % (labels[i], round(best_params[i],2),
                                                                      round(best_params[i]-p16, 2), round(p84 - best_params[i], 2)))
            else:
                # Off-diagonal: plot scatter plots
                data_j = data[:, j]
                data_i = data[:, i]
                p16j = np.percentile(data_j, q=16)
                p84j = np.percentile(data_j, q=84)
                idx_j = np.argwhere((data_j >= p16j) & (data_j <= p84j))
                # get the indexes for the 16th and 84th percentiles
                # p16i = np.percentile(data_i, q=16)
                # p84i = np.percentile(data_i, q=84)
                # data_i = data_i[(data_i >= p16) & (data_i <= p84)]

                h = axes[i, j].hist2d(data_j[idx_j][:, 0], data_i[idx_j][:, 0], bins=bins_heat, cmap='viridis', norm=LogNorm())
                if mappable is None:
                    mappable = h[3]  # Store the mappable object for the colorbar
                #axes[i, j].set_xlabel(labels[j])

            # Remove labels for cleaner look
            if i < num_params - 1:
                axes[i, j].set_xticklabels([])
            if j > 0:
                axes[i, j].set_yticklabels([])

            # Add axis labels
            if i == num_params - 1:
                axes[i, j].set_xlabel(labels[j])
                axes[i, j].xaxis.set_major_locator(ticker.MaxNLocator(4))
                axes[i, j].xaxis.set_minor_locator(ticker.MaxNLocator(15))
            if j == 0:
                axes[i, j].set_ylabel(labels[i])
                axes[i, j].yaxis.set_major_locator(ticker.MaxNLocator(5))
                axes[i, j].yaxis.set_minor_locator(ticker.MaxNLocator(20))


    # Add a colorbar at the bottom for all subplots
    if mappable:
        cbar = fig.colorbar(mappable, ax=axes, orientation='horizontal', fraction=0.05, pad=-0.25, shrink=2.0)
        cbar.set_label('Counts')

    # Adjust layout
    plt.tight_layout(pad=0.1, h_pad=-0.4, w_pad=0.0)
    return fig, axes


def plot_radar_rgb_images(imgs, radar_image_params, rows=10, cols=7, pix_crop=10, dpi=100, name_save=None):
    """
    Plot radar images with MJD labels.

    Parameters:
    imgs (list): List of image pairs (obs, theor)
    radar_image_params (object): Object containing radar image parameters
    rows (int): Number of rows in the plot grid
    cols (int): Number of columns in the plot grid
    pix_crop (int): Number of pixels to crop from each edge of the image
    """
    fig, axarr = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), dpi=dpi)
    for idx_img in range(1, 71):
        obs_img = imgs[idx_img][0]
        theor_img = imgs[idx_img][1]
        rgb = np.zeros([*obs_img.shape, 3])
        rgb[:,:,0] = obs_img
        rgb[:,:, 1] = theor_img
        rgb[:,:, 2] = theor_img
        img_crop = rgb[pix_crop:-pix_crop, pix_crop:-pix_crop]
        axarr[(idx_img-1) // cols][(idx_img-1) % cols].imshow(img_crop, interpolation='nearest')
        img_shape = img_crop.shape
        axarr[(idx_img-1) // cols][(idx_img-1) % cols].text(int(img_shape[0] * 0.5), int(img_shape[1]*0.95),
                                                        f'MJD {round(radar_image_params.jd[idx_img], 3)}', color='white', fontsize=16)
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    if name_save is not None:
        plt.savefig(f'{name_save}', bbox_inches='tight', format='pdf')
    return fig


def plot_radar_gray_images(imgs, radar_image_params, num_rows=35, num_cols=2, figsize=(3, 3), dpi=100, crop=10, name_save=None):
    """
    Plot image differences with MJD labels.

    Parameters:
    imgs (list): List of image pairs (theor, obs)
    radar_image_params (object): Object containing radar image parameters
    num_rows (int): Number of rows in the plot grid
    num_cols (int): Number of columns in the plot grid
    figsize (tuple): Figure size (width, height)
    dpi (int): DPI of the plot
    crop (int): Number of pixels to crop from each edge of the image
    """
    fig, axarr = plt.subplots(num_rows, num_cols + 4, figsize=(figsize[0] * (num_cols+4), figsize[1] * num_rows), dpi=dpi)
    for idx_img in range(1, 71):
        idx_row = (idx_img - 1) // num_cols
        idx_col = (idx_img - 1) % num_cols
        if idx_col == 1:
            idx_col = 3
        img_theor = imgs[idx_img][0][crop:-crop, crop:-crop]
        img_obs = imgs[idx_img][1][crop:-crop, crop:-crop]
        img_diff = img_obs - img_theor
        img_diff_norm = (img_diff-np.min(img_diff))/(np.max(img_diff)-np.min(img_diff))
        axarr[idx_row][idx_col].imshow(img_obs, interpolation='nearest',  cmap='gray')
        axarr[idx_row][idx_col+1].imshow(img_theor, interpolation='nearest',  cmap='gray')
        axarr[idx_row][idx_col+2].imshow(img_diff_norm, interpolation='nearest',  cmap='gray_r')
        img_shape = np.shape(img_theor)
        axarr[idx_row][idx_col].text(
            int(img_shape[0] * 0.4), int(img_shape[1] * 0.95),
            f'MJD {round(radar_image_params.jd[idx_img], 3)}', color='white', fontsize=10)
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        # remove the axis ticks
        for i in range(3):
            axarr[idx_row][idx_col+i].set_xticks([])
            axarr[idx_row][idx_col+i].set_yticks([])
        # set the spacing between subplots
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
    if name_save is not None:
        plt.savefig(f'{name_save}', bbox_inches='tight', format='jpg')
    return fig


class CallbackFunctor:
    def __init__(self, obj_fun):
        self.best_fun_vals = [np.inf]
        self.best_sols = []
        self.num_calls = 0
        self.obj_fun = obj_fun

    def __call__(self, *args, **kwargs):
        fun_val = self.obj_fun(*args, **kwargs)
        self.num_calls += 1
        if 'verbose' in kwargs:
            print(self.num_calls, fun_val)
        if fun_val < self.best_fun_vals[-1]:
            self.best_sols.append(params)
            self.best_fun_vals.append(fun_val)
        return fun_val

    def save_sols(self, filename):
        sols = np.array([sol for sol in self.best_sols])
        np.savetxt(filename, sols)



def bootstrap_lightcurve(lightcurve, frac: float):
    return lightcurve.data.sample(frac=frac).sort_values(by='epoch')
def get_bootstrap_idxs(lightcurve): return lightcurve.data.index