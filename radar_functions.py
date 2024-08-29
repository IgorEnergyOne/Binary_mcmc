import math
import numpy as np
import matplotlib.image as image

# constants
NU0 = 8560000000.
C_MS = 299792458.

class Parameters:
    def __init__  (self, a1=1, b1=1, c1=1, a2=1, b2=1, c2=1, rot_per=5.022/24, l=0, b=0, phi0=0):
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.a2 = a2
        self.b2 = b2
        self.c2 = c2
        self.P = rot_per  # rotational period [h]
        self.ecl_longitude = l  # longitude of the pole
        self.ecl_latitude = b  # latitude of the pole
        self.init_phase = phi0
        self.radar_saturation = 500.
        self.radar_background = 0.1
        self.radar_shift1 = 13
        self.radar_shift2 = 2


class RadarImageParameters:
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

        # self.ix0 = [50906164.4967212 for i in range(self.N_points)]
        # self.iy0 = [148609342.022884 for i in range(self.N_points)]
        # self.iz0 = [537223.181002833 for i in range(self.N_points)]
        # i0 = [(self.ix0[i] ** 2 + self.iy0[i] ** 2 + self.iz0[i] ** 2) ** 0.5 for i in range(self.N_points)]
        # self.ix0 = [self.ix0[i] / i0[i] for i in range(self.N_points)]
        # self.iy0 = [self.iz0[i] / i0[i] for i in range(self.N_points)]
        # self.iz0 = [self.ix0[i] / i0[i] for i in range(self.N_points)]
        #
        # self.ex0 = [50906164.4967212 for i in range(self.N_points)]
        # self.ey0 = [148609342.022884 for i in range(self.N_points)]
        # self.ez0 = [537223.181002833 for i in range(self.N_points)]
        # e0 = [(self.ex0[i] ** 2 + self.ey0[i] ** 2 + self.ez0[i] ** 2) ** 0.5 for i in range(self.N_points)]
        # self.ex0 = [self.ex0[i] / e0[i] for i in range(self.N_points)]
        # self.ey0 = [self.ez0[i] / e0[i] for i in range(self.N_points)]
        # self.ez0 = [self.ex0[i] / e0[i] for i in range(self.N_points)]

        radar_file.close()


class EllipsoidR:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


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

def visible(x,y,z,nx,ny,nz,xc):
    t = (nx*xc-nx*x-ny*y-nz*z)/(nx**2+ny**2+nz**2)
    r_min2 = (x+nx*t-xc)**2+(y+ny*t)**2+(z+nz*t)**2
    r_02 = (x-xc)**2+y**2+z**2
    return ((r_min2>1)or((t<0)and(r_02>1)))


def img_difference(observed, theoretical):
    return np.sum((observed - theoretical)**2, axis=None)

def saturate_theoretical_radar_image(brightness_list, params: Parameters):
    N_radar_pixels = len(brightness_list)
    k = (1 - params.radar_background) / params.radar_saturation
    b = params.radar_background
    for i in range(N_radar_pixels):
        for j in range(N_radar_pixels):
            if brightness_list[i][j] > params.radar_saturation:
                brightness_list[i][j] = params.radar_saturation
            brightness_list[i][j] = k*brightness_list[i][j] + b
    return brightness_list


def pixel_brightness(reflecter, shader, ix, iy, iz, ex, ey, ez,
                     sinalpha, cosalpha, sinbeta, cosbeta, dalphadbeta):
    """calculates pixel brightness"""
    x0 = reflecter.a * cosalpha * cosbeta + reflecter.d
    y0 = reflecter.b * sinalpha * cosbeta
    z0 = reflecter.c * sinbeta

    sx = reflecter.b * reflecter.c * cosalpha * cosbeta ** 2 * dalphadbeta
    sy = reflecter.a * reflecter.c * sinalpha * cosbeta ** 2 * dalphadbeta
    sz = reflecter.a * reflecter.b * sinbeta * cosbeta * dalphadbeta

    s = (sx ** 2 + sy ** 2 + sz ** 2) ** 0.5

    nx = sx / s
    ny = sy / s
    nz = sz / s
    mu_i = nx * ix + ny * iy + nz * iz
    if mu_i < 0:
        mu_i = 0

    mu_e = nx * ex + ny * ey + nz * ez
    if mu_e < 0:
        mu_e = 0

    wiew_i = visible(x0 / shader.a, y0 / shader.b, z0 / shader.c, ix / shader.a, iy / shader.b, iz / shader.c,
                     shader.d / shader.a)
    wiew_e = visible(x0 / shader.a, y0 / shader.b, z0 / shader.c, ex / shader.a, ey / shader.b, ez / shader.c,
                     shader.d / shader.a)

    return s * mu_i * mu_e * wiew_i * wiew_e


def theoretical_radar_image(primary, secondary, ex, ey, ez, P, N_radar_pixels, dr, dnu, Nalpha=50):
    """"""
    brightness_list = [[0. for i in range(N_radar_pixels)] for j in range(N_radar_pixels)]
    dalphadbeta = math.pi ** 2 / Nalpha ** 2
    for j in range(Nalpha):
        beta = math.pi * ((j + 0.5) / Nalpha - 0.5)
        sinbeta = math.sin(beta)
        cosbeta = math.cos(beta)
        for k in range(2 * Nalpha):
            alpha = math.pi * k / Nalpha
            sinalpha = math.sin(alpha)
            cosalpha = math.cos(alpha)

            x0 = primary.a * cosalpha * cosbeta + primary.d
            y0 = primary.b * sinalpha * cosbeta
            z0 = primary.c * sinbeta

            Ny = int(0.5 + 0.5 * N_radar_pixels - (x0 * ex + y0 * ey + z0 * ez) / dr / 2)
            Nx = int(0.5 + 0.5 * N_radar_pixels + 2 * NU0 / C_MS / dnu * 2 * math.pi / P / 84600 * (x0 * ey - y0 * ex))
            additional_brightness = pixel_brightness(primary, secondary, ex, ey, ez, ex, ey, ez, sinalpha, cosalpha,
                                                     sinbeta, cosbeta, dalphadbeta)
            if additional_brightness > 0 and 0 <= Nx < N_radar_pixels and 0 <= Ny < N_radar_pixels:
                brightness_list[Ny][Nx] += additional_brightness

            x0 = secondary.a * cosalpha * cosbeta + secondary.d
            y0 = secondary.b * sinalpha * cosbeta
            z0 = secondary.c * sinbeta

            Ny = int(0.5 + 0.5 * N_radar_pixels - (x0 * ex + y0 * ey + z0 * ez) / dr / 2)
            Nx = int(0.5 + 0.5 * N_radar_pixels + 2 * NU0 / C_MS / dnu * 2 * math.pi / P / 84600 * (x0 * ey - y0 * ex))
            additional_brightness = pixel_brightness(secondary, primary, ex, ey, ez, ex, ey, ez, sinalpha, cosalpha,
                                                     sinbeta, cosbeta, dalphadbeta)
            if additional_brightness > 0 and 0 <= Nx < N_radar_pixels and 0 <= Ny < N_radar_pixels:
                brightness_list[Ny][Nx] += additional_brightness

    return brightness_list