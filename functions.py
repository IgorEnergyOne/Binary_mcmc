import numpy as np
import math
import pandas as pd


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
        rot_per (float) - rotational period of the binary [h]
    Returns:
        density (float): Estimated density of the binary asteroid [g/cm^3].
    """
    G = 6.67408e-11
    density = 3 * np.pi * (a1 + a2) ** 3 / (G * (rot_per * 60 * 60 * 24) ** 2 * (a1 * b1 * c1 + a2 * b2 * c2))
    return density


def normalize_vector(vecs: pd.DataFrame):
    """normalizes given vectors"""
    # Calculate the magnitude of each vector (along axis 1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Replace any zero norms with one to avoid division by zero
    norms[norms == 0] = 1
    # Normalize the vectors
    vecs_normalized = vecs / norms
    return vecs_normalized


def interpolate_vector(vecs: pd.DataFrame, time: float):
    """Computes unit vector directed from asteroid to Earth at given time
       by linear interpolation between the data from JPL horizons."""
    # check if requested time is in range of existing data
    if (time < vecs.reduc_epoch.min()) or (time > vecs.reduc_epoch.max()):
        raise ValueError("Time in earth_vector out of range")

    # get the closest number in table
    n = math.floor((time - vecs.reduc_epoch.iloc[0]) /
                   (vecs.reduc_epoch.iloc[1] - vecs.reduc_epoch.iloc[0]))
    # weight of the lower number
    f = (time - vecs.reduc_epoch.iloc[n]) / (vecs.reduc_epoch.iloc[1] - vecs.reduc_epoch.iloc[0])
    x = (1 - f) * vecs.x.iloc[n] + f * vecs.x.iloc[n + 1]
    y = (1 - f) * vecs.y.iloc[n] + f * vecs.y.iloc[n + 1]
    z = (1 - f) * vecs.z.iloc[n] + f * vecs.z.iloc[n + 1]
    s = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return -x / s, -y / s, -z / s


def compute_phase_angle(vecs_sun: pd.DataFrame,
                        vecs_earth: pd.DataFrame) -> pd.DataFrame:
    """"""
    return np.arccos(vecs_sun.x * vecs_earth.x
                     + vecs_sun.y * vecs_earth.y
                     + vecs_sun.z * vecs_earth.z)


class Ellipsoid:
    """
    Class that represents component of a binary asteroid with an ellipsoid
    """
    # number of points of the surface per 90 degrees
    N_ELLIPSOID_POINTS = 10

    @staticmethod
    def create_meshgrid(n_res: int):
        """Create a 2D grid for beta and alpha angles (build a grid of spherical coordinates)"""
        # set angles of the points for the grid of the ellipsoid
        # beta - latitude, alpha - longitude
        beta = np.pi * (np.arange(-n_res, n_res) + 0.5) / (2 * n_res)
        alpha = np.pi * np.arange(4 * n_res) / (2 * n_res)
        # create grid of spherical coordinates [a * n_res, b * n_res]
        beta_grid, alpha_grid = np.meshgrid(beta, alpha, indexing='ij')
        # Precompute the sine and cosine for the grid angles
        sinbeta = np.sin(beta_grid)
        cosbeta = np.cos(beta_grid)
        sinalpha = np.sin(alpha_grid)
        cosalpha = np.cos(alpha_grid)
        return sinbeta, cosbeta, sinalpha, cosalpha

    # set them as attributes of the class (are the same for every class instance)
    sinbeta, cosbeta, sinalpha, cosalpha = create_meshgrid(N_ELLIPSOID_POINTS)

    def __init__(self, a: float, b: float, c: float) -> None:
        """Creates an ellipsoid which represented by its semi-axes"""
        # sets semi-major axis of ellipsoid
        self.a = a
        self.b = b
        self.c = c
        self._coords_surface = None
        self._normals = None
        self._norm_areas = None

    def __str__(self):
        """Returns a string with ellipsoid parameters"""
        return f"Ellipsis with axes: a={self.a}, b={self.b}, c={self.c}"

    def calc_surf_pos(self):
        """calculates position x,y,z coords of surface points of the ellipsoid"""
        # coords of surface points of the ellipsoid [[x, y, z], [a * n_res], [b * n_res]] dims
        positions_x = self.a * self.cosbeta * self.cosalpha  # x for grid
        positions_y = self.b * self.cosbeta * self.sinalpha  # y for grid
        positions_z = self.c * self.sinbeta  # z for grid
        self._coords_surface = np.array([positions_x, positions_y, positions_z])

    def calc_normals(self):
        """calculates normal vectors and their areas for the ellipsoid
        """
        # orientations of normal vectors for each point on the ellipsoid [[x, y, z], [a * n_res], [b * n_res]] dims
        sx = self.b * self.c * self.cosbeta ** 2 * self.cosalpha  # x for grid
        sy = self.a * self.c * self.cosbeta ** 2 * self.sinalpha  # y for grid
        sz = self.a * self.b * self.cosbeta * self.sinbeta  # z for grid
        # calculate area
        self._norm_areas = (sx ** 2 + sy ** 2 + sz ** 2) ** 0.5 / (math.pi / (2 * self.N_ELLIPSOID_POINTS)) ** 2
        # normalize
        self._normals = np.array([sx, sy, sz]) / self._norm_areas

    @property
    def normals(self):
        if self._normals is None:
            self.calc_normals()
        return self._normals

    @property
    def surface(self):
        if self._coords_surface is None:
            self.calc_surf_pos()
        return self._coords_surface

    @property
    def norm_areas(self):
        if self._norm_areas is None:
            self.calc_normals()
        return self._norm_areas

    @staticmethod
    def visible(position: np.array, direction: np.array, shadower):
        """ Checks the condition if the ray intersects the ellipsoid.
         Ray has coordinates (x0+px*tau, y0+py*tau, z0+pz*tau), tau>0.
         Ellipsoid has the equation x^2/a^2+y^2/b^2+z^2/c^2 = 0.
         Assumes that the ray starts outside the ellipsoid (when tau=0).
         direction (np.array): The direction vector of the ray as a 3d vector
         position (np.array): The initial origin point of the ray as a 3d vector
         """
        tau = (-(direction[0] * position[0] / shadower.a ** 2
                 + direction[1] * position[1] / shadower.b ** 2
                 + direction[2] * position[2] / shadower.c ** 2)
               /
               (direction[0] ** 2 / shadower.a ** 2
                + direction[1] ** 2 / shadower.b ** 2
                + direction[2] ** 2 / shadower.c ** 2))

        # compute the squared distance
        dist_square = ((direction[0] * tau + position[0]) ** 2 / shadower.a ** 2
                       + (direction[1] * tau + position[1]) ** 2 / shadower.b ** 2
                       + (direction[2] * tau + position[2]) ** 2 / shadower.c ** 2)

        dist_zero = (position[0] ** 2 / shadower.a ** 2
                     + position[1] ** 2 / shadower.b ** 2
                     + position[2] ** 2 / shadower.c ** 2)

        results = np.zeros(np.shape(tau), dtype=bool)
        # ray is tangent to the ellipsoid
        # results[(tau < 0)] = True
        # # #results[(tau > 0) & (condition < 1)] = False
        # results[(tau > 0) & (dist_square >= 1)] = True
        results[(dist_square > 1) | ((dist_zero > 1) & (tau < 0))] = True
        return results


class Body:
    """Binary asteroid represented by two components-ellipsoids"""

    def __init__(self, component1: Ellipsoid, component2: Ellipsoid, eq_latitude: float,
                 eq_longitude: float, init_phase: float, rotation_period: float,
                 distance: float = None,
                 ):
        """
        :param component1: first component of a binary represented by an ellipsoid
        :param component2: second component of a binary represented by an ellipsoid
        :param eq_latitude: ecliptic latitude of spin axis (beta) [deg]
        :param eq_longitude: ecliptic longitude of spin axis (lambda) [deg]
        :param init_phase: initial rotation phase [deg]
        :param rotation_period: sidereal rotation period [days]
        :param distance: distances between components
        """
        self.component1 = component1
        self.component2 = component2
        self.ecl_latitude = eq_latitude  # beta
        self.ecl_longitude = eq_longitude  # lambda
        self.init_phase = init_phase
        self.distance = distance
        self.rot_period = rotation_period

    def __str__(self):
        """string representation of a body"""
        return (f"Body consists of:\n{self.component1},\n{self.component2},\nlat={np.degrees(self.ecl_latitude)} deg, "
                f"lon={np.degrees(self.ecl_longitude)} deg, phase={self.init_phase}, distance={self.distance}")

    def component_brightness(self, sun_vecs: np.ndarray,
                             earth_vecs: np.ndarray, switch_places=False):
        """calculates total brightness of a single component of a binary asteroid for
        each instance of vectors"""
        if switch_places:
            component = self.component1
            shadower = self.component2
            distance = self.distance
        else:
            component = self.component2
            shadower = self.component1
            distance = -self.distance

        # get surface coords, normal vectors and areas from the ellipsoid
        positions_nonshift = component.surface
        # shift surface coords by distance between components (along x-axis)
        positions = np.array(positions_nonshift, copy=True)
        positions[0] += distance
        norm_vecs = component.normals
        norm_areas = component.norm_areas
        # array to store brightness for every epoch
        bright_arr = np.zeros(len(sun_vecs))
        # for every point in vectors array:
        for i in range(len(sun_vecs)):
            # check if the component is visible (shaded by another)
            visible_comp1 = component.visible(positions, sun_vecs[i], shadower)
            visible_comp2 = component.visible(positions, earth_vecs[i], shadower)
            # select where visible from both
            both_visible = visible_comp1 & visible_comp2
            # if visible, calculate overall brightness
            norm_trans = np.transpose(norm_vecs, axes=[1, 2, 0])[both_visible]
            scat_law = scattering_law(sun_vecs[i],
                                      earth_vecs[i],
                                      norm_trans)
            # set brightness for every requested epoch
            # this is total brightness of asteroid for one epoch
            bright_arr[i] = np.sum(norm_areas[both_visible] * scat_law)
        return bright_arr

    def total_brightness(self, sun_vecs: np.ndarray, earth_vecs: np.ndarray):
        """calculates total brightness of a single component of a binary asteroid for
        each instance of vectors"""
        total_brightness = self.component_brightness(sun_vecs,
                                                     earth_vecs,
                                                     switch_places=False) \
                           + self.component_brightness(sun_vecs,
                                                       earth_vecs,
                                                       switch_places=True)
        return total_brightness


def rotate_vector(body: Body, vecs: pd.DataFrame):
    """Rotates the positional vectors of the body from the ecliptic coordinate system
    to be in the in asteroid body-fixed frame coords"""
    sin_lon = np.sin(body.ecl_longitude)  # lambda (longitude)
    cos_lon = np.cos(body.ecl_longitude)
    sin_lat = np.sin(np.pi / 2 - body.ecl_latitude)  # beta (latitude)
    cos_lat = np.cos(np.pi / 2 - body.ecl_latitude)
    sin_phi = np.sin(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)
    cos_phi = np.cos(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)

    # rotate vectors by components
    x1 = cos_lon * vecs.x + sin_lon * vecs.y
    y1 = -sin_lon * vecs.x + cos_lon * vecs.y
    z1 = vecs.z

    x2 = cos_lat * x1 - sin_lat * z1
    y2 = y1
    z2 = sin_lat * x1 + cos_lat * z1

    x3 = cos_phi * x2 + sin_phi * y2
    y3 = -sin_phi * x2 + cos_phi * y2
    z3 = z2

    return pd.DataFrame({'x': x3, 'y': y3, 'z': z3})


def rotate_vector_old(body: Body, vecs: pd.DataFrame):
    """Rotates the positional vectors of the body from the ecliptic coordinate system
    to be in the in asteroid body-fixed frame coords"""
    sin_lon = np.sin(body.ecl_longitude)  # lambda (longitude)
    cos_lon = np.cos(body.ecl_longitude)
    sin_lat = np.sin(np.pi / 2 - body.ecl_latitude)  # beta (latitude)
    cos_lat = np.cos(np.pi / 2 - body.ecl_latitude)
    sin_phi = np.sin(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)
    cos_phi = np.cos(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)

    # rotate vectors by components
    x1 = cos_lon * vecs.x + sin_lon * vecs.y
    y1 = -sin_lon * vecs.x + cos_lon * vecs.y
    z1 = vecs.z

    x2 = sin_lat * x1 - cos_lat * z1
    y2 = y1
    z2 = cos_lat * x1 + sin_lat * z1

    x3 = cos_phi * x2 + sin_phi * y2
    y3 = -sin_phi * x2 + cos_phi * y2
    z3 = z2

    return pd.DataFrame({'x': x3, 'y': y3, 'z': z3})


def rotate_vector_mat(body: Body, vecs: pd.DataFrame):
    sin_lon = np.sin(body.ecl_longitude)  # lambda
    cos_lon = np.cos(body.ecl_longitude)
    sin_lat = np.sin(90. - body.ecl_latitude)  # beta
    cos_lat = np.cos(90. - body.ecl_latitude)
    sin_phi = np.sin(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)
    cos_phi = np.cos(2 * np.pi * vecs.reduc_epoch / body.rot_period + body.init_phase)

    # Construct the individual rotation matrices
    lon_mat = np.array([[cos_lon, sin_lon, 0],
                        [-sin_lon, cos_lon, 0],
                        [0, 0, 1]])

    lat_mat = np.array([[cos_lat, 0, -sin_lat],
                        [0, 1, 0],
                        [sin_lat, 0, cos_lat]])

    # Construct the phase rotation matrix for each vector
    phi_mat = np.array([[cos_phi, sin_phi, 0],
                        [-sin_phi, cos_phi, 0],
                        [0, 0, 1]])

    # Compute the total rotation matrix
    vecs_rotated = vecs[['x', 'y', 'z']] @ lat_mat @ lon_mat @ phi_mat
    return vecs_rotated


def visible(position: np.array, direction: np.array, ellips: Ellipsoid):
    """ Checks the condition if the ray intersects the ellipsoid.
     Ray has coordinates (x0+px*tau, y0+py*tau, z0+pz*tau), tau>0.
     Ellipsoid has the equation x^2/a^2+y^2/b^2+z^2/c^2 = 0.
     Assumes that the ray starts outside the ellipsoid (when tau=0).
     direction (np.array): The direction vector of the ray as a 3d vector
     position (np.array): The initial origin point of the ray as a 3d vector
     """
    tau = (-(direction[0] * position[0] / ellips.a ** 2
             + direction[1] * position[1] / ellips.b ** 2
             + direction[2] * position[2] / ellips.c ** 2)
           /
           (direction[0] ** 2 / ellips.a ** 2
            + direction[1] ** 2 / ellips.b ** 2
            + direction[2] ** 2 / ellips.c ** 2))

    # compute the squared distance
    dist_square = ((direction[0] * tau + position[0]) ** 2 / ellips.a ** 2
                   + (direction[1] * tau + position[1]) ** 2 / ellips.b ** 2
                   + (direction[2] * tau + position[2]) ** 2 / ellips.c ** 2)

    dist_zero = (position[0] ** 2 / ellips.a ** 2
                 + position[1] ** 2 / ellips.b ** 2
                 + position[2] ** 2 / ellips.c ** 2)

    results = np.zeros(np.shape(tau), dtype=bool)
    results[(dist_square > 1) | ((dist_zero > 1) & (tau < 0))] = True
    return results


def scattering_law(sun_vecs: np.ndarray,
                   earth_vecs: np.ndarray,
                   norm_vecs: np.ndarray):
    """Calculates the contribution into total brightness
    using Lambert cosine scattering law"""
    p1 = np.sum(sun_vecs * norm_vecs, axis=1)
    p2 = np.sum(earth_vecs * norm_vecs, axis=1)
    # return values where p1 and p2 are positive (and their product)
    # otherwise return 0
    res = p1 * p2
    res[(p1 < 0) | (p2 < 0)] = 0
    return res


def component_brightness(body: Body, sun_vecs_rot: pd.DataFrame,
                         earth_vecs_rot: pd.DataFrame, switch_places=False):
    """calculates total brightness of a single component of a binary asteroid for
    each instance of vectors"""
    if switch_places:
        component = body.component1
        shadower = body.component2
        distance = body.distance
    else:
        component = body.component2
        shadower = body.component1
        distance = -body.distance

    # get surface coords, normal vectors and areas from the ellipsoid
    positions_nonshift = component.surface
    # shift surface coords by distance between components (along x-axis)
    positions = np.array(positions_nonshift, copy=True)
    positions[0] += distance
    norm_vecs = component.normals
    norm_areas = component.norm_areas
    # array to store brightness for every epoch
    bright_arr = np.zeros(len(sun_vecs_rot))
    vis_data = []
    # for every point in vectors array:
    for i in range(len(sun_vecs_rot)):
        # check if the component is visible (shaded by another)
        visible_comp1 = visible(positions, sun_vecs_rot[i], shadower)
        visible_comp2 = visible(positions, earth_vecs_rot[i], shadower)
        # select where visible from both
        both_visible = visible_comp1 & visible_comp2
        # if visible, calculate overall brightness
        norm_trans = np.transpose(norm_vecs, axes=[1, 2, 0])[both_visible]
        scat_law = scattering_law(sun_vecs_rot[i],
                                  earth_vecs_rot[i],
                                  norm_trans)
        # set brightness for every requested epoch
        # this is total brightness of asteroid for one epoch
        bright_arr[i] = np.sum(norm_areas[both_visible] * scat_law)
        # add data for debugging
        vis_data.append([visible_comp1, visible_comp2, both_visible, scat_law])
    return bright_arr, vis_data




