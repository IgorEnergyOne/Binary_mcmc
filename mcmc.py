import math
import numpy as np
import functions as func
from data_structures import LightCurve, ObsData
import radar_functions as rf
import lc_functions as lc


def ln_wrapper(params: tuple, radar_image_list, radar_images: list, position_vecs,
               lightcurve: LightCurve, earth_data: ObsData, sun_data: ObsData):
    """wrapper that sums up the result of two separate processes"""
    chi2_radar = ln_posterior_radar(params, radar_image_list, radar_images, position_vecs)
    chi2_lc = ln_posterior(params, lightcurve, earth_data, sun_data)
    return 6 * chi2_radar + chi2_lc


def ln_posterior(params: tuple, lightcurve: LightCurve, earth_data: ObsData, sun_data: ObsData) -> float:
    """
    calculates chi2 for the modeled lightcurve with the given parameters
    """
    a1, b1, c1, a2, b2, c2, lon, lat, init_phase, rot_period = params[:10]
    # create two ellipsoids with specified parameters
    ellips1 = lc.Ellipsoid(a=a1, b=b1, c=c1)
    ellips2 = lc.Ellipsoid(a=a2, b=b2, c=c2)
    ellips1.distance = -(a1 + a2) * a2 * b2 * c2 / (a1 * b1 * c1 + a2 * b2 * c2)
    ellips2.distance = (a1 + a2) * a1 * b1 * c1 / (a1 * b1 * c1 + a2 * b2 * c2)
    # create body from two ellipsis
    body = lc.Body(ellips1, ellips2, distance=a1 + a2,
                     eq_latitude=lat,
                     eq_longitude=lon,
                     init_phase=init_phase, rotation_period=rot_period)
    # initialize
    theor_mag_data = ObsData()
    chi2 = 0
    # calculate theoretical lightcurve for every part of the lightcurve
    # compare with the observed, and calculate the nesessary shift up/down
    for part_idx, part_curve in enumerate(lightcurve):
        # transform vectors asteroid body-fixed frame coords
        vecs_sun_rot = lc.rotate_vector(body, sun_data[part_idx]).to_numpy()
        vecs_earth_rot = lc.rotate_vector(body, earth_data[part_idx]).to_numpy()
        brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)
        mag_theor = -2.5 * np.log10(brightness)
        # calculate residuals (mag - mag_theor for every partial lightcurve point)
        residuals = part_curve.data['mag'] - mag_theor
        # amount of vertical shift for the lightcurve
        shift = np.sum(residuals / part_curve.data['mag_err'] ** 2) / np.sum(part_curve.data['mag_err'] ** (-2))
        chi2_part = ((part_curve.data['mag'] - shift - mag_theor) ** 2 / part_curve.data['mag_err'] ** 2
                     * part_curve.data['weight'].iloc[0]).sum()
        part_curve.shift = shift
        theor_mag_data += mag_theor
        chi2 += float(chi2_part)

    lightcurve.shift_curves()
    # calculate chi-squared
    # chi2 = np.sum((lightcurve.joined['mag_shifted']
    #                - theor_mag_data.joined) ** 2 / lightcurve.joined['mag_err'] ** 2)
    return -0.5 * chi2


def ln_posterior_radar(params: tuple, radar_image_params: rf.RadarImageParameters, radar_images: list, position_vecs,
                       mult=450):
    """
    calculates chi2 for the modeled lightcurve with the given parameters
    """
    lt_num = 30
    ps = rf.Parameters(a1=params[0] * mult, b1=params[1] * mult, c1=params[2] * mult,
                    a2=params[3] * mult, b2=params[4] * mult, c2=params[5] * mult,
                    l=params[6], b=params[7], init_phase=params[8], rot_per=params[9], radar_sat=params[10],
                    pow_coeff=params[11])
    # number of points in latitude and longitude used in ray tracing
    cosl = math.cos(ps.ecl_longitude)
    sinl = math.sin(ps.ecl_longitude)
    cosb = math.cos(ps.ecl_latitude)
    sinb = math.sin(ps.ecl_latitude)
    # create two components of the binary
    body = rf.create_body(parameters=ps, n_points=int(lt_num / 2))
    diff_total = []
    # subplot(r,c) provide the no. of rows and columns
    for idx_img in range(radar_image_params.N_points):
        if radar_image_params.presence[idx_img] == 1:
            phi = 2 * math.pi * (radar_image_params.jd[idx_img] - 59185.77807) / ps.rot_per + ps.init_phase
            sinphi = math.sin(phi)
            cosphi = math.cos(phi)
            ex0 = position_vecs['x'].iloc[idx_img]
            ey0 = position_vecs['y'].iloc[idx_img]
            ez0 = position_vecs['z'].iloc[idx_img]
            e_vecs = np.array(rf.rotate(ex0, ey0, ez0,
                                     sinl, cosl, sinb, cosb,
                                     sinphi, cosphi))
            observed_radar_brightness = radar_images[idx_img]
            # pad the observed radar image with 15 pixels on each side
            observed_radar_brightness = np.pad(observed_radar_brightness, 10, 'constant', constant_values=0)

            theor_radar_img = rf.theor_radar_image_vec(body=body,
                                                    e_vecs=e_vecs,
                                                    P=ps.rot_per,
                                                    N_radar_pixels=observed_radar_brightness.shape[0],
                                                    dr=radar_image_params.dr[idx_img],
                                                    dnu=0.96, pow_coeff=ps.pow_coeff)
            theor_radar_img = rf.saturate_img(theor_radar_img, ps) - 0.1

            # shift theoretical image
            x_center_theor, y_center_theor = func.center_of_mass(theor_radar_img)
            x_center_obs, y_center_obs = func.center_of_mass(
                observed_radar_brightness - np.median(observed_radar_brightness))
            shift_x = x_center_obs - x_center_theor
            shift_y = y_center_obs - y_center_theor
            theor_radar_img = np.roll(theor_radar_img, int(shift_y), axis=1)
            theor_radar_img = np.roll(theor_radar_img, int(shift_x), axis=0)
            # calculate difference between theoretical and observed images
            diff = rf.img_difference(observed_radar_brightness, theor_radar_img)
            diff_total.append(diff)

    chi2 = np.sum(np.array(diff_total))
    return -0.5 * chi2


def ln_prior(params: tuple, bounds: tuple=None) -> float:
    """constrains parameters to be in bounds, returns -inf if not in bounds,
    does not allow MCMC walkers to wander off"""
    if bounds is None:
        bounds = [[0.9, 2.1], [0.8, 1.2], [0.5, 1.0],
                  [0.45, 1.4], [0.57, 1.1], [0.2, 1.1],
                  [0, 2 * math.pi], [-2.1, -0.6], [0, 2 * math.pi],
                  [5.022 * 0.9 / 24, 5.022 * 1.1 / 24], [1000, 3000],
                  [0.1, 10]]
    a1, b1, c1, a2, b2, c2, lon, lat, init_phase, rot_period, radar_saturation, pow_coeff = params
    # if any parameter is out of bounds, return -inf
    a1_cond = 0.9 <= a1 <= 2.1
    b1_cond = (0.8 <= b1 <= 1.2) & (b1 <= a1)
    c1_cond = (0.5 <= c1 <= 1.0) & (c1 <= b1)
    a2_cond = (0.45 <= a2 <= 1.4) & (a2 <= a1)
    b2_cond = (0.57 <= b2 <= 1.1) & (b2 <= a2)
    c2_cond = (0.2 <= c2 <= 1.1) & (c2 <= b2)
    lon_cond = 0 <= lon <= 2 * math.pi
    lat_cond = -2.1 <= lat <= -0.6
    init_phase_cond = 0 <= init_phase <= 2 * math.pi
    rad_sat_cond = 1000 <= radar_saturation <= 3000
    rot_per_cond = 5.022 * 0.9 / 24 <= rot_period <= 5.022 * 1.1 / 24
    pow_coeff_cond = 0.1 <= pow_coeff <= 10
    if (a1_cond and b1_cond and c1_cond
            and a2_cond and b2_cond and c2_cond
            and lon_cond and lat_cond and init_phase_cond
            and rot_per_cond and rad_sat_cond and pow_coeff_cond):
        return 0
    # if not, return -inf (this ensures walkers does not walk out of bounds)
    return -np.inf


def calc_chi2(obs_curve, theor_curve) -> float:
    """calculate chi squared metric between the observational and theoretical lightcurve"""
    return np.sum((obs_curve['mag_shifted'] - theor_curve) ** 2 / obs_curve['mag_err'] ** 2)


def ln_prob(params: tuple, lightcurve: LightCurve, earth_data: ObsData, sun_data: ObsData) -> float:
    """wrapper for ln posterior, returns chi2 if in bounds, else -inf"""
    # check if parameters are in bounds
    lp = ln_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_posterior(params, lightcurve, earth_data, sun_data)


def ln_prob_both(params: tuple, radar_image_list, radar_images: list, position_vecs,
                 lightcurve: LightCurve, earth_data: ObsData, sun_data: ObsData, params_min, params_max):
    """wrapper for ln posterior, returns chi2 if in bounds, else -inf"""
    # function with the limits
    params = params * (params_max - params_min) + params_min
    ln_p = ln_prior(params)
    if not np.isfinite(ln_p):
        return -np.inf
    return ln_p + ln_wrapper(params, radar_image_list, radar_images, position_vecs,
                             lightcurve, earth_data, sun_data)

def ln_prob_radar(params: tuple, radar_image_list, radar_images: list, position_vecs):
    """wrapper for ln posterior, returns chi2 if in bounds, else -inf"""
    ln_p = ln_prior(params)
    if not np.isfinite(ln_p):
        return -np.inf
    return ln_p + ln_posterior_radar(params, radar_image_list, radar_images, position_vecs)



def theor_curve(params: tuple, earth_data: ObsData, sun_data: ObsData) -> ObsData:
    """builds theoretical lightcurve for the given parameters"""
    a1, b1, c1, a2, b2, c2, lon, lat, init_phase, rot_period = params
    # create two ellipsoids with specified parameters
    ellips1 = lc.Ellipsoid(a=a1, b=b1, c=c1)
    ellips2 = lc.Ellipsoid(a=a2, b=b2, c=c2)
    ellips1.distance = -(a1 + a2) * a2 * b2 * c2 / (a1 * b1 * c1 + a2 * b2 * c2)
    ellips2.distance = (a1 + a2) * a1 * b1 * c1 / (a1 * b1 * c1 + a2 * b2 * c2)
    # create body from two ellipsis
    body = lc.Body(ellips1, ellips2, distance=a1 + a2,
                     eq_latitude=lat,
                     eq_longitude=lon,
                     init_phase=init_phase, rotation_period=rot_period)
    # create object to store theoretical curve data
    theor_mag_data = ObsData()
    # calculate theoretical lightcurve for every part of the lightcurve
    # compare with the observed, and calculate the necessary shift up/down
    for part_idx, part_curve in enumerate(earth_data):
        # transform vectors asteroid body-fixed frame coords
        vecs_sun_rot = lc.rotate_vector(body, sun_data[part_idx]).to_numpy()
        vecs_earth_rot = lc.rotate_vector(body, earth_data[part_idx]).to_numpy()
        brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)
        # transform to magnitudes
        mag_theor = -2.5 * np.log10(brightness)
        theor_mag_data += mag_theor
    return theor_mag_data
