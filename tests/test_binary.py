import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as func
import pytest


def test_same_position():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    ellips1 = func.Ellipsoid(a=2, b=1, c=1)
    ellips2 = func.Ellipsoid(a=1, b=1, c=1)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=3,
                     eq_latitude=0,
                     eq_longitude=np.pi / 2,
                     init_phase=0., rotation_period=5.0 / 24)
    # same position for earth and sun
    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun[:, 0] = 1
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)
    assert np.equal(total_brightness, np.loadtxt('../test_data/test_1.txt')).all()


def test_opposite_position():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    ellips1 = func.Ellipsoid(a=2, b=1, c=1)
    ellips2 = func.Ellipsoid(a=1, b=1, c=1)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=3,
                     eq_latitude=0,
                     eq_longitude=np.pi / 2,
                     init_phase=0., rotation_period=5.0 / 24)

    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun[:, 0] = -1
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])
    state_vecs_earth['x'] = 1

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)
    assert np.equal(total_brightness, 0).all()


def test_constant_brightness():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch
    # create two ellipsoids with specified parameters
    ellips1 = func.Ellipsoid(a=2, b=1, c=1)
    ellips2 = func.Ellipsoid(a=1, b=1, c=1)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=3,
                     eq_latitude=np.pi / 2,
                     eq_longitude=0,
                     init_phase=0., rotation_period=5.0/24)


    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['z'] = -1
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_sun
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)

    assert (total_brightness - 6.7 < 0.1).all()


def test_constant_brightness2():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    # create two ellipsoids with specified parameters
    ellips1 = func.Ellipsoid(a=2, b=1, c=1)
    ellips2 = func.Ellipsoid(a=1, b=1, c=1)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=3,
                     eq_latitude=np.pi / 4,
                     eq_longitude=np.pi / 4,
                     init_phase=0., rotation_period=5.0 / 24)

    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['x'] = 1 / 2
    state_vecs_sun['y'] = 1 / 2
    state_vecs_sun['z'] = 1 / (2 ** 0.5)
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)

    assert (total_brightness - 6.7 < 0.1).all()


def test_small_component():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    ellips1 = func.Ellipsoid(a=1, b=1, c=1)
    ellips2 = func.Ellipsoid(a=0.001, b=0.001, c=0.001)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=3,
                     eq_latitude=np.pi / 4,
                     eq_longitude=np.pi / 4,
                     init_phase=0., rotation_period=5.0 / 24)

    # same position for earth and sun
    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun[:, 0] = -1
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)

    assert (total_brightness - 2.096 < 0.1).all()


def test_separated_components():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    # create two ellipsoids with specified parameters
    ellips1 = func.Ellipsoid(a=1, b=1, c=1)
    ellips2 = func.Ellipsoid(a=1, b=1, c=1)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=100,
                     eq_latitude=np.pi / 4,
                     eq_longitude=np.pi / 4,
                     init_phase=0., rotation_period=5.0 / 24)

    # same position for earth and sun
    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun[:, 0] = 1
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)

    assert (total_brightness - 4.19 < 0.1).all()


def test_component_inside():
    # generate requested epochs for the query
    start_epoch = 2459180.5
    epochs = start_epoch + np.arange(5, 5.4, 0.002)
    reduc_epoch = epochs - start_epoch

    ellips1 = func.Ellipsoid(a=1, b=1, c=1)
    ellips2 = func.Ellipsoid(a=0.99, b=0.99, c=0.99)
    # create body from two ellipsis
    body = func.Body(ellips1, ellips2, distance=0.0,
                     eq_latitude=np.pi / 4,
                     eq_longitude=np.pi / 4,
                     init_phase=0., rotation_period=5.0 / 24)

    # same position for earth and sun
    state_vecs_sun = np.zeros([201, 3])
    state_vecs_sun[:, 0] = 1
    state_vecs_sun = pd.DataFrame(state_vecs_sun, columns=['x', 'y', 'z'])
    state_vecs_sun['reduc_epoch'] = reduc_epoch
    state_vecs_earth = np.copy(state_vecs_sun)
    state_vecs_earth = pd.DataFrame(state_vecs_earth, columns=['x', 'y', 'z', 'reduc_epoch'])

    # transform vectors asteroid body-fixed frame coords
    vecs_sun_rot = func.rotate_vector(body, state_vecs_sun).to_numpy()
    vecs_earth_rot = func.rotate_vector(body, state_vecs_earth).to_numpy()

    total_brightness = body.total_brightness(sun_vecs=vecs_sun_rot, earth_vecs=vecs_earth_rot)

    assert (total_brightness - 2.09 < 0.1).all()