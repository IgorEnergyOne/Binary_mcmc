#!/usr/bin/env python3

import argparse
import math
from multiprocessing import Pool
import pickle

import numpy as np
import pandas as pd
import emcee

import radar_functions as rf
import mcmc
from data_structures import ObsData, LightCurve

# suppress warnings from emcee and futurewarings
import warnings
warnings.filterwarnings("ignore")


def run_mcmc(num_steps: int=1000, num_walkers=3, num_burnin:int=100,
             num_cores=12, out_path="mcmc_results.obj"):
    """wrapper for mcmc pipeline"""
    # read necessary lightcurve data
    lightcurve = LightCurve()  # container to store partial lightcurves
    earth_data = ObsData()  # store earth vectors
    sun_data = ObsData()  # store sun vectors
    # load data
    lightcurve.load_data('input_data/lightcurve_data.csv')
    earth_data.load_data('input_data/state_vecs_earth.csv')
    sun_data.load_data('input_data/state_vecs_sun.csv')
    # calculate shifts
    lightcurve.calculate_shifts(earth_data)
    lightcurve.calculate_shifts(sun_data)

    # read radar data
    radar_image_params = rf.RadarImageParameters("radar_data/data.txt")
    radar_images_obs = [np.array(rf.read_observed_radar_image(radar_image_params, idx_img))
                        for idx_img in range(radar_image_params.N_points)]
    # query_dates = [radar_image_params.jd[idx_img] + 2400000.5 for idx_img in range(len(radar_image_params.jd))]
    # state_vecs_earth = func.get_object_orbit('2000 WO107', epochs=query_dates, location='@399', data_type='vectors')
    # state_vecs_earth['r'] = np.sum(state_vecs_earth[['x', 'y', 'z']] ** 2, axis=1) ** 0.5
    # position_vecs = (state_vecs_earth[['x', 'y', 'z']].T / state_vecs_earth['r']).T
    position_vecs = pd.read_csv('input_data/position_vecs.csv')

    # create initial guess for the parameters
    nwalkers = num_cores * num_walkers  # number of probe points
    coeff = 0.2
    init_ps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0.0]
    init_params = np.array([init_ps[0] + coeff * np.random.randn(nwalkers),  # b1
                            init_ps[1] + coeff * np.random.randn(nwalkers),  # c1
                            init_ps[2] + coeff * np.random.randn(nwalkers),  # a2
                            init_ps[3] + coeff * np.random.randn(nwalkers),  # b2
                            init_ps[4] + coeff * np.random.randn(nwalkers),  # c2
                            init_ps[5] + coeff * np.random.randn(nwalkers),  # c2
                            0 + np.random.uniform(low=0, high=2 * math.pi, size=nwalkers),
                            0 + np.random.uniform(low=-math.pi / 2, high=math.pi / 2, size=nwalkers),
                            0 + np.random.uniform(low=0, high=2 * math.pi, size=nwalkers)]).T
    ndim = np.size(init_params, axis=1)  # number of parameters
    init_params = init_params.astype(float)

    try:
        # run MCMC
        with Pool(processes=num_cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc.ln_prob_both,
                                            pool=pool,
                                            args=(radar_image_params, radar_images_obs, position_vecs,
                                                  lightcurve, earth_data, sun_data),
                                            moves=[(emcee.moves.DEMove(), 0.8),
                                                   (emcee.moves.DESnookerMove(), 0.2)])
            # Run the MCMC sampler
            # starting points for every walker
            pos, prob, state = sampler.run_mcmc(init_params, num_steps, progress=True)
            # Get the samples after burn-in
            samples = sampler.get_chain(discard=num_burnin, flat=True)

    except KeyboardInterrupt:
        fname = f'{out_path[:-4]}_keyinterrupt.obj'
        print(f"The mcmc was interrupted by the user. The samples have been saved to {fname}.")
        # save samples
        file_pick = open(fname, 'wb')
        pickle.dump(sampler, file=file_pick)

    # save samples
    file_pick = open(out_path, 'wb')
    pickle.dump(sampler, file=file_pick)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCMC for binary asteroid')
    parser.add_argument('-n_samples', help='number of samples to run MCMC',
                        default=1000, type=int)
    parser.add_argument('-n_burnin', help='number of burn-in samples',
                        default=100, type=int)
    parser.add_argument('-n_cores', help="number of processor's cores to use",
                        default=12, type=int)
    parser.add_argument('-n_walkers', help="number of mcmc walkers per core",
                        default=3, type=int)
    parser.add_argument('-out_path',
                        help='path where to output the mcmc sampler',
                        default="mcmc_results.obj")
    args = parser.parse_args()
    num_samples = args.n_samples
    num_burnin = args.n_burnin
    num_cores = args.n_cores
    num_walkers = args.n_walkers
    out_path = args.out_path

    if num_burnin >= num_samples:
        num_burnin = int(num_samples * 0.1)
        num_burnin = max(num_burnin, 10)
        print("Error: number of burn-in samples must be less than number of samples")
        print(f"Setting number of burn-in samples to {num_burnin}")

    # run MCMC
    run_mcmc(num_steps=num_samples, num_burnin=num_burnin,
             num_cores=num_cores, num_walkers=num_walkers, out_path=out_path)
    print("Results saved to", out_path)




