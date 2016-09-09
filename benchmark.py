import config
from system import mcmc_system
from frontend import frontend
import numpy as np
import cPickle
import os
import logging, pdb
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import sys
from shutil import copyfile
import time


logging.getLogger().setLevel(logging.INFO)

n_mixtures, n_runs, description = 8, 100, 'mcm_benchmarking'

manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)

save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logging.info('Saving script..')
src = __file__
dest = os.path.join(save_dir, 'script.py')

if os.path.exists(dest):
    logging.info('Overwriting previous run.....')
copyfile(src, dest)


X = manager.get_background_data()

system = mcmc_system.MCMC_ML_System(n_mixtures=n_mixtures, n_runs=n_runs)

filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'ubm' + '.pickle')

system.train_ubm(manager.get_background_data())
"""
try:
    with open(filename) as fp:
        ubm = cPickle.load(fp)
    system.load_background(ubm)
    logging.info('Loaded background model from file')
except:
    logging.info('Training background model...')
    system.train_background(manager.get_background_data())
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as fp:
        cPickle.dump(system.ubm, fp, cPickle.HIGHEST_PROTOCOL)
    logging.info('Finished, saved background model to file...')
"""

# prior = GMMPrior(MeansUniformPrior(X.min(), X.max(), n_mixtures, X.shape[1]),
#                 CovarsStaticPrior(np.array(system.ubm.covars_)),
#                 WeightsStaticPrior(np.array(system.ubm.weights_)))

prior = GMMPrior(MeansGaussianPrior(np.array(system.ubm.means), 3 * np.ones((n_mixtures, X.shape[1]))),
                 CovarsStaticPrior(np.array(system.ubm.covars)),
                 WeightsStaticPrior(np.array(system.ubm.weights)))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.01, 0.05, 0.1]),
                                      propose_covars=None,
                                      propose_weights=None)

system.set_params(proposal, prior)

logging.info('Beginning Monte Carlo Sampling')

times = []
jobs =  [1,4,8,16,32,48,64,128,256]

for n_jobs in jobs:
    start = time.time()
    system.train_speakers(manager.get_enrolment_data(), n_jobs, save_dir)
    times.append(time.time() - start)

results = zip(jobs,times)

print results
