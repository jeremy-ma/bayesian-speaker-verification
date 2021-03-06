import os
import cPickle
from frontend import frontend
import sys, pdb
import config
import time
import numpy as np
from scipy.misc import logsumexp
from gmmmc import GMM
import sklearn.mixture
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging
import bob.bio.gmm.algorithm
from bob.bio.gmm.algorithm import GMM as BobGMM
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior,\
    DiagCovarsWishartPrior, WeightsUniformPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import multiprocessing
import copy
import bob
from system.full_system import KLDivergenceMLStartSystem, KLDivergenceMAPStartSystem
from shutil import copyfile

logging.getLogger().setLevel(logging.INFO)
n_mixtures, n_runs, description = 8, 10, 'all_params'
relevance_factor = 150
n_procs = 1
n_jobs = -1
gender = 'female'
description += '_' + gender

if gender == 'male':
    enrolment = config.reddots_part4_enrol_male
    trials = config.reddots_part4_trial_male
    background = config.background_data_directory_male
else:
    enrolment = config.reddots_part4_enrol_female
    trials = config.reddots_part4_trial_female
    background = config.background_data_directory_female

manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=enrolment,
                               trial_file=trials)

save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                        "samples")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logging.info('Saving script..')
src = __file__
dest = os.path.join(save_dir, 'script.py')
copyfile(src, dest)

system = KLDivergenceMAPStartSystem(n_mixtures, n_runs, relevance_factor)

print "reading background data"
X = manager.get_background_data()
print "obtained background data"

ubm_path = os.path.join(save_dir, 'ubm.pickle')

try:
    with open(ubm_path) as fp:
        ubm = cPickle.load(fp)
        system.train_ubm(X[:1000])  # hack to initialise the ubm
        system.load_ubm(ubm)
    logging.info("Loaded background model")
except IOError:
    logging.info('Training background model...')
    system.train_ubm(manager.get_background_data())
    with open(ubm_path, 'wb') as fp:
        cPickle.dump(system.ubm, fp, cPickle.HIGHEST_PROTOCOL)
    logging.info('Finished, saved background model to file...')

prior = GMMPrior(MeansGaussianPrior(np.array(system.ubm.means), np.array(system.ubm.covars) / relevance_factor),
                 DiagCovarsWishartPrior(relevance_factor, 1/ (system.ubm.covars * relevance_factor)),
                 WeightsUniformPrior(n_mixtures))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0002, 0.001]),
                                      propose_covars=GaussianStepCovarProposal(step_sizes=[0.00001]),
                                      propose_weights=GaussianStepWeightsProposal(n_mixtures, step_sizes=[0.0001]))

logging.info('Beginning Monte Carlo Sampling')

system.set_params(proposal, prior)
system.sample_background(manager.get_background_data(), n_jobs, save_dir)
system.sample_speakers(manager.get_enrolment_data(), n_procs, n_jobs, save_dir)
#system.sample_trials(manager.get_unique_trials(), n_procs, n_jobs, save_dir)

#system.evaluate_forward_unnormalised(manager.get_trial_data(), manager.get_enrolment_data(), manager.get_background_data(),
#                        n_jobs, save_dir, n_runs/2, 1)