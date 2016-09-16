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
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import multiprocessing
import copy
import bob
from system.full_system import KLDivergenceMLStartSystem
from shutil import copyfile

logging.getLogger().setLevel(logging.INFO)
n_mixtures, n_runs, description = 8, 10, 'pairwise_test'
relevance_factor = 150
n_procs = 4
n_jobs = 1
gender = 'female'
description += '_' + gender

if gender == 'male':
    enrolment = config.reddots_part4_enrol_male
    trials = config.reddots_part4_trial_male
else:
    enrolment = config.reddots_part4_enrol_female
    trials = config.reddots_part4_trial_female

manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)

save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                        "samples")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

system = KLDivergenceMLStartSystem(n_mixtures, n_runs)

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
                 CovarsStaticPrior(np.array(system.ubm.covars)),
                 WeightsStaticPrior(np.array(system.ubm.weights)))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0002, 0.001]),
                                      propose_covars=None,
                                      propose_weights=None)

logging.info('Beginning Evaluation')

system.set_params(proposal, prior)
system.evaluate_forward(manager.get_trial_data(), manager.get_enrolment_data(), manager.get_background_data(),
                        n_jobs, save_dir, n_runs/2, 1)

logging.info('Saving script..')
src = __file__
dest = os.path.join(save_dir, 'eval_script_kl.py')
copyfile(src, dest)