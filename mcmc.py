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


logging.getLogger().setLevel(logging.INFO)
n_mixtures, n_runs, description = 8, 100, 'mcmc_rel150_bigubm'
relevance_factor = 150

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
                               trial_file=config.reddots_part4_trial_female,
                               background_data_directory=config.background_data_directory)

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

print "reading background data"
X = manager.get_background_data()
print "obtained background data"

system = mcmc_system.MCMC_ML_System(n_mixtures=n_mixtures, n_runs=n_runs)

logging.info('Training background model...')

system.train_background(manager.get_background_data())
with open(os.path.join(save_dir, 'ubm.pickle'), 'wb') as fp:
    cPickle.dump(system.ubm, fp, cPickle.HIGHEST_PROTOCOL)

logging.info('Finished, saved background model to file...')

# prior = GMMPrior(MeansUniformPrior(X.min(), X.max(), n_mixtures, X.shape[1]),
#                 CovarsStaticPrior(np.array(system.ubm.covars_)),
#                 WeightsStaticPrior(np.array(system.ubm.weights_)))


prior = GMMPrior(MeansGaussianPrior(np.array(system.ubm.means), np.array(system.ubm.covars) / relevance_factor),
                 CovarsStaticPrior(np.array(system.ubm.covars)),
                 WeightsStaticPrior(np.array(system.ubm.weights)))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0002, 0.001]),
                                      propose_covars=None,
                                      propose_weights=None)

system.set_params(proposal, prior)

logging.info('Beginning Monte Carlo Sampling')

system.train_speakers(manager.get_enrolment_data(), -1, save_dir)

logging.info('Saving system to file')
with open(os.path.join(save_dir, 'system.pickle'), 'w') as fp:
    cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)
