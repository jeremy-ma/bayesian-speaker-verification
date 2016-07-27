__author__ = 'jeremyma'
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

def enrol_mcmc(n_mixtures, n_runs, description):
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = manager.get_background_data()

    system = mcmc_system.MCMC_ML_System(n_mixtures=n_mixtures, n_runs=n_runs)

    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'ubm' + '.pickle')

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

    #prior = GMMPrior(MeansUniformPrior(X.min(), X.max(), n_mixtures, X.shape[1]),
    #                 CovarsStaticPrior(np.array(system.ubm.covars_)),
    #                 WeightsStaticPrior(np.array(system.ubm.weights_)))

    prior = GMMPrior(MeansGaussianPrior(np.array(system.ubm.means_), np.ones((n_mixtures, X.shape[1]))),
                     CovarsStaticPrior(np.array(system.ubm.covars_)),
                     WeightsStaticPrior(np.array(system.ubm.weights_)))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.01, 0.05, 0.1]),
                                      propose_covars=None,
                                      propose_weights=None)

    system.set_params(proposal, prior)

    logging.info('Beginning Monte Carlo Sampling')

    save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    system.train_speakers(manager.get_enrolment_data(), -1, save_dir)

    logging.info('Saving system to file')
    with open(os.path.join(save_dir, 'system.pickle'), 'w') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)

def enrol_ais(n_mixtures, n_runs, description):
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = manager.get_background_data()

    betas = [0.0]
    betas.extend(np.logspace(-2, 0, 200))
    system = mcmc_system.AIS_System(n_mixtures, n_runs, betas)

    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'ubm' + '.pickle')

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

    # prior = GMMPrior(MeansUniformPrior(X.min(), X.max(), n_mixtures, X.shape[1]),
    #                 CovarsStaticPrior(np.array(system.ubm.covars_)),
    #                 WeightsStaticPrior(np.array(system.ubm.weights_)))

    prior = GMMPrior(MeansGaussianPrior(np.array(system.ubm.means_), np.ones((n_mixtures, X.shape[1]))),
                     CovarsStaticPrior(np.array(system.ubm.covars_)),
                     WeightsStaticPrior(np.array(system.ubm.weights_)))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.005, 0.01, 0.1]),
                                          propose_covars=None,
                                          propose_weights=None,
                                          propose_iterations=10)

    system.set_params(proposal, prior)

    logging.info('Beginning Monte Carlo Sampling')

    save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    system.train_speakers(manager.get_enrolment_data(), -1, save_dir)

    logging.info('Saving system to file')
    with open(os.path.join(save_dir, 'system.pickle'), 'w') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    enrol_ais(8, 1, 'ais_gaussian_priors')
