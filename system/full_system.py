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


def get_samples(initial, prior, proposal, X, n_jobs, n_runs):
    """
    GMM monte carlo
    :param X:
    :return: monte carlo samples
    """
    initial_gmm = GMM(means=initial.means, covariances=initial.covars, weights=initial.weights)
    mc = MarkovChain(proposal, prior, initial_gmm)
    # make samples
    gmm_samples = mc.sample(X, n_samples=n_runs, n_jobs=n_jobs)
    if proposal.propose_mean is not None:
        logging.info('Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))

    if proposal.propose_covars is not None:
        logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))

    if proposal.propose_weights is not None:
        logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))

    return gmm_samples

def multithreaded_get_samples(params):
    trial, destination_directory, ubm, prior, proposal, n_runs, n_jobs = params
    features = np.load(trial.feature_file)
    base = os.path.basename(trial.feature_file)
    base = base.split('.')[0]
    path = os.path.join(destination_directory, str(trial.actual_speaker), base + '.pickle')
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

    with open(path, 'w') as fp:
        cPickle.dump(get_samples(ubm, prior, proposal, features, n_jobs, n_runs),
                     fp, cPickle.HIGHEST_PROTOCOL)

class PairwiseSystem(object):

    def __init__(self, n_mixtures, n_runs):
        self.n_mixtures = n_mixtures
        self.n_runs = n_runs

    def set_params(self, proposal, prior):
        self.proposal = proposal
        self.prior = prior

    def train_ubm(self, background_features):
        bobmodel = BobGMM(self.n_mixtures)
        bobmodel.train_ubm(background_features)
        self.ubm = GMM(np.array(bobmodel.ubm.means), np.array(bobmodel.ubm.variances),
                       np.array(bobmodel.ubm.weights))

    def train_speakers(self, speaker_data, n_procs, n_jobs, destination_directory):
        for speaker_id, features in speaker_data.iteritems():
            logging.info('Sampling Speaker:{0}'.format(str(speaker_id)))
            path = os.path.join(destination_directory, str(speaker_id), 'speaker' + '.pickle')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as fp:
                cPickle.dump(get_samples(self.ubm, self.prior, self.proposal, features, n_jobs, self.n_runs),
                             fp, cPickle.HIGHEST_PROTOCOL)

    def train_trials(self, unique_trials, n_procs, n_jobs, destination_directory):
        all_trials = []
        for speaker_id, trials in unique_trials.iteritems():
            for trial in trials:
                all_trials.append(trial)

        dirs = [destination_directory for _ in all_trials]
        ubms = [copy.deepcopy(self.ubm) for _ in all_trials]
        priors = [copy.deepcopy(self.prior) for _ in all_trials]
        proposals = [copy.deepcopy(self.proposal) for _ in all_trials]
        runs = [self.n_runs for _ in all_trials]
        jobs = [n_jobs for _ in all_trials]


        params = zip(all_trials, dirs, ubms, priors, proposals, runs, jobs)

        pool = multiprocessing.Pool(n_procs)
        pool.map(multithreaded_get_samples, params)

    def load_ubm(self, ubm):
        self.ubm = ubm


if __name__ == '__main__':
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

    system = PairwiseSystem(n_mixtures, n_runs)

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

    logging.info('Beginning Monte Carlo Sampling')

    system.set_params(proposal, prior)
    system.train_speakers(manager.get_enrolment_data(), n_procs, n_jobs, save_dir)
    system.train_trials(manager.get_unique_trials(), n_procs, n_jobs, save_dir)