__author__ = 'jeremyma'
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
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging

class MCMC_ML_System():

    def __init__(self, n_mixtures=8, n_runs=10000):
        self.n_mixtures = n_mixtures
        self.n_runs = n_runs
        self.model_samples = {}

    def get_samples(self, X, n_jobs):
        """
        GMM monte carlo
        :param X:
        :return: monte carlo samples
        """

        prior = GMMPrior(MeansUniformPrior(X.min(), X.max(), self.n_mixtures, X.shape[1]),
                         CovarsStaticPrior(np.array(self.ubm.covars_)),
                         WeightsStaticPrior(np.array(self.ubm.weights_)))

        proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.005, 0.01, 0.1]),
                                          propose_covars=None,
                                          propose_weights=None)

        initial_gmm = GMM(means=self.ubm.means_, covariances=self.ubm.covars_, weights=self.ubm.weights_)

        mc = MarkovChain(proposal, prior, initial_gmm)
        # make samples
        gmm_samples = mc.sample(X, n_samples=self.n_runs, n_jobs=n_jobs)
        if proposal.propose_mean is not None:
            logging.info('Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))

        if proposal.propose_covars is not None:
            logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))

        if proposal.propose_weights is not None:
            logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))

        return gmm_samples

    def train_background(self, background_features):
        self.ubm = sklearn.mixture.GMM(n_components=self.n_mixtures, n_init=10, n_iter=1000, tol=0.00001)
        self.ubm.fit(background_features)

    def train_speakers(self, speaker_data, n_jobs, destination_directory=None):
        for speaker_id, features in speaker_data.iteritems():
            logging.info('Sampling Speaker:{0}'.format(str(speaker_id)))
            self.model_samples[speaker_id] = self.get_samples(features, n_jobs)
            if destination_directory is not None:
                with open(os.path.join(destination_directory, str(speaker_id) + '.pickle'), 'w') as fp:
                    cPickle.dump(self.model_samples[speaker_id], fp, cPickle.HIGHEST_PROTOCOL)

    def load_background(self, filename):
        with open(filename, 'rb') as fp:
            self.ubm = cPickle.load(fp)

    def load_speakers(self, speaker_data):
        for speaker_id, filename in speaker_data.iteritems():
            logging.info('Loading Speaker:{0}'.format(str(speaker_id)))
            with open(filename, 'rb') as fp:
                self.model_samples[speaker_id] = cPickle.load(fp)

    def verify(self, claimed_speaker, features):
        gmm_samples = self.model_samples[claimed_speaker]
        claimed_likelihoods = []
        for gmm in gmm_samples:
            claimed_likelihoods.append(gmm.log_likelihood(features))
        claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(gmm_samples))
        background = np.sum(self.ubm.score(features))
        likelihood_ratio = claimed - background
        return likelihood_ratio

def save_enrolment_samples(model_data, save_path, n_runs, n_mixtures):
    model_name = model_data['name']
    model_features = model_data['features']
    system = MCMC_ML_System(n_mixtures=n_mixtures, n_runs=n_runs)
    samples = system.get_samples(model_features, -1)
    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs), model_name + '.pickle')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as fp:
        cPickle.dump(samples, fp, cPickle.HIGHEST_PROTOCOL)

def calculate_samples(trial, save_path, num_iterations, num_gaussians):
    print "starting calculation"
    system = MCMC_ML_System(n_mixtures=num_gaussians, n_runs=num_iterations)
    start = time.time()
    system.compute_samples_save(trial, save_path)
    print time.time() - start