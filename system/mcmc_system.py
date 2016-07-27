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
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging

class MCSystem(object):

    def __init__(self, n_mixtures=8):
        self.n_mixtures = n_mixtures
        self.model_samples = {}

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

    def load_background(self, background):
        self.ubm = background

    def load_speakers(self, speaker_data):
        self.model_samples = speaker_data

    def get_samples(self, features, n_jobs):
        pass


class AIS_System(MCSystem):
    def __init__(self, n_mixtures, n_runs, betas):
        super(AIS_System, self).__init__(n_mixtures)
        self.n_runs = n_runs
        self.betas = betas

    def set_params(self, proposal, prior):
        self.proposal = proposal
        self.prior = prior

    def get_samples(self, X, n_jobs):
        ais = AnnealedImportanceSampling(self.proposal, self.prior, self.betas)
        samples = ais.sample(X, self.n_runs, n_jobs)

        if self.proposal.propose_mean is not None:
            logging.info('Means Acceptance: {0}'.format(self.proposal.propose_mean.get_acceptance()))

        if self.proposal.propose_covars is not None:
            logging.info('Covars Acceptance: {0}'.format(self.proposal.propose_covars.get_acceptance()))

        if self.proposal.propose_weights is not None:
            logging.info('Weights Acceptance: {0}'.format(self.proposal.propose_weights.get_acceptance()))

        return samples

    def verify(self, claimed_speaker, features, n_jobs):
        ais_samples = self.model_samples[claimed_speaker]
        numerator = logsumexp([logweight + gmm.log_likelihood(features, n_jobs) for gmm, logweight in ais_samples])
        denominator = logsumexp([logweight for _, logweight in ais_samples])
        claimed = numerator - denominator
        background = np.sum(self.ubm.score(features))
        likelihood_ratio = claimed - background
        return likelihood_ratio

class MCMC_ML_System(MCSystem):

    def __init__(self, n_mixtures, n_runs):
        super(MCMC_ML_System, self).__init__(n_mixtures)
        self.n_runs = n_runs

    def set_params(self, proposal, prior):
        self.proposal = proposal
        self.prior = prior

    def get_samples(self, X, n_jobs):
        """
        GMM monte carlo
        :param X:
        :return: monte carlo samples
        """

        initial_gmm = GMM(means=self.ubm.means_, covariances=self.ubm.covars_, weights=self.ubm.weights_)

        mc = MarkovChain(self.proposal, self.prior, initial_gmm)
        # make samples
        gmm_samples = mc.sample(X, n_samples=self.n_runs, n_jobs=n_jobs)
        if self.proposal.propose_mean is not None:
            logging.info('Means Acceptance: {0}'.format(self.proposal.propose_mean.get_acceptance()))

        if self.proposal.propose_covars is not None:
            logging.info('Covars Acceptance: {0}'.format(self.proposal.propose_covars.get_acceptance()))

        if self.proposal.propose_weights is not None:
            logging.info('Weights Acceptance: {0}'.format(self.proposal.propose_weights.get_acceptance()))

        return gmm_samples


    def verify(self, claimed_speaker, features, burn_in, lag, n_jobs):
        gmm_samples = self.model_samples[claimed_speaker][burn_in::lag]
        claimed_likelihoods = []
        for gmm in gmm_samples:
            claimed_likelihoods.append(gmm.log_likelihood(features, n_jobs))
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