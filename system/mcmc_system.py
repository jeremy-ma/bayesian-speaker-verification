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
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging
import bob.bio.gmm.algorithm
from bob.bio.gmm.algorithm import GMM as BobGMM
import bob


class MCSystem(object):

    def __init__(self, n_mixtures):
        self.n_mixtures = n_mixtures
        self.model_samples = {}

    def train_background(self, background_features):
        bobmodel = BobGMM(self.n_mixtures)
        bobmodel.train_ubm(background_features)
        self.ubm = GMM(np.array(bobmodel.ubm.means), np.array(bobmodel.ubm.variances), np.array(bobmodel.ubm.weights))

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
        background = self.ubm.log_likelihood(features, n_jobs)
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
        initial_gmm = GMM(means=self.ubm.means, covariances=self.ubm.covars, weights=self.ubm.weights)
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

    def verify(self, claimed_speaker, features, n_jobs, burn_in=0, lag=50):
        gmm_samples = self.model_samples[claimed_speaker][burn_in::lag]
        claimed_likelihoods = []
        for gmm in gmm_samples:
            claimed_likelihoods.append(gmm.log_likelihood(features, n_jobs) / features.shape[0])
        claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(gmm_samples))
        background = self.ubm.log_likelihood(features, n_jobs) / features.shape[0]
        likelihood_ratio = claimed - background

        return likelihood_ratio

class MCMC_MAP_System(MCSystem):
    def __init__(self, n_mixtures, n_runs):
        super(MCMC_MAP_System, self).__init__(n_mixtures)
        self.n_runs = n_runs
        self.model = BobGMM(n_mixtures, gmm_enroll_iterations=2)

    def train_background(self, background_features):
        self.model.train_ubm(background_features)
        self.ubm = GMM(np.array(self.model.ubm.means),
                       np.array(self.model.ubm.variances),
                       np.array(self.model.ubm.weights))
        self.model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.model.ubm,
                                                                relevance_factor=self.model.relevance_factor,
                                                                update_means=True, update_variances=False)


    def set_params(self, proposal, prior):
        self.proposal = proposal
        self.prior = prior

    def get_samples(self, X, n_jobs):
        """
        GMM monte carlo
        :param X:
        :return: monte carlo samples
        """
        map = self.model.enroll_gmm(X)
        initial_gmm = GMM(means=map.means, covariances=map.variances, weights=map.weights)
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

    def verify(self, claimed_speaker, features, n_jobs, burn_in=0, lag=50):
        gmm_samples = self.model_samples[claimed_speaker][burn_in::lag]
        claimed_likelihoods = []
        for gmm in gmm_samples:
            claimed_likelihoods.append(gmm.log_likelihood(features, n_jobs)/ features.shape[0])
        claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(gmm_samples))
        background = self.ubm.log_likelihood(features, n_jobs) /features.shape[0]
        likelihood_ratio = claimed - background

        return likelihood_ratio
