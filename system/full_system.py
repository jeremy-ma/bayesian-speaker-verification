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
from collections import defaultdict

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
            os.makedirs(os.path.dirname(path))
        except OSError:
            if not os.path.isdir(path):
                print "whoops!"
                raise

    with open(path, 'w') as fp:
        cPickle.dump(get_samples(ubm, prior, proposal, features, n_jobs, n_runs),
                     fp, cPickle.HIGHEST_PROTOCOL)

class KLDivergenceMLStartSystem(object):

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

    def sample_background(self, background_features, n_jobs, destination_directory):
        path = os.path.join(destination_directory, 'background.pickle')
        with open(path, 'w') as fp:
            cPickle.dump(get_samples(self.ubm, self.prior, self.proposal, background_features, n_jobs, self.n_runs), fp)

    def sample_speakers(self, speaker_data, n_procs, n_jobs, destination_directory):
        for speaker_id, features in speaker_data.iteritems():
            logging.info('Sampling Speaker:{0}'.format(str(speaker_id)))
            path = os.path.join(destination_directory, str(speaker_id), 'speaker' + '.pickle')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as fp:
                cPickle.dump(get_samples(self.ubm, self.prior, self.proposal, features, n_jobs, self.n_runs),
                             fp, cPickle.HIGHEST_PROTOCOL)

    def sample_trials(self, unique_trials, n_procs, n_jobs, destination_directory):
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

    def evaluate_forward(self, all_trials, speaker_data, background_data, n_jobs, samples_directory, burn_in, lag):
        #evaluate using speaker/background posterior samples
        with open(os.path.join(samples_directory, 'background.pickle')) as fp:
            background_samples = cPickle.load(fp)

        background_posteriors = np.array([gmm.log_likelihood(background_data, n_jobs) / background_data.shape[0] +
                                          self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
        background_posteriors = background_posteriors - logsumexp(background_posteriors)
        numer_background = np.sum(background_posteriors)

        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples
            speaker_posteriors = np.array([gmm.log_likelihood(speaker_data[speaker_id], n_jobs) /
                                                     speaker_data[speaker_id].shape[0] +
                                                     self.prior.log_prob(gmm) for gmm in
                                                     speaker_samples[speaker_id][burn_in::lag]])
            # normalise the log probs by dividing by the sum
            speaker_posteriors = speaker_posteriors - logsumexp(speaker_posteriors)
            speaker_numerators[speaker_id] = np.sum(speaker_posteriors)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                denom_speaker_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0] +
                                self.prior.log_prob(gmm) for gmm in speaker_samples[trial.claimed_speaker][burn_in::lag]])
                denom_speaker_posteriors = denom_speaker_posteriors - logsumexp(denom_speaker_posteriors)
                denom_speaker = np.sum(denom_speaker_posteriors)
                kl_speaker = (speaker_numerators[trial.claimed_speaker] - denom_speaker) / \
                             len(speaker_samples[trial.claimed_speaker][burn_in::lag])

                denom_background_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]  +
                                           self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
                denom_background_posteriors = denom_background_posteriors - logsumexp(denom_background_posteriors)
                denom_background = np.sum(denom_background_posteriors)
                kl_background = (numer_background - denom_background) / len(background_samples[burn_in::lag])
                score = kl_speaker - kl_background
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "KLForwardScores.npy"),scores)
        np.save(os.path.join(samples_directory, "KLForwardAnswers.npy"), truth)
        with open(samples_directory + "books.pickle", 'w') as fp:
            cPickle.dump(books, fp)

    def evaluate_forward_unnormalised(self, all_trials, speaker_data, background_data, n_jobs, samples_directory, burn_in, lag):

        #evaluate using speaker/background posterior samples
        with open(os.path.join(samples_directory, 'background.pickle')) as fp:
            background_samples = cPickle.load(fp)

        background_posteriors = np.array([gmm.log_likelihood(background_data, n_jobs) / background_data.shape[0] +
                                          self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
        # background_posteriors = background_posteriors - logsumexp(background_posteriors)
        numer_background = np.sum(background_posteriors)

        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples
            speaker_posteriors = np.array([gmm.log_likelihood(speaker_data[speaker_id], n_jobs) /
                                                     speaker_data[speaker_id].shape[0] +
                                                     self.prior.log_prob(gmm) for gmm in
                                                     speaker_samples[speaker_id][burn_in::lag]])
            # normalise the log probs by dividing by the sum
            # speaker_posteriors = speaker_posteriors - logsumexp(speaker_posteriors)
            speaker_numerators[speaker_id] = np.sum(speaker_posteriors)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                denom_speaker_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0] +
                                self.prior.log_prob(gmm) for gmm in speaker_samples[trial.claimed_speaker][burn_in::lag]])
                # denom_speaker_posteriors = denom_speaker_posteriors - logsumexp(denom_speaker_posteriors)
                denom_speaker = np.sum(denom_speaker_posteriors)
                kl_speaker = (speaker_numerators[trial.claimed_speaker] - denom_speaker) / \
                             len(speaker_samples[trial.claimed_speaker][burn_in::lag])

                denom_background_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]  +
                                           self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
                # denom_background_posteriors = denom_background_posteriors - logsumexp(denom_background_posteriors)
                denom_background = np.sum(denom_background_posteriors)
                kl_background = (numer_background - denom_background) / len(background_samples[burn_in::lag])
                score = kl_speaker - kl_background
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "KLForwardUnnormScores.npy"),scores)
        np.save(os.path.join(samples_directory, "KLForwardUnnormAnswers.npy"), truth)
        with open(samples_directory + "books.pickle", 'w') as fp:
            cPickle.dump(books, fp)

    def evaluate_backward(self, all_trials):
        raise NotImplementedError

    def evaluate_bayes_factor(self, all_trials, n_jobs, samples_directory, burn_in, lag):
        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples

        with open(os.path.join(samples_directory, 'ubm.pickle')) as fp:
            ubm = cPickle.load(fp)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                background_likelihood = ubm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]
                claimed_likelihoods = [gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]
                                       for gmm in speaker_samples[speaker_id]]
                claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(speaker_samples[speaker_id]))
                score = claimed - background_likelihood
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "BayesFactorScores.npy"), scores)
        np.save(os.path.join(samples_directory, "BayesFactorAnswers.npy"), truth)
        with open(samples_directory + "BayesFactorBooks.pickle", 'w') as fp:
            cPickle.dump(books, fp)

class KLDivergenceMAPStartSystem(object):

    def __init__(self, n_mixtures, n_runs, relevance_factor):
        self.n_mixtures = n_mixtures
        self.n_runs = n_runs
        self.model = BobGMM(n_mixtures)
        self.model.relevance_factor = relevance_factor

    def set_params(self, proposal, prior):
        self.proposal = proposal
        self.prior = prior

    def train_ubm(self, background_features):
        self.model = BobGMM(self.n_mixtures)
        self.model.train_ubm(background_features)
        self.ubm = GMM(np.array(self.model.ubm.means), np.array(self.model.ubm.variances),
                       np.array(self.model.ubm.weights))
        self.model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.model.ubm,
                                                                relevance_factor=self.model.relevance_factor,
                                                                update_means=True, update_variances=False)

    def sample_background(self, background_features, n_jobs, destination_directory):
        path = os.path.join(destination_directory, 'background.pickle')
        with open(path, 'w') as fp:
            cPickle.dump(get_samples(self.ubm, self.prior, self.proposal, background_features, n_jobs, self.n_runs), fp)

    def sample_speakers(self, speaker_data, n_procs, n_jobs, destination_directory):
        for speaker_id, features in speaker_data.iteritems():
            logging.info('Sampling Speaker:{0}'.format(str(speaker_id)))
            map = self.model.enroll_gmm(features)
            initial_gmm = GMM(means=map.means, covariances=map.variances, weights=map.weights)
            path = os.path.join(destination_directory, str(speaker_id), 'speaker' + '.pickle')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as fp:
                cPickle.dump(get_samples(initial_gmm, self.prior, self.proposal, features, n_jobs, self.n_runs),
                             fp, cPickle.HIGHEST_PROTOCOL)

    def sample_trials(self, unique_trials, n_procs, n_jobs, destination_directory):
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

    def evaluate_forward(self, all_trials, speaker_data, background_data, n_jobs, samples_directory, burn_in, lag):
        #evaluate using speaker/background posterior samples
        with open(os.path.join(samples_directory, 'background.pickle')) as fp:
            background_samples = cPickle.load(fp)

        background_posteriors = np.array([gmm.log_likelihood(background_data, n_jobs) / background_data.shape[0] +
                                          self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
        background_posteriors = background_posteriors - logsumexp(background_posteriors)
        numer_background = np.sum(background_posteriors)

        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples
            speaker_posteriors = np.array([gmm.log_likelihood(speaker_data[speaker_id], n_jobs) /
                                                     speaker_data[speaker_id].shape[0] +
                                                     self.prior.log_prob(gmm) for gmm in
                                                     speaker_samples[speaker_id][burn_in::lag]])
            # normalise the log probs by dividing by the sum
            speaker_posteriors = speaker_posteriors - logsumexp(speaker_posteriors)
            speaker_numerators[speaker_id] = np.sum(speaker_posteriors)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                denom_speaker_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0] +
                                self.prior.log_prob(gmm) for gmm in speaker_samples[trial.claimed_speaker][burn_in::lag]])
                denom_speaker_posteriors = denom_speaker_posteriors - logsumexp(denom_speaker_posteriors)
                denom_speaker = np.sum(denom_speaker_posteriors)
                kl_speaker = (speaker_numerators[trial.claimed_speaker] - denom_speaker) / \
                             len(speaker_samples[trial.claimed_speaker][burn_in::lag])

                denom_background_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]  +
                                           self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
                denom_background_posteriors = denom_background_posteriors - logsumexp(denom_background_posteriors)
                denom_background = np.sum(denom_background_posteriors)
                kl_background = (numer_background - denom_background) / len(background_samples[burn_in::lag])
                score = kl_speaker - kl_background
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "KLForwardScores.npy"),scores)
        np.save(os.path.join(samples_directory, "KLForwardAnswers.npy"), truth)
        with open(samples_directory + "books.pickle", 'w') as fp:
            cPickle.dump(books, fp)

    #def evaluate_mean_estimate(self, all_trials, speaker_data, background_data, n_jobs, samples_directory, burn_in, lag):
    #    with open(os.path.join(samples_directory, 'background'))

    def evaluate_forward_unnormalised(self, all_trials, speaker_data, background_data, n_jobs, samples_directory, burn_in, lag):

        #evaluate using speaker/background posterior samples
        with open(os.path.join(samples_directory, 'background.pickle')) as fp:
            background_samples = cPickle.load(fp)

        background_posteriors = np.array([gmm.log_likelihood(background_data, n_jobs) / background_data.shape[0] +
                                          self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
        # background_posteriors = background_posteriors - logsumexp(background_posteriors)
        numer_background = np.sum(background_posteriors)

        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples
            speaker_posteriors = np.array([gmm.log_likelihood(speaker_data[speaker_id], n_jobs) /
                                                     speaker_data[speaker_id].shape[0] +
                                                     self.prior.log_prob(gmm) for gmm in
                                                     speaker_samples[speaker_id][burn_in::lag]])
            # normalise the log probs by dividing by the sum
            # speaker_posteriors = speaker_posteriors - logsumexp(speaker_posteriors)
            speaker_numerators[speaker_id] = np.sum(speaker_posteriors)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                denom_speaker_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0] +
                                self.prior.log_prob(gmm) for gmm in speaker_samples[trial.claimed_speaker][burn_in::lag]])
                denom_speaker = np.sum(denom_speaker_posteriors)
                kl_speaker = (speaker_numerators[trial.claimed_speaker] - denom_speaker) / \
                             len(speaker_samples[trial.claimed_speaker][burn_in::lag])

                denom_background_posteriors = np.array([gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]  +
                                           self.prior.log_prob(gmm) for gmm in background_samples[burn_in::lag]])
                denom_background = np.sum(denom_background_posteriors)
                kl_background = (numer_background - denom_background) / len(background_samples[burn_in::lag])
                score = kl_speaker - kl_background
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "KLForwardUnnormScores.npy"),scores)
        np.save(os.path.join(samples_directory, "KLForwardUnnormAnswers.npy"), truth)
        with open(samples_directory + "KLForwardUnnormBooks.pickle", 'w') as fp:
            cPickle.dump(books, fp)

    def evaluate_bayes_factor(self, all_trials, n_jobs, samples_directory, burn_in, lag):
        scores = []
        truth = []

        speaker_samples = {}
        speaker_numerators = {}

        num_trials = 0
        for speaker_id in all_trials:
            num_trials += len(all_trials[speaker_id])
            speaker_path = os.path.join(samples_directory, str(speaker_id), 'speaker.pickle')
            with open(speaker_path) as fp:
                samples = cPickle.load(fp)
            speaker_samples[speaker_id] = samples[burn_in::lag]
            print len(speaker_samples[speaker_id])

        with open(os.path.join(samples_directory, 'ubm.pickle')) as fp:
            ubm = cPickle.load(fp)

        books = defaultdict(dict)

        num_processed = 0
        for speaker_id, trials in all_trials.iteritems():
            for trial in trials:
                if num_processed % 100 == 0:
                    print "{0} done".format(float(num_processed) / num_trials)
                num_processed += 1
                trial_data = np.load(trial.feature_file)
                background_likelihood = ubm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]
                claimed_likelihoods = [gmm.log_likelihood(trial_data, n_jobs) / trial_data.shape[0]
                                       for gmm in speaker_samples[speaker_id]]
                claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(speaker_samples[speaker_id]))
                score = claimed - background_likelihood
                scores.append(score)
                truth.append(trial.answer)
                books[speaker_id][trial.actual_speaker] = score

        scores = np.array(scores)
        truth = np.array(truth)

        np.save(os.path.join(samples_directory, "BayesFactorScores.npy"), scores)
        np.save(os.path.join(samples_directory, "BayesFactorAnswers.npy"), truth)
        with open(samples_directory + "BayesFactorBooks.pickle", 'w') as fp:
            cPickle.dump(books, fp)


def verify(self, claimed_speaker, features, n_jobs, burn_in=0, lag=50):
    gmm_samples = self.model_samples[claimed_speaker][burn_in::lag]
    claimed_likelihoods = []
    for gmm in gmm_samples:
        claimed_likelihoods.append(gmm.log_likelihood(features, n_jobs) / features.shape[0])
    claimed = logsumexp(np.array(claimed_likelihoods)) - np.log(len(gmm_samples))
    background = self.ubm.log_likelihood(features, n_jobs) / features.shape[0]
    likelihood_ratio = claimed - background

    return likelihood_ratio