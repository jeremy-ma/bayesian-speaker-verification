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

        proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.001, 0.005]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.0001]),
                                          propose_weights=GaussianStepWeightsProposal(self.n_mixtures, step_sizes=[0.01, 0.1]))

        initial_gmm = GMM(means=self.ubm.means_, covariances=self.ubm.covars_, weights=self.ubm.weights_)

        mc = MarkovChain(proposal, prior, initial_gmm)
        # make samples
        gmm_samples = mc.sample(X, n_samples=self.n_runs, n_jobs=n_jobs)
        logging.info('Means Acceptance: {0}'.format(proposal.propose_mean.get_acceptance()))
        logging.info('Covars Acceptance: {0}'.format(proposal.propose_covars.get_acceptance()))
        logging.info('Weights Acceptance: {0}'.format(proposal.propose_weights.get_acceptance()))

        return gmm_samples

    def compute_samples_save(self, trial, destination_directory):
        """
        Compute samples for given utterance trials and save to file
        :param trial_list:
        :param destination_directory:
        """
        output_filename = os.path.basename(trial.feature_file).split('.')[0] + '.pickle'
        destination_folder = trial.actual_speaker
        output_filename = os.path.join(destination_directory, destination_folder, output_filename)
        features = trial.get_data()
        samples = self.get_samples(features)

        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

        with open(output_filename, 'wb') as fp:
            cPickle.dump(samples, fp, cPickle.HIGHEST_PROTOCOL)

    def train_background(self, background_features):
        self.ubm = sklearn.mixture.GMM(n_components=self.n_mixtures, n_init=10, n_iter=1000, tol=0.00001)
        self.ubm.fit(background_features)

    def train_speakers(self, speaker_data):
        for speaker_id, features in speaker_data.iteritems():
            self.model_samples[speaker_id] = self.get_samples(features)

    def load_background(self, filename):
        with open(filename, 'rb') as fp:
            self.background_samples = cPickle.load(fp)

    def load_speakers(self, speaker_data):
        for speaker_id, filename in speaker_data.iteritems():
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

if __name__ == '__main__':
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)
    speaker_trials = manager.get_trial_data()

    system = MCMC_ML_System(n_mixtures=64, n_runs=20000)
    directory = os.path.join(config.dropbox_directory, 'MonteCarloSamples', 'gaussians8', 'iterations20000')

    pdb.set_trace()
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    system.load_background(os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians8', 'iterations20000', 'background.pickle'))
    speaker_data = {}
    for speaker_id, _ in manager.get_enrolment_data().iteritems():
        speaker_data[speaker_id] = os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians8', 'iterations20000', speaker_id + '.pickle')

    system.load_speakers(speaker_data)
    """
    answer_array = []
    likelihood_array = []
    count = 0
    for speaker_id, trials in manager.speaker_trials.iteritems():
        for trial in trials:
            count += 1
            if count % 50 == 0:
                print "iteration {0}".format(count)
            answer_array.append(trial.answer)
            likelihood_array.append(system.verify(trial.claimed_speaker, trial.get_data()))

    #save results
    np.save(os.path.join(config.dropbox_directory, 'scoresMonte20000.npy'), likelihood_array)
    np.save(os.path.join(config.dropbox_directory, 'answersMonte20000.npy'), answer_array)

    with open('systemMonte.pickle', 'wb') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)