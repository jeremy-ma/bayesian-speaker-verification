__author__ = 'jeremyma'
from varun.MCMC import MCMCRun
import os
import cPickle
from frontend import frontend
import sys, pdb
import config
import time
import numpy as np
from varun.RobustLikelihoodClass import Likelihood
from scipy.misc import logsumexp


class MonteCarloSystem():

    def __init__(self, num_gaussians=8, num_iterations=10000):
        self.num_mixtures = num_gaussians
        self.num_iterations = num_iterations
        self.background_samples = None
        self.model_samples = {}

    def get_samples(self, X):
        """
        GMM monte carlo
        :param X:
        :return: monte carlo samples
        """
        return MCMCRun(X, numRuns=self.num_iterations, numMixtures=self.num_mixtures)

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
        self.background_samples = self.get_samples(background_features)

    def train_speakers(self, speaker_features):
        for speaker_id, features in speaker_features.iteritems():
            self.model_samples[speaker_id] = self.get_samples(features)

    def load_background(self, filename):
        with open(filename, 'rb') as fp:
            self.background_samples = cPickle.load(fp)

    def load_speakers(self, speaker_samples):
        for speaker_id, filename in speaker_samples.iteritems():
            with open(filename, 'rb') as fp:
                self.model_samples[speaker_id] = cPickle.load(fp)

    def verify(self, claimed_speaker, features):
        claimed_samples = self.model_samples[claimed_speaker]
        likelihood_calculator = Likelihood(features, self.num_mixtures)

        means, covars, weights = self.model_samples[claimed_speaker]
        claimed_likelihoods = []
        for i in xrange(len(means)):
            claimed_likelihoods.append(likelihood_calculator.loglikelihood(means[i], covars[i], weights[i]))

        means, covars, weights = self.background_samples
        background_likelihoods = []
        for i in xrange(len(means)):
            background_likelihoods.append(likelihood_calculator.loglikelihood(means[i], covars[i], weights[i]))


        claimed = logsumexp(np.array(claimed_likelihoods))
        background = logsumexp(np.array(background_likelihoods))
        likelihood_ratio = claimed - background
        return likelihood_ratio


def save_enrolment_samples(model_data, save_path, num_iterations, num_gaussians):
    model_name = model_data['name']
    model_features = model_data['features']
    system = MonteCarloSystem(num_gaussians=num_gaussians, num_iterations=num_iterations)
    samples = system.get_samples(model_features)
    filename = os.path.join(save_path, 'gaussians' +  str(num_gaussians), 'iterations' + str(num_iterations), model_name + '.npy' )
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as fp:
        cPickle.dump(samples, fp, cPickle.HIGHEST_PROTOCOL)


def calculate_samples(trial, save_path, num_iterations, num_gaussians=8):
    print "starting calculation"
    system = MonteCarloSystem(num_gaussians=num_gaussians, num_iterations=num_iterations)
    start = time.time()
    system.compute_samples_save(trial, save_path)
    print time.time() - start

if __name__ == '__main__':
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)
    speaker_trials = manager.get_trial_data()

    system = MonteCarloSystem(num_gaussians=8, num_iterations=200000)
    system.load_background(os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians8', 'iterations200000', 'background.npy'))

    speaker_data = {}
    for speaker_id, _ in manager.get_enrolment_data().iteritems():
        speaker_data[speaker_id] = os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians8', 'iterations200000', speaker_id + '.npy')

    system.load_speakers(speaker_data)

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
    np.save(os.path.join(config.dropbox_directory, 'scoresMonte200000.npy'), likelihood_array)
    np.save(os.path.join(config.dropbox_directory, 'answersMonte200000.npy'), answer_array)
    with open('systemMonte.pickle', 'wb') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)