__author__ = 'jeremyma'
from varun.MCMC import MCMCRun
import os
import cPickle
from frontend import frontend
import sys, pdb
import config
import time


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
    speaker_trials = manager.get_unique_trials()
    count = 0
    for speaker_id, trials in speaker_trials.iteritems():
        print "{0}: {1}".format(speaker_id, len(trials))
        count += len(trials)

    print count