__author__ = 'jeremyma'
from bob.bio.gmm.algorithm import GMM
import gmmmc
import logging
from frontend import  frontend
import config, os
import pdb
import bob
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle


class BobGmmSystem():

    def __init__(self, num_gaussians=8):
        self.model = GMM(num_gaussians)
        self.individuals = {}

    def train_background(self, X):
        self.model.train_ubm(X)
        # enroll trainer because train_ubm has a bug
        self.model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.model.ubm,
                                    relevance_factor = self.model.relevance_factor,
                                    update_means = True, update_variances = False)
        #self.model.gmm_enroll_iterations = 1
        #self.model.training_threshold = 1e-8
        self.ubm = gmmmc.GMM(np.array(self.model.ubm.means),
                             np.array(self.model.ubm.variances),
                             np.array(self.model.ubm.weights))

    def train_speakers(self, speaker_features):
        """
        enrol speakers
        :param speaker_features: dictionary of feature vectors corresponding to enrolment data for each speaker
        """
        for speaker_id, features in speaker_features.iteritems():
            logging.info("BobGmmSystem: Training speaker: {0}".format(speaker_id))
            print "BobGmmSystem: Training speaker: {0}".format(speaker_id)
            #print features.shape
            bobgmm = self.model.enroll_gmm(features)
            self.individuals[speaker_id] = gmmmc.GMM(bobgmm.means, bobgmm.variances, bobgmm.weights)

    def verify(self, claimed_speaker, features):
        """

        :param claimed_speaker: string
        :param features: numpy array
        :return:
        """
        likelihood_ratio = self.individuals[claimed_speaker].log_likelihood(features) / features.shape[0] - \
                           self.ubm.log_likelihood(features) / features.shape[0]
        #likelihood_ratio = self.individuals[claimed_speaker].log_likelihood(features) - np.sum(ubm.score(features))

        return likelihood_ratio

if __name__ == '__main__':

    n_mixtures = 8

    relevance_factor = 150
    ubm_size = 'smallubm'
    gender = 'female'

    if gender == 'male':
        enrolment = config.reddots_part4_enrol_male
        trials = config.reddots_part4_trial_male
        background = config.background_data_directory_male
    else:
        enrolment = config.reddots_part4_enrol_female
        trials = config.reddots_part4_trial_female
        background = config.background_data_directory_female

    if ubm_size == 'smallubm':
        manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                       enrol_file=enrolment,
                                       trial_file=trials)
    else:
        manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                       enrol_file=enrolment,
                                       trial_file=trials,
                                       background_data_directory=background)

    # total = sum([len(trials) for _, trials in manager.speaker_trials.iteritems()])


    back = manager.get_background_data()
    print back.shape
    for relevance_factor in [4, 10, 20, 50, 100, 150, 200, 250]:
        system = BobGmmSystem(num_gaussians=n_mixtures)
        print "training background"
        system.train_background(back)
        print "training speaker models"

        system.model.relevance_factor=relevance_factor

        system.train_speakers(manager.get_enrolment_data())

        numCorrect = 0
        numTrues = 0

        answer_array = []
        likelihood_array = []
        llset = set()
        for speaker_id, trials in manager.speaker_trials.iteritems():
            for trial in trials:
                answer_array.append(trial.answer)
                ll = system.verify(trial.claimed_speaker, trial.get_data())
                likelihood_array.append(ll)

        #save results

        np.save('../gmm_ubm_results/map_scores' + str(n_mixtures) + '_{0}'.format(gender) +
                '_rel' + str(relevance_factor) + '_' + ubm_size + '.npy',
                likelihood_array)
        np.save('../gmm_ubm_results/map_answers' + str(n_mixtures) + '_{0}'.format(gender) +
                '_rel' + str(relevance_factor) + '_' + ubm_size + '.npy', answer_array)
