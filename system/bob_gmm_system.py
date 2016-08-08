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
        self.model.relevance_factor = 10
        self.model.gmm_enroll_iterations = 2
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

    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    total = sum([len(trials) for _, trials in manager.speaker_trials.iteritems()])

    system = BobGmmSystem(num_gaussians=8)
    print "training background"
    system.train_background(manager.get_background_data())
    print "training speaker models"
    system.train_speakers(manager.get_enrolment_data())

    numCorrect = 0
    numTrues = 0

    answer_array = []
    likelihood_array = []
    for speaker_id, trials in manager.speaker_trials.iteritems():
        for trial in trials:
            answer_array.append(trial.answer)
            likelihood_array.append(system.verify(trial.claimed_speaker, trial.get_data()))

    #save results
    np.save('../map_scores.npy', likelihood_array)
    np.save('../map_answers.npy', answer_array)

