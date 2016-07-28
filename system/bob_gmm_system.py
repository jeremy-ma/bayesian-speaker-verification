__author__ = 'jeremyma'
from bob.bio.gmm.algorithm import GMM
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
        self.likelihood_threshold = 1.0

    def train_background(self, X):
        self.model.train_ubm(X)
        pdb.set_trace()
        # enroll trainer because train_ubm has a bug
        self.model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.model.ubm,
                                    relevance_factor = self.model.relevance_factor,
                                    update_means = True, update_variances = False)
        self.model.gmm_enroll_iterations = 5

    def train_speakers(self, speaker_features):
        """
        enrol speakers
        :param speaker_features: dictionary of feature vectors corresponding to enrolment data for each speaker
        """
        for speaker_id, features in speaker_features.iteritems():
            logging.info("BobGmmSystem: Training speaker: {0}".format(speaker_id))
            print "BobGmmSystem: Training speaker: {0}".format(speaker_id)
            #print features.shape
            self.individuals[speaker_id] = self.model.enroll_gmm(features)

    def verify(self, claimed_speaker, features):
        """

        :param claimed_speaker: string
        :param features: numpy array
        :return:
        """
        ubm = self.model.ubm
        likelihood_ratio = self.individuals[claimed_speaker].log_likelihood(features) - ubm.log_likelihood(features)

        return likelihood_ratio

if __name__ == '__main__':

    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    total = sum([len(trials) for _, trials in manager.speaker_trials.iteritems()])

    system = BobGmmSystem(num_gaussians=128)
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
    np.save('map_scores.npy', likelihood_array)
    np.save('map_answers.npy', answer_array)
    with open('systemG128.pickle', 'wb') as fp:
        pickle.dump(system, fp, pickle.HIGHEST_PROTOCOL)

