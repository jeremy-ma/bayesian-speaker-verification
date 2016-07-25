__author__ = 'jeremyma'
import config
from system import mcmc_system
import multiprocessing
from functools import partial
from frontend import frontend
import numpy as np
import cPickle
import os
import time
import logging
def batch_evaluate(num_iterations, num_gaussians):
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)
    speaker_trials = manager.get_trial_data()

    system = mcmc_system.MonteCarloSystem(num_gaussians=num_gaussians, num_iterations=num_iterations)
    system.load_background(os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians' + str(num_gaussians), 'iterations' + str(num_iterations), 'background.npy'))

    speaker_data = {}
    for speaker_id, _ in manager.get_enrolment_data().iteritems():
        speaker_data[speaker_id] = os.path.join(config.dropbox_directory, 'MonteCarloSamples',
                                        'gaussians' + str(num_gaussians), 'iterations' + str(num_iterations), speaker_id + '.npy')

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
    np.save(os.path.join(config.dropbox_directory, 'scoresMonte' + str(num_iterations) + '.npy'), likelihood_array)
    np.save(os.path.join(config.dropbox_directory, 'answersMonte' + str (num_iterations) + '.npy'), answer_array)
    with open('systemMonte.pickle', 'wb') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)


def batch_enrol(n_mixtures, n_runs):
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)
    save_path = os.path.join(config.dropbox_directory, config.computer_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    system = mcmc_system.MCMC_ML_System(n_mixtures=n_mixtures, n_runs=n_runs)
    logging.info('Training background model...')
    system.train_background(manager.get_background_data())
    for speaker_id, features in manager.get_enrolment_data().iteritems():
        logging.info('Sampling Speaker:{0}'.format(str(speaker_id)))
        samples = system.get_samples(features, -1)
        filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs,
                                speaker_id + '.pickle'))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as fp:
            cPickle.dump(samples, fp, cPickle.HIGHEST_PROTOCOL)

    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                                'system' + '.pickle')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    batch_enrol(32, 100)
