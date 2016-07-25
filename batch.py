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

    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                            'ubm' + '.pickle')

    try:
        system.load_background(filename)
        logging.info('Loaded background model from file')
    except:
        logging.info('Training background model...')
        system.train_background(manager.get_background_data())
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as fp:
            cPickle.dump(system.ubm, fp, cPickle.HIGHEST_PROTOCOL)
        logging.info('Finished, saved background model to file...')

    logging.info('Beginning Monte Carlo Sampling')

    save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    system.train_speakers(manager.get_enrolment_data(), -1, save_dir)

    logging.info('Saving system to file')
    with open(os.path.join(save_dir, 'system.pickle'), 'w') as fp:
        cPickle.dump(system, fp, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    batch_enrol(2, 100)
