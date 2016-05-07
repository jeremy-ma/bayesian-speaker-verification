__author__ = 'jeremyma'
import config
from system import prototype_montecarlo_system
import multiprocessing
from functools import partial
from frontend import frontend
import os
import time


def batch_enrol():
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)
    save_path = os.path.join(config.dropbox_directory, config.computer_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #put into map format

    enrolment_data = []
    enrolment_data = [{'name': speaker_id, 'features': features} for speaker_id, features in manager.get_enrolment_data().iteritems() ]
    enrolment_data.append({'name': 'background', 'features': manager.get_background_data()})

    #for el in enrolment_data:
    #    prototype_montecarlo_system.save_enrolment_samples(el, save_path, 100000, 8)
    pool = multiprocessing.Pool(None)
    map_function = partial(prototype_montecarlo_system.save_enrolment_samples,
                           save_path=save_path, num_iterations=100, num_gaussians=8)
    pool.map(map_function, enrolment_data)

def batch_all():
    start = time.time()
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    save_path = os.path.join(config.dropbox_directory, config.computer_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    speaker_trials = manager.get_unique_trials()
    all_trials = []
    for _, trials in speaker_trials.iteritems():
        all_trials.extend(trials)

    pool = multiprocessing.Pool(None)
    map_function = partial(prototype_montecarlo_system.calculate_samples,
                           save_path=save_path, num_iterations=100, num_gaussians=8)
    pool.map(map_function, all_trials)
    print "done"
    print time.time() - start


if __name__ == '__main__':
    batch_enrol()