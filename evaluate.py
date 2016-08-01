__author__ = 'jeremyma'
import config
from system import mcmc_system, bob_gmm_system
from frontend import frontend
import numpy as np
import cPickle
import os
from system.mcmc_system import MCMC_ML_System
import logging, pdb
from gmmmc import GMM

def evaluate_system(system, manager, n_jobs, save_path):
    answer_array = []
    likelihood_array = []

    count = 0
    for speaker_id, trials in manager.speaker_trials.iteritems():
        for trial in trials:
            count += 1
            if count % 50 == 0:
                print "iteration {0}".format(count)
            answer_array.append(trial.answer)
            likelihood_array.append(system.verify(trial.claimed_speaker, trial.get_data(), n_jobs, burn_in=10000, lag=50))

    #save results
    np.save(os.path.join(save_path, 'scoresMHMC.npy'), likelihood_array)

    np.save(os.path.join(save_path, 'answersMHMC.npy'), answer_array)

def evaluate_MCMAP(system, manager, n_jobs, save_path):

    newsys = bob_gmm_system.BobGmmSystem(system.n_mixtures)
    newsys.ubm = system.ubm

    # find map estimate
    for speaker_id, features in manager.get_enrolment_data().iteritems():
        print speaker_id
        samples = system.model_samples[speaker_id]
        map = samples[0]
        max_prob = map.log_likelihood(features)
        for gmm in samples[1:]:
            prob = gmm.log_likelihood(features) + system.prior.log_prob(gmm)
            if prob > max_prob:
                map = gmm
                max_prob = prob
        newsys.individuals[speaker_id] = map

    print "found map"
    answer_array = []
    likelihood_array = []

    count = 0
    for speaker_id, trials in manager.speaker_trials.iteritems():
        for trial in trials:
            count += 1
            if count % 50 == 0:
                print "iteration {0}".format(count)
            answer_array.append(trial.answer)
            likelihood_array.append(newsys.verify(trial.claimed_speaker, trial.get_data()))

    #save results
    np.save(os.path.join(save_path, 'scoresMCMAP.npy'), likelihood_array)
    np.save(os.path.join(save_path, 'answersMCMAP.npy'), answer_array)

def reduce_system(speaker_names, save_path):

    system = MCMC_ML_System(8, 10000)

    for speaker_name in speaker_names:
        with open(os.path.join(save_path, speaker_name + '.pickle')) as fp:
            arr = cPickle.load(fp)
            arr = arr[::50]
            system.model_samples[speaker_name] = arr

    with open(os.path.join(save_path, 'ubm.pickle')) as fp:
        ubm = cPickle.load(fp)
        system.load_background(ubm)

    with open(os.path.join(save_path, 'reducedSystem.pickle'), 'w') as fp:
        cPickle.dump(system, fp, protocol=cPickle.HIGHEST_PROTOCOL)

    return system

if __name__=='__main__':
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    n_mixtures, n_runs = 8, 20000
    data = manager.get_background_data()
    description = 'mcmc_gaussian_prior_gmmcovars'
    save_path = os.path.join(config.dropbox_directory, config.computer_id, description,
                             'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    filename = os.path.join(save_path, 'system.pickle')

    with open(filename, 'r') as fp:
        system = cPickle.load(fp)

    #system = reduce_system(['f0002', 'f0004', 'f0005', 'f0006', 'f0008', 'f0012'], save_path)

    evaluate_system(system, manager, 1, save_path)
    evaluate_MCMAP(system, manager, 1, save_path)

