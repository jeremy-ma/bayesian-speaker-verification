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
            likelihood_array.append(system.verify(trial.claimed_speaker,
                        trial.get_data(), n_jobs, burn_in=25000, lag=50))
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
        map_est = samples[0]
        max_prob = map_est.log_likelihood(features)
        max_ind = 0
        for i, gmm in enumerate(samples):
            prob = gmm.log_likelihood(features) + system.prior.log_prob(gmm)
            if prob > max_prob:
                print "found new map at {0}!".format(i)
                map_est = gmm
                max_prob = prob
                max_ind = i
        print "found map at index {0}".format(max_ind)
        newsys.individuals[speaker_id] = map_est

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

if __name__=='__main__':
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)
    n_jobs = -1
    n_mixtures, n_runs = 8, 50000
    description = 'mcmc_rel150_mapstart_female'
    save_path = os.path.join(config.dropbox_directory, config.computer_id, description,
                             'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    filename = os.path.join(save_path, 'system.pickle')

    with open(filename, 'r') as fp:
        system = cPickle.load(fp)

    evaluate_system(system, manager, n_jobs, save_path)
    evaluate_MCMAP(system, manager, n_jobs, save_path)

