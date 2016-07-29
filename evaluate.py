__author__ = 'jeremyma'
import config
from system import mcmc_system, bob_gmm_system
from frontend import frontend
import numpy as np
import cPickle
import os
import logging, pdb

def evaluate_system(system, manager, n_jobs, description):
    answer_array = []
    likelihood_array = []

    count = 0
    for speaker_id, trials in manager.speaker_trials.iteritems():
        for trial in trials:
            count += 1
            if count % 50 == 0:
                print "iteration {0}".format(count)
            answer_array.append(trial.answer)
            likelihood_array.append(system.verify(trial.claimed_speaker, trial.get_data(), n_jobs, burn_in=0, lag=1))

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)

    #save results
    np.save(os.path.join(save_path,  'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                         'scoresMHMC' + 'G' + str(system.n_mixtures) +
                         '_N' + str(system.n_runs) + '.npy'), likelihood_array)

    np.save(os.path.join(save_path,  'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                         'answersMHMC' + 'G' + str(system.n_mixtures) +
                         '_N' + str(system.n_runs) + '.npy'), answer_array)

def evaluate_MCMAP(system, manager, n_jobs, description):

    newsys = bob_gmm_system.BobGmmSystem(system.n_mixtures)
    newsys.model.ubm = system.ubm
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

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)

    #save results
    np.save(os.path.join(save_path,  'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                         'scoresMCMAPPrior' + 'G' + str(system.n_mixtures) +
                         '_N' + str(system.n_runs) + '.npy'), likelihood_array)

    np.save(os.path.join(save_path,  'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                         'answersMCMAPPrior' + 'G' + str(system.n_mixtures) +
                         '_N' + str(system.n_runs) + '.npy'), answer_array)




if __name__=='__main__':
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    n_mixtures, n_runs = 8, 100
    data = manager.get_background_data()
    description = 'mcmc_gaussian_prior_gmmcovars'
    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
    filename = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                            'system' + '.pickle')

    with open(filename, 'r') as fp:
        system = cPickle.load(fp)
    evaluate_system(system, manager, 1, description)