import sklearn.mixture
import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging
import matplotlib.pyplot as plt
from frontend import frontend
import config
import os,pdb
from system import bob_gmm_system

def bhatta_dist_gaussian(mean1, covar1, mean2, covar2):
    diff = mean1 - mean2
    cov_both = (covar1 + covar2) / 2
    dist = np.sum(np.power(diff, 2) / cov_both) / 8
    dist += np.log(np.sum(cov_both) / np.sqrt(np.sum(covar1) * np.sum(covar2))) / 2
    return dist

def bhatta_dist(gmm1, gmm2):
    # https://en.wikipedia.org/wiki/Bhattacharyya_distance
    dist_set = []
    for mixture in xrange(gmm1.n_mixtures):
        mean1, mean2 = gmm1.means[mixture], gmm2.means[mixture]
        covar1, covar2 = gmm1.covars[mixture], gmm2.covars[mixture]
        dist_set.append(bhatta_dist_gaussian(mean1, covar1, mean2, covar2))

    return np.array(dist_set)

def symmetric_kl_divergence(gmm1, gmm2, n_samples):
    samples1 = gmm1.sample(n_samples)
    samples2 = gmm2.sample(n_samples)

    div = abs(gmm1.log_likelihood(samples1) - gmm2.log_likelihood(samples1) +
              gmm1.log_likelihood(samples2) - gmm2.log_likelihood(samples2))

    return div / (2 * n_samples)

def individual_mixtures():

    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)
    n_mixtures = 8
    np.random.seed(1)

    background = sklearn.mixture.GMM(n_mixtures)

    background.fit(manager.get_background_data())
    background = gmmmc.GMM(background.means_, background.covars_, background.weights_)

    new_means = np.array(background.means)
    for i, mean in enumerate(background.means):
        if i in [0,1,2,3]:
            new_means[i] = mean + 0.2

    speaker = gmmmc.GMM(new_means, np.array(background.covars), np.array(background.weights))

    background_samples = background.sample(10000)
    speaker_samples = speaker.sample(1000)

    logging.getLogger().setLevel(logging.INFO)

    system = bob_gmm_system.BobGmmSystem(n_mixtures)
    system.train_background(background_samples)
    system.model.ubm.means = background.means
    system.model.ubm.variances = background.covars
    system.model.ubm.weights = background.weights
    system.train_speakers({'test': speaker_samples})

    prior = GMMPrior(MeansGaussianPrior(np.array(background.means), np.array(background.covars) / 40),
                     CovarsStaticPrior(np.array(background.covars)),
                     WeightsStaticPrior(np.array(background.weights)))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003, 0.001]),
                                          propose_covars=None,
                                          propose_weights=None)

    mcmc = MarkovChain(proposal, prior, background)

    samples = mcmc.sample(speaker_samples, 5000)

    print "true results"
    print bhatta_dist(background, speaker)



    print "MAP results"
    gmm = system.individuals['test']
    dists = bhatta_dist(background, gmm)
    num_positive = sum(dists > 0.01)
    print dists
    print float(num_positive) / len(dists)

    finals = samples[2500::100]

    print "MCMC results"
    for gmm in finals:
        dists = bhatta_dist(background, gmm)
        num_positive = sum(dists > 0.01)
        print dists


def asWhole():
    manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                                   enrol_file=config.reddots_part4_enrol_female,
                                   trial_file=config.reddots_part4_trial_female)

    n_mixtures = 8
    np.random.seed(1)

    background = sklearn.mixture.GMM(n_mixtures)

    background.fit(manager.get_background_data())
    background = gmmmc.GMM(background.means_, background.covars_, background.weights_)

    new_means = np.array(background.means)
    for i, mean in enumerate(background.means):
        if i in [0, 1, 2, 3]:
            new_means[i] = mean + 0.2

    speaker = gmmmc.GMM(new_means, np.array(background.covars), np.array(background.weights))

    background_samples = background.sample(10000)
    speaker_samples = speaker.sample(1000)

    logging.getLogger().setLevel(logging.INFO)

    system = bob_gmm_system.BobGmmSystem(n_mixtures)
    #system.model.gmm_enroll_iterations = 25
    #system.model.training_threshold = 1e-6
    system.train_background(background_samples)
    system.model.ubm.means = background.means
    system.model.ubm.variances = background.covars
    system.model.ubm.weights = background.weights

    relevance_factor = 100

    prior = GMMPrior(MeansGaussianPrior(np.array(background.means), np.array(background.covars) / relevance_factor),
                     CovarsStaticPrior(np.array(background.covars)),
                     WeightsStaticPrior(np.array(background.weights)))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003]),
                                          propose_covars=None,
                                          propose_weights=None)

    mcmc = MarkovChain(proposal, prior, background)

    samples = mcmc.sample(speaker_samples, 5000)

    print proposal.propose_mean.get_acceptance()

    n_samples = 100000

    background_str = "kl divergence to background: {0}"
    speaker_str = "kl divergence to speaker: {0}"

    print "true results:"
    true_background_div = symmetric_kl_divergence(background, speaker, n_samples)
    print background_str.format(true_background_div)

    map_div_background = []
    map_div_speaker = []
    print "MAP results"
    for _ in xrange(10):
        system.train_speakers({'test': speaker_samples})
        gmm = system.individuals['test']
        back = symmetric_kl_divergence(background, gmm, n_samples)
        speak = symmetric_kl_divergence(background, gmm, n_samples)
        print background_str.format(back)
        print speaker_str.format(speak)
        map_div_background.append(back)
        map_div_speaker.append(speak)


    finals = samples[2500::50]

    print "MCMC results"
    mcmc_div_background = []
    mcmc_div_speaker = []
    for i, gmm in enumerate(finals):
        print "samples: {0}".format(i)
        back = symmetric_kl_divergence(background, gmm, n_samples)
        speak = symmetric_kl_divergence(speaker, gmm, n_samples)
        mcmc_div_background.append(back)
        mcmc_div_speaker.append(speak)
        print background_str.format(back)
        print speaker_str.format(speak)

    f, axarr = plt.subplots(2)
    axarr[0].scatter(true_background_div, 0, color='g')
    axarr[0].scatter(map_div_background, np.ones(len(map_div_background)), color='g')
    axarr[0].scatter(mcmc_div_background, 2 * np.ones(len(mcmc_div_background)), color='g')
    axarr[0].set_title('Distance to Background model')
    axarr[0].set_ylabel('0: true model, 1: map models, 2:mcmc samples')

    axarr[1].scatter(map_div_speaker, np.ones(len(map_div_speaker)), color='r')
    axarr[1].scatter(mcmc_div_speaker, 2 * np.ones(len(mcmc_div_speaker)), color='r')
    axarr[1].set_title("Distance to true speaker model")
    axarr[1].set_ylabel('0: true model, 1: map models, 2:mcmc samples')
    plt.show()


if __name__ == '__main__':
    asWhole()