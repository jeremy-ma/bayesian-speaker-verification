import os
import cPickle
from frontend import frontend
import sys, pdb
import config
import time
import numpy as np
from scipy.misc import logsumexp
from gmmmc import GMM
import sklearn.mixture
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging
import bob.bio.gmm.algorithm
from bob.bio.gmm.algorithm import GMM as BobGMM
import gmmmc
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior,\
    DiagCovarsUniformPrior, DiagCovarsWishartPrior, WeightsUniformPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import multiprocessing
import copy
import bob
from system.full_system import KLDivergenceMLStartSystem, KLDivergenceMAPStartSystem
from shutil import copyfile
from sklearn.metrics import mean_squared_error

def diffexp(x, y):
    a = np.maximum(x,y)
    x_ = x - a
    y_ = y - a
    return (np.exp(x_) - np.exp(y_)) * np.exp(a)

def mean_squared_error_log(true, test):
    # accepts log probabilities
    return np.average(diffexp(true, test) ** 2)

ml_rmse = []
mcmc_rmse = []

ml_rmse2 = []
mcmc_rmse2 = []


counts = []

logging.getLogger().setLevel(logging.INFO)
n_mixtures, n_runs, n_samples, n_features = 8, 10, 1000, 60
gender = 'female'

if gender == 'male':
    enrolment = config.reddots_part4_enrol_male
    trials = config.reddots_part4_trial_male
    background = config.background_data_directory_male
else:
    enrolment = config.reddots_part4_enrol_female
    trials = config.reddots_part4_trial_female
    background = config.background_data_directory_female

manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=enrolment,
                               trial_file=trials,
                               background_data_directory=background)

system = KLDivergenceMAPStartSystem(n_mixtures, 2, 2)

system.train_ubm(manager.get_background_data())

truth_gmm = system.ubm

#print truth_gmm.means, truth_gmm.covars, truth_gmm.weights
# draw samples from the true distribution

stringresults = ''

X = truth_gmm.sample(n_samples)
for _ in xrange(10):
    ML_gmm = sklearn.mixture.GMM(n_mixtures)
    ML_gmm.fit(X)
    ML_gmm = gmmmc.GMM(ML_gmm.means_, covariances=ML_gmm.covars_, weights=ML_gmm.weights_)
    print "##############"
    print ML_gmm.means, ML_gmm.covars, ML_gmm.weights

    prior = GMMPrior(MeansUniformPrior(-10,10,n_mixtures,n_features),
                     #MeansUniformPrior(-1,1,2,1),
                     DiagCovarsUniformPrior(0.00001, 10.0, n_mixtures, n_features),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0001, 0.001, 0.005]),
                                  propose_covars=GaussianStepCovarProposal(step_sizes=[0.000001, 0.00001, 0.0001]),
                                  propose_weights=GaussianStepWeightsProposal(n_mixtures, step_sizes=[0.0001, 0.001, 0.01]))

    #ais = AnnealedImportanceSampling(proposal, priors, )
    mcmc = MarkovChain(proposal, prior, ML_gmm)

    mcmc_samples = mcmc.sample(X, 10000, -1)

    print proposal.propose_mean.get_acceptance()
    print proposal.propose_covars.get_acceptance()
    print proposal.propose_weights.get_acceptance()

    X_test = truth_gmm.sample(1000)

    count = 0

    ll_mcmc = []
    ll_ml = []
    ll_true = []

    for sample in X_test:
        sample = np.array([sample])
        ll_mcmc.append(logsumexp([gmm.log_likelihood(sample, -1) for gmm in mcmc_samples[::100]]) \
                       - np.log(len(mcmc_samples[::100])))
        ll_ml.append(ML_gmm.log_likelihood(sample, -1))
        ll_true.append(truth_gmm.log_likelihood(sample, -1))
        #print ll_mcmc, ll_ml, ll_true

    ll_true = np.array(ll_true)
    ll_mcmc = np.array(ll_mcmc)
    ll_ml = np.array(ll_ml)

    l_true = np.exp(ll_true)
    l_mcmc = np.exp(ll_mcmc)
    l_ml = np.exp(ll_ml)


    count = np.sum(np.abs(l_true - l_mcmc) < np.abs(l_true - l_ml))

    stringresults += "\nMCMC MAE:{0} RMSE:{1}".format(np.mean(np.abs(l_true - l_mcmc)), mean_squared_error(l_true,l_mcmc)**0.5)
    stringresults += "\nML MAE:{0} RMSE:{1}".format(np.mean(np.abs(l_true - l_ml)), mean_squared_error(l_true,l_ml)**0.5)
    stringresults += "\n{0}".format(np.sum(np.abs(l_true - l_mcmc) < np.abs(l_true - l_ml)))


    ml_rmse.append(mean_squared_error(l_true,l_ml)**0.5)
    mcmc_rmse.append(mean_squared_error(l_true,l_mcmc)**0.5)

    counts.append(count)

    print float(count) / float(len(X_test))

print stringresults

print ml_rmse
print mcmc_rmse
print counts

print "ml_rmse mean:{0} std:{1}".format(np.mean(ml_rmse), np.std(ml_rmse))
print "mcmc_rmse mean:{0} std:{1}".format(np.mean(mcmc_rmse), np.std(mcmc_rmse))
print "count mean:{0} std:{1}".format(np.mean(counts), np.std(counts))
