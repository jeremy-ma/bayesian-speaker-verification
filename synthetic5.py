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
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.datasets import load_iris,load_digits
from sklearn.cross_validation import  StratifiedKFold, train_test_split
from sklearn.preprocessing import Normalizer


logging.getLogger().setLevel(logging.INFO)

n_mixtures = 4
n_classes = 10
iris = load_digits(n_class=n_classes)
norm = Normalizer()
X = norm.fit_transform(iris.data)
Y = iris.target
#X,Y = make_classification(n_samples=1800, n_features=n_features, n_classes=3, n_informative=10)
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y)
MLstuff = []
MCMCstuff = []
train_data = X_train
train_labels = y_train

for target in range(n_classes):
    class_data = train_data[train_labels==target]
    print class_data.shape
    ML_gmm = sklearn.mixture.GMM(n_mixtures)
    ML_gmm.fit(class_data)
    ML_gmm = gmmmc.GMM(ML_gmm.means_, covariances=ML_gmm.covars_, weights=ML_gmm.weights_)

    MLstuff.append(ML_gmm)

    prior = GMMPrior(MeansUniformPrior(-10, 10, n_mixtures, n_features),
                     # MeansUniformPrior(-1,1,2,1),
                     DiagCovarsUniformPrior(0.0000000001, 10.0, n_mixtures, n_features),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.00001, 0.000001, 0.0000005]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.0000001, 0.00000001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.002]))

    # ais = AnnealedImportanceSampling(proposal, priors, )
    mcmc = MarkovChain(proposal, prior, ML_gmm)

    mcmc_samples = mcmc.sample(class_data, 5000, 1)

    MCMCstuff.append(mcmc_samples)

    print proposal.propose_mean.get_acceptance()
    print proposal.propose_covars.get_acceptance()
    print proposal.propose_weights.get_acceptance()

guesses = []
ml_answers = []
mcmc_answers = []
for x in X_train:
    ml_scores = []
    mcmc_scores = []
    for target in range(n_classes):
        ml_scores.append(MLstuff[target].log_likelihood(np.array([x]),1))
        mcmc_samples = MCMCstuff[target]
        mcmc_scores.append(logsumexp([gmm.log_likelihood(np.array([x]), 1) for gmm in mcmc_samples[::10]]) \
                       - np.log(len(mcmc_samples[::10])))
    ml_answers.append(np.argmax(ml_scores))
    mcmc_answers.append(np.argmax(mcmc_scores))

print "ml train accuracy:{0}".format(accuracy_score(ml_answers, y_train))
print "mcmc train accuracy:{0}".format(accuracy_score(mcmc_answers, y_train))

for x in X_test:
    ml_scores = []
    mcmc_scores = []
    for target in range(n_classes):
        ml_scores.append(MLstuff[target].log_likelihood(np.array([x]),1))
        mcmc_samples = MCMCstuff[target]
        mcmc_scores.append(logsumexp([gmm.log_likelihood(np.array([x]), 1) for gmm in mcmc_samples[::10]]) \
                       - np.log(len(mcmc_samples[::10])))
    ml_answers.append(np.argmax(ml_scores))
    mcmc_answers.append(np.argmax(mcmc_scores))

print "ml accuracy:{0}".format(accuracy_score(ml_answers, y_test))
print "mcmc accuracy:{0}".format(accuracy_score(mcmc_answers, y_test))

