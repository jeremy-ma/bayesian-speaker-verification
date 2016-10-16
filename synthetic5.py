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

from sklearn.datasets import load_iris

logging.getLogger().setLevel(logging.INFO)

n_mixtures = 4
n_features = 4

iris = load_iris()
X = iris.data
Y = iris.target

for target in [0,1,2]:
    class_data = X[Y==target]
    print class_data.shape
    ML_gmm = sklearn.mixture.GMM(n_mixtures)
    ML_gmm.fit(X)
    ML_gmm = gmmmc.GMM(ML_gmm.means_, covariances=ML_gmm.covars_, weights=ML_gmm.weights_)


    prior = GMMPrior(MeansUniformPrior(-10, 10, n_mixtures, n_features),
                     # MeansUniformPrior(-1,1,2,1),
                     DiagCovarsUniformPrior(0.00001, 10.0, n_mixtures, n_features),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0001, 0.001, 0.005]),
                                          propose_covars=GaussianStepCovarProposal(step_sizes=[0.000001, 0.00001, 0.0001]),
                                          propose_weights=GaussianStepWeightsProposal(n_mixtures,
                                                                                      step_sizes=[0.0001, 0.001, 0.01]))

    # ais = AnnealedImportanceSampling(proposal, priors, )
    mcmc = MarkovChain(proposal, prior, ML_gmm)

    mcmc_samples = mcmc.sample(X, 10000, -1)

    print proposal.propose_mean.get_acceptance()
    print proposal.propose_covars.get_acceptance()
    print proposal.propose_weights.get_acceptance()