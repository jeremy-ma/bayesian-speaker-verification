import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior,\
    DiagCovarsUniformPrior, DiagCovarsWishartPrior, WeightsUniformPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging
import matplotlib.pyplot as plt
import pdb
import sklearn.mixture.gmm
from scipy.misc import logsumexp
import scipy.stats
from sklearn.metrics import mean_squared_error

stringresults = ''

ml_rmse = []

mcmc_rmse = []

counts = []


#np.random.seed(3)
n_mixtures = 4
n_features = 2
n_samples = 1000

logging.getLogger().setLevel(logging.INFO)


truth_gmm = gmmmc.GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
            covariances=np.random.uniform(low=0.001, high=0.01, size=(n_mixtures, n_features)),
            weights=np.random.dirichlet(np.ones((n_mixtures))))

truth2_gmm = gmmmc.GMM(means=np.random.uniform(low=-1, high=1, size=(n_mixtures, n_features)),
            covariances=np.random.uniform(low=0.001, high=0.01, size=(n_mixtures, n_features)),
            weights=np.random.dirichlet(np.ones((n_mixtures))))

#print truth_gmm.means, truth_gmm.covars, truth_gmm.weights
# draw samples from the true distribution
X = truth_gmm.sample(n_samples)
X_2 = truth2_gmm.sample(n_samples)

for _ in xrange(10):

    temp = sklearn.mixture.GMM(n_mixtures)
    temp.fit(X)
    ML_gmm = gmmmc.GMM(temp.means_, covariances=temp.covars_, weights=temp.weights_)
    temp = sklearn.mixture.GMM(n_mixtures)
    temp.fit(X_2)
    ML_gmm2 = gmmmc.GMM(temp.means_, covariances=temp.covars_, weights=temp.weights_)


    print "##############"
    print ML_gmm.means, ML_gmm.covars, ML_gmm.weights

    prior = GMMPrior(MeansUniformPrior(-10,10,n_mixtures,n_features),
                     #MeansUniformPrior(-1,1,2,1),
                     DiagCovarsUniformPrior(0.00001, 1.0, n_mixtures, n_features),
                     WeightsUniformPrior(n_mixtures))

    proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0001, 0.001, 0.005]),
                                  propose_covars=GaussianStepCovarProposal(step_sizes=[0.000001, 0.00001, 0.0001]),
                                  propose_weights=GaussianStepWeightsProposal(n_mixtures, step_sizes=[0.0001, 0.001, 0.01]))

    #ais = AnnealedImportanceSampling(proposal, priors, )
    mcmc = MarkovChain(proposal, prior, ML_gmm)

    mcmc_samples = mcmc.sample(X, 1000, -1)
    mcmc_samples2 = mcmc.sample(X_2, 1000, -1)

    print proposal.propose_mean.get_acceptance()
    print proposal.propose_covars.get_acceptance()
    print proposal.propose_weights.get_acceptance()

    print "#####################################"

    X_test = truth_gmm.sample(1000)

    count = 0

    ll_mcmc = []
    ll_ml = []
    ll_true = []

    for sample in X_test:
        sample = np.array([sample])

        ratio_mcmc = logsumexp([gmm.log_likelihood(sample, -1) for gmm in mcmc_samples]) \
                       - np.log(len(mcmc_samples)) - \
                     logsumexp([gmm.log_likelihood(sample, -1) for gmm in mcmc_samples2]) \
                     - np.log(len(mcmc_samples))
        ll_mcmc.append(ratio_mcmc)
        ll_ml.append(ML_gmm.log_likelihood(sample, -1) - ML_gmm2.log_likelihood(sample, -1))
        ll_true.append(truth_gmm.log_likelihood(sample, -1) - truth2_gmm.log_likelihood(sample))
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