import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging
import matplotlib.pyplot as plt

np.random.seed(3)
logging.getLogger().setLevel(logging.INFO)


true_background = gmmmc.GMM(means=np.array([[-0.5], [-0.1]]),
                            covariances=np.array([[0.01], [0.01]]),
                            weights=np.array([0.5, 0.5]))

true_speaker = gmmmc.GMM(means=np.array([[0.8], [0.5]]),
                         covariances=np.array([[0.01], [0.01]]),
                         weights=np.array([0.5, 0.5]))

background_X = true_background.sample(10000)
speaker_X = true_speaker.sample(1000)
#speaker_X = np.concatenate((speaker_X, background_X[:1000]))

model = bob.bio.gmm.algorithm.GMM(2)

model.train_ubm(background_X)

print model.ubm.means

# enroll trainer because train_ubm has a bug
model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(model.ubm,
                                                   relevance_factor=model.relevance_factor,
                                                   update_means=True, update_variances=False)
ubm = gmmmc.GMM(np.array(model.ubm.means),
                np.array(model.ubm.variances),
                np.array(model.ubm.weights))
bobgmm = model.enroll_gmm(speaker_X)
speaker_gmm = gmmmc.GMM(np.array(bobgmm.means), np.array(bobgmm.variances), np.array(bobgmm.weights))

prior = GMMPrior(MeansGaussianPrior(ubm.means, ubm.covars/2),
                 #MeansUniformPrior(-1,1,2,1),
                 CovarsStaticPrior(np.array(ubm.covars)),
                 WeightsStaticPrior(np.array(ubm.weights)))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003, 0.001]),
                                      propose_covars=None,
                                      propose_weights=None)


mcmc = MarkovChain(proposal, prior, ubm)

samples = mcmc.sample(speaker_X, 10000)
print proposal.propose_mean.get_acceptance()

mapest = samples[0]
mapprob = mapest.log_likelihood(speaker_X) + prior.log_prob(mapest)
for sample in samples:
    prob = sample.log_likelihood(speaker_X) + prior.log_prob(sample)
    if prob > mapprob:
        mapprob = prob
        mapest = sample


final = samples[-1]

mc_means = [[s.means[0][0], s.means[1][0]] for s in samples[::10]]
mc_means = np.array(mc_means)
"""
mcmc = plt.scatter(mc_means[:,0], mc_means[:,1], color= 'b')
map = plt.scatter(speaker_gmm.means[0][0], speaker_gmm.means[1][0], color='r', s=500.0)

true = plt.scatter(true_speaker.means[0][0], true_speaker.means[1][0], color='g', s=500)
prior = plt.scatter(ubm.means[0][0], ubm.means[1][0], color= 'y', s=500)
plt.title('Samples from Posterior Distribution of GMM Means', fontsize=22)
plt.xlabel('Mixture 1 mean', fontsize=22)
plt.ylabel('Mixture 2 mean', fontsize=22)

plt.legend((map, mcmc, prior, true),
           ('MAP estimate', 'Monte Carlo Samples', 'Prior Means', 'Data Means'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=22)
"""
x = np.linspace(-1, 1, 1000)
import matplotlib.mlab as mlab

for s in mc_means[700:710]:
    plt.plot(x,mlab.normpdf(x, s[0], 0.01) + mlab.normpdf(x, s[1], 0.01))
    plt.show()
