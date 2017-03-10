import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain, AnnealedImportanceSampling
import logging
import matplotlib.pyplot as plt
import pdb
from Hamiltonian import LeapFrogMeansProposal

np.random.seed(3)
logging.getLogger().setLevel(logging.INFO)


true_background = gmmmc.GMM(means=np.array([[-0.1], [-0.5]]),
                            covariances=np.array([[0.01], [0.01]]),
                            weights=np.array([0.5, 0.5]))

true_speaker = gmmmc.GMM(means=np.array([[0.8], [0.5]]),
                         covariances=np.array([[0.01], [0.01]]),
                         weights=np.array([0.5, 0.5]))

background_X = true_background.sample(10000)
speaker_X = true_speaker.sample(1000)
#speaker_X = np.concatenate((speaker_X, background_X[:1000]))

model = bob.bio.gmm.algorithm.GMM(2)
model.relevance_factor=2

model.train_ubm(background_X)

model.ubm.means = true_background.means
model.ubm.variances = true_background.covars
model.ubm.weights = true_background.weights

# enroll trainer because train_ubm has a bug
model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(model.ubm,
                                                   relevance_factor=model.relevance_factor,
                                                   update_means=True, update_variances=False, update_weights=False)
ubm = gmmmc.GMM(np.array(model.ubm.means),
                np.array(model.ubm.variances),
                np.array(model.ubm.weights))
bobgmm = model.enroll_gmm(speaker_X)
speaker_gmm = gmmmc.GMM(np.array(bobgmm.means), np.array(bobgmm.variances), np.array(bobgmm.weights))

print model.relevance_factor

prior = GMMPrior(MeansGaussianPrior(ubm.means, ubm.covars/model.relevance_factor),
                 #MeansUniformPrior(-1,1,2,1),
                 CovarsStaticPrior(np.array(ubm.covars)),
                 WeightsStaticPrior(np.array(ubm.weights)))


proposal = GMMBlockMetropolisProposal(propose_mean=LeapFrogMeansProposal(10,0.1),
                                      propose_covars=None,
                                      propose_weights=None)


mcmc = MarkovChain(proposal, prior, ubm)
#betas = np.concatenate(([0], np.logspace(-3,-1,300), np.linspace(0.001,1,300)))
#ais = AnnealedImportanceSampling(proposal, prior, betas)

#samples = ais.sample(speaker_X, 500)
samples = mcmc.sample(speaker_X, 5000)
print proposal.propose_mean.get_acceptance()
"""
mapest = samples[0]
mapprob = mapest.log_likelihood(speaker_X) + prior.log_prob(mapest)
for sample in samples:
    prob = sample.log_likelihood(speaker_X) + prior.log_prob(sample)
    if prob > mapprob:
        mapprob = prob
        mapest = sample
"""
mc_means = [[s.means[0][0], s.means[1][0]] for s in samples]
#pdb.set_trace()
#mc_means = [[s[0].means[0][0], s[0].means[1][0]] for s in samples]
mc_means = np.array(mc_means)
plt.figure(figsize=(14,11))

posterior_means = np.mean(mc_means,axis=0)
pm = gmmmc.GMM(np.array([[posterior_means[0]], [posterior_means[1]]]),
                np.array(model.ubm.variances),
                np.array(model.ubm.weights))

print pm.log_likelihood(speaker_X) + prior.log_prob(pm)
print speaker_gmm.log_likelihood(speaker_X) + prior.log_prob(speaker_gmm)

mcmc = plt.scatter(mc_means[:,0], mc_means[:,1], color= 'b')
map = plt.scatter(speaker_gmm.means[0][0], speaker_gmm.means[1][0], color='r', s=500.0)

true = plt.scatter(true_speaker.means[0][0], true_speaker.means[1][0], color='g', s=500)
prior = plt.scatter(ubm.means[0][0], ubm.means[1][0], color= 'y', s=500)
posterior = plt.scatter(posterior_means[0], posterior_means[1], color='m', s=500)

plt.title('Samples from Posterior Distribution of GMM Means', fontsize=28)
plt.xlabel('Mixture 1 mean', fontsize=28)
plt.ylabel('Mixture 2 mean', fontsize=28)

plt.legend((map, mcmc, prior, true, posterior),
           ('MAP estimate', 'Monte Carlo Samples', 'Prior Means', 'True Means', 'Posterior Mean'),
           scatterpoints=1,
           loc='upper left',
           ncol=2,
           fontsize=22)




#mcmc.set_size_inches(14,11,forward=True)
plt.show()

"""
x = np.linspace(-1, 1, 1000)
import matplotlib.mlab as mlab

for s in mc_means[700:710]:
    plt.plot(x,mlab.normpdf(x, s[0], 0.01) + mlab.normpdf(x, s[1], 0.01))
    plt.show()
"""

