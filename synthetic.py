import gmmmc
import numpy as np
import bob.bio.gmm
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import logging
import matplotlib.pyplot as plt

np.random.seed(2)
logging.getLogger().setLevel(logging.INFO)

true_background = gmmmc.GMM(means=np.array([[-0.5], [-0.1]]),
                            covariances=np.array([[0.01], [0.01]]),
                            weights=np.array([0.5, 0.5]))

true_speaker = gmmmc.GMM(means=np.array([[0.8], [0.5]]),
                         covariances=np.array([[0.01], [0.01]]),
                         weights=np.array([0.5, 0.5]))

background_X = true_background.sample(10000)
speaker_X = true_speaker.sample(1000)

model = bob.bio.gmm.algorithm.GMM(2)

model.train_ubm(background_X)
# enroll trainer because train_ubm has a bug
model.enroll_trainer = bob.learn.em.MAP_GMMTrainer(model.ubm,
                                                   relevance_factor=model.relevance_factor,
                                                   update_means=True, update_variances=False)
model.gmm_enroll_iterations = 25
model.training_threshold = 1e-5,
ubm = gmmmc.GMM(np.array(model.ubm.means),
                np.array(model.ubm.variances),
                np.array(model.ubm.weights))
bobgmm = model.enroll_gmm(speaker_X)
speaker_gmm = gmmmc.GMM(np.array(bobgmm.means), np.array(bobgmm.variances), np.array(bobgmm.weights))

prior = GMMPrior(MeansGaussianPrior(np.array(ubm.means), np.array(ubm.covars)),
                 CovarsStaticPrior(np.array(ubm.covars)),
                 WeightsStaticPrior(np.array(ubm.weights)))

proposal = GMMBlockMetropolisProposal(propose_mean=GaussianStepMeansProposal(step_sizes=[0.0003, 0.001]),
                                      propose_covars=None,
                                      propose_weights=None)

mcmc = MarkovChain(proposal, prior, ubm)

samples = mcmc.sample(speaker_X, 1000)

mapest = samples[0]
mapprob = mapest.log_likelihood(speaker_X) + prior.log_prob(mapest)
for sample in samples:
    prob = sample.log_likelihood(speaker_X) + prior.log_prob(sample)
    if prob > mapprob:
        mapprob = prob
        mapest = sample


final = samples[-1]

means = [s.means[0][0] for s in samples[::10]]

plt.scatter(mapest.means[0][0], 0, color='g')
plt.scatter(means, np.ones((len(means))), color= 'b')
plt.scatter(speaker_gmm.means[0][0], -1, color='r')
plt.scatter(ubm.means[0][0], -1, color= 'y')
print ubm.covars

plt.show()

