import numpy as np
from gmmmc.gmm import GMM
from gmmmc.proposals.proposals import Proposal
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
import pdb
import logging

def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr

def means_derivative(X, gmm, means_prior):
    n_samples, n_dim = X.shape
    lpr = _log_multivariate_normal_density_diag(X, gmm.means, gmm.covars) + np.log(gmm.weights)
    logprob = logsumexp(lpr)

    derivative_weights = np.exp(lpr - logprob)
    derivative = -np.sum(derivative_weights.T[:,:,np.newaxis] * (X - gmm.means[:,np.newaxis])/gmm.covars[:,np.newaxis], axis=1)
    # add the infludence of the prior
    prior_derivative = (gmm.means - means_prior.means) / means_prior.covars

    #pdb.set_trace()
    derivative += prior_derivative

    #for i in xrange(gmm.n_mixtures):
    #    for j in xrange(n_samples):
    #        derivative[i] += derivative_weights[j][i] * (X[j] - means[i]) / covars[i]

    return derivative

class LeapFrogMeansProposal(Proposal):
    """Gaussian Proposal distribution for means of a GMM"""

    def __init__(self, num_steps, step_size):
        """
        Gaussian proposal distribution for the means. The multivariate Gaussian is centered at the means of the current
        state in the Markov Chain and has covariance given by step_sizes. Multiple step sizes can be specified.
        The proposal algorithm will take these steps in the sequence specified in step_sizes.

        Parameters
        ----------
        step_sizes : 1-D array_like
            Iterable containing the sequence of step sizes (covariances of the Gaussian proposal distribution"
        """
        super(LeapFrogMeansProposal, self).__init__()
        self.step_size = step_size
        self.num_steps = num_steps
        self.count_accepted = 0
        self.count_illegal =  0
        self.count_proposed = 0

    def propose(self, X, gmm, target, n_jobs=1):
        """
        Propose a new set of GMM means.

        Parameters
        ----------
        X : 2-D array like of shape (n_samples, n_features)
            The observed data or evidence.
        gmm : GMM object
            The current state (set of gmm parameters) in the Markov Chain
        target : GMMPosteriorTarget object
            The target distribution to be found. Must just have Gaussian Mean prior
        n_jobs : int
            Number of cpu cores to use in the calculation of log probabilities.

        Returns
        -------
            : GMM
            A new GMM object initialised with new mean parameters.
        """
        num_mixtures, dimension = gmm.means.shape

        #sample new momentums for each mixture
        momentum = np.array([multivariate_normal(np.zeros(dimension), np.diag(np.ones((dimension,)))).rvs()
                    for _ in xrange(num_mixtures)])
        updated_gmm = GMM(gmm.means, gmm.covars, gmm.weights)

        means_prior = target.prior.means_prior

        for i in xrange(self.num_steps):
            for j in xrange(num_mixtures):
                self.count_proposed += 1
                # leapfrog algorithm
                derivative = means_derivative(X, updated_gmm, means_prior)
                proposed_momentum = np.array(momentum)
                proposed_means = np.array(updated_gmm.means)

                proposed_momentum[j] = momentum[j] - self.step_size / 2 * derivative[j]
                proposed_means[j] = updated_gmm.means[j] + self.step_size * momentum[j]

                proposed_gmm = GMM(proposed_means, np.array(gmm.covars), np.array(gmm.weights))

                derivative = means_derivative(X, proposed_gmm, means_prior)
                proposed_momentum[j] = momentum[j] - self.step_size / 2 * derivative[j]
                acceptance = proposed_gmm.log_likelihood(X,n_jobs=1) - updated_gmm.log_likelihood(X, n_jobs=1) + \
                            np.dot(momentum.ravel(), proposed_momentum.ravel()) / 2

                if acceptance > 0 or acceptance > np.log(np.random.uniform()):
                    self.count_accepted += 1
                    updated_gmm = proposed_gmm
                    momentum = proposed_momentum

        return updated_gmm

if __name__=='__main__':
    np.random.seed(3)
    g = GMM(np.array([[0.5,0.5],[0.45,0.45],[0.55,0.55]]), np.array(np.array([[0.01,0.01],[0.01,0.01],[0.01,0.01]])),
            np.array([0.4,0.4,0.2]))
    X = np.random.random((100,2))


