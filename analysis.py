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
from gmmmc.priors import MeansUniformPrior, CovarsStaticPrior, WeightsStaticPrior, GMMPrior, MeansGaussianPrior
from gmmmc.proposals import GMMBlockMetropolisProposal, GaussianStepCovarProposal, GaussianStepWeightsProposal, GaussianStepMeansProposal
from gmmmc import MarkovChain
import multiprocessing
import copy
import bob
from system.full_system import KLDivergenceMLStartSystem
from shutil import copyfile
import matplotlib.pyplot as plt


def effectiveSampleSize(data, stepSize=1):
    """ Effective sample size, as computed by BEAST Tracer."""
    samples = len(data)

    assert len(data) > 1, "no stats for short sequences"

    maxLag = min(samples // 3, 1000)

    gammaStat = [0, ] * maxLag
    # varGammaStat = [0,]*maxLag

    varStat = 0.0;

    if type(data) != np.ndarray:
        data = np.array(data)

    normalizedData = data - data.mean()

    for lag in range(maxLag):
        v1 = normalizedData[:samples - lag]
        v2 = normalizedData[lag:]
        v = v1 * v2
        gammaStat[lag] = sum(v) / len(v)
        # varGammaStat[lag] = sum(v*v) / len(v)
        # varGammaStat[lag] -= gammaStat[0] ** 2

        # print lag, gammaStat[lag], varGammaStat[lag]

        if lag == 0:
            varStat = gammaStat[0]
        elif lag % 2 == 0:
            s = gammaStat[lag - 1] + gammaStat[lag]
            if s > 0:
                varStat += 2.0 * s
            else:
                break

    # standard error of mean
    # stdErrorOfMean = Math.sqrt(varStat/samples);

    # auto correlation time
    act = stepSize * varStat / gammaStat[0]

    # effective sample size
    ess = (stepSize * samples) / act

    return ess

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def check_same(x, n_bins):
    out = chunkIt(x, n_bins)
    acceptance_rates = []
    for bin in out:
        count = 0
        for i, num in enumerate(bin):
            if i == 0:
                continue
            else:
                if num != bin[i-1]:
                    count += 1
        acceptance_rates.append(float(count) / len(bin))
    return acceptance_rates

def check_ess(x, n_bins):
    out = chunkIt(x, n_bins)
    ess_rates = []
    for bin in out:
        ess_rates.append(effectiveSampleSize(bin))

    return ess_rates


logging.getLogger().setLevel(logging.INFO)
n_mixtures, n_runs, description = 128, 100000, 'mapstart20_fullbackground'
relevance_factor = 20
n_procs = 4
n_jobs = -1
gender = 'female'
description += '_' + gender

if gender == 'male':
    enrolment = config.reddots_part4_enrol_male
    trials = config.reddots_part4_trial_male
else:
    enrolment = config.reddots_part4_enrol_female
    trials = config.reddots_part4_trial_female

manager = frontend.DataManager(data_directory=os.path.join(config.data_directory, 'preprocessed'),
                               enrol_file=config.reddots_part4_enrol_female,
                               trial_file=config.reddots_part4_trial_female)

save_path = os.path.join(config.dropbox_directory, 'superstation_results', description)
save_dir = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs),
                        "samples")
speakertest = os.path.join(save_dir, 'speaker.pickle')
# speakertest = os.path.join(save_dir, 'speaker.pickle')
print speakertest
with open(speakertest) as fp:
    samples = cPickle.load(fp)

means = [gmm.means[0][1] for gmm in samples]
# pdb.set_trace()
print check_same(means, 100)
print check_ess(means, 100)
plt.acorr(means, maxlags=100,  normed=True, usevlines=False);
print effectiveSampleSize(means)
plt.show()
#system.evaluate_forward_unnormalised(manager.get_trial_data(), manager.get_enrolment_data(), manager.get_background_data(),
#                        n_jobs, save_dir, n_runs/2, 1)
#system.evaluate_bayes_factor(manager.get_trial_data(), n_jobs, save_dir, n_runs/2, 50)
