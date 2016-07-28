import os
import config
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pdb

def detection_error_tradeoff(y_true, probas_pred, pos_label=None,
                             sample_weight=None):
    """Compute error rates for different probability thresholds
    Note: this implementation is restricted to the binary classification task.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification in range {-1, 1} or {0, 1}.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int, optional (default=None)
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    fns : array, shape = [n_thresholds]
        A count of false negatives, at index i being the number of positive
        samples assigned a score < thresholds[i]. The total number of
        positive samples is equal to tps[-1] (thus false negatives are given by
        tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.

    """
    fps, tps, thresholds = metrics.ranking._binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    #pdb.set_trace()
    fns = tps[-1] - tps
    tp_count = tps[-1]
    tn_count = (fps[-1] - fps)[0]

    # start with false positives is zero and stop with false negatives zero
    # and reverse the outputs so list of false positives is decreasing
    last_ind = tps.searchsorted(tps[-1]) + 1
    first_ind = fps[::-1].searchsorted(fps[0])
    sl = range(first_ind, last_ind)[::-1]
    return fps[sl] / fps[-1], fns[sl] / tps[-1], thresholds[sl]

def EER(fps, fns):
    min_diff = np.inf
    min_ind = 0
    for i in xrange(len(fps)):
        diff = abs(fps[i] - fns[i])
        if diff < min_diff:
            min_diff = diff
            min_ind = i
    return (fps[min_ind], fns[min_ind])


n_mixtures = 8
n_runs = 100
description = 'ais_gaussian_priors'
type = 'AIS'

save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
save_path = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))


filename = os.path.join(save_path, 'scores' + type + 'G' + str(n_mixtures) +
                         '_N' + str(n_runs) + '.npy')
scoresMHMC = np.load(filename)
filename = os.path.join(save_path, 'answers'+ type + 'G' + str(n_mixtures) +
                        '_N' + str(n_runs) + '.npy')
answersMHMC = np.load(filename)

scoresMAP = np.load('map_scores.npy')
answersMAP = np.load('map_answers.npy')

fps_MHMC, fns_MHMC, _ = detection_error_tradeoff(answersMHMC, scoresMHMC)

fps_MAP, fns_MAP, _ = detection_error_tradeoff(answersMAP, scoresMAP)

plt.title('Detection Error Tradeoff')
plt.plot(fps_MHMC, fns_MHMC, 'r', label='GMM-MHMC')

plt.plot(fps_MAP, fns_MAP, 'g', label='MAP')

plt.ylabel('False Negative Rate')
plt.xlabel('False Positive Rate')

print EER(fps_MHMC, fns_MHMC)
print EER(fps_MAP, fns_MAP)


plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()