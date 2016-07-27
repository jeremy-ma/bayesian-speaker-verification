import os
import config
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


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




save_path = os.path.join(config.dropbox_directory, config.computer_id)


filename = os.path.join(save_path, 'scoresMHMC' + 'G' + str(128) +
                         '_N' + str(10000) + '.npy')
scoresMHMC = np.load(filename)
filename = os.path.join(save_path, 'answersMHMC' + 'G' + str(128) +
                        '_N' + str(10000) + '.npy')
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

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()