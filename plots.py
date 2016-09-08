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
    return fps[min_ind]

def plot_curves(file_list):

    colour_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    print file_list
    for i, (scores_name, answers_name, algo) in enumerate(file_list):
        # remove identical scores (due to files with just noise in them)
        scores, answers = np.load(scores_name), np.load(answers_name)
        print scores
        scores, indices = np.unique(scores, return_index=True)
        answers = answers[indices]
        fps, fns, _ = detection_error_tradeoff(answers, scores)
        eer = EER(fps, fns)
        label = "{0} (EER:{1})".format(algo, eer)
        plt.plot(fps, fns, colour_list[i], label=label)

    plt.legend(loc='upper right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()

def plot_regular():
    n_mixtures = 8
    n_runs = 1000
    gender = 'female'
    description = 'UBM/GMM' + '_' + gender

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
    save_path = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))
    plotstuff = []

    for rel in [ 10, 20, 50, 100, 150, 200, 250]:

        regular_scores = os.path.join('gmm_ubm_results', 'map_scores8_female_rel{0}_smallubm.npy'.format(rel))
        regular_answers = os.path.join('gmm_ubm_results', 'map_answers8_female_rel{0}_smallubm.npy'.format(rel))
        plotstuff.append((regular_scores, regular_answers, 'GMM/UBM rel{0}'.format(rel)))

    plt.title('Det Curve:' + description + '/n_mixtures{0}'.format(n_mixtures) + '/n_runs{0}'.format(n_runs),
              fontsize=22)

    plot_curves(plotstuff)


if __name__ == '__main__':
    #plot_regular()

    n_mixtures = 8
    n_runs = 50000
    gender = 'female'
    description = 'mcmc_rel150' + '_mapstart' + '_' + 'bigubm' '_' + gender

    save_path = os.path.join(config.dropbox_directory, config.computer_id, description)
    save_path = os.path.join(save_path, 'gaussians' + str(n_mixtures), 'iterations' + str(n_runs))

    regular_scores = os.path.join('gmm_ubm_results', 'map_scores8_relevance150.npy')
    regular_answers = os.path.join('gmm_ubm_results', 'map_answers8_relevance150.npy')

    plotstuff = [
        (regular_scores, regular_answers, 'GMM/UBM'),
        (os.path.join(save_path, 'scoresMHMC.npy'), os.path.join(save_path, 'answersMHMC.npy'), 'MC GMM/UBM'),
        (os.path.join(save_path, 'scoresMCMAP.npy'), os.path.join(save_path, 'answersMCMAP.npy'), 'MC MAP GMM/UBM')
    ]
    plt.title('Det Curve:' + description + '/n_mixtures{0}'.format(n_mixtures) + '/n_runs{0}'.format(n_runs),
              fontsize=22)

    plot_curves(plotstuff)