import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GMM

import config
from features import mfcc


if __name__ == '__main__':

    data = np.memmap('/Users/jeremyma/Dropbox/reddots_r2015q4_v1/pcm/f0004/20150302231221864_f0004_377.pcm', dtype='h', mode='r')
    print data
    #wavwriter = wave.open('/Users/jeremyma/Documents/UNSW/THESIS/ThesisPrototype/test.wav', mode = 'wb')
    wavfile.write('/Users/jeremyma/Documents/UNSW/THESIS/ThesisPrototype/test.wav', 16000, data)
    features = mfcc(data, config.reddots_fs)
    print features.shape
    gmm = GMM()
    gmm.fit(features)
    print gmm