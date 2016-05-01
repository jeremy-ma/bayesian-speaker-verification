__author__ = 'jeremyma'
import sys
sys.path.append('/Users/jeremyma/Documents/UNSW/THESIS/ThesisPrototype')
import os
import config
import numpy as np
from bob.bio.spear.preprocessor import Mod_4Hz
from bob.bio.spear.extractor import Cepstral
from scipy.io import wavfile
from collections import defaultdict

def process_all_and_save(data_directory, dest_directory, test_id='test'):
    """
    Preprocess all media files, perform voice activity detection and MFCC
    feature extraction. Save files to dest_directory. Expects .pcm files.

    :param data_directory: str
    :param dest_directory: str
    :param test_id: str
    :return:
    """

    voice_activity_detector = Mod_4Hz()
    feature_extractor = Cepstral()

    for subdir, dirs, files in os.walk(data_directory):
        for pcmfile in files:
            if pcmfile == '.DS_Store':
                continue
            else:
                print pcmfile
                output_file = pcmfile.split('.')[0] + '.npy'
                # save to same relative path in dest_directory
                relative_path = os.path.relpath(subdir, data_directory)
                path = os.path.join(dest_directory,relative_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                newsavefile = os.path.join(path, output_file)
                input_file = os.path.join(subdir, pcmfile)

                if os.path.isfile(newsavefile) or os.path.getsize(input_file) < 1000:
                    print "skipping"
                    continue


                # load data into memory
                data = np.memmap(input_file, dtype='h', mode='r')
                data = np.array(data, dtype=float)
                input_signal = (config.data_fs, data)
                input_data = voice_activity_detector(input_signal) # Perform voice activity detection with 'Mod_4Hz'
                normalised_features = feature_extractor(input_data) # Perform cepstral feature extraction

                #save
                np.save(newsavefile, normalised_features)


def parse_trials(enrol_file, trial_file):

    speaker_enrol_files = {}
    with open(enrol_file) as enrol_fp:
        for entry in enrol_fp:
            entry = entry.rstrip()
            parts = entry.split(' ')
            speaker_id = parts[0]
            filelist = parts[1].split(',')
            filelist = [relative_filepath + '.npy' for relative_filepath in filelist]
            speaker_enrol_files[speaker_id] = filelist

    speaker_trial_files = defaultdict(set)
    with open(trial_file) as trial_fp:
        for entry in trial_fp:
            entry = entry.rstrip()
            parts = entry.split(',')
            speaker_id = parts[0].split('_')[0]
            relative_filepath = parts[1] + '.npy'
            speaker_trial_files[speaker_id].add(relative_filepath)

    return (speaker_enrol_files, speaker_trial_files)


if __name__ == '__main__':
    # process_all_and_save(data_directory=os.path.join(config.reddots_directory,'pcm/'), \
    #            dest_directory=os.path.join(config.reddots_directory,'preprocessed_vad/'))

    speaker_enrol, speaker_trial = parse_trials(enrol_file=config.reddots_part4_enrol_female,
                                                trial_file=config.reddots_part4_trial_female)
