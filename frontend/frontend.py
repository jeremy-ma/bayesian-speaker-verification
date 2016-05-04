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

def process_all_and_save(data_directory, dest_directory):
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

class Trial():

    def __init__(self, filename, claimed_speaker, actual_speaker):
        self.feature_file = filename
        self.claimed_speaker = claimed_speaker
        self.actual_speaker = actual_speaker
        self.answer = (self.claimed_speaker == self.actual_speaker)

    def get_data(self):
        return np.load(self.feature_file)

class DataManager():

    def __init__(self, data_directory, enrol_file, trial_file):
        self.data_directory = data_directory
        self.speaker_enrolment_files = self.parse_enrol(enrol_file)
        self.speaker_trials = self.parse_trials(trial_file)

    def parse_enrol(self, enrol_file):
        """
        parse enrolment file
        :param enrol_file:
        :return:
        """
        speaker_enrol_files = {}
        with open(enrol_file) as enrol_fp:
            for entry in enrol_fp:
                entry = entry.rstrip()
                parts = entry.split(' ')
                speaker_id = parts[0]
                filelist = parts[1].split(',')
                # get list of file names
                filelist = [os.path.join(self.data_directory, relative_filepath + '.npy') for relative_filepath in filelist]
                # tag with
                speaker_enrol_files[speaker_id] = filelist

        return speaker_enrol_files

    def parse_trials(self, trial_file):
        """
        parse trial file
        :param trial_file:
        :return: speaker_trials, dictionary { speaker_id: [trials]}
        """

        speaker_trial_temp = defaultdict(set)
        with open(trial_file) as trial_fp:
            for entry in trial_fp:
                entry = entry.rstrip()
                parts = entry.split(',')
                speaker_id = parts[0].split('_')[0]
                relative_filepath = parts[1] + '.npy'
                # want to get rid of duplicate trials
                claimed_speaker = relative_filepath.split('/')[0]
                speaker_trial_temp[speaker_id].add((os.path.join(self.data_directory, relative_filepath),claimed_speaker))

        speaker_trials = defaultdict(list)
        for speaker_id, files in speaker_trial_temp.iteritems():
            for filepath, actual_speaker in files:
                trial = Trial(filename=os.path.join(self.data_directory, filepath), claimed_speaker=speaker_id,
                              actual_speaker=actual_speaker)
                speaker_trials[speaker_id].append(trial)

        return speaker_trials

    def get_features(self, filelist):
        """
        helper function
        :param filelist:
        :return:
        """
        feature_list = []
        for fname in filelist:
            arr = np.load(os.path.join(self.data_directory, fname))
            feature_list.append(arr)
        return np.concatenate(feature_list)

    def get_background_data(self):
        """
        Get data for training background model
        :return: 2d array of all enrolment data
        """
        array_list = []
        for speaker, enrol_filelist in self.speaker_enrolment_files.iteritems():
            array_list.append(self.get_features(enrol_filelist))
        return np.concatenate(array_list)

    def get_enrolment_data(self):
        """
        Get enrolment data
        :return: dictionary speaker_id->feature array
        """
        enrolment_data = {}
        for speaker_id, filelist in self.speaker_enrolment_files.iteritems():
            array_list = [np.load(filename) for filename in filelist]
            enrolment_data[speaker_id] = np.concatenate(array_list)

        return enrolment_data

    def get_trial_data(self):
        return self.speaker_trials

    def get_unique_trials(self):
        """
        Get unique (by filename) trials
        :return: unique_trial { actual_speaker_id : [trials including their recording] }
        """
        file_set = set()
        unique_trials = defaultdict(list)
        for speaker_id, trial_list in self.speaker_trials.iteritems():
            for trial in trial_list:
                if trial.feature_file not in file_set:
                    unique_trials[trial.actual_speaker].append(trial)
        return unique_trials

if __name__ == '__main__':
    # process_all_and_save(data_directory=os.path.join(config.reddots_directory,'pcm/'), \
    #            dest_directory=os.path.join(config.reddots_directory,'preprocessed_vad/'))

    #speaker_enrol, speaker_trial = parse_trials(enrol_file=config.reddots_part4_enrol_female,
    #                                            trial_file=config.reddots_part4_trial_female)
    pass