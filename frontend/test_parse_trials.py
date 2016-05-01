from unittest import TestCase
from frontend.frontend import parse_trials
import config, pdb
__author__ = 'jeremyma'


class TestParse_trials(TestCase):
    def test_parse_trials (self):
        speaker_enrol, speaker_trial = parse_trials(enrol_file=config.reddots_part4_enrol_female,
                                       trial_file=config.reddots_part4_trial_female)

        # 6 female speakers
        self.assertEquals(len(speaker_enrol.keys()), 6)
        self.assertEquals(len(speaker_trial.keys()), 6)

        # 1704 samples to compare each speaker against, 10 samples per enrolement
        self.assertEquals(len(speaker_enrol['f0002']), 10)
        self.assertEquals(len(speaker_trial['f0002']), 1704)

        self.assertTrue('f0004/20150518230322699_f0004_15484.npy' in speaker_enrol['f0004'])
        self.assertTrue('f0015/20150809205711766_f0015_887.npy' in speaker_trial['f0002'])
        self.assertTrue('f0008/20150605103543978_f0008_34.npy' in speaker_trial['f0002'])
        self.assertTrue('f0015/20150809205711766_f0015_887.npy' in speaker_trial['f0012'])


