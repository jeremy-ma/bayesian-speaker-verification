__author__ = 'jeremyma'
import os
reddots_directory = '/mnt/hgfs/jeremyma/Dropbox/reddots_r2015q4_v1'
computer_id = 'macvm'
dropbox_directory = '/mnt/hgfs/jeremyma/Dropbox/'
reddots_fs = 16000
reddots_nchannels = 1
reddots_part4_enrol_female = os.path.join(reddots_directory, 'ndx/f_part_04_tp.trn')
reddots_part4_enrol_male = os.path.join(reddots_directory, 'ndx/m_part_04_tp.trn')
reddots_part4_trial_female = os.path.join(reddots_directory, 'ndx/f_part_04.ndx')
reddots_part4_trial_male = os.path.join(reddots_directory, 'ndx/m_part_04.ndx')

data_directory = reddots_directory
data_fs = reddots_fs
enrol_file = reddots_part4_enrol_female
trial_file = reddots_part4_trial_female


