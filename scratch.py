import mne
import os.path as op
from glob import glob
from mne.preprocessing import compute_proj_ecg, compute_proj_eog


data_path = r'D:\Dropbox (Personal)\vaina'

subjects = ['LMV2016_N01', 'LMV2016_N01b', 'LMV2016_N02', 'LMV2016_N02b', 'LMV2016_N03', 'LMV2016_N03b',
            'LMV2016_N05b', 'LMV2016_N06', 'LMV2016_N06b']

meg_files = ['swa01_raw.fif', 'swa02_raw.fif', 'swa03_raw.fif']

subjects_mri_dir = op.join(data_path,'anatomy')
subjects_meg_dir = op.join(data_path, 'meg')



subject = subjects[0]

subject_mri_dir = op.join(subjects_mri_dir, subject)
subject_meg_dir = op.join(subjects_meg_dir, subject)



bem_dir = op.join(subject_mri_dir, 'bem')


trans_file = op.join(subject_meg_dir, 'coreg', 'COR-' + subject + '.fif')
events_files = sorted(glob('D:\\Dropbox (Personal)\\vaina\\meg\\LMV2016_N01\\events\\*_raw_tagged_1.eve'))
proj_file = op.join(subject_meg_dir, 'proj', 'swa02-proj.fif')

raws = []

for meg_file in meg_files:
    raws.append(mne.io.read_raw(op.join(subject_meg_dir, meg_file)))

raw = mne.concatenate_raws(raws)

projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog = mne.read_proj(proj_file)


raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog

raw.apply_proj()
