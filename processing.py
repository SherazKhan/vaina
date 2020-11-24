import mne
import os.path as op
from glob import glob
from mne.preprocessing import compute_proj_ecg
import numpy as np
from mne.viz import circular_layout, plot_connectivity_circle
import matplotlib.pyplot as plt
plt.ion()


data_path = r'D:\Dropbox (Personal)\vaina'

subjects = ['LMV2016_N01', 'LMV2016_N01b', 'LMV2016_N02', 'LMV2016_N02b', 'LMV2016_N03', 'LMV2016_N03b',
            'LMV2016_N05b', 'LMV2016_N06', 'LMV2016_N06b']

meg_files = ['swa01_raw.fif', 'swa02_raw.fif', 'swa03_raw.fif']

subjects_mri_dir = op.join(data_path, 'anatomy')
subjects_meg_dir = op.join(data_path, 'meg')



subject = subjects[0]

subject_mri_dir = op.join(subjects_mri_dir, subject)
subject_meg_dir = op.join(subjects_meg_dir, subject)



bem_dir = op.join(subject_mri_dir, 'bem')


trans_file = op.join(subject_meg_dir, 'coreg', 'COR-' + subject + '.fif')
events_files = sorted(glob('D:\\Dropbox (Personal)\\vaina\\meg\\LMV2016_N01\\events\\*_raw_tagged_1.eve'))
proj_file = op.join(subject_meg_dir, 'proj', 'swa02-proj.fif')




all_epochs = []
event_dict = {'300': 300, '800': 800, '1300': 1300}
reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13)    # 3000 fT/cm

tmin, tmax = (-0.2, 3.5)  # epoch from 200 ms before event to 3500 ms after it
baseline = (None, 0)      # baseline period from start of epoch to time=0

raws = []

for index, meg_file in enumerate(meg_files):
    raw = mne.io.read_raw(op.join(subject_meg_dir, meg_file))
    raws.append(raw)

raw = mne.concatenate_raws(raws)
projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog = mne.read_proj(proj_file)
cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)


for index, meg_file in enumerate(meg_files):
    raw = mne.io.read_raw(op.join(subject_meg_dir, meg_file), preload=True)
    raw.info['projs'] += projs_ecg
    raw.info['projs'] += projs_eog
    raw.apply_proj()
    events = mne.read_events(events_files[index])
    raw = raw.filter(0.1, 110, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
               filter_length='auto', phase='zero', fir_window='hann')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True,
                        baseline=baseline, reject=reject_criteria, preload=True)
    epochs.resample(330.)
    all_epochs.append(epochs)

epochs = mne.concatenate_epochs(all_epochs)

raw = mne.concatenate_raws(raws)
bem_fname = op.join(bem_dir, subject + '-20480-bem-sol.fif')
bem = mne.read_bem_solution(bem_fname)
src = mne.setup_source_space(subject, spacing='ico5',
                             add_dist=False, subjects_dir=subjects_mri_dir)



fwd = mne.make_forward_solution(raw.info, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, n_jobs=2)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2, depth=0.8)

snr = 1.0           # use smaller SNR for raw data
inv_method = 'dSPM'
parc = 'aparc.a2009s'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
lambda2 = 1.0 / snr ** 2


stcs = mne.minimum_norm.apply_inverse_epochs(epochs['300'], inv, lambda2, inv_method,
                            pick_ori=None, return_generator=True)



labels_parc = mne.read_labels_from_annot(subject, parc=parc,
                                         subjects_dir=subjects_mri_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub structures contained in the src space
# If mode = 'mean_flip' this option is used only for the cortical label
src = inv['src']
label_ts = mne.extract_label_time_course(
    stcs, labels_parc, src, mode='mean_flip', allow_empty=True,
    return_generator=True)

fmin = 8.
fmax = 13.
sfreq = epochs.info['sfreq']  # the sampling frequency
con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

node_colors = [label.color for label in labels_parc]

# We reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels_parc]
lh_labels = [name for name in label_names if name.endswith('lh')]
rh_labels = [name for name in label_names if name.endswith('rh')]

# Get the y-location of the label
label_ypos_lh = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels_parc[idx].pos[:, 1])
    label_ypos_lh.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order
node_order = lh_labels[::-1] + rh_labels

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) // 2])

conmat = con[:, :, 0]
fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity 300 Condition (PLI)', fig=fig)