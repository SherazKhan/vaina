import mne
import os.path as op
from glob import glob
from mne.preprocessing import compute_proj_ecg
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper, tfr_array_multitaper

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
reject_criteria = dict(mag=3000e-15,  # 3000 fT
                       grad=3000e-13)  # 3000 fT/cm

tmin, tmax = (-0.2, 1.1)  # epoch from 200 ms before event to 1000 ms after it
baseline = (None, 0)  # baseline period from start of epoch to time=0

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
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8)

labels = mne.read_labels_from_annot(subject, 'aparc', subjects_dir=subjects_mri_dir)
labels_name = np.array([label.name for label in labels])
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal', return_generator=True)

label_ts = np.array(mne.extract_label_time_course(stcs, labels, inv['src'], return_generator=False))

psds, freqs = psd_array_multitaper(label_ts, epochs.info['sfreq'], fmin=2, fmax=55)

tfr_alpha = tfr_array_multitaper(label_ts, epochs.info['sfreq'], freqs=np.arange(8, 13), output='avg_power', n_jobs=4)
tfr_beta = tfr_array_multitaper(label_ts, epochs.info['sfreq'], freqs=np.arange(16, 30), output='avg_power', n_jobs=4)
tfr_lgamma = tfr_array_multitaper(label_ts, epochs.info['sfreq'], freqs=np.arange(30, 55), output='avg_power', n_jobs=4)
tfr_hgamma = tfr_array_multitaper(label_ts, epochs.info['sfreq'], freqs=np.arange(65, 100), output='avg_power', n_jobs=4)


for ix, inds in enumerate(np.split(np.arange(68), 4)):
    plt.figure(figsize=(15, 20))
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    lineObjects = plt.plot(freqs, 20 * np.log10(psds.mean(0).T)[:, inds], linewidth=4)
    plt.xlabel('Frequency (Hz)', fontsize=30)
    plt.ylabel('Power (20*log10)', fontsize=30)
    plt.xlim(2, 55)
    plt.legend(iter(lineObjects), labels_name[inds], fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(ix) + '.png')

def plot_tf(data, stem):
    times = epochs.times
    for ix, inds in enumerate(np.split(np.arange(68), 4)):
        plt.figure(figsize=(15, 20))
        plt.rc('xtick', labelsize=25)
        plt.rc('ytick', labelsize=25)
        lineObjects = plt.plot(times, data.mean(1).T[:, inds], linewidth=4)
        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Power', fontsize=30)
        plt.xlim(0, 1)
        plt.legend(iter(lineObjects), labels_name[inds], fontsize=18)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(ix) + '_' + stem + '.png')


plot_tf(tfr_alpha, 'alpha')
plot_tf(tfr_beta, 'beta')
plot_tf(tfr_lgamma, 'low_gamma')
plot_tf(tfr_hgamma, 'high_gamma')











