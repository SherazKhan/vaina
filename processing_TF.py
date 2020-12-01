import mne
import os.path as op
from glob import glob
from mne.preprocessing import compute_proj_ecg
import numpy as np
import matplotlib.pyplot as plt
from mne.minimum_norm import source_induced_power

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

tmin, tmax = (-0.2, 3.5)  # epoch from 200 ms before event to 3500 ms after it
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

snr = 1.0  # use smaller SNR for raw data
inv_method = 'dSPM'
parc = 'aparc.a2009s'  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
lambda2 = 1.0 / snr ** 2

stcs = mne.minimum_norm.apply_inverse_epochs(epochs['300'], inv, lambda2, inv_method,
                                             pick_ori=None, return_generator=True)

labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_mri_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub structures contained in the src space
# If mode = 'mean_flip' this option is used only for the cortical label
src = inv['src']
label_ts = mne.extract_label_time_course(
    stcs, labels_parc, src, mode='mean_flip', allow_empty=True,
    return_generator=True)

freqs = np.arange(2, 55)
n_cycles = freqs / 3.

label = labels_parc[10]
power, itc = source_induced_power(epochs['300'], inv, freqs, label, baseline=(-0.1, 0), baseline_mode='percent',
                                  n_cycles=n_cycles, n_jobs=1)

power = np.mean(power, axis=0)  # average over sources
itc = np.mean(itc, axis=0)  # average over sources
times = epochs.times

##########################################################################
# View time-frequency plots
plt.subplots_adjust(0.1, 0.08, 0.96, 0.94, 0.2, 0.43)
plt.subplot(2, 2, 2 * ii + 1)
plt.imshow(20 * power,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Power (%s)' % title)
plt.colorbar()

plt.subplot(2, 2, 2 * ii + 2)
plt.imshow(itc,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=0, vmax=0.7,
           cmap='RdBu_r')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('ITC (%s)' % title)
plt.colorbar()

plt.show()
