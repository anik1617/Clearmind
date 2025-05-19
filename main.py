import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import mne
from mne.datasets import sample
import os
from data_clean import EEGDataConverter

# Enable parallel processing
os.environ['MNE_USE_CUDA'] = 'true'
n_jobs = -1  # Use all available CPU cores

channels = ["Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
                   "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"]

# Change this for output or input path
input_path = "./input/DLR_3_1.tsv"
output_path = "./output/final_EEG.csv"

# Use EEGDataConverter to process the data
converter = EEGDataConverter(input_path=input_path, output_path=output_path)

time_values, eeg_data = converter.convert()

# Create DataFrame from the processed data
df = pd.read_csv(output_path)

print(f"EEG Data Shape: {eeg_data.shape}") 

sfreq = 256
info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=["eeg"] * len(channels))

# Create MNE RawArray
raw = mne.io.RawArray(eeg_data, info)
raw.set_eeg_reference('average', projection=True)

# Print dataset info
print(raw.info)

# Set a standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

print(sample.data_path())


# Load MNE's default fsaverage brain
subjects_dir =  "C:/Users/anik2/mne_data/MNE-fsaverage-data/"
subject = "fsaverage"
print(f"fsaverage subject directory: {subjects_dir}")

# Setup standard head model
src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model(subject, ico=3, subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(bem)

# Downsample the data for faster processing

# Compute the forward model
trans = "fsaverage"  # Use template transformation
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True, n_jobs=n_jobs)

# Compute noise covariance from your EEG data
cov = mne.compute_raw_covariance(raw, method="shrunk", n_jobs=n_jobs)

# Compute the inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)

# Apply the inverse operator to your raw EEG data
method = "dSPM"  # You can also use "sLORETA" or "MNE"
stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0 / 3.0 ** 2, method=method)



brain_kwargs = dict(alpha=0.1, background="white", cortex="low_contrast")

brain = mne.viz.Brain(subject, subjects_dir=subjects_dir, **brain_kwargs)


stc.crop(0.05, 0.5)


kwargs = dict(
    fmin=stc.data.min(),
    fmax=stc.data.max(),
    alpha=0.25,
    smoothing_steps="nearest",
    time=stc.times,
)

brain.add_data(stc.lh_data, hemi="lh", vertices=stc.lh_vertno, **kwargs)
brain.add_data(stc.rh_data, hemi="rh", vertices=stc.rh_vertno, **kwargs)

raw.plot_sensors(kind="topomap", show_names=True)
plt.show(block=True)
brain.show()