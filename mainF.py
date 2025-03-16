import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import mne

channels = ["Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
                   "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"]

csv_file = "your_eeg_data.csv"  # Update with your actual file path
df = pd.read_csv("final_EEG.csv")

# Extract time values (assuming the first column is time)
time_values = df.iloc[:, 0].values  # First column for timestamps
eeg_data = df.iloc[:, 1:].values.T
print(f"EEG Data Shape: {eeg_data.shape}") 

sfreq = 256
info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=["eeg"] * len(channels))

# Create MNE RawArray
raw = mne.io.RawArray(eeg_data, info)
raw.set_eeg_reference(projection=True)

# Print dataset info
print(raw.info)

# Set a standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Plot sensor locations
raw.plot_sensors(kind="topomap", show_names=True)

# Load MNE's default fsaverage brain
subjects_dir = "C:/Users/anik2/mne_data/MNE-fsaverage-data"
subject = "fsaverage"
print(f"fsaverage subject directory: {subjects_dir}")

# Setup standard head model
src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(bem)

# Compute the forward model
trans = "fsaverage"  # Use template transformation
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True)

# Compute noise covariance from your EEG data
cov = mne.compute_raw_covariance(raw, method="shrunk")

# Compute the inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)

# Apply the inverse operator to your raw EEG data
method = "dSPM"  # You can also use "sLORETA" or "MNE"
stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1/9, method=method)

brain_kwargs = dict(alpha=0.1, background="white", cortex="low_contrast")

brain = mne.viz.Brain(subject, subjects_dir=subjects_dir, **brain_kwargs)
stc.crop(0.09, 0.1)
kwargs = dict(
    fmin=stc.data.min(),
    fmax=stc.data.max(),
    alpha=0.25,
    smoothing_steps="nearest",
    time=stc.times,
)

brain.add_data(stc.lh_data, hemi="lh", vertices=stc.lh_vertno, **kwargs)
brain.add_data(stc.rh_data, hemi="rh", vertices=stc.rh_vertno, **kwargs)


input()
