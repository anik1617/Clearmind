import numpy as np
import pandas as pd
import mne

# Load data
df = pd.read_csv("hello_how_are_you_eeg.csv")
eeg_data = df.iloc[:, 1:].values.T  # shape = (channels, timepoints)

# Safety checks
eeg_data = np.nan_to_num(eeg_data)
print(f"EEG shape: {eeg_data.shape}")  # should be (16, 66000)

# Create MNE Raw object
channels = ["Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
            "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"]
sfreq = 256
info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=["eeg"] * len(channels))
raw = mne.io.RawArray(eeg_data, info)
raw.set_eeg_reference('average', projection=True)

# Optional: Crop data for debug (e.g., 10 seconds = 2560 samples)
raw.crop(tmin=0, tmax=10)

# Standard montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Covariance matrix (SAFE)
cov = mne.compute_raw_covariance(raw, method="empirical", n_jobs=1)

print("Covariance matrix computed safely.")

inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)

# Apply the inverse operator to your raw EEG data
method = "dSPM"  # You can also use "sLORETA" or "MNE"
stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0 / 3.0 ** 2, method=method)

print("Inverse solution applied successfully.")
