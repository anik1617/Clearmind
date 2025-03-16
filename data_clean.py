import mne
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

headers = [
    "Index", "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2", "F7", "F8", "C3", "C4", 
    "T3", "T4", "P3", "P4", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2", 
    "Other", "Other", "Other", "Other", "Other", "Other", "Other", 
    "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"
]

print(len(headers))
tsv_file = "DLR_3_1.tsv"
tf=pd.read_table(tsv_file,sep='\t', header=None)
tf.columns = headers
tf.to_csv("EEG_scan.csv", index=False)

columns_to_keep = ["Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
                   "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"]
df=pd.read_csv("EEG_scan.csv")
df = df[columns_to_keep]
df.insert(0, "Time", np.arange(0, len(df) / 255, 1 / 255))
df.to_csv("final_EEG.csv", index=False)

time_val = df.iloc[:, 0].values
eeg_data = df.iloc[:, 1:].values.T



print(f"EEG Data Shape: {eeg_data.shape}")

sfreq = 256

