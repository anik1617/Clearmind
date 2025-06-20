import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import mne
from mne.datasets import sample
import os
import threading
import sys
import subprocess

class EEGVisualizer:
    def __init__(self, csv_path: str, subjects_dir: str = "C:/Users/anik2/mne_data/MNE-fsaverage-data/"):
        """
        Initialize the EEG visualizer.
        
        Args:
            csv_path (str): Path to the input CSV file containing EEG data
            subjects_dir (str): Path to the fsaverage subject directory
        """
        self.csv_path = csv_path
        self.subjects_dir = subjects_dir
        self.channels = ["Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
                        "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"]
        self.sfreq = 256
        self.n_jobs = -1  # Use all available CPU cores

    def visualize(self):
        """
        Process the EEG data and create visualizations.
        """
        # Read and prepare data
        df = pd.read_csv(self.csv_path)
        eeg_data = df.iloc[:, 2:].values.T
        print(f"EEG Data Shape: {eeg_data.shape}")

        # Create MNE info and raw array
        info = mne.create_info(ch_names=self.channels, sfreq=self.sfreq, ch_types=["eeg"] * len(self.channels))
        raw = mne.io.RawArray(eeg_data, info)
        raw.set_eeg_reference('average', projection=True)
        print(raw.info)

        # Set montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)

        # Setup source space and BEM
        subject = "fsaverage"
        src = mne.setup_source_space(subject, spacing="ico4", subjects_dir=self.subjects_dir, add_dist=False)
        bem = mne.make_bem_model(subject, ico=3, subjects_dir=self.subjects_dir)
        bem_sol = mne.make_bem_solution(bem)

        # Compute forward model
        trans = "fsaverage"
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True, n_jobs=self.n_jobs)

        # Handle covariance based on data length
        print("N_TIMES:", raw.n_times)
        if raw.n_times < 12000:
            cov = mne.compute_raw_covariance(raw, method="shrunk", n_jobs=self.n_jobs)
        else:
            cov = mne.compute_raw_covariance(raw, method="empirical", n_jobs=self.n_jobs)
            raw.crop(tmin=0, tmax=5)

        # Compute inverse operator and apply
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
        stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0 / 3.0 ** 2, method="dSPM")

        # Create brain visualization
        brain_kwargs = dict(alpha=0.4, background="white", cortex="classic")
        brain = mne.viz.Brain(subject, subjects_dir=self.subjects_dir, **brain_kwargs)

        # Crop and prepare data for visualization
        stc.crop(0.05, 0.5)
        kwargs = dict(
            fmin=stc.data.min(),
            fmax=stc.data.max(),
            alpha=0.25,
            smoothing_steps="nearest",
            time=stc.times,
        )

        # Add data to brain visualization
        brain.add_data(stc.lh_data, hemi="lh", vertices=stc.lh_vertno, **kwargs)
        brain.add_data(stc.rh_data, hemi="rh", vertices=stc.rh_vertno, **kwargs)

        
        raw.plot_sensors(kind="topomap", show_names=True)
        plt.show(block=True)
        brain.show()


def launch_in_subprocess(csv_path: str):
    subprocess.Popen(
        [sys.executable, __file__, csv_path]
        # creationflags=subprocess.CREATE_NO_WINDOW  # optional: hide terminal popup
    )

        



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <csv_path>")
    else:
        path = sys.argv[1]
        EEGVisualizer(path).visualize()
