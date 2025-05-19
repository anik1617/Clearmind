import mne
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import os

class EEGDataConverter:
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the EEG data converter with input and output paths.
        
        Args:
            input_path (str): Path to the input TSV file
            output_path (str): Path to save the output CSV file
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define headers for the TSV file
        self.headers = [
            "Index", "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2", "F7", "F8", "C3", "C4", 
            "T3", "T4", "P3", "P4", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2", 
            "Other", "Other", "Other", "Other", "Other", "Other", "Other", 
            "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"
        ]
        
        # Define EEG channels to keep
        self.columns_to_keep = [
            "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
            "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"
        ]

    def convert(self) -> tuple:
        """
        Convert the TSV file to processed CSV format.
        
        Returns:
            tuple: (time_values, eeg_data) where:
                - time_values: numpy array of time points
                - eeg_data: numpy array of EEG data (channels x time)
        """
        # Read TSV file
        df = pd.read_table(self.input_path, sep='\t', header=None)
        df.columns = self.headers
        
        # Extract timestamp column
        timestamp_col = df["Timestamp"].values
        start_time = timestamp_col[0]
        elapsed_time = timestamp_col - start_time  # Time relative to first sample

        # Select only the EEG channels
        df_eeg = df[self.columns_to_keep].copy()

        # Add time column (based on actual timestamps)
        df_eeg.insert(0, "Time", elapsed_time)

        # Save to CSV
        df_eeg.to_csv(self.output_path, index=False)

        # Return arrays
        time_val = df_eeg["Time"].values
        eeg_data = df_eeg.iloc[:, 1:].values.T

        print(f"EEG Data Shape: {eeg_data.shape}")
        return time_val, eeg_data


# if __name__ == "__main__":
#     converter = EEGDataConverter(
#         input_path="./input/DLR_3_1.tsv",
#         output_path="./output/final_EEG.csv"
#     )
#     time_values, eeg_data = converter.convert()
