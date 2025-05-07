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
        
        # Select only the EEG channels we want
        df = df[self.columns_to_keep]
        
        # Add time column
        df.insert(0, "Time", np.arange(0, len(df) / 255, 1 / 255))
        
        # Save to CSV
        df.to_csv(self.output_path, index=False)
        
        # Extract time values and EEG data
        time_val = df.iloc[:, 0].values
        eeg_data = df.iloc[:, 1:].values.T
        
        print(f"EEG Data Shape: {eeg_data.shape}")
        
        return time_val, eeg_data

# Example usage:
if __name__ == "__main__":
    converter = EEGDataConverter(
        input_path="./input/DLR_3_1.tsv",
        output_path="./output/final_EEG.csv"
    )
    time_values, eeg_data = converter.convert()
