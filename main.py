import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
print(data_path)
subjects_dir = data_path / "subjects"
sample_dir = data_path / "MEG" / "sample"

brain_kwargs = dict(alpha=0.1, background="white", cortex="low_contrast")
brain = mne.viz.Brain("sample", subjects_dir=subjects_dir, **brain_kwargs)

stc = mne.read_source_estimate(sample_dir / "sample_audvis-eeg")
stc.crop(0.09, 0.1)

kwargs = dict(
    fmin=stc.data.min(),
    fmax=stc.data.max(),
    alpha=0.25,
    smoothing_steps="nearest",
    time=stc.times,
)
print(stc.data.min())
print(stc.data.max())
brain.add_data(stc.lh_data, hemi="lh", vertices=stc.lh_vertno, **kwargs)
brain.add_data(stc.rh_data, hemi="rh", vertices=stc.rh_vertno, **kwargs)

input()