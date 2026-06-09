EEG Stem Separation — Dataset Layout
====================================

Use train/ and val/ splits under this folder. Each split needs the same six files.

dataset/
├── train/
│   ├── mix.npy
│   ├── target_eeg.npy
│   ├── target_ecg.npy
│   ├── target_emg.npy
│   ├── target_eog.npy
│   └── target_noise.npy
└── val/
    ├── mix.npy
    ├── target_eeg.npy
    ├── target_ecg.npy
    ├── target_emg.npy
    ├── target_eog.npy
    └── target_noise.npy

Tensor shape per file: (num_channels, num_time_samples)
Optional batch dimension: (num_recordings, num_channels, num_time_samples)
All six files in a split must share the same shape.
