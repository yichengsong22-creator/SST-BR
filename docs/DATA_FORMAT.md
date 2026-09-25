# Data format

Set `dataset.rhb_root` in `configs/rhb_full.yaml`. The root contains
source-segment directories named `<subject>_<segment>`:

```text
RHB_train/
└── <subject>_<segment>/
    ├── <recording>_complex_range_matrix.mat
    └── vital_dict.npy
```

The MAT file exposes a two-dimensional `complex_range_matrix` and a matching
one-dimensional `range_grid`. The PPG file contains one finite numeric vector.
Raw data are not distributed here.

Tables are joined by explicit subject, source-segment, and window identifiers.
The 10-second reference is estimated directly from the matching PPG interval.
