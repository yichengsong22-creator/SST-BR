# Data format

The public adapter expects source-segment directories named
`<subject>_<segment>`, each containing one radar MAT file and one
`vital_dict.npy` PPG vector. The MAT file exposes `complex_range_matrix` and
`range_grid`.

Local tables use subject, source-segment, and local-window identifiers. Direct
evaluation references use subject, source-segment, and evaluation-window
identifiers. Explicit keys prevent row-order joins.

The Binder creates synthetic arrays in memory. No RHB record, subject ID,
feature cache, prediction file, or model is distributed.
