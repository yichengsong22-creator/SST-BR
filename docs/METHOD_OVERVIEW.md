# SST-BR method overview

```text
radar observation T -> N local windows Δt
each local window   -> spatial representation
                    -> spectral representation
                    -> local representation
all local vectors   -> time–feature fusion -> subject baseline
current local vector                         -> window residual
baseline + residual                         -> local HR
two local HR predictions                    -> 10-second prediction
```

The full computations and parameters are public:

- Spatial: [`sstbr/features/spatial.py`](../sstbr/features/spatial.py)
- Spectral: [`sstbr/features/spectral.py`](../sstbr/features/spectral.py)
- Ordered extractor: [`sstbr/features/extractor.py`](../sstbr/features/extractor.py)
- Temporal fusion: [`sstbr/temporal/fusion.py`](../sstbr/temporal/fusion.py)
- Baseline–residual regression: [`sstbr/models/baseline_residual.py`](../sstbr/models/baseline_residual.py)
- 10-s aggregation: [`sstbr/evaluation/aggregation.py`](../sstbr/evaluation/aggregation.py)
- Pooled metrics: [`sstbr/evaluation/metrics.py`](../sstbr/evaluation/metrics.py)

The final evaluation averages each consecutive pair of 5-s predictions. The
reference is estimated independently from the matching 10-second PPG samples.
