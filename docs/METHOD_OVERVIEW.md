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

The paper endpoint averages consecutive local predictions. Its reference is
estimated directly from the matching 10-second PPG samples.
