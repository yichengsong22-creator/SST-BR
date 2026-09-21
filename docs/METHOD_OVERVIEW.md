# SST-BR method overview

```text
Radar observation T
      -> N local windows Δt

Each local window
      -> spatial representation
      -> spectral representation
      -> local representation

All local representations
      -> short-to-long temporal fusion
      -> baseline branch

Current local representation
      -> residual branch

baseline + residual
      -> local HR
      -> 10-second aggregation and direct PPG evaluation
```

The module names, interfaces, shapes, and scientific data flow are public. The
Binder uses `DemoCore` to make every stage executable. Its generic summaries
are educational and are not the representation, temporal computation, or
regression implementation used for the manuscript result.

The exact computational formulas, feature ordering, fitted-model settings, and
learned artifacts are intentionally absent during review.
