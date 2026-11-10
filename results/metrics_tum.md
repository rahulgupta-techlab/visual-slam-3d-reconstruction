# Visual odometry evaluation

Dataset: TUM monocular fallback.
KITTI uses metric stereo scale and no scale correction in evo. TUM uses one global Sim(3) alignment because monocular scale is unobservable.
Plots and raw logs are written only for sequences whose evo runs complete successfully.

| Sequence | ATE RMSE (m) | RPE RMSE (m) | Status |
|---|---:|---:|---|
| fr1_xyz | 0.148051 | 0.011285 | Completed |
| fr1_desk | 0.466446 | 0.031539 | Completed |
