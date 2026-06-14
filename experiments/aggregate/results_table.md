# Multi-seed results (mean +/- std)

## 7-day forecast (training horizon)

| Model | n seeds | mean RMSE (degC) | mean skill | params |
|-------|---------|------------------|-----------|--------|
| Patch Transformer | 3 | 0.5108 +/- 0.0014 | +0.1085 +/- 0.0020 | 712,821 |
| Tubelet Transformer | 3 | 0.5037 +/- 0.0019 | +0.1216 +/- 0.0046 | 1,440,647 |

## Autoregressive long horizon

| Model | n seeds | useful horizon (days) |
|-------|---------|----------------------|
| Patch Transformer | 3 | 13.0 +/- 6.2 (per seed: [6, 18, 15]) |
| Tubelet Transformer | 3 | 6.7 +/- 0.6 (per seed: [7, 7, 6]) |
