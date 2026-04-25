# Hyperparameter Pivot Tables

????????????????????????????????????? `best_val_acc`?

## activation x hidden_dim

| activation \ hidden_dim | 128 | 256 | 512 | 768 |
|---|---:|---:|---:|---:|
| relu | 0.908833 | 0.910333 | 0.912833 | 0.912167 |
| sigmoid | 0.897000 | 0.906667 | 0.907000 | 0.903167 |
| tanh | 0.902500 | 0.909333 | 0.907500 | 0.905167 |

Best settings for each filled cell:

| row | col | best_val_acc | activation | hidden_dim | lr | lr_decay | weight_decay | run_name |
|---|---|---:|---|---:|---:|---:|---:|---|
| relu | 128 | 0.908833 | relu | 128 | 0.2 | 0.95 | 0.001 | hd128_relu_lr0p2_decay0p95_wd0p001 |
| relu | 256 | 0.910333 | relu | 256 | 0.2 | 0.95 | 0.001 | hd256_relu_lr0p2_decay0p95_wd0p001 |
| relu | 512 | 0.912833 | relu | 512 | 0.2 | 0.95 | 0.001 | hd512_relu_lr0p2_decay0p95_wd0p001 |
| relu | 768 | 0.912167 | relu | 768 | 0.2 | 0.95 | 0.001 | hd768_relu_lr0p2_decay0p95_wd0p001 |
| sigmoid | 128 | 0.897000 | sigmoid | 128 | 0.2 | 0.98 | 0.0001 | hd128_sigmoid_lr0p2_decay0p98_wd0p0001 |
| sigmoid | 256 | 0.906667 | sigmoid | 256 | 0.2 | 0.98 | 0 | hd256_sigmoid_lr0p2_decay0p98_wd0p0 |
| sigmoid | 512 | 0.907000 | sigmoid | 512 | 0.2 | 0.98 | 1e-05 | hd512_sigmoid_lr0p2_decay0p98_wd1e-05 |
| sigmoid | 768 | 0.903167 | sigmoid | 768 | 0.2 | 1 | 0.0001 | hd768_sigmoid_lr0p2_decay1p0_wd0p0001 |
| tanh | 128 | 0.902500 | tanh | 128 | 0.2 | 0.98 | 0.001 | hd128_tanh_lr0p2_decay0p98_wd0p001 |
| tanh | 256 | 0.909333 | tanh | 256 | 0.2 | 0.95 | 0.001 | hd256_tanh_lr0p2_decay0p95_wd0p001 |
| tanh | 512 | 0.907500 | tanh | 512 | 0.2 | 0.95 | 0.001 | hd512_tanh_lr0p2_decay0p95_wd0p001 |
| tanh | 768 | 0.905167 | tanh | 768 | 0.2 | 0.95 | 0.0001 | hd768_tanh_lr0p2_decay0p95_wd0p0001 |


## lr_decay x lr

| lr_decay \ lr | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|
| 0.9 | 0.880333 | 0.899167 | 0.906167 | 0.909167 |
| 0.95 | 0.886500 | 0.905167 | 0.910500 | 0.912833 |
| 0.98 | 0.892167 | 0.907167 | 0.910333 | 0.911500 |
| 1 | 0.898500 | 0.907167 | 0.907500 | 0.910000 |

Best settings for each filled cell:

| row | col | best_val_acc | activation | hidden_dim | lr | lr_decay | weight_decay | run_name |
|---|---|---:|---|---:|---:|---:|---:|---|
| 0.9 | 0.01 | 0.880333 | relu | 768 | 0.01 | 0.9 | 0.001 | hd768_relu_lr0p01_decay0p9_wd0p001 |
| 0.9 | 0.05 | 0.899167 | relu | 768 | 0.05 | 0.9 | 0.001 | hd768_relu_lr0p05_decay0p9_wd0p001 |
| 0.9 | 0.1 | 0.906167 | relu | 512 | 0.1 | 0.9 | 0.001 | hd512_relu_lr0p1_decay0p9_wd0p001 |
| 0.9 | 0.2 | 0.909167 | relu | 768 | 0.2 | 0.9 | 0.001 | hd768_relu_lr0p2_decay0p9_wd0p001 |
| 0.95 | 0.01 | 0.886500 | relu | 768 | 0.01 | 0.95 | 0.001 | hd768_relu_lr0p01_decay0p95_wd0p001 |
| 0.95 | 0.05 | 0.905167 | relu | 512 | 0.05 | 0.95 | 0.001 | hd512_relu_lr0p05_decay0p95_wd0p001 |
| 0.95 | 0.1 | 0.910500 | relu | 768 | 0.1 | 0.95 | 0.001 | hd768_relu_lr0p1_decay0p95_wd0p001 |
| 0.95 | 0.2 | 0.912833 | relu | 512 | 0.2 | 0.95 | 0.001 | hd512_relu_lr0p2_decay0p95_wd0p001 |
| 0.98 | 0.01 | 0.892167 | relu | 768 | 0.01 | 0.98 | 0.001 | hd768_relu_lr0p01_decay0p98_wd0p001 |
| 0.98 | 0.05 | 0.907167 | relu | 512 | 0.05 | 0.98 | 0.001 | hd512_relu_lr0p05_decay0p98_wd0p001 |
| 0.98 | 0.1 | 0.910333 | relu | 512 | 0.1 | 0.98 | 0.001 | hd512_relu_lr0p1_decay0p98_wd0p001 |
| 0.98 | 0.2 | 0.911500 | relu | 512 | 0.2 | 0.98 | 0 | hd512_relu_lr0p2_decay0p98_wd0p0 |
| 1 | 0.01 | 0.898500 | relu | 768 | 0.01 | 1 | 0.001 | hd768_relu_lr0p01_decay1p0_wd0p001 |
| 1 | 0.05 | 0.907167 | relu | 768 | 0.05 | 1 | 0.001 | hd768_relu_lr0p05_decay1p0_wd0p001 |
| 1 | 0.1 | 0.907500 | relu | 768 | 0.1 | 1 | 0 | hd768_relu_lr0p1_decay1p0_wd0p0 |
| 1 | 0.2 | 0.910000 | relu | 768 | 0.2 | 1 | 1e-05 | hd768_relu_lr0p2_decay1p0_wd1e-05 |


## weight_decay x lr

| weight_decay \ lr | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|
| 0 | 0.897000 | 0.905000 | 0.907500 | 0.911500 |
| 1e-05 | 0.897333 | 0.905667 | 0.907167 | 0.910000 |
| 0.0001 | 0.897500 | 0.906500 | 0.907000 | 0.909333 |
| 0.001 | 0.898500 | 0.907167 | 0.910500 | 0.912833 |

Best settings for each filled cell:

| row | col | best_val_acc | activation | hidden_dim | lr | lr_decay | weight_decay | run_name |
|---|---|---:|---|---:|---:|---:|---:|---|
| 0 | 0.01 | 0.897000 | relu | 768 | 0.01 | 1 | 0 | hd768_relu_lr0p01_decay1p0_wd0p0 |
| 0 | 0.05 | 0.905000 | relu | 512 | 0.05 | 0.98 | 0 | hd512_relu_lr0p05_decay0p98_wd0p0 |
| 0 | 0.1 | 0.907500 | relu | 768 | 0.1 | 1 | 0 | hd768_relu_lr0p1_decay1p0_wd0p0 |
| 0 | 0.2 | 0.911500 | relu | 512 | 0.2 | 0.98 | 0 | hd512_relu_lr0p2_decay0p98_wd0p0 |
| 1e-05 | 0.01 | 0.897333 | relu | 768 | 0.01 | 1 | 1e-05 | hd768_relu_lr0p01_decay1p0_wd1e-05 |
| 1e-05 | 0.05 | 0.905667 | relu | 512 | 0.05 | 0.98 | 1e-05 | hd512_relu_lr0p05_decay0p98_wd1e-05 |
| 1e-05 | 0.1 | 0.907167 | relu | 768 | 0.1 | 0.98 | 1e-05 | hd768_relu_lr0p1_decay0p98_wd1e-05 |
| 1e-05 | 0.2 | 0.910000 | relu | 768 | 0.2 | 1 | 1e-05 | hd768_relu_lr0p2_decay1p0_wd1e-05 |
| 0.0001 | 0.01 | 0.897500 | relu | 768 | 0.01 | 1 | 0.0001 | hd768_relu_lr0p01_decay1p0_wd0p0001 |
| 0.0001 | 0.05 | 0.906500 | relu | 512 | 0.05 | 1 | 0.0001 | hd512_relu_lr0p05_decay1p0_wd0p0001 |
| 0.0001 | 0.1 | 0.907000 | relu | 768 | 0.1 | 0.95 | 0.0001 | hd768_relu_lr0p1_decay0p95_wd0p0001 |
| 0.0001 | 0.2 | 0.909333 | relu | 512 | 0.2 | 1 | 0.0001 | hd512_relu_lr0p2_decay1p0_wd0p0001 |
| 0.001 | 0.01 | 0.898500 | relu | 768 | 0.01 | 1 | 0.001 | hd768_relu_lr0p01_decay1p0_wd0p001 |
| 0.001 | 0.05 | 0.907167 | relu | 512 | 0.05 | 0.98 | 0.001 | hd512_relu_lr0p05_decay0p98_wd0p001 |
| 0.001 | 0.1 | 0.910500 | relu | 768 | 0.1 | 0.95 | 0.001 | hd768_relu_lr0p1_decay0p95_wd0p001 |
| 0.001 | 0.2 | 0.912833 | relu | 512 | 0.2 | 0.95 | 0.001 | hd512_relu_lr0p2_decay0p95_wd0p001 |


## lr x hidden_dim

| lr \ hidden_dim | 128 | 256 | 512 | 768 |
|---|---:|---:|---:|---:|
| 0.01 | 0.892833 | 0.893667 | 0.896500 | 0.898500 |
| 0.05 | 0.901833 | 0.905167 | 0.907167 | 0.907167 |
| 0.1 | 0.904333 | 0.908500 | 0.910333 | 0.910500 |
| 0.2 | 0.908833 | 0.910333 | 0.912833 | 0.912167 |

Best settings for each filled cell:

| row | col | best_val_acc | activation | hidden_dim | lr | lr_decay | weight_decay | run_name |
|---|---|---:|---|---:|---:|---:|---:|---|
| 0.01 | 128 | 0.892833 | relu | 128 | 0.01 | 1 | 0.0001 | hd128_relu_lr0p01_decay1p0_wd0p0001 |
| 0.01 | 256 | 0.893667 | relu | 256 | 0.01 | 1 | 0 | hd256_relu_lr0p01_decay1p0_wd0p0 |
| 0.01 | 512 | 0.896500 | relu | 512 | 0.01 | 1 | 0 | hd512_relu_lr0p01_decay1p0_wd0p0 |
| 0.01 | 768 | 0.898500 | relu | 768 | 0.01 | 1 | 0.001 | hd768_relu_lr0p01_decay1p0_wd0p001 |
| 0.05 | 128 | 0.901833 | relu | 128 | 0.05 | 1 | 0.001 | hd128_relu_lr0p05_decay1p0_wd0p001 |
| 0.05 | 256 | 0.905167 | tanh | 256 | 0.05 | 1 | 0.001 | hd256_tanh_lr0p05_decay1p0_wd0p001 |
| 0.05 | 512 | 0.907167 | relu | 512 | 0.05 | 0.98 | 0.001 | hd512_relu_lr0p05_decay0p98_wd0p001 |
| 0.05 | 768 | 0.907167 | relu | 768 | 0.05 | 1 | 0.001 | hd768_relu_lr0p05_decay1p0_wd0p001 |
| 0.1 | 128 | 0.904333 | relu | 128 | 0.1 | 0.98 | 0.001 | hd128_relu_lr0p1_decay0p98_wd0p001 |
| 0.1 | 256 | 0.908500 | relu | 256 | 0.1 | 0.98 | 0.001 | hd256_relu_lr0p1_decay0p98_wd0p001 |
| 0.1 | 512 | 0.910333 | relu | 512 | 0.1 | 0.98 | 0.001 | hd512_relu_lr0p1_decay0p98_wd0p001 |
| 0.1 | 768 | 0.910500 | relu | 768 | 0.1 | 0.95 | 0.001 | hd768_relu_lr0p1_decay0p95_wd0p001 |
| 0.2 | 128 | 0.908833 | relu | 128 | 0.2 | 0.95 | 0.001 | hd128_relu_lr0p2_decay0p95_wd0p001 |
| 0.2 | 256 | 0.910333 | relu | 256 | 0.2 | 0.95 | 0.001 | hd256_relu_lr0p2_decay0p95_wd0p001 |
| 0.2 | 512 | 0.912833 | relu | 512 | 0.2 | 0.95 | 0.001 | hd512_relu_lr0p2_decay0p95_wd0p001 |
| 0.2 | 768 | 0.912167 | relu | 768 | 0.2 | 0.95 | 0.001 | hd768_relu_lr0p2_decay0p95_wd0p001 |


## weight_decay x hidden_dim

| weight_decay \ hidden_dim | 128 | 256 | 512 | 768 |
|---|---:|---:|---:|---:|
| 0 | 0.902167 | 0.906667 | 0.911500 | 0.907500 |
| 1e-05 | 0.902333 | 0.906667 | 0.909333 | 0.910000 |
| 0.0001 | 0.902500 | 0.906167 | 0.909333 | 0.909000 |
| 0.001 | 0.908833 | 0.910333 | 0.912833 | 0.912167 |

Best settings for each filled cell:

| row | col | best_val_acc | activation | hidden_dim | lr | lr_decay | weight_decay | run_name |
|---|---|---:|---|---:|---:|---:|---:|---|
| 0 | 128 | 0.902167 | relu | 128 | 0.2 | 0.95 | 0 | hd128_relu_lr0p2_decay0p95_wd0p0 |
| 0 | 256 | 0.906667 | sigmoid | 256 | 0.2 | 0.98 | 0 | hd256_sigmoid_lr0p2_decay0p98_wd0p0 |
| 0 | 512 | 0.911500 | relu | 512 | 0.2 | 0.98 | 0 | hd512_relu_lr0p2_decay0p98_wd0p0 |
| 0 | 768 | 0.907500 | relu | 768 | 0.1 | 1 | 0 | hd768_relu_lr0p1_decay1p0_wd0p0 |
| 1e-05 | 128 | 0.902333 | relu | 128 | 0.2 | 0.95 | 1e-05 | hd128_relu_lr0p2_decay0p95_wd1e-05 |
| 1e-05 | 256 | 0.906667 | sigmoid | 256 | 0.2 | 0.98 | 1e-05 | hd256_sigmoid_lr0p2_decay0p98_wd1e-05 |
| 1e-05 | 512 | 0.909333 | relu | 512 | 0.2 | 1 | 1e-05 | hd512_relu_lr0p2_decay1p0_wd1e-05 |
| 1e-05 | 768 | 0.910000 | relu | 768 | 0.2 | 1 | 1e-05 | hd768_relu_lr0p2_decay1p0_wd1e-05 |
| 0.0001 | 128 | 0.902500 | relu | 128 | 0.2 | 0.9 | 0.0001 | hd128_relu_lr0p2_decay0p9_wd0p0001 |
| 0.0001 | 256 | 0.906167 | sigmoid | 256 | 0.2 | 1 | 0.0001 | hd256_sigmoid_lr0p2_decay1p0_wd0p0001 |
| 0.0001 | 512 | 0.909333 | relu | 512 | 0.2 | 1 | 0.0001 | hd512_relu_lr0p2_decay1p0_wd0p0001 |
| 0.0001 | 768 | 0.909000 | relu | 768 | 0.2 | 1 | 0.0001 | hd768_relu_lr0p2_decay1p0_wd0p0001 |
| 0.001 | 128 | 0.908833 | relu | 128 | 0.2 | 0.95 | 0.001 | hd128_relu_lr0p2_decay0p95_wd0p001 |
| 0.001 | 256 | 0.910333 | relu | 256 | 0.2 | 0.95 | 0.001 | hd256_relu_lr0p2_decay0p95_wd0p001 |
| 0.001 | 512 | 0.912833 | relu | 512 | 0.2 | 0.95 | 0.001 | hd512_relu_lr0p2_decay0p95_wd0p001 |
| 0.001 | 768 | 0.912167 | relu | 768 | 0.2 | 0.95 | 0.001 | hd768_relu_lr0p2_decay0p95_wd0p001 |


