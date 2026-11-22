## Held-out test set (headline)

| config   |   runs |   accuracy |   acc_injected |   acc_control |   acc_real |   acc_identify |   acc_explain |   detection |   false_alarm |   confident_wrong |   abstain |   aurc |   usd_per_task |   calls_per_task |
|:---------|-------:|-----------:|---------------:|--------------:|-----------:|---------------:|--------------:|------------:|--------------:|------------------:|----------:|-------:|---------------:|-----------------:|
| skeptic  |    189 |      0.757 |          0.775 |         0.684 |        nan |          0.857 |           nan |       0.838 |         0.088 |             0.069 |     0.185 |  0.060 |          0.018 |           15.778 |
| react    |    189 |      0.778 |          0.721 |         0.807 |        nan |          1.000 |           nan |       0.811 |         0.140 |             0.127 |     0.101 |  0.168 |          0.010 |           10.423 |
| fixed    |     63 |      0.762 |          0.838 |         0.842 |        nan |          0.143 |           nan |       0.919 |         0.158 |             0.190 |     0.000 |  0.175 |          0.000 |            2.540 |


### Accuracy by family

| family     |   fixed |   react |   skeptic |
|:-----------|--------:|--------:|----------:|
| audit      |    0.5  |    0.39 |      0.22 |
| clock      |    0.75 |    0.79 |      0.75 |
| depth      |    1    |    0.67 |      0.81 |
| drift      |    1    |    0.78 |      1    |
| friction   |    0.14 |    1    |      0.86 |
| mud_weight |    1    |    0.93 |      0.93 |
| planted    |    1    |    1    |      1    |
| stuck      |    0.9  |    0.93 |      0.9  |
| survey     |    0.5  |    0.78 |      0.61 |
| unit       |    1    |    0.4  |      0.53 |


### Skeptic vs ReAct, paired by task

| slice    | compare         |   tasks |   skeptic |   react |   diff |   ci95_low |   ci95_high |     p |   skeptic_better |   react_better |
|:---------|:----------------|--------:|----------:|--------:|-------:|-----------:|------------:|------:|-----------------:|---------------:|
| all      | skeptic - react |      63 |     0.757 |   0.778 | -0.021 |     -0.101 |       0.053 | 0.700 |               11 |             11 |
| injected | skeptic - react |      37 |     0.775 |   0.721 |  0.054 |     -0.027 |       0.135 | 0.306 |                8 |              4 |
| control  | skeptic - react |      19 |     0.684 |   0.807 | -0.123 |     -0.298 |       0.018 | 0.262 |                3 |              6 |
| identify | skeptic - react |       7 |     0.857 |   1.000 | -0.143 |     -0.429 |       0.000 | 1.000 |                0 |              1 |




Total cost: $5.357 over 441 runs.