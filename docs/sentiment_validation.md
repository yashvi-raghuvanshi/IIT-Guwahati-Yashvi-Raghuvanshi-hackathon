# Sentiment validation (300 labelled sentences)

Majority-class baseline accuracy: 57.3%


## finbert: accuracy 85.7%, macro-F1 0.86 (threshold ±0.3)

Rows = true label, columns = predicted

```
col_0     negative  neutral  positive
label                                
negative        42        1         0
neutral         11      134        27
positive         0        4        81
```

| class | precision | recall | F1 |
|---|---|---|---|
| negative | 0.79 | 0.98 | 0.88 |
| neutral | 0.96 | 0.78 | 0.86 |
| positive | 0.75 | 0.95 | 0.84 |

## vader: accuracy 59.3%, macro-F1 0.49 (threshold ±0.3)

Rows = true label, columns = predicted

```
col_0     negative  neutral  positive
label                                
negative         7       24        12
neutral          4      118        50
positive         4       28        53
```

| class | precision | recall | F1 |
|---|---|---|---|
| negative | 0.47 | 0.16 | 0.24 |
| neutral | 0.69 | 0.69 | 0.69 |
| positive | 0.46 | 0.62 | 0.53 |