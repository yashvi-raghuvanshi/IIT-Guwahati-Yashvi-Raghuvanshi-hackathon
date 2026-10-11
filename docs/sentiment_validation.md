# Sentiment validation (300 labelled sentences)

Majority-class baseline accuracy: 57.3%


## finbert: accuracy 78.0%, macro-F1 0.79 (threshold ±0.15)

Rows = true label, columns = predicted

```
col_0     negative  neutral  positive
label                                
negative        43        0         0
neutral         19      107        46
positive         1        0        84
```

| class | precision | recall | F1 |
|---|---|---|---|
| negative | 0.68 | 1.00 | 0.81 |
| neutral | 1.00 | 0.62 | 0.77 |
| positive | 0.65 | 0.99 | 0.78 |

## vader: accuracy 53.7%, macro-F1 0.49 (threshold ±0.15)

Rows = true label, columns = predicted

```
col_0     negative  neutral  positive
label                                
negative        12       15        16
neutral          8       90        74
positive         5       21        59
```

| class | precision | recall | F1 |
|---|---|---|---|
| negative | 0.48 | 0.28 | 0.35 |
| neutral | 0.71 | 0.52 | 0.60 |
| positive | 0.40 | 0.69 | 0.50 |