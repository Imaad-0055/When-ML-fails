# When ML Fails: Paddy Yield Regression

## Research question and chosen dataset

Dataset: Paddy (UCI 1186 style tabular agronomic dataset), with 2789 rows and 44 input features. The task is regression: predict `Paddy yield(in Kg)`.

Research question: **Does a Random Forest trained to predict total paddy yield rely on cultivated-area and raw input-quantity features as a scale shortcut, producing excellent random-split performance but failing when evaluated on unseen cultivated-area groups?**

This question is falsifiable. If performance remains stable when one `Hectares` group is held out, or if removing/normalizing scale features does not improve the held-out-area failure, the shortcut hypothesis is not supported.

## Reference model and observed symptom

The reference model is a non-linear `RandomForestRegressor` with one-hot encoding for categorical variables and fixed seeds. Under ordinary random splits, the model looks extremely strong:

| model | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std |
| --- | --- | --- | --- | --- | --- | --- |
| corrected_per_hectare_rf | 578.375 | 27.156 | 808.456 | 41.076 | 0.992 | 0.001 |
| reference_total_yield_rf | 596.945 | 29.619 | 836.548 | 43.002 | 0.992 | 0.001 |

The symptom appears when the split is changed to leave one cultivated-area group out. The same model has much larger errors on unseen area values:

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| corrected_per_hectare_rf | 784.041 | 945.823 | -0.497 |
| reference_total_yield_rf | 4979.719 | 5075.480 | -271.524 |

Figure `results/leave_one_hectare_mae.png` shows the failure for each held-out `Hectares` value. Figure `results/target_scale_by_hectare.png` shows why the failure is plausible: total yield rises almost mechanically with cultivated area, while yield per hectare varies much less.

## Causal hypothesis and controlled experiment

Hypothesis: the model is not mainly learning agronomic productivity. It is learning the scale of the field and associated raw input quantities. This is a shortcut because it is highly predictive in-distribution but does not encode a stable causal relation for productivity.

Controlled test: compare three random-split ablations:

| split | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| all_features | 632.453 | 886.793 | 0.990 |
| drop_hectares_only | 632.371 | 886.665 | 0.990 |
| drop_all_raw_scale_inputs | 8026.732 | 9246.056 | -0.054 |

Dropping `Hectares` alone barely changes the random-split score, because other raw input quantities still encode field scale. Dropping all raw scale inputs causes performance to collapse, which supports the hypothesis that the reference model depends on scale information rather than robust productivity features.

The stronger out-of-distribution test is leave-one-hectare-out evaluation. In that setting the reference model has to predict an area group absent from training, and its mean MAE rises sharply.

## Proposed correction and evaluation

The correction targets the cause rather than the symptom:

1. Convert raw input quantities to per-hectare quantities.
2. Train the model to predict yield per hectare instead of total yield.
3. Convert predictions back to total yield by multiplying by hectares at inference time.
4. Add mild Random Forest regularization (`max_depth=10`, `min_samples_leaf=5`).

This directly changes the representation so the model learns productivity patterns instead of memorizing total scale. On leave-one-hectare-out evaluation, mean MAE decreases from 4980 kg to 784 kg.

The corrected model keeps random-split performance comparable to the reference model, so the repair does not simply trade away ordinary predictive accuracy.

## Threats to validity

The dataset has only six distinct `Hectares` values, so leave-one-group-out tests are informative but coarse. Differences could still depend on the specific distribution of farms in each area group.

The correction uses `Hectares` again when converting predicted kg/ha back to total kg. This is intentional for total-yield prediction, but it means the final total-yield metric still benefits from knowing field area. The important change is that the learned model no longer treats raw quantities as direct total-scale signals.

The data may contain repeated or near-repeated agronomic recipes. If rows are not independent farms or seasons, random-split performance may be overoptimistic for reasons beyond the scale shortcut.

The experiment tests one representation failure on this CSV. It does not prove that the same correction would work under real seasonal drift, new varieties, or new geographic regions.

No external validation set is available, and the analysis uses the provided features only. A stronger study would include real deployment-time data and confidence intervals from more repeated group splits.

## Conclusion and what we learned

The reference Random Forest appears excellent under a random split, with `R2` near 0.99, but this hides a serious failure: it depends on field-scale variables and raw input quantities. When asked to generalize to unseen cultivated-area groups, errors become much larger.

Normalizing quantities and the target per hectare is a principled correction because it aligns the feature representation with agronomic productivity. The result is not perfect, but it makes the failure visible, tests a causal explanation, and repairs the specific cause more directly than simple hyperparameter tuning.
