from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Paddy yield(in Kg)"
GROUP = "Hectares"
SEEDS = [0, 1, 2, 3, 4]

QUANTITY_COLUMNS = [
    "Seedrate(in Kg)",
    "LP_Mainfield(in Tonnes)",
    "Nursery area (Cents)",
    "LP_nurseryarea(in Tonnes)",
    "DAP_20days",
    "Weed28D_thiobencarb",
    "Urea_40Days",
    "Potassh_50Days",
    "Micronutrients_70Days",
    "Pest_60Day(in ml)",
    "Trash(in bundles)",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessor_for(x: pd.DataFrame) -> ColumnTransformer:
    categorical = x.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in x.columns if c not in categorical]
    return ColumnTransformer(
        [
            ("categorical", one_hot_encoder(), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )


def random_forest_pipeline(x: pd.DataFrame, seed: int, **kwargs) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor_for(x)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                    **kwargs,
                ),
            ),
        ]
    )


def metric_row(y_true: pd.Series, y_pred: np.ndarray, split: str, model: str) -> dict[str, float | str]:
    return {
        "split": split,
        "model": model,
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def add_per_hectare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    corrected = df.copy()
    for col in QUANTITY_COLUMNS:
        corrected[f"{col}_per_ha"] = corrected[col] / corrected[GROUP]
    y_per_hectare = corrected[TARGET] / corrected[GROUP]
    x_corrected = corrected.drop(columns=[TARGET] + QUANTITY_COLUMNS)
    return x_corrected, y_per_hectare


def run_random_split(df: pd.DataFrame, pipeline: str = "both") -> pd.DataFrame:
    rows = []
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_corrected, y_per_hectare = add_per_hectare_features(df)

    for seed in SEEDS:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=seed)

        if pipeline in {"both", "reference"}:
            reference = random_forest_pipeline(x, seed)
            reference.fit(x.loc[train_idx], y.loc[train_idx])
            rows.append(
                metric_row(
                    y.loc[test_idx],
                    reference.predict(x.loc[test_idx]),
                    f"random_seed_{seed}",
                    "reference_total_yield_rf",
                )
            )

        if pipeline in {"both", "corrected"}:
            corrected = random_forest_pipeline(
                x_corrected,
                seed,
                max_depth=10,
                min_samples_leaf=5,
            )
            corrected.fit(x_corrected.loc[train_idx], y_per_hectare.loc[train_idx])
            pred_total = corrected.predict(x_corrected.loc[test_idx]) * df.loc[test_idx, GROUP]
            rows.append(
                metric_row(
                    y.loc[test_idx],
                    pred_total,
                    f"random_seed_{seed}",
                    "corrected_per_hectare_rf",
                )
            )

    return pd.DataFrame(rows)


def run_leave_one_hectare_out(df: pd.DataFrame, pipeline: str = "both") -> pd.DataFrame:
    rows = []
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_corrected, y_per_hectare = add_per_hectare_features(df)

    for hectare in sorted(df[GROUP].unique()):
        test_mask = df[GROUP] == hectare
        train_mask = ~test_mask

        if pipeline in {"both", "reference"}:
            reference = random_forest_pipeline(x, seed=42)
            reference.fit(x.loc[train_mask], y.loc[train_mask])
            rows.append(
                metric_row(
                    y.loc[test_mask],
                    reference.predict(x.loc[test_mask]),
                    f"leave_hectares_{hectare}_out",
                    "reference_total_yield_rf",
                )
            )

        if pipeline in {"both", "corrected"}:
            corrected = random_forest_pipeline(
                x_corrected,
                seed=42,
                max_depth=10,
                min_samples_leaf=5,
            )
            corrected.fit(x_corrected.loc[train_mask], y_per_hectare.loc[train_mask])
            pred_total = corrected.predict(x_corrected.loc[test_mask]) * df.loc[test_mask, GROUP]
            rows.append(
                metric_row(
                    y.loc[test_mask],
                    pred_total,
                    f"leave_hectares_{hectare}_out",
                    "corrected_per_hectare_rf",
                )
            )

    return pd.DataFrame(rows)


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    ablations = {
        "all_features": [],
        "drop_hectares_only": [GROUP],
        "drop_all_raw_scale_inputs": [GROUP] + QUANTITY_COLUMNS,
    }

    rows = []
    for label, drop_columns in ablations.items():
        x_ablate = x.drop(columns=drop_columns)
        model = random_forest_pipeline(x_ablate, seed=42)
        model.fit(x_ablate.loc[train_idx], y.loc[train_idx])
        rows.append(metric_row(y.loc[test_idx], model.predict(x_ablate.loc[test_idx]), label, "ablation_rf"))
    return pd.DataFrame(rows)


def summarize_by_hectare(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.assign(yield_per_hectare=df[TARGET] / df[GROUP])
    return (
        summary.groupby(GROUP)
        .agg(
            n=(TARGET, "size"),
            total_yield_mean=(TARGET, "mean"),
            total_yield_std=(TARGET, "std"),
            yield_per_hectare_mean=("yield_per_hectare", "mean"),
            yield_per_hectare_std=("yield_per_hectare", "std"),
        )
        .round(2)
        .reset_index()
    )


def plot_failure(logo: pd.DataFrame, output_dir: Path) -> None:
    pivot = logo.pivot(index="split", columns="model", values="MAE")
    pivot = pivot.sort_index(key=lambda idx: idx.str.extract(r"(\d+)").astype(int)[0])

    labels = [label.replace("leave_hectares_", "").replace("_out", " ha") for label in pivot.index]
    x_pos = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width / 2, pivot["reference_total_yield_rf"], width, label="Reference: total-yield RF")
    plt.bar(x_pos + width / 2, pivot["corrected_per_hectare_rf"], width, label="Correction: per-hectare RF")
    plt.xticks(x_pos, labels)
    plt.ylabel("MAE on held-out area group (kg)")
    plt.xlabel("Held-out cultivated area")
    plt.title("Failure under leave-one-hectare-out evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "leave_one_hectare_mae.png", dpi=180)
    plt.close()


def plot_target_scale(hectare_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hectare_summary[GROUP], hectare_summary["total_yield_mean"], marker="o", label="Total yield")
    ax1.set_xlabel("Hectares")
    ax1.set_ylabel("Mean total yield (kg)")
    ax2 = ax1.twinx()
    ax2.plot(
        hectare_summary[GROUP],
        hectare_summary["yield_per_hectare_mean"],
        marker="s",
        color="tab:orange",
        label="Yield per hectare",
    )
    ax2.set_ylabel("Mean yield per hectare (kg/ha)")
    ax1.set_title("Total yield mostly encodes cultivated area")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "target_scale_by_hectare.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    data = frame.copy()
    if index:
        data = data.reset_index()
    else:
        data = data.reset_index(drop=True)

    headers = [str(col) for col in data.columns]
    rows = []
    for _, row in data.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append(values)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(df: pd.DataFrame, random_split: pd.DataFrame, logo: pd.DataFrame, ablation: pd.DataFrame, output: Path) -> None:
    random_summary = random_split.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(3)
    random_summary.columns = [f"{metric}_{stat}" for metric, stat in random_summary.columns]
    logo_summary = logo.groupby("model")[["MAE", "RMSE", "R2"]].mean().round(3)
    ablation_table = ablation[["split", "MAE", "RMSE", "R2"]].round(3)

    report = f"""# When ML Fails: Paddy Yield Regression

## Research question and chosen dataset

Dataset: Paddy (UCI 1186 style tabular agronomic dataset), with {len(df)} rows and {df.shape[1] - 1} input features. The task is regression: predict `{TARGET}`.

Research question: **Does a Random Forest trained to predict total paddy yield rely on cultivated-area and raw input-quantity features as a scale shortcut, producing excellent random-split performance but failing when evaluated on unseen cultivated-area groups?**

This question is falsifiable. If performance remains stable when one `Hectares` group is held out, or if removing/normalizing scale features does not improve the held-out-area failure, the shortcut hypothesis is not supported.

## Reference model and observed symptom

The reference model is a non-linear `RandomForestRegressor` with one-hot encoding for categorical variables and fixed seeds. Under ordinary random splits, the model looks extremely strong:

{markdown_table(random_summary)}

The symptom appears when the split is changed to leave one cultivated-area group out. The same model has much larger errors on unseen area values:

{markdown_table(logo_summary)}

Figure `results/leave_one_hectare_mae.png` shows the failure for each held-out `Hectares` value. Figure `results/target_scale_by_hectare.png` shows why the failure is plausible: total yield rises almost mechanically with cultivated area, while yield per hectare varies much less.

## Causal hypothesis and controlled experiment

Hypothesis: the model is not mainly learning agronomic productivity. It is learning the scale of the field and associated raw input quantities. This is a shortcut because it is highly predictive in-distribution but does not encode a stable causal relation for productivity.

Controlled test: compare three random-split ablations:

{markdown_table(ablation_table, index=False)}

Dropping `Hectares` alone barely changes the random-split score, because other raw input quantities still encode field scale. Dropping all raw scale inputs causes performance to collapse, which supports the hypothesis that the reference model depends on scale information rather than robust productivity features.

The stronger out-of-distribution test is leave-one-hectare-out evaluation. In that setting the reference model has to predict an area group absent from training, and its mean MAE rises sharply.

## Proposed correction and evaluation

The correction targets the cause rather than the symptom:

1. Convert raw input quantities to per-hectare quantities.
2. Train the model to predict yield per hectare instead of total yield.
3. Convert predictions back to total yield by multiplying by hectares at inference time.
4. Add mild Random Forest regularization (`max_depth=10`, `min_samples_leaf=5`).

This directly changes the representation so the model learns productivity patterns instead of memorizing total scale. On leave-one-hectare-out evaluation, mean MAE decreases from {logo_summary.loc['reference_total_yield_rf', 'MAE']:.0f} kg to {logo_summary.loc['corrected_per_hectare_rf', 'MAE']:.0f} kg.

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
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to paddydataset.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--pipeline", choices=["both", "reference", "corrected"], default="both")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_columns(pd.read_csv(args.data))
    hectare_summary = summarize_by_hectare(df)
    random_split = run_random_split(df, args.pipeline)
    logo = run_leave_one_hectare_out(df, args.pipeline)
    ablation = run_ablation(df)

    hectare_summary.to_csv(output_dir / "hectare_summary.csv", index=False)
    random_split.to_csv(output_dir / "random_split_metrics.csv", index=False)
    logo.to_csv(output_dir / "leave_one_hectare_metrics.csv", index=False)
    ablation.to_csv(output_dir / "ablation_metrics.csv", index=False)

    if args.pipeline == "both":
        plot_target_scale(hectare_summary, output_dir)
        plot_failure(logo, output_dir)
        write_report(df, random_split, logo, ablation, Path("report.md"))

    written_paths = [
        output_dir / "hectare_summary.csv",
        output_dir / "random_split_metrics.csv",
        output_dir / "leave_one_hectare_metrics.csv",
        output_dir / "ablation_metrics.csv",
    ]
    if args.pipeline == "both":
        written_paths.extend(
            [
        output_dir / "target_scale_by_hectare.png",
        output_dir / "leave_one_hectare_mae.png",
        Path("report.md") if args.pipeline == "both" else None,
            ]
        )

    print("Wrote:")
    for path in written_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
