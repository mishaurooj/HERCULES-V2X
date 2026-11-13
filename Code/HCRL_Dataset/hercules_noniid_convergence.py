
#!/usr/bin/env python3
"""
HERCULES-V2X: matched Non-IID convergence experiment.

Uses the same:
  - HCRL data loader
  - 12-feature preprocessing
  - train/validation/test split
  - PyTorch neural network
  - training-only class balancing
  - Dirichlet client partitioning
  - equal/FedAvg/FedProx implementations

as hercules_reviewer_experiments(2).py.

Outputs:
  convergence_roundwise.csv
  convergence_per_seed.csv
  convergence_summary.csv
  convergence_table_blue.tex

IMPORTANT:
This experiment measures actual performance.
It does not guarantee high F1 or faster convergence.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


CLIENT_COUNTS = [2, 4, 8, 16, 32]
ALPHAS = [0.1, 0.5, 1.0, 10.0]
SEEDS = [11, 22, 33, 44, 55]
METHODS = ["equal", "fedavg", "fedprox"]


def load_reference_script(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference experiment script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "hercules_reference", str(path)
    )
    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "load_data",
        "temporal_split",
        "noniid_clients",
        "fed_training",
        "seed_all",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Reference script is missing functions: "
            + ", ".join(missing)
        )

    return module


def prepare_data(ref, args):
    print("Loading HCRL data with reference loader...")

    X, y, timestamps, sources, scenarios, audit = (
        ref.load_data(
            Path(args.data_root),
            args.max_rows_per_file,
            SEEDS[0],
        )
    )

    parts = ref.temporal_split(
        timestamps,
        sources,
        y=y,
    )

    train_mask = parts == 0
    val_mask = parts == 1
    test_mask = parts == 2

    for name, mask in [
        ("train", train_mask),
        ("validation", val_mask),
        ("test", test_mask),
    ]:
        if np.unique(y[mask]).size != 2:
            raise RuntimeError(
                f"{name} split does not contain both classes."
            )

    scaler = StandardScaler()
    scaler.fit(X[train_mask])

    X_train = scaler.transform(
        X[train_mask]
    ).astype(np.float32)

    X_val = scaler.transform(
        X[val_mask]
    ).astype(np.float32)

    y_train = y[train_mask].astype(np.int8)
    y_val = y[val_mask].astype(np.int8)

    print(
        "Split strategy:",
        ref.temporal_split.last_strategy,
    )
    print("Features:", X_train.shape[1])
    print("Train:", len(y_train))
    print("Validation:", len(y_val))
    print(
        "Train attack prevalence:",
        round(float(y_train.mean()), 4),
    )
    print(
        "Validation attack prevalence:",
        round(float(y_val.mean()), 4),
    )

    return X_train, y_train, X_val, y_val


def run_one(
    ref, X_train, y_train, X_val, y_val,
    clients, k, alpha, method, seed, args
):
    # fed_training is the SAME implementation used by
    # hercules_reviewer_experiments(2).py.
    #
    # It uses the reference neural network, Adam optimizer,
    # training-only balancing, and weighted aggregation.

    model, trace, elapsed, total_bytes = (
        ref.fed_training(
            X_train,
            y_train,
            clients,
            method,
            seed,
            args.rounds,
            args.local_epochs,
            args.batch,
            args.lr,
            evalx=X_val,
            evaly=y_val,
        )
    )

    if len(trace) != args.rounds:
        raise RuntimeError(
            f"Expected {args.rounds} rounds, "
            f"received {len(trace)}."
        )

    rows = []

    for item in trace:
        rows.append({
            "k": k,
            "alpha": alpha,
            "method": method,
            "seed": seed,
            "round": int(item["round"]),
            "f1": float(item["f1"]),
            "accuracy": float(item["accuracy"]),
            "total_training_sec": elapsed,
            "total_communication_bytes": total_bytes,
        })

    return rows


def summarize_run(group, target):
    group = group.sort_values("round")

    f1 = group["f1"].to_numpy(dtype=float)
    rounds = group["round"].to_numpy(dtype=int)

    best = float(f1.max())
    final = float(f1[-1])

    own_hits = np.flatnonzero(
        f1 >= 0.95 * best
    )

    fixed_hits = np.flatnonzero(
        f1 >= target
    )

    r95 = (
        float(rounds[own_hits[0]])
        if len(own_hits)
        else np.nan
    )

    rt = (
        float(rounds[fixed_hits[0]])
        if len(fixed_hits)
        else np.nan
    )

    return {
        "k": int(group["k"].iloc[0]),
        "alpha": float(group["alpha"].iloc[0]),
        "method": group["method"].iloc[0],
        "seed": int(group["seed"].iloc[0]),
        "final_f1": final,
        "best_f1": best,
        "r95": r95,
        "rounds_to_target": rt,
        "auc_roundwise": float(f1.mean()),
        "time_sec": float(
            group["total_training_sec"].iloc[0]
        ),
        "communication_bytes": float(
            group["total_communication_bytes"].iloc[0]
        ),
        "target_reached": int(len(fixed_hits) > 0),
    }


def add_paired_metrics(df):
    result = df.copy()

    base = (
        result[result["method"] == "equal"]
        .set_index(["k", "alpha", "seed"])
    )

    gains = []

    for _, row in result.iterrows():
        key = (
            row["k"],
            row["alpha"],
            row["seed"],
        )

        if key not in base.index:
            gains.append({
                "delta_auc": np.nan,
                "delta_final_f1": np.nan,
                "round_reduction_pct": np.nan,
            })
            continue

        b = base.loc[key]

        delta_auc = (
            row["auc_roundwise"]
            - b["auc_roundwise"]
        )

        delta_f1 = (
            row["final_f1"]
            - b["final_f1"]
        )

        if (
            pd.notna(row["rounds_to_target"])
            and pd.notna(b["rounds_to_target"])
            and b["rounds_to_target"] > 0
        ):
            reduction = 100.0 * (
                b["rounds_to_target"]
                - row["rounds_to_target"]
            ) / b["rounds_to_target"]
        else:
            reduction = np.nan

        gains.append({
            "delta_auc": delta_auc,
            "delta_final_f1": delta_f1,
            "round_reduction_pct": reduction,
        })

    return pd.concat(
        [
            result.reset_index(drop=True),
            pd.DataFrame(gains),
        ],
        axis=1,
    )


def latex_table(summary, path, target):
    """
    Generate a full-width blue table:
    3 subtables in row 1, 2 centered in row 2.
    """

    def fmt(value, digits=3):
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"

    def panel(k, label):
        subset = (
            summary[summary["k"] == k]
            .sort_values(["alpha", "method"])
        )

        method_names = {
            "equal": "Equal",
            "fedavg": "FedAvg",
            "fedprox": "FedProx",
        }

        lines = [
            r"\begin{minipage}[t]{0.32\textwidth}",
            r"\centering",
            rf"\textbf{{({label}) {k} Clients}}\\[3pt]",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"$\alpha$ & Method & $R_{95}$ & "
            r"AUC$_R$ & Time (s)\\",
            r"\midrule",
        ]

        for _, row in subset.iterrows():
            values = [
                fmt(row["alpha"], 1),
                method_names[row["method"]],
                fmt(row["r95"], 1),
                fmt(row["auc_roundwise"], 4),
                fmt(row["time_sec"], 2),
            ]

            lines.append(
                " & ".join(values) + r"\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{minipage}",
        ]

        return "\n".join(lines)

    lines = [
        r"% Required: xcolor, graphicx, booktabs",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begingroup",
        r"\color{blue}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{\textcolor{blue}{"
        r"Non-IID federated convergence analysis "
        r"using the matched neural-network "
        r"experimental configuration.}}",
        r"\label{tab:noniid_convergence}",
        panel(2, "a"),
        r"\hfill",
        panel(4, "b"),
        r"\hfill",
        panel(8, "c"),
        r"\par\vspace{0.35cm}",
        r"\makebox[\textwidth][c]{%",
        panel(16, "d"),
        r"\hspace{0.03\textwidth}",
        panel(32, "e"),
        r"}",
        r"\par\vspace{0.15cm}",
        r"\begin{minipage}{\textwidth}",
        r"\raggedright\footnotesize",
        r"\textcolor{blue}{%",
        r"\textit{Note:} Results are averaged "
        r"over matched random seeds.",
        r"$R_{95}$ is the first round reaching "
        r"95\% of each run's best validation F1.",
        r"AUC$_R$ is the mean validation F1 "
        r"across communication rounds.",
        r"Time denotes total measured training "
        r"and validation execution time.",
        r"Because $R_{95}$ uses a run-specific "
        r"target, smaller values do not "
        r"necessarily imply superior convergence.",
        r"}",
        r"\end{minipage}",
        r"\endgroup",
        r"\end{table*}",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        "--dataset",
        dest="data_root",
        default=r"E:\other\hercules\dataset",
    )

    parser.add_argument("--reference",default=r"E:\other\hercules\hercules_reviewer_experiments.py",)
    parser.add_argument(
        "--output",
        default=r"E:\other\hercules\results\tables",
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=300000,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--balance-mode",
        choices=["batch", "loss", "none"],
        default="batch",
    )

    args = parser.parse_args()

    if not 1 <= args.seeds <= len(SEEDS):
        parser.error("--seeds must be between 1 and 5")

    if not 0 < args.target_f1 <= 1:
        parser.error("--target-f1 must be in (0, 1]")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ref = load_reference_script(
        Path(args.reference)
    )

    ref.ACTIVE_DEVICE = (
        "cuda"
        if args.device == "auto"
        and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    if (
        ref.ACTIVE_DEVICE == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA selected but unavailable."
        )

    ref.ACTIVE_BALANCE = args.balance_mode

    print("=" * 65)
    print("MATCHED HERCULES NON-IID CONVERGENCE")
    print("=" * 65)
    print("Device:", ref.ACTIVE_DEVICE)
    print("Balance:", ref.ACTIVE_BALANCE)
    print("Rounds:", args.rounds)
    print("Seeds:", args.seeds)

    X_train, y_train, X_val, y_val = (
        prepare_data(ref, args)
    )

    logs = []
    skipped = []

    total = (
        len(CLIENT_COUNTS)
        * len(ALPHAS)
        * args.seeds
        * len(METHODS)
    )

    completed = 0

    for k in CLIENT_COUNTS:
        for alpha in ALPHAS:
            for seed in SEEDS[:args.seeds]:

                # Same reference client partitioning.
                clients = ref.noniid_clients(
                    y_train,
                    k,
                    alpha,
                    seed,
                    min_size=32,
                )

                if clients is None:
                    message = (
                        f"K={k}, alpha={alpha}, "
                        f"seed={seed}: client too small"
                    )
                    skipped.append(message)
                    print("SKIP:", message)
                    continue

                for method in METHODS:
                    completed += 1

                    print(
                        f"[{completed}/{total}] "
                        f"K={k}, alpha={alpha}, "
                        f"seed={seed}, method={method}",
                        flush=True,
                    )

                    rows = run_one(
                        ref,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        clients,
                        k,
                        alpha,
                        method,
                        seed,
                        args,
                    )

                    logs.extend(rows)

                    print(
                        "Final F1:",
                        f"{rows[-1]['f1']:.4f}",
                        flush=True,
                    )

    if not logs:
        raise RuntimeError(
            "No valid configurations completed. "
            "Inspect client sizes and dataset coverage."
        )

    roundwise = pd.DataFrame(logs)

    roundwise.to_csv(
        output / "convergence_roundwise.csv",
        index=False,
    )

    per_seed = pd.DataFrame([
        summarize_run(group, args.target_f1)
        for _, group in roundwise.groupby(
            ["k", "alpha", "method", "seed"]
        )
    ])

    per_seed = add_paired_metrics(per_seed)

    per_seed.to_csv(
        output / "convergence_per_seed.csv",
        index=False,
    )

    numeric = [
        "final_f1",
        "best_f1",
        "r95",
        "rounds_to_target",
        "auc_roundwise",
        "time_sec",
        "communication_bytes",
        "target_reached",
        "delta_auc",
        "delta_final_f1",
        "round_reduction_pct",
    ]

    summary = (
        per_seed.groupby(
            ["k", "alpha", "method"],
            as_index=False,
        )[numeric]
        .mean()
    )

    counts = (
        per_seed.groupby(
            ["k", "alpha", "method"]
        )
        .size()
        .reset_index(name="seeds")
    )

    summary = summary.merge(
        counts,
        on=["k", "alpha", "method"],
    )

    summary.to_csv(
        output / "convergence_summary.csv",
        index=False,
    )

    latex_table(
        summary,
        output / "convergence_table_blue.tex",
        args.target_f1,
    )

    (
        output / "convergence_skipped.txt"
    ).write_text(
        "\n".join(skipped),
        encoding="utf-8",
    )

    print("\nEXPERIMENT COMPLETED")
    print("Completed runs:", completed)
    print("Skipped partitions:", len(skipped))
    print("Results:", output)
    print("LaTeX:", output / "convergence_table_blue.tex")


if __name__ == "__main__":
    main()
