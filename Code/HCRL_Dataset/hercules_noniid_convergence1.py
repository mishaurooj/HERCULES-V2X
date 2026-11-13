
#!/usr/bin/env python3
"""
HERCULES-V2X MATCHED NON-IID CONVERGENCE EXPERIMENT

Reuses hercules_reviewer_experiments.py:
    load_data
    temporal_split
    noniid_clients
    fed_training

Fast mode:
    8, 16, 32 clients
    alpha = 0.5, 1.0, 10.0
    2 seeds, 10 rounds
    20,000 rows per recording

Full mode:
    2, 4, 8, 16, 32 clients
    alpha = 0.1, 0.5, 1.0, 10.0
    5 seeds, 20 rounds
    300,000 rows per recording

Results are measured, never fabricated.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent

METHODS = ["equal", "fedavg", "fedprox"]
ALL_CLIENTS = [2, 4, 8, 16, 32]
ALL_ALPHAS = [0.1, 0.5, 1.0, 10.0]
ALL_SEEDS = [11, 22, 33, 44, 55]


def load_reference(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file missing: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "hercules_reference", str(path)
    )
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)

    required = [
        "load_data",
        "temporal_split",
        "noniid_clients",
        "fed_training",
    ]

    missing = [
        name for name in required
        if not hasattr(ref, name)
    ]

    if missing:
        raise RuntimeError(
            "Missing reference functions: "
            + ", ".join(missing)
        )

    return ref


def prepare_data(ref, args):
    print("\nLoading CAN recordings...", flush=True)

    X, y, timestamps, sources, scenarios, audit = (
        ref.load_data(
            Path(args.dataset),
            args.max_rows_per_file,
            11,
        )
    )

    parts = ref.temporal_split(
        timestamps,
        sources,
        y=y,
    )

    tr = parts == 0
    va = parts == 1

    if (
        len(np.unique(y[tr])) < 2
        or len(np.unique(y[va])) < 2
    ):
        raise RuntimeError(
            "Training and validation must contain "
            "both normal and attack classes."
        )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(
        X[tr]
    ).astype(np.float32)

    Xva = scaler.transform(
        X[va]
    ).astype(np.float32)

    ytr = y[tr].astype(np.int8)
    yva = y[va].astype(np.int8)

    # Deterministic stratified validation subsample
    # for fast mode. Never sample from the test set.
    if (
        args.validation_limit > 0
        and len(yva) > args.validation_limit
    ):
        rng = np.random.default_rng(2026)

        selected = []
        for label in [0, 1]:
            ids = np.flatnonzero(yva == label)
            quota = max(
                1,
                round(
                    args.validation_limit
                    * len(ids) / len(yva)
                ),
            )
            selected.extend(
                rng.choice(
                    ids,
                    min(quota, len(ids)),
                    replace=False,
                ).tolist()
            )

        selected = np.asarray(
            sorted(set(selected)),
            dtype=int,
        )

        Xva = Xva[selected]
        yva = yva[selected]

    print("Training rows:", len(ytr))
    print("Validation rows:", len(yva))
    print("Features:", Xtr.shape[1])
    print(
        "Split:",
        ref.temporal_split.last_strategy,
    )
    print(
        "Training attack fraction:",
        round(float(ytr.mean()), 4),
    )
    print(
        "Validation attack fraction:",
        round(float(yva.mean()), 4),
    )

    return Xtr, ytr, Xva, yva


def run_experiment(
    ref, Xtr, ytr, Xva, yva,
    clients, k, alpha, method, seed, args
):
    # Uses the exact reference federated trainer.
    # Equal, FedAvg and FedProx are evaluated on
    # identical client partitions and validation data.

    model, trace, elapsed, communication = (
        ref.fed_training(
            Xtr,
            ytr,
            clients,
            method,
            seed,
            args.rounds,
            args.local_epochs,
            args.batch,
            args.lr,
            evalx=Xva,
            evaly=yva,
        )
    )

    if len(trace) != args.rounds:
        raise RuntimeError(
            "Round-wise trace is incomplete."
        )

    return [
        {
            "k": k,
            "alpha": alpha,
            "method": method,
            "seed": seed,
            "round": int(item["round"]),
            "f1": float(item["f1"]),
            "accuracy": float(item["accuracy"]),
            "total_time_sec": float(elapsed),
            "total_communication_bytes":
                int(communication),
        }
        for item in trace
    ]


def calculate_metrics(group, target):
    group = group.sort_values("round")

    f1 = group["f1"].to_numpy(dtype=float)
    rounds = group["round"].to_numpy(dtype=int)

    best = float(np.max(f1))

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
        "method": str(group["method"].iloc[0]),
        "seed": int(group["seed"].iloc[0]),
        "final_f1": float(f1[-1]),
        "best_f1": best,
        "r95": r95,
        "rounds_to_target": rt,
        "auc_roundwise": float(f1.mean()),
        "time_sec": float(
            group["total_time_sec"].iloc[0]
        ),
        "communication_mb": float(
            group["total_communication_bytes"].iloc[0]
        ) / 1e6,
        "target_reached": int(
            len(fixed_hits) > 0
        ),
    }


def summarize(roundwise, target):
    rows = [
        calculate_metrics(group, target)
        for _, group in roundwise.groupby(
            ["k", "alpha", "method", "seed"]
        )
    ]

    per_seed = pd.DataFrame(rows)

    base = (
        per_seed[per_seed.method == "equal"]
        .set_index(["k", "alpha", "seed"])
    )

    improvements = []

    for _, row in per_seed.iterrows():
        key = (
            row["k"],
            row["alpha"],
            row["seed"],
        )

        if key not in base.index:
            improvements.append(
                [np.nan, np.nan, np.nan]
            )
            continue

        b = base.loc[key]

        delta_f1_pp = 100 * (
            row["final_f1"] - b["final_f1"]
        )

        delta_auc = (
            row["auc_roundwise"]
            - b["auc_roundwise"]
        )

        if (
            pd.notna(row["rounds_to_target"])
            and pd.notna(b["rounds_to_target"])
            and b["rounds_to_target"] > 0
        ):
            round_reduction = 100 * (
                b["rounds_to_target"]
                - row["rounds_to_target"]
            ) / b["rounds_to_target"]
        else:
            round_reduction = np.nan

        improvements.append([
            delta_f1_pp,
            delta_auc,
            round_reduction,
        ])

    per_seed[
        [
            "delta_f1_pp",
            "delta_auc",
            "round_reduction_pct",
        ]
    ] = improvements

    numeric = [
        "final_f1",
        "best_f1",
        "r95",
        "rounds_to_target",
        "auc_roundwise",
        "time_sec",
        "communication_mb",
        "target_reached",
        "delta_f1_pp",
        "delta_auc",
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

    return per_seed, summary


def create_latex(summary, output, args):
    """
    Generate a blue full-width MDPI table with
    three panels in row one and two centered below.
    """

    def fmt(x, n=2):
        if pd.isna(x):
            return "--"
        return f"{float(x):.{n}f}"

    def panel(k, letter):
        sub = (
            summary[summary.k == k]
            .sort_values(["alpha", "method"])
        )

        lines = [
            r"\begin{minipage}[t]{0.32\textwidth}",
            r"\centering",
            rf"\textbf{{({letter}) {k} Clients}}\\[3pt]",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"$\alpha$ & Method & $R_{95}$ & "
            r"AUC$_R$ & Time (s)\\",
            r"\midrule",
        ]

        names = {
            "equal": "Equal",
            "fedavg": "FedAvg",
            "fedprox": "FedProx",
        }

        for alpha in ALL_ALPHAS:
            for method in METHODS:
                found = sub[
                    (np.isclose(sub.alpha, alpha))
                    & (sub.method == method)
                ]

                if found.empty:
                    values = [
                        fmt(alpha, 1),
                        names[method],
                        "--",
                        "--",
                        "--",
                    ]
                else:
                    r = found.iloc[0]
                    values = [
                        fmt(alpha, 1),
                        names[method],
                        fmt(r["r95"], 1),
                        fmt(r["auc_roundwise"], 4),
                        fmt(r["time_sec"], 2),
                    ]

                lines.append(
                    " & ".join(values) + r" \\"
                )

        lines += [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{minipage}",
        ]

        return "\n".join(lines)

    lines = [
        r"% Packages: xcolor, graphicx, booktabs",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begingroup",
        r"\color{blue}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{\textcolor{blue}{"
        r"Non-IID federated convergence analysis "
        r"using the matched neural-network detector.}}",
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
        rf"\textit{{Note:}} {args.rounds} communication "
        r"rounds were evaluated.",
        r"$R_{95}$ denotes the first round reaching "
        r"95\% of each run's best validation F1.",
        r"AUC$_R$ is the mean validation F1 "
        r"across communication rounds.",
        r"Time denotes measured execution time "
        r"in seconds.",
        r"Values are averaged across completed seeds.",
        r"Missing entries indicate configurations "
        r"not evaluated or without valid partitions.",
        r"A lower $R_{95}$ does not necessarily "
        r"indicate faster convergence to a common "
        r"performance level.",
        r"}",
        r"\end{minipage}",
        r"\endgroup",
        r"\end{table*}",
    ]

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        default=str(
            HERE / "hercules_reviewer_experiments.py"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(HERE / "dataset"),
    )
    parser.add_argument(
        "--output",
        default=str(
            HERE / "results" / "tables" / "convergence"
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all configurations and five seeds.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
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
        "--target-f1",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--balance-mode",
        choices=["batch", "loss", "none"],
        default="batch",
    )

    args = parser.parse_args()

    if args.full:
        clients_list = ALL_CLIENTS
        alpha_list = ALL_ALPHAS
        default_rounds = 20
        default_seeds = 5
        default_rows = 300000
        default_validation = 0
    else:
        clients_list = [8, 16, 32]
        alpha_list = [0.5, 1.0, 10.0]
        default_rounds = 10
        default_seeds = 2
        default_rows = 20000
        default_validation = 5000

    args.rounds = (
        args.rounds
        if args.rounds is not None
        else default_rounds
    )
    args.seeds = (
        args.seeds
        if args.seeds is not None
        else default_seeds
    )
    args.max_rows_per_file = (
        args.max_rows_per_file
        if args.max_rows_per_file is not None
        else default_rows
    )
    args.validation_limit = (
        args.validation_limit
        if args.validation_limit is not None
        else default_validation
    )

    if not 1 <= args.seeds <= 5:
        parser.error("--seeds must be 1 through 5")
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    torch.set_num_threads(
        max(1, args.threads)
    )

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but not available."
        )

    ref = load_reference(
        Path(args.reference)
    )

    ref.ACTIVE_DEVICE = device
    ref.ACTIVE_BALANCE = args.balance_mode

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("MATCHED NON-IID CONVERGENCE")
    print("=" * 65)
    print("Mode:", "FULL" if args.full else "FAST")
    print("Device:", device)
    print("Rounds:", args.rounds)
    print("Seeds:", args.seeds)
    print("Data rows/file:", args.max_rows_per_file)
    print("Output:", out)

    Xtr, ytr, Xva, yva = prepare_data(
        ref, args
    )

    # Use a run-specific directory to avoid mixing
    # fast and full results or different settings.
    config = {
        "full": args.full,
        "rounds": args.rounds,
        "seeds": args.seeds,
        "max_rows_per_file":
            args.max_rows_per_file,
        "validation_limit":
            args.validation_limit,
        "lr": args.lr,
        "batch": args.batch,
        "local_epochs": args.local_epochs,
        "balance_mode": args.balance_mode,
        "device": device,
        "clients": clients_list,
        "alphas": alpha_list,
        "target_f1": args.target_f1,
    }

    (out / "configuration.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    total = (
        len(clients_list)
        * len(alpha_list)
        * args.seeds
        * len(METHODS)
    )

    counter = 0
    skipped = []
    all_rows = []

    for k in clients_list:
        for alpha in alpha_list:
            for seed in ALL_SEEDS[:args.seeds]:

                clients = ref.noniid_clients(
                    ytr,
                    k,
                    alpha,
                    seed,
                    min_size=32,
                )

                if clients is None:
                    skipped.append(
                        f"K={k}, alpha={alpha}, "
                        f"seed={seed}: insufficient "
                        f"client samples"
                    )
                    continue

                for method in METHODS:
                    counter += 1

                    print(
                        f"\n[{counter}/{total}] "
                        f"K={k} alpha={alpha} "
                        f"seed={seed} {method}",
                        flush=True,
                    )

                    rows = run_experiment(
                        ref,
                        Xtr,
                        ytr,
                        Xva,
                        yva,
                        clients,
                        k,
                        alpha,
                        method,
                        seed,
                        args,
                    )

                    all_rows.extend(rows)

                    # Save each completed experiment.
                    checkpoint = (
                        checkpoint_dir /
                        f"K{k}_a{alpha}_s{seed}_"
                        f"{method}.csv"
                    )
                    pd.DataFrame(rows).to_csv(
                        checkpoint,
                        index=False,
                    )

                    print(
                        "Final F1:",
                        f"{rows[-1]['f1']:.4f}",
                        "Time:",
                        f"{rows[-1]['total_time_sec']:.2f}s",
                        flush=True,
                    )

    if not all_rows:
        raise RuntimeError(
            "No valid configurations completed."
        )

    roundwise = pd.DataFrame(all_rows)

    roundwise.to_csv(
        out / "convergence_roundwise.csv",
        index=False,
    )

    per_seed, summary = summarize(
        roundwise,
        args.target_f1,
    )

    per_seed.to_csv(
        out / "convergence_per_seed.csv",
        index=False,
    )

    summary.to_csv(
        out / "convergence_summary.csv",
        index=False,
    )

    create_latex(
        summary,
        out / "convergence_table_blue.tex",
        args,
    )

    (
        out / "convergence_skipped.txt"
    ).write_text(
        "\n".join(skipped),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print("Runs completed:", counter)
    print("Skipped partitions:", len(skipped))
    print("Results:", out)
    print("LaTeX:", out / "convergence_table_blue.tex")


if __name__ == "__main__":
    main()
