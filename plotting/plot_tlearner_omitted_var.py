from __future__ import annotations

from datetime import datetime
from pathlib import Path

from hdpolicy.io.save_load import load_results
from hdpolicy.plots.common import save_fig
from hdpolicy.plots.common import new_fig, style_axes


def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    res_path = Path("results/2026-03-31_170557_tlearner_omitted_var")
    res = load_results(res_path)

    fig1, fig2 = plot_graphs(res)

    out_path = Path(f"graphs/tl_ov_welfare_{timestamp}.png")
    save_fig(fig1, out_path)

    out_path = Path(f"graphs/tl_ov_risk_{timestamp}.png")
    save_fig(fig2, out_path)


    print(f"Graphs saved to {out_path}.")



def plot_graphs(
        res: dict
):
    # Load results
    tl_train_welfare = res["tl_train_welfare"]
    tl_test_welfare = res["tl_test_welfare"]
    clf_train_welfare = res["clf_train_welfare"]
    clf_test_welfare = res["clf_test_welfare"]
    sb_oracle_welfare = res["sb_oracle_welfare"]
    oracle_train_welfare = res["oracle_train_welfare"]
    oracle_test_welfare = res["oracle_test_welfare"]
    clf_train_loss = res["clf_train_loss"]
    clf_train_risk = res["clf_train_risk"]
    clf_test_risk = res["clf_test_risk"]
    p_grid = res["p_grid"]


    ### Training test welfare
    title = "Welfare Performance of T-learner vs Weighted Classification (Omitted Variable Model)"
    fig1, ax = new_fig()
    ax.plot(p_grid, tl_train_welfare, label = "TL Training welfare", ls = "-.", color = "Orange")
    ax.plot(p_grid, tl_test_welfare, label = "TL Test welfare", color = "Orange")

    ax.plot(p_grid, clf_train_welfare, label="Clf Training welfare", ls = "-.", color="C0")
    ax.plot(p_grid, clf_test_welfare, label="Clf Test welfare", color="C0")

    ax.plot(p_grid, sb_oracle_welfare, label = "Second best oracle", ls = "--", color = "r")
    ax.axhline(oracle_test_welfare, ls = "--", label = "Oracle", color = "r")

    ax.set_yscale("log")
    ax.set_ylabel("Welfare (log)")
    ax.set_title(title)
    style_axes(ax)


    ### Risk plot
    title = "Training Loss and Risk (Omitted Variable Model)"
    fig2, ax = new_fig()
    ax.plot(p_grid, clf_train_risk, label="Training risk (left axis)", ls = "-.", color="C0")
    ax.plot(p_grid, clf_test_risk, label="Test risk (left axis)", color="C0")
    ax.set_ylabel("Classification risk")

    ax2 = ax.twinx()
    ax2.plot(p_grid, clf_train_loss, label="Training loss (right axis)", ls = "--", color="Black")
    ax2.set_ylabel("Logistic loss")

    ax.set_title(title)

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2)
    ax.set_xlabel("Number of features (p)")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)


    return fig1, fig2


if __name__ == "__main__":
    main()





