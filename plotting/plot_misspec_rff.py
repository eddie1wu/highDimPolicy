from __future__ import annotations

from datetime import datetime
from pathlib import Path

from hdpolicy.io.save_load import load_results
from hdpolicy.plots.common import save_fig
from hdpolicy.plots.common import new_fig, style_axes


def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    res_path = Path("results/2026-03-31_134353_misspec_rff")
    res = load_results(res_path)

    fig1, fig2, fig3 = plot_graphs(res)

    out_path = Path(f"graphs/welfare_misspec_rff_{timestamp}.png")
    save_fig(fig1, out_path)

    out_path = Path(f"graphs/risk_misspec_rff_{timestamp}.png")
    save_fig(fig2, out_path)

    out_path = Path(f"graphs/norm_misspec_rff_{timestamp}.png")
    save_fig(fig3, out_path)

    print(f"Graphs saved to {out_path}.")



def plot_graphs(
        res: dict
):
    # Load results
    train_welfare = res["train_welfare"]
    test_welfare = res["test_welfare"]
    misspec_X_welfare = res["misspec_X_welfare"]
    correct_X_welfare = res["correct_X_welfare"]
    oracle_train_welfare = res["oracle_train_welfare"]
    oracle_test_welfare = res["oracle_test_welfare"]
    train_loss = res["train_loss"]
    train_risk = res["train_risk"]
    test_risk = res["test_risk"]
    beta_norm = res["beta_norm"]
    svc_test_welfare = res["svc_test_welfare"]
    svc_w_norm = res["svc_w_norm"]
    p_grid = res["p_grid"]
    sample_size = 100


    ### Training test welfare
    title = "Welfare Double Ascent under Model Misspecification (Random Fourier Sieves)"
    fig1, ax = new_fig()
    ax.plot(p_grid, train_welfare, label = "Training welfare", color = "Orange")
    ax.plot(p_grid, test_welfare, label = "Test welfare", color = "C0")

    ax.axhline(oracle_test_welfare, ls = "--", label = "Oracle", color = "r")
    # ax.axhline(oracle_train_welfare, ls="--", label="Oracle", color="yellow")
    ax.axhline(misspec_X_welfare, ls = "-.", label = "Misspecified X's", color = "C0")
    ax.axhline(correct_X_welfare, ls="--", label="Correct X's", color="C0")
    ax.axvline(sample_size, linestyle="--", lw = 0.7, label="Training sample size", color = "black")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Welfare (log)")
    ax.set_title(title)
    style_axes(ax)


    ### Risk plot
    title = "Training loss and risk under Model Misspecification (Random Fourier Sieves)"
    fig2, ax = new_fig()
    ax.plot(p_grid, train_risk, label="Training risk (left axis)", color="C0")
    ax.plot(p_grid, test_risk, label="Test risk (left axis)", color="Orange")
    ax.set_ylabel("Classification risk")

    ax2 = ax.twinx()
    ax2.plot(p_grid, train_loss, label="Training loss (right axis)", ls = "--", color="Black")
    ax2.set_ylabel("Logistic loss")

    ax.set_title(title)

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2)
    ax.set_xlabel("Number of features (p)")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)


    ### Parameter vector norm plot
    title = "Norm of fitted parameter vector (Random Fourier Sieves)"
    fig3, ax = new_fig()
    ax.plot(p_grid, beta_norm, label = "Norm", color="C0")
    ax.axvline(sample_size, linestyle="--", lw = 0.7, label="Training sample size", color = "black")
    ax.set_ylabel("Norm")
    ax.set_title(title)
    style_axes(ax)

    return fig1, fig2, fig3



if __name__ == "__main__":
    main()


