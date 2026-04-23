from __future__ import annotations

from datetime import datetime
from pathlib import Path

from hdpolicy.io.save_load import load_results
from hdpolicy.plots.common import save_fig
from hdpolicy.plots.common import new_fig, style_axes


def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    res_path = Path("results/2026-03-31_221921_isotropic_X")
    res = load_results(res_path)

    fig1, fig2, fig3, fig4 = plot_graphs(res)

    out_path = Path(f"graphs/welfare_isotropic_X_{timestamp}.png")
    save_fig(fig1, out_path)

    out_path = Path(f"graphs/shrinkage_isotropic_X_{timestamp}.png")
    save_fig(fig2, out_path)

    out_path = Path(f"graphs/risk_isotropic_X_{timestamp}.png")
    save_fig(fig3, out_path)

    out_path = Path(f"graphs/norm_isotropic_X_{timestamp}.png")
    save_fig(fig4, out_path)

    print(f"Graphs saved to {out_path}.")



def plot_graphs(
        res: dict
):
    # Load results
    train_welfare = res["train_welfare"]
    test_welfare = res["test_welfare"]
    oracle_train_welfare = res["oracle_train_welfare"]
    oracle_test_welfare = res["oracle_test_welfare"]
    train_loss = res["train_loss"]
    train_risk = res["train_risk"]
    test_risk = res["test_risk"]
    beta_norm = res["beta_norm"]
    svc_test_welfare = res["svc_test_welfare"]
    svc_w_norm = res["svc_w_norm"]
    ridge_test_welfare = res["ridge_test_welfare"]
    lasso_test_welfare = res["lasso_test_welfare"]
    p_grid = res["p_grid"]
    interpolating_threshold = 105


    ### Training test welfare
    title = "Double Ascent in Welfare Maximization (Isotropic X)"
    fig1, ax = new_fig()
    ax.plot(p_grid, train_welfare, label = "Training welfare", color = "Orange")
    ax.plot(p_grid, test_welfare, label = "Test welfare", color = "C0")
    ax.plot(p_grid, oracle_train_welfare, ls="-.", label="Training oracle", color="r")
    ax.plot(p_grid, oracle_test_welfare, ls = "--", label = "Oracle", color = "r")

    ax.axvline(interpolating_threshold, linestyle="--", lw = 0.7, label="Interpolating threshold", color = "black")

    # ax.set_yscale("log")
    ax.set_ylabel("Welfare")
    ax.set_title(title)
    style_axes(ax)


    ### Compare to shrinkage
    title = "Comparing to Shrinkage Estimators (Isotropic X)"
    fig2, ax = new_fig()
    ax.plot(p_grid, test_welfare, label = "No-shrinkage welfare", color = "C0")
    ax.plot(p_grid, ridge_test_welfare, label="Ridge welfare", color="Orange")
    ax.plot(p_grid, lasso_test_welfare, label="LASSO welfare", color="Green")
    ax.plot(p_grid, oracle_test_welfare, ls="--", label="Oracle", color="r")

    # ax.plot(p_grid, svc_test_welfare, ls="--", label="SVC welfare", color="Purple")

    ax.axvline(interpolating_threshold, linestyle="--", lw=0.7, label="Interpolating threshold", color="black")

    # ax.set_yscale("log")
    ax.set_ylabel("Welfare")
    ax.set_title(title)
    style_axes(ax)


    ### Risk plot
    title = "Training Loss and Risk (Isotropic X)"
    fig3, ax = new_fig()
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
    title = "Norm of fitted parameter vector (Isotropic X)"
    fig4, ax = new_fig()
    ax.plot(p_grid, svc_w_norm, label = "Max margin SVM norm", color="C0")
    # ax.axvline(interpolating_threshold, linestyle="--", lw=0.7, label="Interpolating threshold", color="black")
    ax.set_ylabel("Norm")
    ax.set_title(title)
    style_axes(ax)

    return fig1, fig2, fig3, fig4



if __name__ == "__main__":
    main()


