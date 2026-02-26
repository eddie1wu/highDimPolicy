from __future__ import annotations

from hdpolicy.plots.common import new_fig, style_axes

def plot_welfare_omitted_var(
        res: dict,
        interpolating_threshold: int,   # 75
        title: str = "Double Ascent in Welfare Maximization (Omitted Variables)"
):
    """
    Expects res to contain:
        - "train_welfare"
        - "test_welfare"
        - "sb_oracle_welfare"
        - "oracle_welfare"
        - "p_grid"
    """

    # Load results
    train_welfare = res["train_welfare"]
    test_welfare = res["test_welfare"]
    sb_oracle_welfare = res["sb_oracle_welfare"]
    oracle = res["oracle_welfare"]
    p_grid = res["p_grid"]

    # Make plot
    fig, ax = new_fig()

    ax.plot(p_grid, train_welfare, label = "Training welfare", color = "Orange")
    ax.plot(p_grid, test_welfare, label = "Test welfare", color = "C0")
    ax.plot(p_grid, sb_oracle_welfare, label = "Second best oracle", color = "C0", ls = "--")

    ax.axhline(oracle, ls = "--", label = "Oracle", color = "r")
    ax.axvline(interpolating_threshold, linestyle="--", lw = 0.7, label="Interpolating threshold", color = "black")

    ax.set_yscale("log")
    ax.set_ylabel("Welfare (log)")
    ax.set_title(title)
    style_axes(ax)

    return fig, ax
