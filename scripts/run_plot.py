from datetime import datetime
from pathlib import Path

from hdpolicy.io.save_load import load_results
# from hdpolicy.plots.plot_simple_linear import plot_welfare_simple_linear
from hdpolicy.plots.plot_misspecified import plot_welfare_misspecified
from hdpolicy.plots.common import save_fig

def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    res_path = Path("results/2026-03-01_231942_misspecified")
    out_path = Path(f"graphs/welfare_misspecified_{timestamp}.png")

    res = load_results(res_path)
    fig, ax = plot_welfare_misspecified(res, 75)

    save_fig(fig, out_path)

    print(f"Graph saved to {out_path}.")


if __name__ == "__main__":
    main()


