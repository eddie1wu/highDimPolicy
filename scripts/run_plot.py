from pathlib import Path

from hdpolicy.io.save_load import load_results
from hdpolicy.plots.plot_omitted_var import plot_welfare_omitted_var
from hdpolicy.plots.common import save_fig

def main():

    res_path = Path("results/2026-02-25_234436_omitted_var")
    out_path = Path("graphs/welfare_omitted_var.png")

    res = load_results(res_path)
    fig, ax = plot_welfare_omitted_var(res, 75)

    save_fig(fig, out_path)

    print(f"Graph saved to {out_path}.")


if __name__ == "__main__":
    main()


