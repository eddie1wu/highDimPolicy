from pathlib import Path

from hdpolicy.config import Config
# from hdpolicy.experiments.simple_linear import run_simple_linear
# from hdpolicy.experiments.random_ReLU import run_random_ReLU
# from hdpolicy.experiments.omitted_var import run_omitted_var
from hdpolicy.experiments.misspecified import run_misspecified
# from hdpolicy.experiments.misspecified_friedman import run_misspecified
# from hdpolicy.experiments.misspecified_highfreq import run_misspecified
from hdpolicy.io.save_load import make_run_dir, save_results, save_config


def main():

    cfg = Config(n_rep = 6)

    # results = run_simple_linear(cfg, max_workers = 6)
    results = run_misspecified(cfg, max_workers = 6)

    run_dir = make_run_dir(Path("results"), tag = "misspecified")

    save_results(run_dir, results)
    save_config(run_dir, cfg)

    print(f"Saved to {run_dir}")

if __name__ == '__main__':
    main()
