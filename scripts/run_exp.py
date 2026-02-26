from pathlib import Path

from hdpolicy.config import Config
# from hdpolicy.experiments.random_ReLU import run_random_ReLU
from hdpolicy.experiments.omitted_var import run_omitted_var
# from hdpolicy.experiments.misspecified import run_misspecified
# from hdpolicy.experiments.misspecified_friedman import run_misspecified
# from hdpolicy.experiments.misspecified_highfreq import run_misspecified
from hdpolicy.io.save_load import make_run_dir, save_results


def main():

    cfg = Config(n_rep = 100)

    results = run_omitted_var(cfg, max_workers = 6)

    run_dir = make_run_dir(Path("results"), tag = "omitted_var")
    save_results(run_dir, results)
    print(f"Saved to {run_dir}")

if __name__ == '__main__':
    main()

