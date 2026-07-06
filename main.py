from taylor_swft import run_BRAS_eval, save_statistics, make_table, detail_computation_times_on_BRAS_CR3
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path("data/BRAS")
    exp_dir = Path("run/test_new_evaluation")
    # run_BRAS_eval(data_dir, exp_dir, recompute_metrics=False)
    # save_statistics(exp_dir)
    make_table(exp_dir)
    detail_computation_times_on_BRAS_CR3()
