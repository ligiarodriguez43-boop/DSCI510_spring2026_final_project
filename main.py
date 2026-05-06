"""
Run the full project pipeline: all four data sources end-to-end.

Usage:
    python main.py                   # run everything, show plots
    python main.py --no-plots        # skip matplotlib plots
    python main.py --skip-api        # skip API-based analyses
    python main.py --only crc        # run only the local CRC analysis
                                     # options: crc | global | trials | wonder

Note:
    - Local CSV analyses require crc_dataset.csv and
      colorectal_cancer_dataset.csv in ../data/.
    - The clinical trials analysis requires NCI_API_KEY in .env or env vars.
    - The CDC WONDER analysis requires internet access.
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root: `python main.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()  # populate os.environ from .env if present

# Run a pipeline stage, log failures, but never crash the whole run
def _safe_run(name, fn, **kwargs):
    print(f"\n{'=' * 70}")
    print(f"Running: {name}")
    print(f"{'=' * 70}")
    try:
        return fn(**kwargs)
    except Exception as e:
        print(f"\n[!] {name} failed: {type(e).__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-plots", action="store_true",
                        help="Suppress matplotlib plots")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API-based analyses (trials, wonder)")
    parser.add_argument("--only",
                        choices=["crc", "global", "trials", "wonder"],
                        help="Run only one specific analysis")
    args = parser.parse_args()

    show_plots = not args.no_plots

    import crc_analysis
    import global_rankings
    import crc_clinical_trials
    import cdc_wonder

    stages = []
    if args.only == "crc" or args.only is None:
        stages.append(("CRC Analysis (CSV)",
                       crc_analysis.run_full_pipeline))
    if args.only == "global" or args.only is None:
        stages.append(("Global Rankings (CSV)",
                       global_rankings.run_full_pipeline))
    if not args.skip_api and (args.only == "trials" or args.only is None):
        stages.append(("NCI Clinical Trials (API)",
                       crc_clinical_trials.run_full_pipeline))
    if not args.skip_api and (args.only == "wonder" or args.only is None):
        stages.append(("CDC WONDER (API)",
                       cdc_wonder.run_full_pipeline))

    results = {}
    for name, fn in stages:
        results[name] = _safe_run(name, fn, show_plots=show_plots)

    print(f"\n{'=' * 70}")
    print("Pipeline complete")
    print(f"{'=' * 70}")
    for name, result in results.items():
        status = "OK" if result is not None else "FAILED"
        print(f"  {name:<35}  [{status}]")


if __name__ == "__main__":
    main()
