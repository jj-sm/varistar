"""
varistar.ml.pipeline
====================
CLI-friendly pipeline that discovers all `.dat` files in a directory,
runs the full TimeSeries → LightCurve → FeatureExtractor chain in parallel,
and saves a timestamped CSV catalogue.

Usage (script)
--------------
    python -m varistar.ml.pipeline --data /path/to/dat/files --limit 5000

Usage (library)
---------------
    from varistar.ml.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(data_path="/data/ogle", limit=1000)
    df = pipeline.run()
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm

    _TQDM = True
except ImportError:
    _TQDM = False


# ---------------------------------------------------------------------------
# Per-star worker  (module-level → picklable on macOS / Windows)
# ---------------------------------------------------------------------------


def _process_star(file_path: Path) -> dict:
    """
    Full single-star pipeline:  load → clean → period → extract features.

    Runs inside a worker process; all imports are local so the function
    is picklable and does not require the parent to import varistar first.

    Returns a feature dict on success, or
    {"id": filename, "error": message} on failure.
    """
    try:
        from varistar.timeseries import TimeSeries
        from varistar.lightcurve import LightCurve
        from varistar.ml.features import FeatureExtractor

        # 1. Load and clean
        ts = TimeSeries(magnitude="mag I", time_scale="HJD")
        ts.load_data_from_file(file_path)

        # IQR-based outlier removal (k=3 is more conservative than the default 1.5)
        outlier_mask = ts.stats_outlier_clipping(k=3)
        ts.apply_mask_to_df(outlier_mask)

        # 2. Period analysis
        #    find_best_period() is called lazily inside FeatureExtractor.extract()
        #    if lc.periods is still empty, so this step is intentionally lightweight.
        lc = LightCurve(ts)

        # 3. Feature extraction (all 18 features)
        return FeatureExtractor.extract(ts_obj=ts, lc_obj=lc)

    except Exception as exc:  # noqa: BLE001
        return {"id": Path(file_path).name, "error": str(exc)}


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """
    Orchestrates parallel feature extraction over a directory of `.dat` files.

    Parameters
    ----------
    data_path : str | Path
        Directory containing OGLE-style `.dat` photometry files.
    limit : int | None
        Cap the number of files processed (useful for testing).
    n_cores : int | None
        Worker process count. Defaults to (CPU count - 1).
    selected_indices : list[int] | None
        Feature indices (0–17) to extract. None → all 18.
    output_dir : str | Path | None
        Directory where the CSV is saved. Defaults to the current directory.
    """

    def __init__(
        self,
        data_path: str | Path,
        limit: int | None = None,
        n_cores: int | None = None,
        selected_indices: list[int] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.limit = limit
        self.n_cores = n_cores or max(1, multiprocessing.cpu_count() - 1)
        self.selected_indices = selected_indices
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, save: bool = True) -> pd.DataFrame:
        """
        Discover files, run the parallel pipeline, and return a feature DataFrame.

        Parameters
        ----------
        save : bool
            Write results to a timestamped CSV in ``output_dir``.

        Returns
        -------
        pd.DataFrame
            Rows = stars, columns = feature names, index = timeseries_id.
            Failed stars are collected in ``self.errors``.
        """
        files = self._discover_files()
        print(
            f"\n{'─' * 55}\n"
            f"  varistar FeaturePipeline  —  {datetime.now():%Y-%m-%d %H:%M}\n"
            f"{'─' * 55}\n"
            f"  Files        : {len(files)}\n"
            f"  Workers      : {self.n_cores}\n"
            f"  Features     : {self.selected_indices or 'all (0–17)'}\n"
            f"{'─' * 55}"
        )

        raw_results = self._run_parallel(files)

        successful = [r for r in raw_results if "error" not in r]
        self.errors: list[dict] = [r for r in raw_results if "error" in r]

        if not successful:
            raise RuntimeError("All stars failed. Check errors via pipeline.errors.")

        df = pd.DataFrame(successful).set_index("id").fillna(0.0)

        self._print_summary(len(files), len(successful))

        if save:
            self._save(df)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        files = sorted(self.data_path.glob("*.dat"))
        if not files:
            raise FileNotFoundError(f"No .dat files found in: {self.data_path}")
        if self.limit:
            files = files[: self.limit]
            print(f"Limiting to first {self.limit} files.")
        return files

    def _run_parallel(self, files: list[Path]) -> list[dict]:
        """Execute _process_star across all files using a process pool."""
        results: list[dict] = []
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            iterator = pool.imap_unordered(_process_star, files)
            if _TQDM:
                iterator = tqdm(iterator, total=len(files), desc="Stars", unit="star")
            for result in iterator:
                results.append(result)
        return results

    def _save(self, df: pd.DataFrame) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"features_{timestamp}.csv"
        df.to_csv(out_path)
        print(f"\nCatalogue saved to: {out_path}")
        return out_path

    def _print_summary(self, total: int, successful: int) -> None:
        n_errors = total - successful
        print(
            f"\n{'─' * 55}\n"
            f"  Completed    : {successful}/{total} stars\n"
            f"  Errors       : {n_errors}\n"
            f"{'─' * 55}"
        )
        if self.errors:
            print("  First 5 errors:")
            for err in self.errors[:5]:
                print(f"    [{err['id']}] {err['error']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="varistar.ml.pipeline",
        description="Extract photometric features from a directory of .dat files.",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="DIR",
        help="Path to directory containing .dat photometry files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N files (for testing).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes (default: CPU count - 1).",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Feature indices to extract, e.g. --features 0 1 7 11 15 (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory for the CSV file (default: current directory).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the CSV (useful when using as a library call).",
    )
    return parser


def main() -> None:
    """CLI entry point.  Called by ``python -m varistar.ml.pipeline``."""
    # Used for MP on macOS and Windows
    multiprocessing.freeze_support()

    parser = _build_parser()
    args = parser.parse_args()

    pipeline = FeaturePipeline(
        data_path=args.data,
        limit=args.limit,
        n_cores=args.cores,
        selected_indices=args.features,
        output_dir=args.output,
    )
    pipeline.run(save=not args.no_save)


if __name__ == "__main__":
    main()
