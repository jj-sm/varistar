"""
varistar.ml.data
================
Dataset container for variable star objects.

Handles batch feature extraction (parallel), unsupervised clustering,
and dimensionality-reduction visualisation (PCA / t-SNE).
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Multiprocessing worker
# Defined at module level so it is picklable on all platforms.
# ---------------------------------------------------------------------------


def _extract_worker(args: tuple) -> dict:
    """
    Top-level worker function for multiprocessing.Pool.

    Parameters
    ----------
    args : (ts_obj, lc_obj, selected_indices)
        Passed as a single tuple so pool.map / pool.imap can be used directly.
    """
    ts, lc, indices = args
    from varistar.ml.features import FeatureExtractor

    return FeatureExtractor.extract(ts_obj=ts, lc_obj=lc, selected_indices=indices)


class Dataset:
    """
    Container for a collection of (TimeSeries, LightCurve) pairs.

    Typical workflow
    ----------------
    >>> ds = Dataset(data_dir="/path/to/survey")
    >>> ds.add_object(ts1, lc1)
    >>> ds.add_object(ts2, lc2)
    >>> df = ds.build_features(selected_indices=[0, 1, 7, 11, 12, 13, 15])
    >>> ds.cluster_data(n_clusters=5)
    >>> ds.visualize_clusters()
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        # Each element is (TimeSeries, LightCurve)
        self.objects: list[tuple] = []
        self.df_features: pd.DataFrame | None = None

    # Data ingestion
    def add_object(self, ts, lc) -> None:
        """Register a (TimeSeries, LightCurve) pair for feature extraction."""
        self.objects.append((ts, lc))

    def clear_objects(self) -> None:
        """Remove all registered objects and reset the feature DataFrame."""
        self.objects.clear()
        self.df_features = None

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:
        n_feat = len(self.df_features.columns) if self.df_features is not None else 0
        return (
            f"Dataset(objects={len(self.objects)}, "
            f"features_extracted={self.df_features is not None}, "
            f"n_features={n_feat})"
        )

    # Feature extraction
    def build_features(
        self,
        n_cores: int | None = None,
        selected_indices: list[int] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features for all registered objects using a process pool.

        Parameters
        ----------
        n_cores : int | None
            Number of worker processes. Defaults to (CPU count - 1).
        selected_indices : list[int] | None
            Feature indices to compute (0–17). None → all 18 features.
        show_progress : bool
            Show a tqdm progress bar when tqdm is installed.

        Returns
        -------
        pd.DataFrame
            Rows = stars, columns = feature names, index = timeseries_id.
        """
        if not self.objects:
            raise RuntimeError("No objects registered. Call add_object() first.")

        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)

        n_selected = len(selected_indices) if selected_indices else 18
        print(
            f"Extracting {n_selected} features for {len(self.objects)} stars "
            f"using {n_cores} worker(s)..."
        )

        worker_args = [(ts, lc, selected_indices) for ts, lc in self.objects]

        results = self._run_parallel(worker_args, n_cores, show_progress)

        self.df_features = (
            pd.DataFrame(results)
            .set_index("id")
            .fillna(0.0)  # Guard against partial fit failures
        )
        print(f"Done. Feature matrix shape: {self.df_features.shape}")
        return self.df_features

    @staticmethod
    def _run_parallel(
        worker_args: list,
        n_cores: int,
        show_progress: bool,
    ) -> list[dict]:
        """Run _extract_worker across worker_args, optionally with tqdm."""
        try:
            from tqdm import tqdm

            _tqdm_available = True
        except ImportError:
            _tqdm_available = False

        results: list[dict] = []
        with multiprocessing.Pool(processes=n_cores) as pool:
            iterator = pool.imap_unordered(_extract_worker, worker_args)
            if show_progress and _tqdm_available:
                iterator = tqdm(iterator, total=len(worker_args), desc="Features")
            for res in iterator:
                results.append(res)
        return results

    # Clustering
    def cluster_data(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Run K-Means clustering on the extracted feature matrix.

        Parameters
        ----------
        n_clusters : int
            Number of clusters for K-Means.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with an added 'cluster' column.
        """
        if self.df_features is None:
            raise RuntimeError("No features available. Call build_features() first.")

        X = self._scale_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.df_features["cluster"] = kmeans.fit_predict(X)

        distribution = self.df_features["cluster"].value_counts().to_dict()
        print(f"K-Means complete. Cluster distribution: {distribution}")
        return self.df_features

    # Visualisation
    def visualize_clusters(
        self,
        figsize: tuple[int, int] = (10, 7),
        save_path: str | Path | None = None,
    ) -> None:
        """
        PCA projection of the feature space, coloured by cluster label.

        Also prints the top-5 features with the largest loading on PC1
        so you can interpret what drives the main axis of separation.

        Parameters
        ----------
        figsize : tuple
            Matplotlib figure size.
        save_path : str | Path | None
            If provided, saves the figure instead of calling plt.show().
        """
        self._require_clusters()

        feature_cols = self._pure_feature_columns()
        X = StandardScaler().fit_transform(self.df_features[feature_cols])

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        self.df_features["pca_1"] = coords[:, 0]
        self.df_features["pca_2"] = coords[:, 1]

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=self.df_features,
            x="pca_1",
            y="pca_2",
            hue="cluster",
            palette="turbo",
            alpha=0.7,
            s=60,
            edgecolor="black",
            ax=ax,
        )
        ax.set_title("PCA Projection — Star Clustering", fontsize=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.grid(True, alpha=0.2)
        ax.legend(title="Cluster", loc="best")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
        else:
            plt.show()

        # Report top PC1 drivers
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2"],
            index=feature_cols,
        )
        print("\nTop 5 features driving PC1 separation:")
        print(loadings["PC1"].abs().sort_values(ascending=False).head(5).to_string())

    def visualize_clusters_tsne(
        self,
        method: str = "pca",
        perplexity: float = 30.0,
        figsize: tuple[int, int] = (10, 7),
        save_path: str | Path | None = None,
    ) -> None:
        """
        2-D visualisation using either PCA or t-SNE, coloured by cluster.

        Parameters
        ----------
        method : {'pca', 'tsne'}
            Dimensionality reduction method.
        perplexity : float
            t-SNE perplexity parameter (ignored for PCA).
        figsize : tuple
            Matplotlib figure size.
        save_path : str | Path | None
            If provided, saves the figure instead of calling plt.show().
        """
        self._require_clusters()

        feature_cols = self._pure_feature_columns()
        X = StandardScaler().fit_transform(self.df_features[feature_cols])

        if method == "tsne":
            print("Running t-SNE (this may take a while)...")
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            reducer = PCA(n_components=2)

        coords = reducer.fit_transform(X)
        self.df_features["x"] = coords[:, 0]
        self.df_features["y"] = coords[:, 1]

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=self.df_features,
            x="x",
            y="y",
            hue="cluster",
            palette="turbo",
            s=60,
            ax=ax,
        )
        ax.set_title(f"Star Catalogue Visualisation ({method.upper()})")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
        else:
            plt.show()

    # Export helpers
    def export_features(self, path: str | Path) -> None:
        """Save the feature DataFrame to CSV."""
        if self.df_features is None:
            raise RuntimeError("No features to export. Call build_features() first.")
        pd.DataFrame(self.df_features).to_csv(path)
        print(f"Features saved to: {path}")

    def get_cluster(self, cluster_id: int) -> pd.DataFrame:
        """Return the subset of the feature DataFrame belonging to one cluster."""
        self._require_clusters()
        return self.df_features[self.df_features["cluster"] == cluster_id].copy()

    # Private helpers
    def _scale_features(self) -> np.ndarray:
        """Return a scaled numpy array of numeric feature columns only."""
        cols = self._pure_feature_columns()
        return StandardScaler().fit_transform(self.df_features[cols])

    def _pure_feature_columns(self) -> list[str]:
        """All numeric columns, excluding auxiliary visualisation and cluster columns."""
        exclude = {"cluster", "pca_1", "pca_2", "x", "y"}
        return [
            c
            for c in self.df_features.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(self.df_features[c])
        ]

    def _require_clusters(self) -> None:
        if self.df_features is None:
            raise RuntimeError("No features available. Call build_features() first.")
        if "cluster" not in self.df_features.columns:
            raise RuntimeError("No cluster labels found. Call cluster_data() first.")
