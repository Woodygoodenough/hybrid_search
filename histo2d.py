import numpy as np
from datetime import datetime
from typing import List, Tuple
from dbManagement import DBRecord
from shared_dataclasses import Predicate


class Histo2D:
    def __init__(
        self, token_bins: np.ndarray, revdate_bins: np.ndarray, counts: np.ndarray
    ):
        self.token_bins = token_bins
        self.revdate_bins = revdate_bins
        self.counts = counts
        self.n_token = len(token_bins)
        self.n_revdate = len(revdate_bins)

    @classmethod
    def from_records(
        cls, records: List[DBRecord], n_token: int = 10, n_revdate: int = 10
    ) -> "Histo2D":
        """Construct histogram from records with configurable dimensions."""
        # Extract data
        token_counts = [r.token_count for r in records]
        revdates = [
            datetime.strptime(r.revdate, "%Y-%m-%d %H:%M:%S").timestamp()
            for r in records
        ]

        # Create bins using percentiles. linespace returns "points" that are evenly spaced between start and end inclusive, thus +1.
        token_percentiles = np.linspace(0, 100, n_token + 1)
        revdate_percentiles = np.linspace(0, 100, n_revdate + 1)

        token_bins = np.percentile(token_counts, token_percentiles)
        revdate_bins = np.percentile(revdates, revdate_percentiles)

        # Count entries in each bin
        counts = np.zeros((n_token, n_revdate))
        for r in records:
            token_val = r.token_count
            revdate_val = datetime.strptime(r.revdate, "%Y-%m-%d %H:%M:%S").timestamp()

            # Find bin indices
            token_idx = np.searchsorted(token_bins, token_val, side="left") - 1
            revdate_idx = np.searchsorted(revdate_bins, revdate_val, side="left") - 1

            # Clamp to valid range
            token_idx = max(0, min(token_idx, n_token - 1))
            revdate_idx = max(0, min(revdate_idx, n_revdate - 1))

            counts[token_idx, revdate_idx] += 1

        return cls(token_bins, revdate_bins, counts)

    def rectangle_sum(
        self, token_range: Tuple[float, float], revdate_range: Tuple[float, float]
    ) -> int:
        """Conservative estimate of entries in rectangle defined by ranges."""
        token_min, token_max = token_range
        revdate_min, revdate_max = revdate_range

        # Find bin ranges
        token_start = np.searchsorted(self.token_bins, token_min, side="left")
        token_end = np.searchsorted(self.token_bins, token_max, side="right")
        revdate_start = np.searchsorted(self.revdate_bins, revdate_min, side="left")
        revdate_end = np.searchsorted(self.revdate_bins, revdate_max, side="right")

        # Clamp to valid ranges
        token_start = max(0, min(token_start, self.n_token))
        token_end = max(0, min(token_end, self.n_token))
        revdate_start = max(0, min(revdate_start, self.n_revdate))
        revdate_end = max(0, min(revdate_end, self.n_revdate))

        # Sum counts in rectangle (conservative: include partial bins)
        if token_start >= token_end or revdate_start >= revdate_end:
            return 0

        return int(
            np.sum(self.counts[token_start:token_end, revdate_start:revdate_end])
        )

    def __repr__(self) -> str:
        """String representation for Jupyter notebook display."""
        lines = [f"Histo2D({self.counts.shape[0]}x{self.counts.shape[1]})"]
        lines.append(
            "Token bins: "
            + " ".join([f"{x:.0f}" for x in self.token_bins[:5]])
            + ("..." if len(self.token_bins) > 5 else "")
        )
        lines.append(
            "Revdate bins: "
            + " ".join(
                [
                    f"{datetime.fromtimestamp(x).strftime('%Y-%m-%d')}"
                    for x in self.revdate_bins[:5]
                ]
            )
            + ("..." if len(self.revdate_bins) > 5 else "")
        )
        lines.append(f"Total entries: {int(np.sum(self.counts))}")
        return "\n".join(lines)

    def draw_histo(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create the heatmap with improved formatting
        plt.figure(figsize=(12, 8))

        # Create custom tick labels with predicate-friendly semantics
        # First bin: ≤ hi; others: (lo, ≤ hi)
        token_labels = []
        for i in range(self.counts.shape[0]):
            lo = self.token_bins[i]
            hi = self.token_bins[i + 1]
            if i == 0:
                token_labels.append(f"≤{hi:.0f}")
            else:
                token_labels.append(f"({lo:.0f},\n≤{hi:.0f})")

        revdate_labels = []
        for i in range(self.counts.shape[1]):
            lo = datetime.fromtimestamp(self.revdate_bins[i]).strftime("%Y-%m-%d")
            hi = datetime.fromtimestamp(self.revdate_bins[i + 1]).strftime("%Y-%m-%d")
            if i == 0:
                revdate_labels.append(f"≤{hi}")
            else:
                revdate_labels.append(f"({lo},\n≤{hi})")

        sns.heatmap(
            self.counts,
            cmap="Blues",
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Number of Entries"},
            xticklabels=revdate_labels,
            yticklabels=token_labels,
        )

        plt.title(
            "2D Histogram: Token Count Distribution by Revision Date",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Revision Date (YYYY-MM-DD)", fontsize=12)
        plt.ylabel("Token Count", fontsize=12)

        # Improve layout
        plt.tight_layout()
        plt.show()
        return plt.gcf()

    def save_to_disk(self, path: str) -> None:
        """Save histogram arrays to disk (numpy .npz)."""
        np.savez_compressed(
            path,
            token_bins=self.token_bins.astype(float),
            revdate_bins=self.revdate_bins.astype(float),
            counts=self.counts.astype(float),
        )

    @classmethod
    def load_from_disk(cls, path: str) -> "Histo2D":
        """Load histogram arrays from disk (numpy .npz)."""
        data = np.load(path)
        token_bins = data["token_bins"]
        revdate_bins = data["revdate_bins"]
        counts = data["counts"]
        return cls(token_bins=token_bins, revdate_bins=revdate_bins, counts=counts)

    def to_table(self) -> str:
        """Create a styled HTML table representation of the histogram data (requires pandas)."""
        import pandas as pd

        # Create column headers (revdate bins) with first using ≤max and others (lo, ≤hi)
        columns = ["Token Range"]
        for i in range(self.counts.shape[1]):
            lo_dt = datetime.fromtimestamp(self.revdate_bins[i]).strftime("%Y-%m-%d")
            hi_dt = datetime.fromtimestamp(self.revdate_bins[i + 1]).strftime(
                "%Y-%m-%d"
            )
            if i == 0:
                columns.append(f"≤{hi_dt}")
            else:
                columns.append(f"({lo_dt}, ≤{hi_dt})")

        # Create rows
        data = []
        index = []
        for i in range(self.counts.shape[0]):
            lo = self.token_bins[i]
            hi = self.token_bins[i + 1]
            if i == 0:
                row_label = f"≤{hi:.0f}"
            else:
                row_label = f"({lo:.0f}, ≤{hi:.0f})"

            row_data = [row_label] + [
                int(self.counts[i, j]) for j in range(self.counts.shape[1])
            ]
            data.append(row_data)
            index.append(row_label)

        df = pd.DataFrame(data, columns=columns, index=index)
        df.index.name = "Token Count Range"

        styled = (
            df.style.background_gradient(cmap="Blues", axis=None)
            .format(precision=0)
            .set_properties(**{"text-align": "center"})
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [("text-align", "center"), ("font-weight", "bold")],
                    },
                    {
                        "selector": "caption",
                        "props": [
                            ("caption-side", "top"),
                            ("font-weight", "bold"),
                            ("font-size", "14px"),
                        ],
                    },
                ]
            )
        )
        styled.caption = f"2D Histogram Table: {int(np.sum(self.counts))} total entries"
        return styled.to_html()

    def to_dataframe(self):
        """Return the histogram data as a pandas DataFrame."""
        import pandas as pd

        # Create column headers (revdate bins)
        columns = ["Token Range"]
        for i in range(self.counts.shape[1]):
            lo_dt = datetime.fromtimestamp(self.revdate_bins[i]).strftime("%Y-%m-%d")
            hi_dt = datetime.fromtimestamp(self.revdate_bins[i + 1]).strftime(
                "%Y-%m-%d"
            )
            if i == 0:
                columns.append(f"≤{hi_dt}")
            else:
                columns.append(f"({lo_dt}, ≤{hi_dt})")

        data = []
        index = []
        for i in range(self.counts.shape[0]):
            lo = self.token_bins[i]
            hi = self.token_bins[i + 1]
            if i == 0:
                row_label = f"≤{hi:.0f}"
            else:
                row_label = f"({lo:.0f}, ≤{hi:.0f})"

            row_data = [row_label] + [
                int(self.counts[i, j]) for j in range(self.counts.shape[1])
            ]
            data.append(row_data)
            index.append(row_label)

        df = pd.DataFrame(data, columns=columns, index=index)
        df.index.name = "Token Count Range"
        return df

    def estimate_survivors(self, predicates: List[Predicate]) -> int:
        """Estimate survivors count given predicates by summing bins consistent with predicates.
        Uses linear interpolation for bins that are partially included.

        Semantics:
        - revdate predicates compare against bin timestamps. Predicate value is already normalized to 'YYYY-MM-DD HH:MM:SS' so we can compare directly.
        - token_count predicates compare numerically to bin ranges.
        - We forbid IN and = for now, since they do not fit into histogram semantics.

        """
        if not predicates:
            return int(self.counts.sum())

        # Precompute bin edge vectors
        token_lo = self.token_bins[:-1].astype(float)
        token_hi = self.token_bins[1:].astype(float)
        rev_lo = self.revdate_bins[:-1].astype(float)
        rev_hi = self.revdate_bins[1:].astype(float)

        # Initialize fractional weights (rows for token, cols for revdate)
        row_weights = np.ones(self.counts.shape[0], dtype=float)
        col_weights = np.ones(self.counts.shape[1], dtype=float)

        for p in predicates:
            if p.key == "token_count":
                v = float(p.value)
                token_widths = token_hi - token_lo
                new_row_weights = np.zeros(self.counts.shape[0], dtype=float)
                if p.operator == ">":
                    # Fully included if token_lo > v, partially if token_lo <= v < token_hi
                    full_mask = token_lo > v
                    partial_mask = (token_lo <= v) & (token_hi > v)
                    new_row_weights[full_mask] = 1.0
                    new_row_weights[partial_mask] = np.maximum(
                        0.0, (token_hi[partial_mask] - v) / token_widths[partial_mask]
                    )
                elif p.operator == ">=":
                    # Fully included if token_lo >= v, partially if token_lo < v <= token_hi
                    full_mask = token_lo >= v
                    partial_mask = (token_lo < v) & (token_hi >= v)
                    new_row_weights[full_mask] = 1.0
                    new_row_weights[partial_mask] = np.maximum(
                        0.0, (token_hi[partial_mask] - v) / token_widths[partial_mask]
                    )
                elif p.operator == "<":
                    # Fully included if token_hi < v, partially if token_lo < v <= token_hi
                    full_mask = token_hi < v
                    partial_mask = (token_lo < v) & (token_hi >= v)
                    new_row_weights[full_mask] = 1.0
                    new_row_weights[partial_mask] = np.maximum(
                        0.0, (v - token_lo[partial_mask]) / token_widths[partial_mask]
                    )
                elif p.operator == "<=":
                    # Fully included if token_hi <= v, partially if token_lo <= v < token_hi
                    full_mask = token_hi <= v
                    partial_mask = (token_lo <= v) & (token_hi > v)
                    new_row_weights[full_mask] = 1.0
                    new_row_weights[partial_mask] = np.maximum(
                        0.0, (v - token_lo[partial_mask]) / token_widths[partial_mask]
                    )
                elif p.operator == "=":
                    # Treat = as a narrow range query (v-0.5 to v+0.5)
                    new_row_weights = ((token_lo <= v) & (token_hi >= v)).astype(float)
                elif p.operator == "IN":
                    # For IN operator, create a mask for any of the values
                    in_mask = np.zeros(self.counts.shape[0], dtype=bool)
                    for val in p.value if isinstance(p.value, list) else [p.value]:
                        val = float(val)
                        in_mask |= (token_lo <= val) & (token_hi >= val)
                    new_row_weights = in_mask.astype(float)
                row_weights = row_weights * new_row_weights
            elif p.key == "revdate":
                # p.value already normalized to 'YYYY-MM-DD HH:MM:SS'
                t = datetime.strptime(str(p.value), "%Y-%m-%d %H:%M:%S").timestamp()
                rev_widths = rev_hi - rev_lo
                new_col_weights = np.zeros(self.counts.shape[1], dtype=float)
                if p.operator == ">":
                    # Fully included if rev_lo > t, partially if rev_lo <= t < rev_hi
                    full_mask = rev_lo > t
                    partial_mask = (rev_lo <= t) & (rev_hi > t)
                    new_col_weights[full_mask] = 1.0
                    new_col_weights[partial_mask] = np.maximum(
                        0.0, (rev_hi[partial_mask] - t) / rev_widths[partial_mask]
                    )
                elif p.operator == ">=":
                    # Fully included if rev_lo >= t, partially if rev_lo < t <= rev_hi
                    full_mask = rev_lo >= t
                    partial_mask = (rev_lo < t) & (rev_hi >= t)
                    new_col_weights[full_mask] = 1.0
                    new_col_weights[partial_mask] = np.maximum(
                        0.0, (rev_hi[partial_mask] - t) / rev_widths[partial_mask]
                    )
                elif p.operator == "<":
                    # Fully included if rev_hi < t, partially if rev_lo < t <= rev_hi
                    full_mask = rev_hi < t
                    partial_mask = (rev_lo < t) & (rev_hi >= t)
                    new_col_weights[full_mask] = 1.0
                    new_col_weights[partial_mask] = np.maximum(
                        0.0, (t - rev_lo[partial_mask]) / rev_widths[partial_mask]
                    )
                elif p.operator == "<=":
                    # Fully included if rev_hi <= t, partially if rev_lo <= t < rev_hi
                    full_mask = rev_hi <= t
                    partial_mask = (rev_lo <= t) & (rev_hi > t)
                    new_col_weights[full_mask] = 1.0
                    new_col_weights[partial_mask] = np.maximum(
                        0.0, (t - rev_lo[partial_mask]) / rev_widths[partial_mask]
                    )
                elif p.operator == "=":
                    # Treat = as a narrow range query (t to t + small delta)
                    new_col_weights = ((rev_lo <= t) & (rev_hi >= t)).astype(float)
                elif p.operator == "IN":
                    # For IN operator, create a mask for any of the values
                    in_mask = np.zeros(self.counts.shape[1], dtype=bool)
                    for val in p.value if isinstance(p.value, list) else [p.value]:
                        t_val = datetime.strptime(
                            str(val), "%Y-%m-%d %H:%M:%S"
                        ).timestamp()
                        in_mask |= (rev_lo <= t_val) & (rev_hi >= t_val)
                    new_col_weights = in_mask.astype(float)
                col_weights = col_weights * new_col_weights
            else:
                raise ValueError(
                    f"Invalid predicate key: {p.key}. Only 'revdate' and 'token_count' are supported."
                )

        if not row_weights.any() or not col_weights.any():
            return 0

        # Apply weights: multiply counts by row and column weights
        weighted_counts = (
            self.counts * row_weights[:, np.newaxis] * col_weights[np.newaxis, :]
        )
        return int(weighted_counts.sum())

    def revdate_edges_df(self):
        """Return a DataFrame with exact revdate bin separating points with full timestamp."""
        import pandas as pd

        formatted = [
            datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
            for v in self.revdate_bins
        ]
        return pd.DataFrame(
            {
                "edge_index": list(range(len(self.revdate_bins))),
                "revdate_ts": self.revdate_bins.astype(float),
                "revdate_str": formatted,
            }
        )

    def display_summary(self) -> str:
        """Display a summary of the histogram statistics."""
        total = int(np.sum(self.counts))
        max_count = int(np.max(self.counts))
        min_count = int(np.min(self.counts))
        non_empty_bins = int(np.sum(self.counts > 0))

        lines = []
        lines.append("Histogram Summary:")
        lines.append("-" * 30)
        lines.append(f"Total entries: {total}")
        lines.append(
            f"Non-empty bins: {non_empty_bins}/{self.counts.shape[0] * self.counts.shape[1]} ({100*non_empty_bins/(self.counts.shape[0] * self.counts.shape[1]):.1f}%)"
        )
        lines.append(f"Max count per bin: {max_count}")
        lines.append(f"Min count per bin: {min_count}")
        lines.append(
            f"Average count per bin: {total/(self.counts.shape[0] * self.counts.shape[1]):.2f}"
        )
        lines.append(
            f"Average count per non-empty bin: {total/non_empty_bins:.2f}"
            if non_empty_bins > 0
            else "Average count per non-empty bin: N/A"
        )

        return "\n".join(lines)
