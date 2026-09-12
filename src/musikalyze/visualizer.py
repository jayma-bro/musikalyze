"""
MusicEDA - Exploratory data analysis toolkit for music feature DataFrames.

Outputs the exact same figures and logic as ``demo/musiceda.py``,
adapted to live inside the musikalyze package.

Usage
-----
>>> import pandas as pd
>>> from musikalyze.visualizer import MusicEDA
>>> df = batch.analyze(["metas_all_pct", "genre400_all"])
>>> eda = MusicEDA(df, score_cols=["acousticness", "danceability"],
...                genre_cols="genre400_all")
>>> eda.plot_histogram("acousticness")

Supported inputs
----------------
- ``score_cols`` : numeric features on a 0-100 scale (auto-detected if None)
- ``other_cols`` : numeric features with arbitrary ranges (auto-detected if None)
- ``genre_cols`` : one or more columns of dicts ``{genre_name: score}``
- ``id_cols``    : track identifier columns used in hover tooltips
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MusicEDA:
    """EDA helper for music feature DataFrames with embedded genre dicts."""

    def __init__(
            self,
            df: pd.DataFrame,
            *,
            score_cols: list[str] | None = None,
            other_cols: list[str] | None = None,
            genre_cols: list[str] | str | None = None,
            id_cols: list[str] | None = None,
            skip_cols: list[str] | None = None,
            default_height: int = 700,
            violin_max_per_fig: int = 11,
        ):
        """
        Parameters
        ----------
        df : input dataframe (one row per track)
        score_cols : numeric 0-100 feature columns (auto-detected if None)
        other_cols : numeric non-0-100 feature columns (auto-detected if None)
        genre_cols : column(s) containing dicts of {genre: score}
        id_cols : column(s) used to identify tracks in hover tooltips
        skip_cols : columns to exclude from auto-detection
        default_height : default figure height in px
        violin_max_per_fig : max violins per figure before splitting
        """
        self.df = df.copy()
        self.height = default_height
        self.violin_max_per_fig = violin_max_per_fig

        if isinstance(genre_cols, str):
            genre_cols = [genre_cols]
        self.genre_cols = list(genre_cols or [])
        self.id_cols = list(id_cols or [])
        skip = set(skip_cols or []) | set(self.genre_cols) | set(self.id_cols)

        # Auto-detection
        remaining = [c for c in df.columns if c not in skip]
        if score_cols is None:
            score_cols = [
                c for c in remaining
                if pd.api.types.is_numeric_dtype(df[c])
                and df[c].dropna().between(0, 100).all()
            ]
        if other_cols is None:
            other_cols = [
                c for c in remaining
                if pd.api.types.is_numeric_dtype(df[c]) and c not in score_cols
            ]
        self.score_cols = list(score_cols)
        self.other_cols = list(other_cols)

        # Cache for exploded genre dataframes
        self._genre_cache: dict[str, pd.DataFrame] = {}

        # Single hover label from id_cols + original df index
        if self.id_cols:
            ids = (
                self.df[self.id_cols]
                .apply(lambda col: col.map(str))
                .agg(" - ".join, axis=1)
            )
        else:
            ids = pd.Series("", index=self.df.index)
        self.df["__id__"] = "#" + self.df.index.astype(str) + " \u00b7 " + ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _genres(self, col: str) -> pd.DataFrame:
        """Explode a genre dict column into a wide numeric dataframe."""
        if col not in self._genre_cache:
            self._genre_cache[col] = pd.json_normalize(self.df[col]).fillna(0)
        return self._genre_cache[col]

    @staticmethod
    def _split_genre_key(key: str) -> tuple[str, str]:
        """Return (category, subgenre). Plain keys map to (key, key)."""
        if "---" in key:
            cat, sub = key.split("---", 1)
            return cat.strip(), sub.strip()
        return key, key

    def _resolve_col(self, name: str) -> pd.Series:
        """Resolve a column name, including 'genre_col.Genre Key' references."""
        if name in self.df.columns:
            return self.df[name]
        if "." in name:
            gcol, gkey = name.split(".", 1)
            if gcol in self.genre_cols:
                return self._genres(gcol)[gkey]
        for gcol in self.genre_cols:
            gf = self._genres(gcol)
            if name in gf.columns:
                return gf[name]
        raise KeyError(f"Column '{name}' not found in df or genre columns.")

    def _col_axis_range(self, col: str) -> list[float] | None:
        """Fixed [0, 100] range for score cols / genre refs, None otherwise."""
        if col in self.score_cols or col not in self.df.columns:
            return [0, 100]
        return None

    def _split_by_scale(self, cols: list[str]) -> tuple[list[str], list[str]]:
        """Split columns into (score-scale, other-scale) groups."""
        scores = [c for c in cols if self._col_axis_range(c) is not None]
        others = [c for c in cols if self._col_axis_range(c) is None]
        return scores, others

    def _base_layout(self, title: str, height: int | None) -> dict:
        return {"title": title, "height": height or self.height,
                "template": "plotly_white"}

    def _hover_labels(self, index: pd.Index) -> pd.Series:
        return self.df.loc[index, "__id__"]

    def _kde(self, values: np.ndarray, n_points: int = 300,
             bw_adjust: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian KDE with adjustable bandwidth (Scott's rule * bw_adjust)."""
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        kde.set_bandwidth(kde.scotts_factor() * bw_adjust)
        xs = np.linspace(values.min(), values.max(), n_points)
        return xs, kde(xs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Overview of detected column roles and basic stats."""
        rows = []
        for role, cols in [("score", self.score_cols),
                           ("other", self.other_cols),
                           ("genre_dict", self.genre_cols),
                           ("id", self.id_cols)]:
            for c in cols:
                if role in ("score", "other"):
                    s = self.df[c]
                    rows.append({"column": c, "role": role,
                                 "dtype": str(s.dtype),
                                 "n_missing": int(s.isna().sum()),
                                 "min": s.min(), "max": s.max(),
                                 "mean": round(s.mean(), 2)})
                elif role == "genre_dict":
                    gf = self._genres(c)
                    rows.append({"column": c, "role": role,
                                 "dtype": "dict",
                                 "n_missing": int(self.df[c].isna().sum()),
                                 "min": None, "max": None,
                                 "mean": f"{gf.shape[1]} genres"})
                else:
                    rows.append({"column": c, "role": role,
                                 "dtype": str(self.df[c].dtype),
                                 "n_missing": int(self.df[c].isna().sum()),
                                 "min": None, "max": None, "mean": None})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Score / numeric feature plots
    # ------------------------------------------------------------------

    def plot_histogram(
            self,
            col: str,
            *,
            color_by: str | None = None,
            bins: int = 50,
            bw_adjust: float = 1.0,
            height: int | None = None,
        ) -> go.Figure:
        """Histogram of one numeric column with an overlaid KDE curve.

        bw_adjust < 1 gives a more detailed KDE, > 1 a smoother one.
        """
        series = self._resolve_col(col)
        data = pd.DataFrame({"value": series})
        if color_by:
            data["group"] = self.df[color_by].astype(str)

        fig = px.histogram(
            data, x="value", color="group" if color_by else None,
            nbins=bins, barmode="overlay", opacity=0.75,
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(bargap=0.05)

        # KDE overlay (scaled to histogram counts)
        colors = px.colors.qualitative.Plotly
        if color_by:
            groups = data["group"].unique()
            for i, g in enumerate(groups):
                vals = data.loc[data["group"] == g, "value"].dropna().values
                if len(vals) > 2:
                    xs, ys = self._kde(vals, bw_adjust=bw_adjust)
                    bin_w = (vals.max() - vals.min()) / bins
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys * len(vals) * bin_w, mode="lines",
                        name=f"{g} KDE", line=dict(color=colors[i % len(colors)]),
                    ))
        else:
            vals = series.dropna().values
            if len(vals) > 2:
                xs, ys = self._kde(vals, bw_adjust=bw_adjust)
                bin_w = (vals.max() - vals.min()) / bins
                fig.add_trace(go.Scatter(
                    x=xs, y=ys * len(vals) * bin_w, mode="lines",
                    name="KDE", line=dict(color="black", width=2),
                ))

        rng = self._col_axis_range(col)
        if rng:
            fig.update_xaxes(range=rng)
        fig.update_layout(**self._base_layout(f"Distribution of {col}", height),
                          xaxis_title=col, yaxis_title="Count")
        return fig

    def plot_violin(
            self,
            cols: list[str] | None = None,
            *,
            color_by: str | None = None,
            bandwidth: float | None = None,
            max_per_fig: int | None = None,
            height: int | None = None,
        ) -> go.Figure | list[go.Figure]:
        """Violin plots with embedded boxplots.

        Score-scale (0-100) and other-scale columns are always rendered in
        separate figures. Figures are split when exceeding max_per_fig.
        Returns a single figure when possible, else a list of figures.
        Outlier points carry the track id in their hover tooltip.
        """
        cols = cols or (self.score_cols + self.other_cols)
        max_per_fig = max_per_fig or self.violin_max_per_fig

        scores, others = self._split_by_scale(cols)
        groups = [g for g in (scores, others) if g]

        # Split each scale group into chunks of max_per_fig
        chunks: list[list[str]] = []
        for g in groups:
            for i in range(0, len(g), max_per_fig):
                chunks.append(g[i:i + max_per_fig])

        figures = []
        for chunk in chunks:
            long_rows = []
            for c in chunk:
                s = self._resolve_col(c)
                sub = pd.DataFrame({
                    "variable": c, "value": s,
                    "__id__": self.df["__id__"],
                })
                if color_by:
                    sub["group"] = self.df[color_by].astype(str)
                long_rows.append(sub)
            long = pd.concat(long_rows, ignore_index=True).dropna(subset=["value"])

            hover = {"__id__": True, "variable": False, "value": ":.1f"}
            if color_by:
                hover["group"] = False

            fig = px.violin(
                long, x="variable", y="value",
                color="group" if color_by else "variable",
                box=True, points="outliers",
                hover_data=hover,
            )

            if bandwidth is not None:
                fig.update_traces(bandwidth=bandwidth)

            fig.update_traces(meanline_visible=True)
            rng = self._col_axis_range(chunk[0])
            if rng:
                fig.update_yaxes(range=rng)

            scale_note = "0-100 scale" if rng else "auto scale"
            fig.update_layout(
                **self._base_layout(
                    f"Violin distributions ({scale_note})", height),
                yaxis_title="Value", xaxis_title="",
                showlegend=bool(color_by),
            )
            figures.append(fig)

        return figures[0] if len(figures) == 1 else figures

    def plot_scatter(
            self,
            x: str,
            y: str,
            *,
            color_by: str | None = None,
            force_0_100: bool = False,
            trendline: bool = True,
            height: int | None = None,
        ) -> go.Figure:
        """Scatter plot of two columns (genre refs allowed, e.g. 'gc.Rock---Metal').

        The OLS trendline is always red, dashed, and shows its R-squared in the legend.
        """
        xs, ys = self._resolve_col(x), self._resolve_col(y)
        data = pd.DataFrame({"x": xs, "y": ys, "__id__": self.df["__id__"]})
        if color_by:
            data["group"] = self.df[color_by].astype(str)
        data = data.dropna(subset=["x", "y"])

        # R-squared on the full sample
        if len(data) > 2:
            slope, intercept = np.polyfit(data["x"], data["y"], 1)
            r = np.corrcoef(data["x"], data["y"])[0, 1]
            r2 = r ** 2
        else:
            slope = intercept = r2 = None

        hover = {"__id__": True, "x": ":.2f", "y": ":.2f"}
        if color_by:
            hover["group"] = False

        fig = px.scatter(
            data, x="x", y="y", color="group" if color_by else None,
            hover_data=hover,
            opacity=0.7,
        )

        if trendline and slope is not None:
            xline = np.array([data["x"].min(), data["x"].max()])
            fig.add_trace(go.Scatter(
                x=xline, y=slope * xline + intercept, mode="lines",
                name=f"Trend (R²={r2:.3f})",
                line=dict(color="red", dash="dash", width=2),
            ))

        if force_0_100:
            fig.update_xaxes(range=[0, 100])
            fig.update_yaxes(range=[0, 100])
        else:
            rx, ry = self._col_axis_range(x), self._col_axis_range(y)
            if rx:
                fig.update_xaxes(range=rx)
            if ry:
                fig.update_yaxes(range=ry)

        fig.update_layout(**self._base_layout(f"{y} vs {x}", height),
                          xaxis_title=x, yaxis_title=y)
        return fig

    def plot_pairplot(
            self,
            cols: list[str] | None = None,
            *,
            color_by: str | None = None,
            bins: int = 25,
            height: int | None = None,
        ) -> go.Figure:
        """Pairplot grid: scatter off-diagonal (with r annotation),
        histogram on the diagonal."""
        cols = cols or self.score_cols
        n = len(cols)
        if n < 2:
            raise ValueError("Need at least 2 columns for a pairplot.")

        palette = px.colors.qualitative.Plotly
        groups = (
            self.df[color_by].astype(str) if color_by
            else pd.Series("all", index=self.df.index)
        )
        group_names = list(groups.unique())

        fig = make_subplots(
            rows=n, cols=n, shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.02, vertical_spacing=0.02,
        )

        for i, cy in enumerate(cols):      # row → y axis
            for j, cx in enumerate(cols):  # col → x axis
                row, colpos = i + 1, j + 1
                show_legend = (i == 0 and j == 0)
                for k, g in enumerate(group_names):
                    mask = groups == g
                    ids = self._hover_labels(self.df.index[mask])
                    if i == j:
                        fig.add_trace(go.Histogram(
                            x=self.df.loc[mask, cx], name=str(g),
                            legendgroup=str(g), showlegend=show_legend,
                            marker_color=palette[k % len(palette)],
                            nbinsx=bins, opacity=0.75,
                        ), row=row, col=colpos)
                    else:
                        fig.add_trace(go.Scatter(
                            x=self.df.loc[mask, cx], y=self.df.loc[mask, cy],
                            mode="markers", name=str(g), legendgroup=str(g),
                            showlegend=show_legend,
                            marker=dict(size=4, opacity=0.6,
                                        color=palette[k % len(palette)]),
                            text=ids, hovertemplate="%{text}<extra></extra>",
                        ), row=row, col=colpos)

                # Pearson r annotation on scatter cells
                if i != j:
                    r = np.corrcoef(self.df[cx], self.df[cy])[0, 1]
                    fig.add_annotation(
                        text=f"r={r:.2f}", xref=f"x{colpos + i*n} domain",
                        yref=f"y{row + j*n - n + n} domain",
                        x=0.05, y=0.95, showarrow=False,
                        font=dict(size=11, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        row=row, col=colpos,
                    )
                    xv, yv = self.df[cx].values, self.df[cy].values
                    if len(xv) > 2:
                        slope, intercept = np.polyfit(xv, yv, 1)
                        xline = np.array([xv.min(), xv.max()])
                        fig.add_trace(go.Scatter(
                            x=xline, y=slope * xline + intercept,
                            mode="lines", showlegend=False,
                            line=dict(color="red", dash="dash", width=1.5),
                            hoverinfo="skip",
                        ), row=row, col=colpos)

                if i == n - 1:
                    fig.update_xaxes(title_text=cx, row=row, col=colpos)
                if j == 0:
                    fig.update_yaxes(title_text=cy, row=row, col=colpos)

        total_height = (height or self.height) + 150 * max(0, n - 3)
        fig.update_layout(**self._base_layout("Pairplot", total_height),
                          bargap=0.05)
        return fig

    def plot_correlation_heatmap(
        self,
        cols: list[str] | None = None,
        *,
        method: str = "pearson",
        height: int | None = None,
    ) -> go.Figure:
        """Correlation heatmap of numeric columns."""
        cols = cols or (self.score_cols + self.other_cols)
        corr = self.df[cols].corr(method=method)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig.update_layout(**self._base_layout(
            f"Correlation heatmap ({method})", height))
        return fig

    # ------------------------------------------------------------------
    # Genre plots
    # ------------------------------------------------------------------

    def plot_genre_threshold_analysis(
            self,
            genre_col: str | None = None,
            *,
            step: int = 5,
            skip_low: int = 1,
            dual_axis: bool = True,
            height: int | None = None,
        ) -> go.Figure:
        """Threshold sweep: genres/track quantiles + distinct genres per threshold.

        step : spacing between tested thresholds (default 5).
        skip_low : skip the N lowest thresholds (saturated zone where every
                   genre passes, which flattens the curves).
        Lines : median, Q1, Q3, P90, min/max of genres per track.
        """
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        thresholds = list(range(0, 100, step))
        counts = pd.DataFrame({t: (gf >= t).sum(axis=1) for t in thresholds})

        qs = counts.quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        distinct = [(gf >= t).any().sum() for t in thresholds]

        keep = thresholds[skip_low:]
        fig = go.Figure()
        lines = [
            ("Max", qs.loc[1.0], "#d62728", "dot"),
            ("P90", qs.loc[0.9], "#ff7f0e", "dash"),
            ("Q3", qs.loc[0.75], "#2ca02c", "dash"),
            ("Median", qs.loc[0.5], "#1f77b4", "solid"),
            ("Mean", counts.mean(), "#17becf", "solid"),
            ("Q1", qs.loc[0.25], "#9467bd", "dash"),
            ("Min", qs.loc[0.0], "#7f7f7f", "dot"),
        ]
        for name, qvals, color, dash in lines:
            fig.add_trace(go.Scatter(
                x=keep, y=[qvals[t] for t in keep], mode="lines+markers",
                name=name, line=dict(color=color, dash=dash),
            ))
        fig.add_trace(go.Scatter(
            x=keep, y=distinct[skip_low:], mode="lines+markers",
            name="Distinct genres", line=dict(color="black", width=2),
            yaxis="y2" if dual_axis else "y",
        ))

        layout = self._base_layout(
            f"Threshold analysis \u2014 {genre_col}", height)
        layout.update(
            xaxis_title="Score threshold",
            yaxis_title="Genres per track",
            yaxis_rangemode="tozero",
        )
        if dual_axis:
            layout["yaxis2"] = dict(
                title="Distinct genres", overlaying="y", side="right",
                rangemode="tozero",
            )
        fig.update_layout(**layout)
        return fig

    def plot_genre_countplot(
            self,
            genre_col: str | None = None,
            *,
            threshold: float = 50,
            stack_step: int = 10,
            top_n: int | None = 30,
            height: int | None = None,
        ) -> go.Figure:
        """Stacked count of genres passing thresholds from 90 down to `threshold`.

        Each bar segment = number of tracks whose genre score is in
        [level - stack_step, level). Gradient colors show score intensity.
        """
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)

        levels = list(range(90, int(threshold) - 1, -stack_step))
        # cumulative counts at each level
        cum = {t: (gf >= t).sum() for t in levels}
        total = cum[levels[-1]].sort_values(ascending=False)
        if top_n:
            total = total.head(top_n)
        genres = total.index.tolist()

        fig = go.Figure()
        n_seg = len(levels)
        colors = px.colors.sample_colorscale(
            "Viridis", [i / max(n_seg - 1, 1) for i in range(n_seg)])
        for i, t in enumerate(levels):
            # exclusive count: >= t but < next level above (else cum double-counts)
            if i == 0:
                seg = cum[t][genres]
                label = f"\u2265 {t}"
            else:
                seg = cum[t][genres] - cum[levels[i - 1]][genres]
                label = f"{t}\u2013{levels[i - 1]}"
            fig.add_trace(go.Bar(
                x=genres, y=seg, name=label,
                marker_color=colors[i],
                hovertemplate="%{x}<br>" + label + ": %{y} tracks<extra></extra>",
            ))

        fig.update_layout(
            **self._base_layout(
                f"Genre counts by score band \u2014 {genre_col} "
                f"(\u2265 {threshold})", height),
            barmode="stack", xaxis_tickangle=-45,
            xaxis_title="Genre", yaxis_title="Number of tracks",
            legend_title="Score band",
        )
        return fig

    def plot_genre_global(
            self,
            genre_col: str | None = None,
            *,
            top_n: int | None = None,
            level: str = "genre",
            height: int | None = None,
        ) -> go.Figure:
        """Total cumulative score per genre (or per category)."""
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        sums = gf.sum().sort_values(ascending=False)

        if level == "category":
            cats = {}
            for g, v in sums.items():
                cat, _ = self._split_genre_key(g)
                cats[cat] = cats.get(cat, 0) + v
            sums = pd.Series(cats).sort_values(ascending=False)

        if top_n:
            sums = sums.head(top_n)
        sums = sums.iloc[::-1]  # largest on top in horizontal bars

        fig = go.Figure(go.Bar(
            x=sums.values, y=sums.index, orientation="h",
            marker_color=sums.values, marker_colorscale="Viridis",
        ))
        fig.update_layout(**self._base_layout(
            f"Cumulative score by {level} \u2014 {genre_col}", height),
            xaxis_title="Cumulative score", yaxis_title="")
        return fig

    def plot_genre_profile(
            self,
            genre_col: str | None = None,
            *,
            sample_size: int | None = 10,
            sample_indices: list[int] | None = None,
            top_n: int = 40,
            stats_only: bool = False,
            height: int | None = None,
        ) -> go.Figure:
        """Per-track genre profiles, genres sorted by cumulative score.

        stats_only : instead of individual tracks, plot mean/Q1/Q3/min/max
                     across all tracks.
        """
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        order = gf.sum().sort_values(ascending=False).head(top_n).index
        gf = gf[order]

        fig = go.Figure()
        if stats_only:
            qs = gf.quantile([0, 0.25, 0.5, 0.75, 1.0])
            for name, qvals, dash in [
                ("Mean", gf.mean(), "solid"), ("Q3", qs.loc[0.75], "dash"),
                ("Median", qs.loc[0.5], "solid"), ("Q1", qs.loc[0.25], "dash"),
                ("Max", qs.loc[1.0], "dot"), ("Min", qs.loc[0.0], "dot"),
            ]:
                fig.add_trace(go.Scatter(
                    x=gf.columns, y=qvals, mode="lines",
                    name=name, line=dict(dash=dash),
                ))
        else:
            if sample_indices is not None:
                idx = sample_indices
            else:
                n = sample_size or 10
                idx = gf.sample(n=min(n, len(gf))).index.tolist()
            for i in idx:
                fig.add_trace(go.Scatter(
                    x=gf.columns, y=gf.loc[i], mode="lines",
                    name=self.df.loc[i, "__id__"], opacity=0.7,
                ))

        fig.update_layout(**self._base_layout(
            f"Genre profiles \u2014 {genre_col} (top {top_n} by cumulated score)",
            height),
            xaxis_tickangle=-45, xaxis_title="Genre", yaxis_title="Score",
            yaxis_range=[0, 100])
        return fig

    def plot_category_stacked(
            self,
            genre_col: str | None = None,
            *,
            sample_size: int = 10,
            top_genres: int = 20,
            height: int | None = None,
        ) -> go.Figure:
        """Stacked genre composition of a random sample of tracks.

        Re-running draws a new random sample.
        """
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        top = gf.sum().sort_values(ascending=False).head(top_genres).index
        idx = gf.sample(n=min(sample_size, len(gf))).index

        fig = go.Figure()
        labels = [self.df.loc[i, "__id__"] for i in idx]
        for g in top:
            fig.add_trace(go.Bar(
                x=labels, y=gf.loc[idx, g], name=g,
            ))
        fig.update_layout(**self._base_layout(
            f"Genre composition per track \u2014 {genre_col} (random sample)",
            height),
            barmode="stack", xaxis_tickangle=-30,
            xaxis_title="Track", yaxis_title="Score")
        return fig

    def plot_genre_cooccurrence(
            self,
            genre_col: str | None = None,
            *,
            threshold: float = 50,
            top_n: int = 30,
            height: int | None = None,
        ) -> go.Figure:
        """Co-occurrence heatmap: how often genre pairs both pass `threshold`."""
        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        freq = (gf >= threshold).sum().sort_values(ascending=False)
        top = freq.head(top_n).index
        mask = (gf[top] >= threshold).astype(int).to_numpy()
        co = mask.T @ mask
        np.fill_diagonal(co, 0)

        fig = px.imshow(
            co, x=top.tolist(), y=top.tolist(),
            labels=dict(color="Co-occurrences"),
            color_continuous_scale="Viridis", aspect="auto",
        )
        fig.update_layout(**self._base_layout(
            f"Genre co-occurrence \u2014 {genre_col} (threshold={threshold})",
            (height or self.height) + 100),
            xaxis_tickangle=-45)
        return fig

    def plot_genre_pca(
            self,
            genre_col: str | None = None,
            *,
            threshold: float = 30,
            color_by: str | None = None,
            top_n: int = 100,
            height: int | None = None,
        ) -> go.Figure:
        """PCA 2D projection of tracks in genre space."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("plot_genre_pca requires scikit-learn.")

        genre_col = genre_col or self.genre_cols[0]
        gf = self._genres(genre_col)
        top = gf.sum().sort_values(ascending=False).head(top_n).index
        X = StandardScaler().fit_transform(gf[top])
        proj = PCA(n_components=2).fit_transform(X)

        data = pd.DataFrame({
            "PC1": proj[:, 0], "PC2": proj[:, 1],
            "__id__": self.df["__id__"].values,
        })
        if color_by:
            data["group"] = self.df[color_by].astype(str).values

        fig = px.scatter(
            data, x="PC1", y="PC2", color="group" if color_by else None,
            hover_data={"__id__": True, "PC1": ":.2f", "PC2": ":.2f"},
            opacity=0.7,
        )
        fig.update_layout(**self._base_layout(
            f"PCA of genre space \u2014 {genre_col} (top {top_n} genres)", height))
        return fig

    # ------------------------------------------------------------------
    # Advanced views
    # ------------------------------------------------------------------

    def plot_radar(
            self,
            cols: list[str] | None = None,
            *,
            n_random: int = 5,
            indices: list[int] | None = None,
            height: int | None = None,
        ) -> go.Figure:
        """Radar chart of score features for a few tracks (random or given)."""
        cols = cols or self.score_cols
        if indices is None:
            indices = self.df.sample(n=min(n_random, len(self.df))).index.tolist()

        fig = go.Figure()
        for i in indices:
            fig.add_trace(go.Scatterpolar(
                r=self.df.loc[i, cols].tolist(),
                theta=cols, fill="toself",
                name=self.df.loc[i, "__id__"], opacity=0.6,
            ))
        fig.update_layout(
            **self._base_layout("Radar \u2014 score features", height),
            polar=dict(radialaxis=dict(range=[0, 100])),
        )
        return fig

    def plot_embedding(
            self,
            cols: list[str] | None = None,
            *,
            color_by: str | None = None,
            method: str = "umap",
            height: int | None = None,
        ) -> go.Figure:
        """2D embedding (UMAP or t-SNE) of tracks in feature space."""
        cols = cols or self.score_cols
        X = self.df[cols].dropna()

        if method == "umap":
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError("pip install umap-learn, or method='tsne'")
            emb = UMAP(n_components=2, random_state=42).fit_transform(X)
        else:
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, random_state=42).fit_transform(X)

        data = pd.DataFrame({
            "x": emb[:, 0], "y": emb[:, 1],
            "__id__": self._hover_labels(X.index).values,
        }, index=X.index)
        if color_by:
            data["group"] = self.df.loc[X.index, color_by].astype(str).values

        hover = {"__id__": True, "x": ":.2f", "y": ":.2f"}
        if color_by:
            hover["group"] = False

        fig = px.scatter(
            data, x="x", y="y", color="group" if color_by else None,
            hover_data=hover,
            opacity=0.7,
        )
        fig.update_layout(**self._base_layout(
            f"{method.upper()} embedding", height),
            xaxis_title="", yaxis_title="")
        return fig


