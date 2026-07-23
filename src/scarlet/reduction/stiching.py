#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raccord automatique de deux courbes SAS sans correction explicite de dQ.

Le raccord est déterminé à partir du plateau du log-ratio :

    r(Q) = ln(I_low(Q)) - ln(I_high(Q))

La convention mathématique du fit est toujours :

    I_low(Q) ~= s * I_high(Q)

Donc :

    s = exp(<r>)

Nouvelle option importante
--------------------------
L'option --scale-target permet de choisir quelle courbe est réellement
multipliée dans la sortie concaténée :

    --scale-target high
        applique s à la courbe high.
        I_high_corr = s * I_high
        I_low reste inchangée.

    --scale-target low
        applique 1/s à la courbe low.
        I_low_corr = I_low / s
        I_high reste inchangée.

Dans les deux cas, les deux courbes sont mises sur la même échelle relative.
Seule la convention d'échelle absolue de la sortie change.

Grilles disponibles
-------------------
    --grid high
    --grid low
    --grid common

Avec grid="common", une grille logarithmique commune est construite dans
l'overlap avec par défaut :

    n_common = min(n_low_overlap, n_high_overlap)

Format d'entrée attendu
-----------------------
Fichiers texte avec au moins 4 colonnes :

    Q   I   dI   dQ

Les lignes commençant par # sont ignorées.
Les lignes contenant NaN ou des valeurs non positives sont ignorées.

Exemple
-------
python raccord_sans_dq_common_grid_scale_target.py \
    ludox_SM30_config_config_9_detector0.txt \
    ludox_SM30_config_config_10_detector0.txt \
    --outdir raccord_ludox \
    --low-name config_9 \
    --high-name config_10 \
    --grid common \
    --scale-target high
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import PchipInterpolator
except ImportError as exc:
    raise ImportError("Ce script nécessite scipy : pip install scipy") from exc


@dataclass(frozen=True)
class SASCurve:
    q: np.ndarray
    i: np.ndarray
    di: np.ndarray
    dq: np.ndarray
    name: str = "curve"

    @classmethod
    def from_txt(cls, path: str | Path, *, name: str | None = None) -> "SASCurve":
        path = Path(path)
        data = np.loadtxt(path, comments="#")

        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"{path} doit contenir au moins 4 colonnes : Q I dI dQ")

        data = data[:, :4]
        data = data[np.all(np.isfinite(data), axis=1)]

        valid = (
            (data[:, 0] > 0)
            & (data[:, 1] > 0)
            & (data[:, 2] > 0)
            & (data[:, 3] > 0)
        )
        data = data[valid]

        if data.shape[0] < 2:
            raise ValueError(f"Pas assez de points valides dans {path}")

        data = data[np.argsort(data[:, 0])]

        return cls(
            q=data[:, 0],
            i=data[:, 1],
            di=data[:, 2],
            dq=data[:, 3],
            name=name or path.stem,
        )

    def scaled(self, factor: float, *, name: str | None = None) -> "SASCurve":
        return SASCurve(
            q=self.q.copy(),
            i=factor * self.i,
            di=factor * self.di,
            dq=self.dq.copy(),
            name=name or f"{self.name}_scaled",
        )


@dataclass(frozen=True)
class OverlapRatio:
    q: np.ndarray
    log_ratio: np.ndarray
    log_ratio_error: np.ndarray
    low_i: np.ndarray
    low_di: np.ndarray
    low_dq: np.ndarray
    high_i: np.ndarray
    high_di: np.ndarray
    high_dq: np.ndarray
    grid: str


@dataclass(frozen=True)
class WindowFit:
    i0: int
    i1: int
    n_points: int
    q_min: float
    q_max: float
    log_scale_high_to_low: float
    log_scale_error: float
    scale_high_to_low: float
    scale_error_high_to_low: float
    chi2_red_flatness: float
    log_ratio_slope: float
    slope_error: float
    slope_z: float
    rho_resolution: float
    log_width: float
    score: float


@dataclass(frozen=True)
class StitchResult:
    method: str
    grid: str
    scale_target: str
    scale_high_to_low: float
    scale_error_high_to_low: float
    applied_scale_to_low: float
    applied_scale_to_high: float
    fit_q_min: float
    fit_q_max: float
    keep_q_min: float
    keep_q_max: float
    chi2_red_flatness: float
    log_ratio_slope: float
    slope_z: float
    rho_resolution: float
    score: float
    stitched_curve: SASCurve
    origin_config: np.ndarray
    best_window: WindowFit


def interpolate_log_intensity(q_source: np.ndarray, i_source: np.ndarray, q_eval: np.ndarray) -> np.ndarray:
    valid = (
        np.isfinite(q_source)
        & np.isfinite(i_source)
        & (q_source > 0)
        & (i_source > 0)
    )
    interpolator = PchipInterpolator(
        np.log(q_source[valid]),
        np.log(i_source[valid]),
        extrapolate=False,
    )
    return np.exp(interpolator(np.log(q_eval)))


def interpolate_linear(q_source: np.ndarray, y_source: np.ndarray, q_eval: np.ndarray) -> np.ndarray:
    valid = np.isfinite(q_source) & np.isfinite(y_source)
    interpolator = PchipInterpolator(
        q_source[valid],
        y_source[valid],
        extrapolate=False,
    )
    return interpolator(q_eval)


def overlap_bounds(
    low: SASCurve,
    high: SASCurve,
    q_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    q_min = max(float(np.min(low.q)), float(np.min(high.q)))
    q_max = min(float(np.max(low.q)), float(np.max(high.q)))

    if q_range is not None:
        q_min = max(q_min, float(q_range[0]))
        q_max = min(q_max, float(q_range[1]))

    if q_min >= q_max:
        raise ValueError("Pas de recouvrement en Q entre les deux courbes")

    return q_min, q_max


def build_evaluation_grid(
    low: SASCurve,
    high: SASCurve,
    *,
    q_min: float,
    q_max: float,
    grid: str = "common",
    n_common: int | None = None,
) -> np.ndarray:
    if grid == "low":
        q = low.q[(low.q >= q_min) & (low.q <= q_max)]

    elif grid == "high":
        q = high.q[(high.q >= q_min) & (high.q <= q_max)]

    elif grid == "common":
        n_low = int(np.count_nonzero((low.q >= q_min) & (low.q <= q_max)))
        n_high = int(np.count_nonzero((high.q >= q_min) & (high.q <= q_max)))

        if n_common is None:
            n_common = min(n_low, n_high)

        if n_common < 2:
            raise ValueError("Pas assez de points pour grid='common'")

        q = np.exp(np.linspace(np.log(q_min), np.log(q_max), int(n_common)))

    else:
        raise ValueError("grid doit valoir 'low', 'high' ou 'common'")

    q = np.asarray(q, dtype=float)
    q = q[np.isfinite(q) & (q > 0)]

    if q.size < 2:
        raise ValueError(f"Pas assez de points sur la grille {grid!r}")

    return q


def compute_log_ratio_on_overlap(
    low: SASCurve,
    high: SASCurve,
    *,
    q_range: tuple[float, float] | None = None,
    grid: str = "common",
    n_common: int | None = None,
) -> OverlapRatio:
    q_min, q_max = overlap_bounds(low, high, q_range=q_range)

    q = build_evaluation_grid(
        low,
        high,
        q_min=q_min,
        q_max=q_max,
        grid=grid,
        n_common=n_common,
    )

    low_i = interpolate_log_intensity(low.q, low.i, q)
    low_di = interpolate_linear(low.q, low.di, q)
    low_dq = interpolate_linear(low.q, low.dq, q)

    high_i = interpolate_log_intensity(high.q, high.i, q)
    high_di = interpolate_linear(high.q, high.di, q)
    high_dq = interpolate_linear(high.q, high.dq, q)

    valid = (
        np.isfinite(q)
        & np.isfinite(low_i)
        & np.isfinite(low_di)
        & np.isfinite(low_dq)
        & np.isfinite(high_i)
        & np.isfinite(high_di)
        & np.isfinite(high_dq)
        & (q > 0)
        & (low_i > 0)
        & (low_di > 0)
        & (low_dq > 0)
        & (high_i > 0)
        & (high_di > 0)
        & (high_dq > 0)
    )

    q = q[valid]
    low_i = low_i[valid]
    low_di = low_di[valid]
    low_dq = low_dq[valid]
    high_i = high_i[valid]
    high_di = high_di[valid]
    high_dq = high_dq[valid]

    if q.size < 2:
        raise ValueError("Pas assez de points valides dans l'overlap")

    log_ratio = np.log(low_i) - np.log(high_i)
    log_ratio_error = np.sqrt((low_di / low_i) ** 2 + (high_di / high_i) ** 2)

    return OverlapRatio(
        q=q,
        log_ratio=log_ratio,
        log_ratio_error=log_ratio_error,
        low_i=low_i,
        low_di=low_di,
        low_dq=low_dq,
        high_i=high_i,
        high_di=high_di,
        high_dq=high_dq,
        grid=grid,
    )


def weighted_constant_fit(y: np.ndarray, yerr: np.ndarray) -> tuple[float, float, float]:
    weights = 1.0 / np.maximum(yerr, 1e-12) ** 2
    level = float(np.sum(weights * y) / np.sum(weights))
    level_error = float(np.sqrt(1.0 / np.sum(weights)))

    residual = y - level
    chi2 = float(np.sum(weights * residual**2))
    chi2_red = chi2 / max(y.size - 1, 1)

    return level, level_error, chi2_red


def weighted_slope(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[float, float]:
    weights = 1.0 / np.maximum(yerr, 1e-12) ** 2
    xbar = float(np.sum(weights * x) / np.sum(weights))
    ybar = float(np.sum(weights * y) / np.sum(weights))

    dx = x - xbar
    denom = float(np.sum(weights * dx**2))

    if denom <= 0:
        return np.nan, np.inf

    slope = float(np.sum(weights * dx * (y - ybar)) / denom)
    slope_error = float(np.sqrt(1.0 / denom))

    return slope, slope_error


def scan_flat_ratio_windows(
    ratio: OverlapRatio,
    *,
    min_points: int = 8,
    min_log_width: float = 0.12,
    slope_weight: float = 0.1,
    width_weight: float = 0.05,
    resolution_weight: float = 0.0,
    rho_ref: float = 0.20,
) -> pd.DataFrame:
    q = ratio.q
    x = np.log(q)
    r = ratio.log_ratio
    dr = ratio.log_ratio_error

    n = q.size
    log_total_width = float(x[-1] - x[0])
    rows: list[dict] = []

    for i0 in range(0, n - min_points + 1):
        for i1 in range(i0 + min_points, n + 1):
            xw = x[i0:i1]
            rw = r[i0:i1]
            drw = dr[i0:i1]

            log_width = float(xw[-1] - xw[0])
            if log_width < min_log_width:
                continue

            log_scale, log_scale_error, chi2_red = weighted_constant_fit(rw, drw)
            slope, slope_error = weighted_slope(xw, rw, drw)

            slope_z = (
                abs(slope / slope_error)
                if np.isfinite(slope_error) and slope_error > 0
                else np.inf
            )

            rho = float(np.nanmedian(
                np.sqrt(
                    (ratio.low_dq[i0:i1] / q[i0:i1]) ** 2
                    + (ratio.high_dq[i0:i1] / q[i0:i1]) ** 2
                )
            ))

            score = (
                chi2_red
                + slope_weight * slope_z**2
                + width_weight * (log_total_width / log_width)
                + resolution_weight * (rho / rho_ref) ** 2
            )

            rows.append(
                {
                    "i0": int(i0),
                    "i1": int(i1),
                    "n_points": int(i1 - i0),
                    "q_min": float(q[i0]),
                    "q_max": float(q[i1 - 1]),
                    "log_scale_high_to_low": float(log_scale),
                    "log_scale_error": float(log_scale_error),
                    "scale_high_to_low": float(np.exp(log_scale)),
                    "scale_error_high_to_low": float(np.exp(log_scale) * log_scale_error),
                    "chi2_red_flatness": float(chi2_red),
                    "log_ratio_slope": float(slope),
                    "slope_error": float(slope_error),
                    "slope_z": float(slope_z),
                    "rho_resolution": float(rho),
                    "log_width": float(log_width),
                    "score": float(score),
                }
            )

    if not rows:
        raise RuntimeError("Aucune fenêtre candidate trouvée")

    return pd.DataFrame(rows)


def row_to_window_fit(row: pd.Series) -> WindowFit:
    return WindowFit(
        i0=int(row["i0"]),
        i1=int(row["i1"]),
        n_points=int(row["n_points"]),
        q_min=float(row["q_min"]),
        q_max=float(row["q_max"]),
        log_scale_high_to_low=float(row["log_scale_high_to_low"]),
        log_scale_error=float(row["log_scale_error"]),
        scale_high_to_low=float(row["scale_high_to_low"]),
        scale_error_high_to_low=float(row["scale_error_high_to_low"]),
        chi2_red_flatness=float(row["chi2_red_flatness"]),
        log_ratio_slope=float(row["log_ratio_slope"]),
        slope_error=float(row["slope_error"]),
        slope_z=float(row["slope_z"]),
        rho_resolution=float(row["rho_resolution"]),
        log_width=float(row["log_width"]),
        score=float(row["score"]),
    )


def compute_keep_overlap(
    q_fit_min: float,
    q_fit_max: float,
    *,
    keep_fraction: float = 0.25,
) -> tuple[float, float]:
    if not (0 < keep_fraction <= 1):
        raise ValueError("keep_fraction doit être dans ]0, 1]")

    log_min = np.log(q_fit_min)
    log_max = np.log(q_fit_max)
    log_mid = 0.5 * (log_min + log_max)
    half_keep = 0.5 * keep_fraction * (log_max - log_min)

    return float(np.exp(log_mid - half_keep)), float(np.exp(log_mid + half_keep))


def scales_from_target(scale_high_to_low: float, scale_target: str) -> tuple[float, float]:
    """Retourne les facteurs effectivement appliqués à low et high.

    Convention du fit :
        low ~= scale_high_to_low * high

    scale_target="high":
        low reste fixe, high est multipliée par scale_high_to_low.

    scale_target="low":
        high reste fixe, low est multipliée par 1 / scale_high_to_low.
    """
    if scale_target == "high":
        return 1.0, float(scale_high_to_low)

    if scale_target == "low":
        return float(1.0 / scale_high_to_low), 1.0

    raise ValueError("scale_target doit valoir 'low' ou 'high'")


def concatenate_curves(
    low: SASCurve,
    high: SASCurve,
    *,
    scale_high_to_low: float,
    scale_target: str,
    q_keep_min: float,
    q_keep_max: float,
) -> tuple[SASCurve, np.ndarray, float, float]:
    """Concatène low et high avec choix de la courbe rescalée."""

    applied_scale_to_low, applied_scale_to_high = scales_from_target(
        scale_high_to_low,
        scale_target,
    )

    keep_low = low.q <= q_keep_max
    keep_high = high.q >= q_keep_min

    q_final = np.concatenate([low.q[keep_low], high.q[keep_high]])

    i_final = np.concatenate([
        applied_scale_to_low * low.i[keep_low],
        applied_scale_to_high * high.i[keep_high],
    ])

    di_final = np.concatenate([
        applied_scale_to_low * low.di[keep_low],
        applied_scale_to_high * high.di[keep_high],
    ])

    dq_final = np.concatenate([
        low.dq[keep_low],
        high.dq[keep_high],
    ])

    origin = np.concatenate([
        np.full(np.count_nonzero(keep_low), 0, dtype=int),
        np.full(np.count_nonzero(keep_high), 1, dtype=int),
    ])

    order = np.argsort(q_final)

    stitched = SASCurve(
        q=q_final[order],
        i=i_final[order],
        di=di_final[order],
        dq=dq_final[order],
        name=f"{low.name}_plus_{high.name}",
    )

    return stitched, origin[order], applied_scale_to_low, applied_scale_to_high


def stitch_by_flat_ratio(
    low: SASCurve,
    high: SASCurve,
    *,
    method: str,
    scale_target: str = "high",
    resolution_weight: float = 0.0,
    q_range: tuple[float, float] | None = None,
    grid: str = "common",
    n_common: int | None = None,
    min_points: int = 8,
    min_log_width: float = 0.12,
    slope_weight: float = 0.1,
    width_weight: float = 0.05,
    rho_ref: float = 0.20,
    keep_fraction: float = 0.25,
) -> tuple[StitchResult, pd.DataFrame, OverlapRatio]:
    ratio = compute_log_ratio_on_overlap(
        low,
        high,
        q_range=q_range,
        grid=grid,
        n_common=n_common,
    )

    scan = scan_flat_ratio_windows(
        ratio,
        min_points=min_points,
        min_log_width=min_log_width,
        slope_weight=slope_weight,
        width_weight=width_weight,
        resolution_weight=resolution_weight,
        rho_ref=rho_ref,
    )

    best = row_to_window_fit(scan.sort_values("score").iloc[0])

    q_keep_min, q_keep_max = compute_keep_overlap(
        best.q_min,
        best.q_max,
        keep_fraction=keep_fraction,
    )

    stitched, origin, applied_low, applied_high = concatenate_curves(
        low,
        high,
        scale_high_to_low=best.scale_high_to_low,
        scale_target=scale_target,
        q_keep_min=q_keep_min,
        q_keep_max=q_keep_max,
    )

    result = StitchResult(
        method=method,
        grid=grid,
        scale_target=scale_target,
        scale_high_to_low=best.scale_high_to_low,
        scale_error_high_to_low=best.scale_error_high_to_low,
        applied_scale_to_low=applied_low,
        applied_scale_to_high=applied_high,
        fit_q_min=best.q_min,
        fit_q_max=best.q_max,
        keep_q_min=q_keep_min,
        keep_q_max=q_keep_max,
        chi2_red_flatness=best.chi2_red_flatness,
        log_ratio_slope=best.log_ratio_slope,
        slope_z=best.slope_z,
        rho_resolution=best.rho_resolution,
        score=best.score,
        stitched_curve=stitched,
        origin_config=origin,
        best_window=best,
    )

    return result, scan, ratio


def result_to_summary_dict(result: StitchResult) -> dict:
    return {
        "method": result.method,
        "grid": result.grid,
        "scale_target": result.scale_target,
        "scale_high_to_low": result.scale_high_to_low,
        "scale_error_high_to_low": result.scale_error_high_to_low,
        "applied_scale_to_low": result.applied_scale_to_low,
        "applied_scale_to_high": result.applied_scale_to_high,
        "fit_q_min": result.fit_q_min,
        "fit_q_max": result.fit_q_max,
        "keep_q_min": result.keep_q_min,
        "keep_q_max": result.keep_q_max,
        "chi2_red_flatness": result.chi2_red_flatness,
        "log_ratio_slope": result.log_ratio_slope,
        "slope_z": result.slope_z,
        "rho_resolution": result.rho_resolution,
        "score": result.score,
        "n_points_fit": result.best_window.n_points,
    }


def compare_grid_sensitivity(
    low: SASCurve,
    high: SASCurve,
    *,
    scale_target: str,
    resolution_weight: float,
    min_points: int,
    min_log_width: float,
    slope_weight: float,
    width_weight: float,
    rho_ref: float,
    keep_fraction: float,
) -> pd.DataFrame:
    rows = []

    for grid in ["low", "high", "common"]:
        result, _, _ = stitch_by_flat_ratio(
            low,
            high,
            method=f"grid_sensitivity_{grid}",
            scale_target=scale_target,
            resolution_weight=resolution_weight,
            grid=grid,
            min_points=min_points,
            min_log_width=min_log_width,
            slope_weight=slope_weight,
            width_weight=width_weight,
            rho_ref=rho_ref,
            keep_fraction=keep_fraction,
        )
        rows.append(result_to_summary_dict(result))

    df = pd.DataFrame(rows)
    mean_scale = float(df["scale_high_to_low"].mean())
    df["relative_scale_high_to_low_deviation_from_mean"] = (
        np.abs(df["scale_high_to_low"] - mean_scale) / mean_scale
    )
    return df


def save_stitched_curve(result: StitchResult, outpath: str | Path) -> None:
    outpath = Path(outpath)

    data = np.column_stack([
        result.stitched_curve.q,
        result.stitched_curve.i,
        result.stitched_curve.di,
        result.stitched_curve.dq,
        result.origin_config,
    ])

    header = (
        "stitched curve generated by raccord_sans_dq_common_grid_scale_target.py\n"
        f"method = {result.method}\n"
        f"grid = {result.grid}\n"
        f"scale_target = {result.scale_target}\n"
        f"scale_high_to_low = {result.scale_high_to_low:.10g}\n"
        f"scale_error_high_to_low = {result.scale_error_high_to_low:.10g}\n"
        f"applied_scale_to_low = {result.applied_scale_to_low:.10g}\n"
        f"applied_scale_to_high = {result.applied_scale_to_high:.10g}\n"
        f"fit_range = {result.fit_q_min:.10g} {result.fit_q_max:.10g}\n"
        f"keep_overlap_range = {result.keep_q_min:.10g} {result.keep_q_max:.10g}\n"
        f"chi2_red_flatness = {result.chi2_red_flatness:.10g}\n"
        f"log_ratio_slope = {result.log_ratio_slope:.10g}\n"
        f"slope_z = {result.slope_z:.10g}\n"
        f"rho_resolution = {result.rho_resolution:.10g}\n"
        "columns: q I I_error q_error origin_curve\n"
        "origin_curve: 0=low, 1=high"
    )

    np.savetxt(outpath, data, header=header)


def plot_ratio_with_windows(
    ratio: OverlapRatio,
    result_no_res: StitchResult,
    result_with_res: StitchResult,
    outpath: str | Path,
) -> None:
    plt.figure(figsize=(7.2, 4.8))

    ratio_linear = np.exp(ratio.log_ratio)
    ratio_error = ratio_linear * ratio.log_ratio_error

    plt.errorbar(
        ratio.q,
        ratio_linear,
        yerr=ratio_error,
        fmt="o",
        markersize=4,
        label=f"I_low / I_high, grid={ratio.grid}",
    )

    plt.axvspan(
        result_no_res.fit_q_min,
        result_no_res.fit_q_max,
        alpha=0.18,
        label="fit sans pond. résolution",
    )

    plt.axvspan(
        result_with_res.fit_q_min,
        result_with_res.fit_q_max,
        alpha=0.18,
        label="fit avec pond. résolution",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("I_low / I_high")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_scaled_curves(
    low: SASCurve,
    high: SASCurve,
    result: StitchResult,
    outpath: str | Path,
    *,
    zoom: bool = False,
) -> None:
    low_scaled = low.scaled(result.applied_scale_to_low)
    high_scaled = high.scaled(result.applied_scale_to_high)

    plt.figure(figsize=(7.2, 4.8))

    plt.errorbar(
        low_scaled.q,
        low_scaled.i,
        yerr=low_scaled.di,
        fmt=".",
        markersize=3,
        alpha=0.45,
        label=f"{low.name}, facteur {result.applied_scale_to_low:.5g}",
    )

    plt.errorbar(
        high_scaled.q,
        high_scaled.i,
        yerr=high_scaled.di,
        fmt=".",
        markersize=3,
        alpha=0.45,
        label=f"{high.name}, facteur {result.applied_scale_to_high:.5g}",
    )

    plt.errorbar(
        result.stitched_curve.q,
        result.stitched_curve.i,
        yerr=result.stitched_curve.di,
        fmt="o",
        markersize=3,
        label="concaténée",
    )

    plt.axvspan(result.fit_q_min, result.fit_q_max, alpha=0.12, label="zone de fit")
    plt.axvspan(result.keep_q_min, result.keep_q_max, alpha=0.20, label="overlap conservé")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("I")
    plt.legend()
    plt.tight_layout()

    if zoom:
        plt.xlim(0.75 * result.fit_q_min, 1.25 * result.fit_q_max)
        mask = (
            (result.stitched_curve.q >= 0.75 * result.fit_q_min)
            & (result.stitched_curve.q <= 1.25 * result.fit_q_max)
        )
        if np.any(mask):
            ymin = np.nanmin(result.stitched_curve.i[mask])
            ymax = np.nanmax(result.stitched_curve.i[mask])
            plt.ylim(0.6 * ymin, 1.8 * ymax)

    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_resolution_relative(ratio: OverlapRatio, outpath: str | Path) -> None:
    plt.figure(figsize=(7.2, 4.8))

    plt.plot(ratio.q, ratio.low_dq / ratio.q, "o-", label="low dQ/Q")
    plt.plot(ratio.q, ratio.high_dq / ratio.q, "o-", label="high dQ/Q")
    plt.plot(
        ratio.q,
        np.sqrt((ratio.low_dq / ratio.q) ** 2 + (ratio.high_dq / ratio.q) ** 2),
        "o-",
        label="résolution combinée",
    )

    plt.xscale("log")
    plt.xlabel("Q")
    plt.ylabel("dQ / Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def run_stitching_test(
    low_path: str | Path,
    high_path: str | Path,
    *,
    outdir: str | Path,
    low_name: str,
    high_name: str,
    grid: str,
    n_common: int | None,
    scale_target: str,
    min_points: int,
    min_log_width: float,
    slope_weight: float,
    width_weight: float,
    resolution_weight: float,
    rho_ref: float,
    keep_fraction: float,
    compare_grids: bool,
) -> tuple[StitchResult, StitchResult]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    low = SASCurve.from_txt(low_path, name=low_name)
    high = SASCurve.from_txt(high_path, name=high_name)

    result_no_res, scan_no_res, ratio = stitch_by_flat_ratio(
        low,
        high,
        method="without_resolution_weight",
        scale_target=scale_target,
        resolution_weight=0.0,
        grid=grid,
        n_common=n_common,
        min_points=min_points,
        min_log_width=min_log_width,
        slope_weight=slope_weight,
        width_weight=width_weight,
        rho_ref=rho_ref,
        keep_fraction=keep_fraction,
    )

    result_with_res, scan_with_res, ratio = stitch_by_flat_ratio(
        low,
        high,
        method="with_resolution_weight",
        scale_target=scale_target,
        resolution_weight=resolution_weight,
        grid=grid,
        n_common=n_common,
        min_points=min_points,
        min_log_width=min_log_width,
        slope_weight=slope_weight,
        width_weight=width_weight,
        rho_ref=rho_ref,
        keep_fraction=keep_fraction,
    )

    pd.DataFrame([
        result_to_summary_dict(result_no_res),
        result_to_summary_dict(result_with_res),
    ]).to_csv(outdir / "stitching_summary.csv", index=False)

    scan_no_res.to_csv(outdir / "scan_without_resolution_weight.csv", index=False)
    scan_with_res.to_csv(outdir / "scan_with_resolution_weight.csv", index=False)

    if compare_grids:
        compare_grid_sensitivity(
            low,
            high,
            scale_target=scale_target,
            resolution_weight=0.0,
            min_points=min_points,
            min_log_width=min_log_width,
            slope_weight=slope_weight,
            width_weight=width_weight,
            rho_ref=rho_ref,
            keep_fraction=keep_fraction,
        ).to_csv(outdir / "grid_sensitivity_without_resolution_weight.csv", index=False)

        compare_grid_sensitivity(
            low,
            high,
            scale_target=scale_target,
            resolution_weight=resolution_weight,
            min_points=min_points,
            min_log_width=min_log_width,
            slope_weight=slope_weight,
            width_weight=width_weight,
            rho_ref=rho_ref,
            keep_fraction=keep_fraction,
        ).to_csv(outdir / "grid_sensitivity_with_resolution_weight.csv", index=False)

    save_stitched_curve(result_no_res, outdir / "stitched_without_resolution_weight.txt")
    save_stitched_curve(result_with_res, outdir / "stitched_with_resolution_weight.txt")

    plot_ratio_with_windows(
        ratio,
        result_no_res,
        result_with_res,
        outdir / "ratio_selected_windows.png",
    )

    plot_resolution_relative(ratio, outdir / "resolution_relative_overlap.png")

    plot_scaled_curves(
        low,
        high,
        result_no_res,
        outdir / "stitched_curve_without_resolution_weight.png",
        zoom=False,
    )
    plot_scaled_curves(
        low,
        high,
        result_no_res,
        outdir / "stitched_curve_without_resolution_weight_zoom.png",
        zoom=True,
    )
    plot_scaled_curves(
        low,
        high,
        result_with_res,
        outdir / "stitched_curve_with_resolution_weight.png",
        zoom=False,
    )
    plot_scaled_curves(
        low,
        high,
        result_with_res,
        outdir / "stitched_curve_with_resolution_weight_zoom.png",
        zoom=True,
    )

    with open(outdir / "stitching_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "without_resolution_weight": result_to_summary_dict(result_no_res),
                "with_resolution_weight": result_to_summary_dict(result_with_res),
                "parameters": {
                    "grid": grid,
                    "n_common": n_common,
                    "scale_target": scale_target,
                    "min_points": min_points,
                    "min_log_width": min_log_width,
                    "slope_weight": slope_weight,
                    "width_weight": width_weight,
                    "resolution_weight": resolution_weight,
                    "rho_ref": rho_ref,
                    "keep_fraction": keep_fraction,
                    "compare_grids": compare_grids,
                },
            },
            f,
            indent=2,
        )

    return result_no_res, result_with_res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raccord automatique de deux courbes SAS par plateau du log-ratio."
    )

    parser.add_argument("low_curve", type=str)
    parser.add_argument("high_curve", type=str)

    parser.add_argument("--outdir", type=str, default="raccord_output")
    parser.add_argument("--low-name", type=str, default="low")
    parser.add_argument("--high-name", type=str, default="high")

    parser.add_argument(
        "--grid",
        type=str,
        default="common",
        choices=["low", "high", "common"],
        help="Grille utilisée pour évaluer le ratio.",
    )

    parser.add_argument(
        "--n-common",
        type=int,
        default=None,
        help="Nombre de points pour grid='common'. Défaut: min(n_low, n_high).",
    )

    parser.add_argument(
        "--scale-target",
        type=str,
        default="high",
        choices=["low", "high"],
        help=(
            "Courbe à laquelle appliquer le facteur dans la sortie. "
            "'high' applique s à high ; 'low' applique 1/s à low."
        ),
    )

    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-log-width", type=float, default=0.12)
    parser.add_argument("--slope-weight", type=float, default=0.1)
    parser.add_argument("--width-weight", type=float, default=0.05)
    parser.add_argument("--resolution-weight", type=float, default=1.0)
    parser.add_argument("--rho-ref", type=float, default=0.20)
    parser.add_argument("--keep-fraction", type=float, default=0.25)

    parser.add_argument(
        "--no-grid-comparison",
        action="store_true",
        help="Désactive la comparaison automatique grid=low/high/common.",
    )

    args = parser.parse_args()

    no_res, with_res = run_stitching_test(
        args.low_curve,
        args.high_curve,
        outdir=args.outdir,
        low_name=args.low_name,
        high_name=args.high_name,
        grid=args.grid,
        n_common=args.n_common,
        scale_target=args.scale_target,
        min_points=args.min_points,
        min_log_width=args.min_log_width,
        slope_weight=args.slope_weight,
        width_weight=args.width_weight,
        resolution_weight=args.resolution_weight,
        rho_ref=args.rho_ref,
        keep_fraction=args.keep_fraction,
        compare_grids=not args.no_grid_comparison,
    )

    print("\n=== Résumé raccord ===")
    for result in [no_res, with_res]:
        print(f"\nMéthode : {result.method}")
        print(f"  grid                     : {result.grid}")
        print(f"  scale_target             : {result.scale_target}")
        print(f"  s high -> low            : {result.scale_high_to_low:.8g} ± {result.scale_error_high_to_low:.3g}")
        print(f"  applied_scale_to_low     : {result.applied_scale_to_low:.8g}")
        print(f"  applied_scale_to_high    : {result.applied_scale_to_high:.8g}")
        print(f"  zone de fit              : {result.fit_q_min:.6g} -> {result.fit_q_max:.6g}")
        print(f"  overlap conservé         : {result.keep_q_min:.6g} -> {result.keep_q_max:.6g}")
        print(f"  chi2_red platitude       : {result.chi2_red_flatness:.4g}")
        print(f"  slope_z                  : {result.slope_z:.4g}")
        print(f"  rho résolution           : {result.rho_resolution:.4g}")
        print(f"  score                    : {result.score:.4g}")


if __name__ == "__main__":
    main()