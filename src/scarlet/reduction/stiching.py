#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raccord automatique multi-segments SAS/SANS/SAXS.

Ce script généralise le raccord deux-courbes à un cas plus réaliste :

    plusieurs configurations
    plusieurs détecteurs par configuration
    plusieurs courbes candidates par configuration

L'idée est de traiter chaque courbe détecteur comme un segment candidat :

    CurveSegment(config_id, detector_id, q, I, dI, dQ)

Puis :
    1. calculer la qualité intrinsèque de chaque segment ;
    2. tester les raccords pair-à-pair dans les zones de recouvrement ;
    3. construire une chaîne de segments utiles ;
    4. rejeter les segments redondants, trop bruités ou sans apport en Q ;
    5. produire une courbe concaténée finale et des diagnostics.

Convention du facteur entre deux segments
-----------------------------------------
Pour une transition A -> B, on ajuste toujours :

    I_A(Q) ~= s * I_B(Q)

Donc :

    s = scale_b_to_a

Dans une chaîne, si A a déjà un facteur global G_A,
alors le facteur global de B est :

    G_B = G_A * s

Par défaut, la courbe finale est exprimée dans l'échelle du premier segment
retenu. On peut ensuite choisir une référence d'échelle avec :

    --reference-segment c10d0

ou :

    --reference-config config_10

Dans ce cas, tous les facteurs globaux sont renormalisés pour que le segment
de référence ait un facteur global égal à 1.

Format des fichiers
-------------------
Fichiers texte à au moins 4 colonnes :

    Q   I   dI   dQ

Les lignes commençant par # sont ignorées.

Exemple
-------
python raccord_multisegment_sans_dq.py \
    ludox_SM30_config_config_9_detector0.txt \
    ludox_SM30_config_config_9_detector1.txt \
    ludox_SM30_config_config_9_detector2.txt \
    ludox_SM30_config_config_10_detector0.txt \
    ludox_SM30_config_config_10_detector1.txt \
    ludox_SM30_config_config_10_detector2.txt \
    --outdir raccord_multi \
    --grid common
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import json
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scarlet.io.nexus_reader import read_processed_data

try:
    from scipy.interpolate import PchipInterpolator
except ImportError as exc:
    raise ImportError("Ce script nécessite scipy : pip install scipy") from exc


# =============================================================================
# Structures
# =============================================================================

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

        return cls.from_array(data, name=name or path.stem, source=str(path))

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        name: str,
        source: str = "array",
    ) -> "SASCurve":
        data = np.asarray(data, dtype=float)

        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"{source} doit contenir au moins 4 colonnes: Q I dI dQ")

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
            raise ValueError(f"Pas assez de points valides dans {source}")

        data = data[np.argsort(data[:, 0])]

        return cls(
            q=data[:, 0],
            i=data[:, 1],
            di=data[:, 2],
            dq=data[:, 3],
            name=name,
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
class CurveSegment:
    name: str
    curve: SASCurve
    config_id: str
    detector_id: str
    path: str | None = None

    @property
    def q_min(self) -> float:
        return float(np.min(self.curve.q))

    @property
    def q_max(self) -> float:
        return float(np.max(self.curve.q))

    @property
    def q_center(self) -> float:
        return float(np.sqrt(self.q_min * self.q_max))

    @property
    def n_points(self) -> int:
        return int(self.curve.q.size)

    @property
    def rel_error(self) -> np.ndarray:
        return self.curve.di / self.curve.i

    @property
    def rel_resolution(self) -> np.ndarray:
        return self.curve.dq / self.curve.q

    def quality_score(self, *, resolution_weight: float = 1.0) -> float:
        """Score de qualité intrinsèque.

        Plus petit = meilleur.
        """
        return float(np.nanmedian(
            self.rel_error**2
            + resolution_weight * self.rel_resolution**2
        ))

    def quality_summary(self, *, resolution_weight: float = 1.0) -> dict:
        return {
            "segment": self.name,
            "config_id": self.config_id,
            "detector_id": self.detector_id,
            "n_points": self.n_points,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "q_center": self.q_center,
            "median_dI_over_I": float(np.nanmedian(self.rel_error)),
            "median_dQ_over_Q": float(np.nanmedian(self.rel_resolution)),
            "fraction_dI_over_I_gt_20pct": float(np.mean(self.rel_error > 0.20)),
            "quality_score": self.quality_score(resolution_weight=resolution_weight),
        }


@dataclass(frozen=True)
class PairFit:
    segment_a: str
    segment_b: str
    scale_b_to_a: float
    scale_error: float
    q_overlap_min: float
    q_overlap_max: float
    n_overlap: int
    fit_q_min: float
    fit_q_max: float
    n_fit: int
    chi2_red: float
    slope_z: float
    log_ratio_slope: float
    rho_resolution: float
    log_width: float
    score: float
    new_log_coverage: float
    accepted_flatness: bool
    accepted_transition: bool
    reject_reason: str


@dataclass(frozen=True)
class SelectedSegment:
    segment: CurveSegment
    global_scale: float
    transition_from_previous: PairFit | None = None
    reason: str = ""


@dataclass(frozen=True)
class MultiStitchResult:
    selected: list[SelectedSegment]
    rejected: pd.DataFrame
    pair_fits: pd.DataFrame
    final_curve: SASCurve
    origin_segment_id: np.ndarray
    origin_map: dict[int, str]


# =============================================================================
# Utilitaires d'interpolation et de fit
# =============================================================================

def infer_ids_from_filename(path: str | Path) -> tuple[str, str, str]:
    """Infère name, config_id et detector_id depuis un nom de fichier.

    Fonction volontairement tolérante. Exemple :
        ludox_SM30_config_config_9_detector2.txt
        -> name='config9_detector2', config_id='config_9', detector_id='detector2'
    """
    stem = Path(path).stem

    config_match = re.search(r"config[_-]?(\d+)", stem)
    detector_match = re.search(r"detector[_-]?(\d+)", stem)

    config_id = f"config_{config_match.group(1)}" if config_match else "config_unknown"
    detector_id = f"detector{detector_match.group(1)}" if detector_match else "detectorunknown"

    compact_config = config_id.replace("config_", "c")
    compact_detector = detector_id.replace("detector", "d")
    name = f"{compact_config}{compact_detector}"

    return name, config_id, detector_id


def load_segments(paths: list[str | Path], *, names: list[str] | None = None) -> list[CurveSegment]:
    segments: list[CurveSegment] = []

    if names is not None and len(names) != len(paths):
        raise ValueError("--names doit contenir autant d'entrées que de fichiers")

    for idx, path in enumerate(paths):
        if names is None:
            name, config_id, detector_id = infer_ids_from_filename(path)
        else:
            name = names[idx]
            config_id = f"config_unknown_{idx}"
            detector_id = f"detectorunknown_{idx}"

        curve = SASCurve.from_txt(path, name=name)
        segments.append(
            CurveSegment(
                name=name,
                curve=curve,
                config_id=config_id,
                detector_id=detector_id,
                path=str(path),
            )
        )

    return sorted(segments, key=lambda s: (s.q_min, s.q_center))


def load_segment_from_nexus(
    path: str | Path,
    *,
    names: list[str] | None = None,
    config_id: str = "unknown",
    entry_name: str = "processed",
) -> list[CurveSegment]:
    path = Path(path)
    processed_data = read_processed_data(path, entry_name=entry_name)
    if processed_data is None:
        return []

    available = [(detector_index, data) for detector_index, data in enumerate(processed_data) if data is not None]
    if names is not None and len(names) != len(available):
        raise ValueError("--names doit contenir autant d'entrées que de détecteurs disponibles")

    normalized_config_id = str(config_id).strip() or "unknown"
    segments: list[CurveSegment] = []
    for idx, (detector_index, detector_data) in enumerate(available):
        detector_id = f"detector{detector_index}"
        if names is None:
            if normalized_config_id.startswith("config_"):
                segment_name = f"{normalized_config_id.replace('config_', 'c')}d{detector_index}"
            else:
                segment_name = f"{normalized_config_id}_{detector_id}"
        else:
            segment_name = names[idx]

        curve = SASCurve.from_array(
            detector_data,
            name=segment_name,
            source=f"{path}:{entry_name}/data{detector_index}",
        )
        segments.append(
            CurveSegment(
                name=segment_name,
                curve=curve,
                config_id=normalized_config_id,
                detector_id=detector_id,
                path=str(path),
            )
        )

    return sorted(segments, key=lambda s: (s.q_min, s.q_center))


def interpolate_log_intensity(q_source: np.ndarray, i_source: np.ndarray, q_eval: np.ndarray) -> np.ndarray:
    valid = (
        np.isfinite(q_source)
        & np.isfinite(i_source)
        & (q_source > 0)
        & (i_source > 0)
    )

    return np.exp(
        PchipInterpolator(
            np.log(q_source[valid]),
            np.log(i_source[valid]),
            extrapolate=False,
        )(np.log(q_eval))
    )


def interpolate_linear(q_source: np.ndarray, y_source: np.ndarray, q_eval: np.ndarray) -> np.ndarray:
    valid = np.isfinite(q_source) & np.isfinite(y_source)

    return PchipInterpolator(
        q_source[valid],
        y_source[valid],
        extrapolate=False,
    )(q_eval)


def build_overlap_grid(
    a: CurveSegment,
    b: CurveSegment,
    *,
    grid: str = "common",
    n_common: int | None = None,
) -> np.ndarray | None:
    q_min = max(a.q_min, b.q_min)
    q_max = min(a.q_max, b.q_max)

    if q_max <= q_min:
        return None

    if grid == "a":
        q = a.curve.q[(a.curve.q >= q_min) & (a.curve.q <= q_max)]

    elif grid == "b":
        q = b.curve.q[(b.curve.q >= q_min) & (b.curve.q <= q_max)]

    elif grid == "common":
        n_a = int(np.count_nonzero((a.curve.q >= q_min) & (a.curve.q <= q_max)))
        n_b = int(np.count_nonzero((b.curve.q >= q_min) & (b.curve.q <= q_max)))

        if n_common is None:
            n_common = min(n_a, n_b)

        if n_common < 2:
            return None

        q = np.exp(np.linspace(np.log(q_min), np.log(q_max), int(n_common)))

    else:
        raise ValueError("grid doit valoir 'common', 'a' ou 'b'")

    q = q[np.isfinite(q) & (q > 0)]
    return q if q.size >= 2 else None


def weighted_constant_fit(y: np.ndarray, yerr: np.ndarray) -> tuple[float, float, float]:
    weights = 1.0 / np.maximum(yerr, 1e-12) ** 2
    level = float(np.sum(weights * y) / np.sum(weights))
    level_error = float(np.sqrt(1.0 / np.sum(weights)))
    chi2_red = float(np.sum(weights * (y - level) ** 2) / max(y.size - 1, 1))
    return level, level_error, chi2_red


def weighted_slope(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[float, float]:
    weights = 1.0 / np.maximum(yerr, 1e-12) ** 2

    x0 = np.sum(weights * x) / np.sum(weights)
    y0 = np.sum(weights * y) / np.sum(weights)

    dx = x - x0
    denom = np.sum(weights * dx**2)

    if denom <= 0:
        return np.nan, np.inf

    slope = float(np.sum(weights * dx * (y - y0)) / denom)
    slope_error = float(np.sqrt(1.0 / denom))
    return slope, slope_error


def best_pair_fit(
    a: CurveSegment,
    b: CurveSegment,
    *,
    grid: str = "common",
    n_common: int | None = None,
    min_points: int = 8,
    min_log_width: float = 0.10,
    slope_weight: float = 0.10,
    width_weight: float = 0.05,
    resolution_weight: float = 0.50,
    rho_ref: float = 0.20,
    max_chi2_red: float = 3.0,
    max_slope_z: float = 2.5,
    min_new_log_coverage: float = 0.05,
) -> PairFit | None:
    """Teste le meilleur raccord possible A -> B.

    Convention :
        I_A ~= s * I_B

    Retourne le meilleur plateau du log-ratio, ou None si overlap insuffisant.
    """

    q = build_overlap_grid(a, b, grid=grid, n_common=n_common)
    if q is None:
        return None

    ca = a.curve
    cb = b.curve

    ai = interpolate_log_intensity(ca.q, ca.i, q)
    adi = interpolate_linear(ca.q, ca.di, q)
    adq = interpolate_linear(ca.q, ca.dq, q)

    bi = interpolate_log_intensity(cb.q, cb.i, q)
    bdi = interpolate_linear(cb.q, cb.di, q)
    bdq = interpolate_linear(cb.q, cb.dq, q)

    valid = (
        np.isfinite(ai)
        & np.isfinite(adi)
        & np.isfinite(adq)
        & np.isfinite(bi)
        & np.isfinite(bdi)
        & np.isfinite(bdq)
        & (ai > 0)
        & (adi > 0)
        & (adq > 0)
        & (bi > 0)
        & (bdi > 0)
        & (bdq > 0)
    )

    q = q[valid]
    ai = ai[valid]
    adi = adi[valid]
    adq = adq[valid]
    bi = bi[valid]
    bdi = bdi[valid]
    bdq = bdq[valid]

    if q.size < min_points:
        return None

    log_ratio = np.log(ai) - np.log(bi)
    log_ratio_error = np.sqrt((adi / ai) ** 2 + (bdi / bi) ** 2)
    rho_resolution = np.sqrt((adq / q) ** 2 + (bdq / q) ** 2)

    x = np.log(q)
    total_log_width = float(x[-1] - x[0])

    best: dict | None = None

    for i0 in range(0, q.size - min_points + 1):
        for i1 in range(i0 + min_points, q.size + 1):
            log_width = float(x[i1 - 1] - x[i0])
            if log_width < min_log_width:
                continue

            level, level_error, chi2_red = weighted_constant_fit(
                log_ratio[i0:i1],
                log_ratio_error[i0:i1],
            )

            slope, slope_error = weighted_slope(
                x[i0:i1],
                log_ratio[i0:i1],
                log_ratio_error[i0:i1],
            )

            slope_z = (
                abs(slope / slope_error)
                if np.isfinite(slope_error) and slope_error > 0
                else np.inf
            )

            rho_med = float(np.nanmedian(rho_resolution[i0:i1]))

            score = (
                chi2_red
                + slope_weight * slope_z**2
                + width_weight * (total_log_width / log_width)
                + resolution_weight * (rho_med / rho_ref) ** 2
            )

            candidate = {
                "scale_b_to_a": float(np.exp(level)),
                "scale_error": float(np.exp(level) * level_error),
                "fit_q_min": float(q[i0]),
                "fit_q_max": float(q[i1 - 1]),
                "n_fit": int(i1 - i0),
                "chi2_red": float(chi2_red),
                "slope_z": float(slope_z),
                "log_ratio_slope": float(slope),
                "rho_resolution": float(rho_med),
                "log_width": float(log_width),
                "score": float(score),
            }

            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        return None

    new_log_coverage = float(max(0.0, np.log(b.q_max / a.q_max)))
    accepted_flatness = bool(best["chi2_red"] < max_chi2_red and best["slope_z"] < max_slope_z)
    accepted_transition = bool(accepted_flatness and new_log_coverage >= min_new_log_coverage)

    if not accepted_flatness:
        reject_reason = "rejected_bad_overlap"
    elif new_log_coverage < min_new_log_coverage:
        reject_reason = "rejected_no_new_q_coverage"
    else:
        reject_reason = ""

    return PairFit(
        segment_a=a.name,
        segment_b=b.name,
        scale_b_to_a=best["scale_b_to_a"],
        scale_error=best["scale_error"],
        q_overlap_min=float(q[0]),
        q_overlap_max=float(q[-1]),
        n_overlap=int(q.size),
        fit_q_min=best["fit_q_min"],
        fit_q_max=best["fit_q_max"],
        n_fit=best["n_fit"],
        chi2_red=best["chi2_red"],
        slope_z=best["slope_z"],
        log_ratio_slope=best["log_ratio_slope"],
        rho_resolution=best["rho_resolution"],
        log_width=best["log_width"],
        score=best["score"],
        new_log_coverage=new_log_coverage,
        accepted_flatness=accepted_flatness,
        accepted_transition=accepted_transition,
        reject_reason=reject_reason,
    )


# =============================================================================
# Sélection de segments
# =============================================================================

def pairfit_to_dict(fit: PairFit) -> dict:
    return {
        "segment_a": fit.segment_a,
        "segment_b": fit.segment_b,
        "scale_b_to_a": fit.scale_b_to_a,
        "scale_error": fit.scale_error,
        "q_overlap_min": fit.q_overlap_min,
        "q_overlap_max": fit.q_overlap_max,
        "n_overlap": fit.n_overlap,
        "fit_q_min": fit.fit_q_min,
        "fit_q_max": fit.fit_q_max,
        "n_fit": fit.n_fit,
        "chi2_red": fit.chi2_red,
        "slope_z": fit.slope_z,
        "log_ratio_slope": fit.log_ratio_slope,
        "rho_resolution": fit.rho_resolution,
        "log_width": fit.log_width,
        "score": fit.score,
        "new_log_coverage": fit.new_log_coverage,
        "accepted_flatness": fit.accepted_flatness,
        "accepted_transition": fit.accepted_transition,
        "reject_reason": fit.reject_reason,
    }


def compute_all_pair_fits(
    segments: list[CurveSegment],
    **fit_kwargs,
) -> pd.DataFrame:
    rows: list[dict] = []

    for a, b in permutations(segments, 2):
        if b.q_max <= a.q_max * 1.02:
            continue

        fit = best_pair_fit(a, b, **fit_kwargs)
        if fit is None:
            continue

        rows.append(pairfit_to_dict(fit))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def choose_start_segment(
    segments: list[CurveSegment],
    *,
    policy: str = "lowest_q",
) -> CurveSegment:
    if policy == "lowest_q":
        return min(segments, key=lambda s: s.q_min)

    if policy == "best_low_q_quality":
        q_min_global = min(s.q_min for s in segments)
        candidates = [s for s in segments if s.q_min <= 1.2 * q_min_global]
        return min(candidates, key=lambda s: s.quality_score())

    raise ValueError("start_policy doit valoir 'lowest_q' ou 'best_low_q_quality'")


def stitch_segments_greedy(
    segments: list[CurveSegment],
    *,
    start_policy: str = "lowest_q",
    grid: str = "common",
    n_common: int | None = None,
    min_points: int = 8,
    min_log_width: float = 0.10,
    slope_weight: float = 0.10,
    width_weight: float = 0.05,
    resolution_weight: float = 0.50,
    rho_ref: float = 0.20,
    max_chi2_red: float = 3.0,
    max_slope_z: float = 2.5,
    min_new_log_coverage: float = 0.05,
    segment_quality_weight: float = 4.0,
    new_coverage_weight: float = 1.5,
    keep_fraction: float = 0.25,
) -> MultiStitchResult:
    """Sélectionne une chaîne de segments utiles par algorithme glouton.

    À chaque étape, on cherche le segment qui :
        - se raccorde correctement au segment courant ;
        - apporte une nouvelle gamme de Q ;
        - a une bonne qualité intrinsèque.

    Ce n'est pas encore une optimisation globale de type graphe/Dijkstra,
    mais c'est robuste et simple pour une première intégration SCARLET.
    """

    segments = sorted(segments, key=lambda s: (s.q_min, s.q_center))
    seg_by_name = {s.name: s for s in segments}

    pair_fits_df = compute_all_pair_fits(
        segments,
        grid=grid,
        n_common=n_common,
        min_points=min_points,
        min_log_width=min_log_width,
        slope_weight=slope_weight,
        width_weight=width_weight,
        resolution_weight=resolution_weight,
        rho_ref=rho_ref,
        max_chi2_red=max_chi2_red,
        max_slope_z=max_slope_z,
        min_new_log_coverage=min_new_log_coverage,
    )

    start = choose_start_segment(segments, policy=start_policy)

    selected: list[SelectedSegment] = [
        SelectedSegment(segment=start, global_scale=1.0, reason="start")
    ]
    used = {start.name}
    current = start

    for _ in range(len(segments) - 1):
        candidates: list[tuple[float, CurveSegment, PairFit]] = []

        for b in segments:
            if b.name in used:
                continue

            if b.q_max <= current.q_max * (1.0 + 1e-12):
                continue

            fit = best_pair_fit(
                current,
                b,
                grid=grid,
                n_common=n_common,
                min_points=min_points,
                min_log_width=min_log_width,
                slope_weight=slope_weight,
                width_weight=width_weight,
                resolution_weight=resolution_weight,
                rho_ref=rho_ref,
                max_chi2_red=max_chi2_red,
                max_slope_z=max_slope_z,
                min_new_log_coverage=min_new_log_coverage,
            )

            if fit is None or not fit.accepted_transition:
                continue

            segment_quality = b.quality_score(resolution_weight=resolution_weight)

            transition_cost = (
                fit.score
                + segment_quality_weight * segment_quality
                - new_coverage_weight * fit.new_log_coverage
            )

            candidates.append((transition_cost, b, fit))

        if not candidates:
            break

        _, chosen_segment, chosen_fit = min(candidates, key=lambda x: x[0])

        previous_scale = selected[-1].global_scale
        chosen_scale = previous_scale * chosen_fit.scale_b_to_a

        selected.append(
            SelectedSegment(
                segment=chosen_segment,
                global_scale=chosen_scale,
                transition_from_previous=chosen_fit,
                reason="greedy_best_transition",
            )
        )

        used.add(chosen_segment.name)
        current = chosen_segment

    selected_names = [s.segment.name for s in selected]
    max_selected_q = max(s.segment.q_max for s in selected)

    rejected_rows = []
    for segment in segments:
        if segment.name in selected_names:
            continue

        if segment.q_max <= max_selected_q * 1.02:
            reason = "rejected_redundant_or_no_new_q_coverage"
        else:
            reason = "rejected_no_valid_transition"

        rejected_rows.append({
            "segment": segment.name,
            "config_id": segment.config_id,
            "detector_id": segment.detector_id,
            "q_min": segment.q_min,
            "q_max": segment.q_max,
            "quality_score": segment.quality_score(resolution_weight=resolution_weight),
            "reason": reason,
        })

    rejected = pd.DataFrame(rejected_rows)

    final_curve, origin_ids, origin_map = build_final_curve(
        selected,
        keep_fraction=keep_fraction,
    )

    return MultiStitchResult(
        selected=selected,
        rejected=rejected,
        pair_fits=pair_fits_df,
        final_curve=final_curve,
        origin_segment_id=origin_ids,
        origin_map=origin_map,
    )


def overlap_keep_range(fit: PairFit, *, keep_fraction: float) -> tuple[float, float]:
    log_min = np.log(fit.fit_q_min)
    log_max = np.log(fit.fit_q_max)
    log_mid = 0.5 * (log_min + log_max)
    half = 0.5 * keep_fraction * (log_max - log_min)
    return float(np.exp(log_mid - half)), float(np.exp(log_mid + half))


def build_final_curve(
    selected: list[SelectedSegment],
    *,
    keep_fraction: float = 0.25,
) -> tuple[SASCurve, np.ndarray, dict[int, str]]:
    pieces = []
    origin_names = []

    for idx, item in enumerate(selected):
        seg = item.segment
        curve = seg.curve
        scale = item.global_scale

        q_low = -np.inf
        q_high = np.inf

        # Début du segment : depuis l'overlap conservé avec le précédent
        if idx > 0:
            previous_fit = item.transition_from_previous
            assert previous_fit is not None
            q_keep_min, _ = overlap_keep_range(previous_fit, keep_fraction=keep_fraction)
            q_low = q_keep_min

        # Fin du segment : jusqu'à l'overlap conservé avec le suivant
        if idx < len(selected) - 1:
            next_fit = selected[idx + 1].transition_from_previous
            assert next_fit is not None
            _, q_keep_max = overlap_keep_range(next_fit, keep_fraction=keep_fraction)
            q_high = q_keep_max

        mask = (curve.q >= q_low) & (curve.q <= q_high)

        piece = np.column_stack([
            curve.q[mask],
            scale * curve.i[mask],
            scale * curve.di[mask],
            curve.dq[mask],
        ])

        pieces.append(piece)
        origin_names.extend([seg.name] * piece.shape[0])

    if not pieces:
        raise RuntimeError("Aucun segment sélectionné")

    data = np.vstack(pieces)
    origin_names_array = np.asarray(origin_names)

    order = np.argsort(data[:, 0])
    data = data[order]
    origin_names_array = origin_names_array[order]

    origin_map = {idx: item.segment.name for idx, item in enumerate(selected)}
    reverse_origin_map = {name: idx for idx, name in origin_map.items()}
    origin_ids = np.asarray([reverse_origin_map[name] for name in origin_names_array], dtype=int)

    curve = SASCurve(
        q=data[:, 0],
        i=data[:, 1],
        di=data[:, 2],
        dq=data[:, 3],
        name="stitched_multisegment",
    )

    return curve, origin_ids, origin_map



# =============================================================================
# Choix de la référence d'échelle
# =============================================================================

def choose_reference_selected_segment(
    selected: list[SelectedSegment],
    *,
    reference_segment: str | None = None,
    reference_config: str | None = None,
    reference_detector: str | None = None,
) -> SelectedSegment:
    """Choisit le segment retenu qui définit l'échelle finale.

    Parameters
    ----------
    selected:
        Liste des segments effectivement retenus dans la chaîne.

    reference_segment:
        Nom exact du segment, par exemple "c10d0".
        Prioritaire sur reference_config.

    reference_config:
        Identifiant de configuration, par exemple "config_10".
        Si plusieurs segments de cette configuration sont retenus, on prend
        celui dont la gamme en Q est la plus large.

    reference_detector:
        Optionnel. Permet de restreindre à un détecteur, par exemple
        "detector0".

    Returns
    -------
    SelectedSegment
        Le segment retenu servant de référence.

    Notes
    -----
    Cette fonction ne change pas la sélection des segments. Elle change
    uniquement l'échelle globale de la courbe finale.
    """

    if reference_segment is None and reference_config is None:
        return selected[0]

    if reference_segment is not None:
        matches = [
            item for item in selected
            if item.segment.name == reference_segment
        ]

        if not matches:
            available = ", ".join(item.segment.name for item in selected)
            raise ValueError(
                f"Le segment de référence {reference_segment!r} n'est pas dans "
                f"les segments retenus. Segments disponibles: {available}"
            )

        return matches[0]

    assert reference_config is not None

    matches = [
        item for item in selected
        if item.segment.config_id == reference_config
    ]

    if reference_detector is not None:
        matches = [
            item for item in matches
            if item.segment.detector_id == reference_detector
        ]

    if not matches:
        available = ", ".join(
            f"{item.segment.name}({item.segment.config_id}, {item.segment.detector_id})"
            for item in selected
        )
        raise ValueError(
            f"Aucun segment retenu ne correspond à reference_config={reference_config!r}"
            + (
                f" et reference_detector={reference_detector!r}."
                if reference_detector is not None
                else "."
            )
            + f" Segments retenus disponibles: {available}"
        )

    # Si plusieurs détecteurs de la configuration de référence sont retenus,
    # on prend celui qui couvre la plus grande largeur logarithmique en Q.
    return max(
        matches,
        key=lambda item: np.log(item.segment.q_max / item.segment.q_min),
    )


def rebase_result_to_reference(
    result: MultiStitchResult,
    *,
    reference_segment: str | None = None,
    reference_config: str | None = None,
    reference_detector: str | None = None,
) -> MultiStitchResult:
    """Renormalise tous les facteurs globaux vers une référence choisie.

    Si le segment de référence a initialement un facteur global G_ref, tous les
    facteurs sont divisés par G_ref :

        G_i,new = G_i / G_ref

    Ainsi, le segment de référence a :

        G_ref,new = 1

    La courbe finale est elle aussi divisée par G_ref.
    """

    ref_item = choose_reference_selected_segment(
        result.selected,
        reference_segment=reference_segment,
        reference_config=reference_config,
        reference_detector=reference_detector,
    )

    reference_scale = ref_item.global_scale

    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Le facteur global du segment de référence est invalide.")

    rebased_selected = [
        SelectedSegment(
            segment=item.segment,
            global_scale=item.global_scale / reference_scale,
            transition_from_previous=item.transition_from_previous,
            reason=(
                item.reason
                if item.segment.name != ref_item.segment.name
                else f"{item.reason}; scale_reference"
            ),
        )
        for item in result.selected
    ]

    rebased_curve = SASCurve(
        q=result.final_curve.q.copy(),
        i=result.final_curve.i / reference_scale,
        di=result.final_curve.di / reference_scale,
        dq=result.final_curve.dq.copy(),
        name=result.final_curve.name + "_rebased",
    )

    return MultiStitchResult(
        selected=rebased_selected,
        rejected=result.rejected,
        pair_fits=result.pair_fits,
        final_curve=rebased_curve,
        origin_segment_id=result.origin_segment_id,
        origin_map=result.origin_map,
    )



# =============================================================================
# Export et figures
# =============================================================================

def selected_to_dataframe(selected: list[SelectedSegment]) -> pd.DataFrame:
    rows = []

    for idx, item in enumerate(selected):
        seg = item.segment
        row = {
            "order": idx,
            "segment": seg.name,
            "config_id": seg.config_id,
            "detector_id": seg.detector_id,
            "global_scale": item.global_scale,
            "q_min": seg.q_min,
            "q_max": seg.q_max,
            "median_dI_over_I": float(np.nanmedian(seg.rel_error)),
            "median_dQ_over_Q": float(np.nanmedian(seg.rel_resolution)),
            "quality_score": seg.quality_score(),
            "reason": item.reason,
        }

        if item.transition_from_previous is not None:
            fit = item.transition_from_previous
            row.update({
                "previous_segment": fit.segment_a,
                "transition_scale_b_to_a": fit.scale_b_to_a,
                "transition_scale_error": fit.scale_error,
                "fit_q_min": fit.fit_q_min,
                "fit_q_max": fit.fit_q_max,
                "transition_chi2_red": fit.chi2_red,
                "transition_slope_z": fit.slope_z,
                "transition_rho_resolution": fit.rho_resolution,
                "transition_new_log_coverage": fit.new_log_coverage,
                "transition_score": fit.score,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    segments: list[CurveSegment],
    result: MultiStitchResult,
    *,
    outdir: str | Path,
    resolution_weight: float = 1.0,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        s.quality_summary(resolution_weight=resolution_weight)
        for s in segments
    ]).to_csv(outdir / "segments_quality_summary.csv", index=False)

    selected_df = selected_to_dataframe(result.selected)
    selected_df.to_csv(outdir / "selected_segments.csv", index=False)

    result.rejected.to_csv(outdir / "rejected_segments.csv", index=False)
    result.pair_fits.to_csv(outdir / "pairwise_overlap_fits.csv", index=False)

    final_data = np.column_stack([
        result.final_curve.q,
        result.final_curve.i,
        result.final_curve.di,
        result.final_curve.dq,
        result.origin_segment_id,
    ])

    header = (
        "q I I_error q_error origin_segment_id\n"
        + "\n".join(f"origin_segment_id {idx}: {name}" for idx, name in result.origin_map.items())
    )

    np.savetxt(outdir / "selected_segments_stitched_curve.txt", final_data, header=header)

    with open(outdir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_segments": selected_df.to_dict(orient="records"),
                "rejected_segments": result.rejected.to_dict(orient="records"),
                "origin_map": result.origin_map,
                "scale_reference_segment": [
                    item.segment.name for item in result.selected
                    if abs(item.global_scale - 1.0) < 1e-12
                ],
            },
            f,
            indent=2,
        )

    plot_raw_segments(segments, outdir / "01_raw_segments.png")
    plot_relative_error(segments, outdir / "02_relative_error.png")
    plot_relative_resolution(segments, outdir / "03_relative_resolution.png")
    plot_final_selected_segments(result, outdir / "04_selected_final_curve.png")
    plot_retained_vs_rejected(segments, result, outdir / "05_retained_vs_rejected.png")
    plot_transition_matrix(segments, result.pair_fits, outdir / "06_transition_matrix.png")


def plot_raw_segments(segments: list[CurveSegment], outpath: str | Path) -> None:
    plt.figure(figsize=(8, 5.5))
    for seg in segments:
        c = seg.curve
        plt.errorbar(c.q, c.i, yerr=c.di, fmt=".", markersize=3, alpha=0.7, label=seg.name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("I")
    plt.title("Segments bruts")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_relative_error(segments: list[CurveSegment], outpath: str | Path) -> None:
    plt.figure(figsize=(8, 5.5))
    for seg in segments:
        c = seg.curve
        plt.plot(c.q, c.di / c.i, ".", markersize=3, label=seg.name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("dI / I")
    plt.title("Erreur relative")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_relative_resolution(segments: list[CurveSegment], outpath: str | Path) -> None:
    plt.figure(figsize=(8, 5.5))
    for seg in segments:
        c = seg.curve
        plt.plot(c.q, c.dq / c.q, ".", markersize=3, label=seg.name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("dQ / Q")
    plt.title("Résolution relative")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_final_selected_segments(result: MultiStitchResult, outpath: str | Path) -> None:
    plt.figure(figsize=(8, 5.5))

    for item in result.selected:
        seg = item.segment
        c = seg.curve.scaled(item.global_scale)
        plt.errorbar(
            c.q,
            c.i,
            yerr=c.di,
            fmt=".",
            markersize=3,
            alpha=0.35,
            label=f"{seg.name} × {item.global_scale:.3g}",
        )

    c = result.final_curve
    plt.errorbar(c.q, c.i, yerr=c.di, fmt="o", markersize=2.5, label="courbe finale")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("I rescalée")
    plt.title("Segments retenus et courbe finale")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_retained_vs_rejected(
    segments: list[CurveSegment],
    result: MultiStitchResult,
    outpath: str | Path,
) -> None:
    selected_names = {item.segment.name for item in result.selected}

    plt.figure(figsize=(8, 5.5))

    for item in result.selected:
        c = item.segment.curve.scaled(item.global_scale)
        plt.plot(c.q, c.i, ".", markersize=3, label=f"retenu {item.segment.name}")

    for seg in segments:
        if seg.name in selected_names:
            continue

        # Échelle approximative pour affichage : on prend le meilleur fit
        # vers un segment retenu, si disponible.
        best = None
        best_global_scale = 1.0
        for item in result.selected:
            fit = best_pair_fit(item.segment, seg)
            if fit is None:
                continue
            if best is None or fit.score < best.score:
                best = fit
                best_global_scale = item.global_scale * fit.scale_b_to_a

        c = seg.curve.scaled(best_global_scale)
        plt.plot(c.q, c.i, ".", markersize=2, alpha=0.35, label=f"rejeté {seg.name}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Q")
    plt.ylabel("I rescalée approximative")
    plt.title("Segments retenus vs rejetés")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_transition_matrix(
    segments: list[CurveSegment],
    pair_fits: pd.DataFrame,
    outpath: str | Path,
) -> None:
    names = [s.name for s in segments]
    matrix = np.full((len(names), len(names)), np.nan)

    if not pair_fits.empty:
        for _, row in pair_fits.iterrows():
            if bool(row["accepted_transition"]):
                i = names.index(row["segment_a"])
                j = names.index(row["segment_b"])
                matrix[i, j] = row["chi2_red"]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.colorbar(im, label="chi2_red des transitions acceptées")
    plt.title("Transitions acceptées A → B")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raccord automatique multi-segments SAS/SANS/SAXS."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Fichiers texte Q I dI dQ à traiter comme segments candidats.",
    )

    parser.add_argument("--outdir", default="raccord_multisegment_output")

    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Noms optionnels des segments, dans le même ordre que les fichiers.",
    )

    parser.add_argument(
        "--grid",
        default="common",
        choices=["common", "a", "b"],
        help="Grille pour les raccords pair-à-pair.",
    )

    parser.add_argument("--n-common", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-log-width", type=float, default=0.10)
    parser.add_argument("--slope-weight", type=float, default=0.10)
    parser.add_argument("--width-weight", type=float, default=0.05)
    parser.add_argument("--resolution-weight", type=float, default=0.50)
    parser.add_argument("--rho-ref", type=float, default=0.20)
    parser.add_argument("--max-chi2-red", type=float, default=3.0)
    parser.add_argument("--max-slope-z", type=float, default=2.5)
    parser.add_argument("--min-new-log-coverage", type=float, default=0.05)
    parser.add_argument("--segment-quality-weight", type=float, default=4.0)
    parser.add_argument("--new-coverage-weight", type=float, default=1.5)
    parser.add_argument("--keep-fraction", type=float, default=0.25)

    parser.add_argument(
        "--reference-segment",
        default=None,
        help=(
            "Nom exact du segment qui définit l'échelle finale, par exemple c10d0. "
            "Prioritaire sur --reference-config."
        ),
    )

    parser.add_argument(
        "--reference-config",
        default=None,
        help=(
            "Configuration qui définit l'échelle finale, par exemple config_10. "
            "Si plusieurs segments retenus appartiennent à cette configuration, "
            "le segment couvrant la plus grande largeur log(Q) est choisi."
        ),
    )

    parser.add_argument(
        "--reference-detector",
        default=None,
        help=(
            "Détecteur optionnel pour préciser --reference-config, par exemple detector0."
        ),
    )

    parser.add_argument(
        "--start-policy",
        default="lowest_q",
        choices=["lowest_q", "best_low_q_quality"],
    )

    args = parser.parse_args()

    segments = load_segments(args.files, names=args.names)

    result = stitch_segments_greedy(
        segments,
        start_policy=args.start_policy,
        grid=args.grid,
        n_common=args.n_common,
        min_points=args.min_points,
        min_log_width=args.min_log_width,
        slope_weight=args.slope_weight,
        width_weight=args.width_weight,
        resolution_weight=args.resolution_weight,
        rho_ref=args.rho_ref,
        max_chi2_red=args.max_chi2_red,
        max_slope_z=args.max_slope_z,
        min_new_log_coverage=args.min_new_log_coverage,
        segment_quality_weight=args.segment_quality_weight,
        new_coverage_weight=args.new_coverage_weight,
        keep_fraction=args.keep_fraction,
    )

    if args.reference_segment is not None or args.reference_config is not None:
        result = rebase_result_to_reference(
            result,
            reference_segment=args.reference_segment,
            reference_config=args.reference_config,
            reference_detector=args.reference_detector,
        )

    save_outputs(
        segments,
        result,
        outdir=args.outdir,
        resolution_weight=args.resolution_weight,
    )

    print("\n=== Segments retenus ===")
    print(selected_to_dataframe(result.selected).to_string(index=False))

    print("\n=== Segments rejetés ===")
    if result.rejected.empty:
        print("Aucun")
    else:
        print(result.rejected.to_string(index=False))

    print(f"\nFichiers écrits dans : {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
