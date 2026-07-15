import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scarlet.reduction.stitching import (
    SASCurve,
    common_resolution_overlap,
    stitch_many,
    stitch_pair,
)


def load_grasp_curve(path: Path) -> SASCurve:
    """Load one GRASP ASCII I(Q) export with columns Q, I, dI, dQ."""
    data = np.loadtxt(path, skiprows=40)
    data = data[data[:, 1] > 0]
    return SASCurve(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        name=path.stem,
    )

def load_txt_curve(path: Path) -> SASCurve:
    """Load one ASCII I(Q) export with columns Q, I, dI, dQ."""
    data = np.loadtxt(path)
    data = data[data[:, 1] > 0]
    return SASCurve(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        name=path.stem,
    )

def plot_results(
    low: SASCurve,
    high: SASCurve,
    result,
    *,
    output_path: Path,
    extra_curves: tuple[SASCurve, ...] = (),
    stitched_plot_scale: float = 10.0,
) -> None:
    scaled_high = high.scaled(result.scale_factor)

    fig, (ax_curve, ax_profile) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    for curve in extra_curves:
        ax_curve.errorbar(
            curve.q,
            curve.i,
            yerr=curve.di,
            fmt="o",
            ms=2,
            lw=0.6,
            capsize=0,
            alpha=0.35,
            label=f"{curve.name} (raw)",
        )

    ax_curve.errorbar(
        low.q,
        low.i,
        yerr=low.di,
        fmt="o",
        ms=3,
        lw=0.8,
        capsize=0,
        alpha=0.85,
        label=f"{low.name} input",
    )
    ax_curve.errorbar(
        scaled_high.q,
        scaled_high.i,
        yerr=scaled_high.di,
        fmt="o",
        ms=3,
        lw=0.8,
        capsize=0,
        alpha=0.85,
        label=f"{high.name} scaled",
    )
    ax_curve.loglog(
        result.curve.q,
        result.curve.i * stitched_plot_scale,
        "-",
        color="black",
        lw=1.4,
        label=f"Stitched curve x{stitched_plot_scale:g}",
    )
    ax_curve.axvspan(
        result.retained_overlap[0],
        result.retained_overlap[1],
        color="tab:green",
        alpha=0.12,
        label="Retained overlap",
    )
    ax_curve.axvspan(
        result.fit_range[0],
        result.fit_range[1],
        color="tab:orange",
        alpha=0.08,
        label="Fit range",
    )
    ax_curve.set_xscale("log")
    ax_curve.set_yscale("log")
    ax_curve.set_ylabel("Intensity")
    ax_curve.grid(True, which="both", alpha=0.2)
    ax_curve.legend()

    profile = result.local_scale_profile
    ax_profile.errorbar(
        profile.q,
        profile.factor,
        yerr=profile.factor_error,
        fmt="o",
        ms=3,
        lw=0.8,
        capsize=0,
        color="tab:purple",
        label="Local scale",
    )
    ax_profile.axhline(result.scale_factor, color="black", lw=1.2, label="Global fit")
    ax_profile.set_xscale("log")
    ax_profile.set_xlabel("Q")
    ax_profile.set_ylabel("Scale")
    ax_profile.grid(True, which="both", alpha=0.2)
    ax_profile.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_many_results(
    curves: tuple[SASCurve, ...],
    result,
    *,
    output_path: Path,
    stitched_plot_scale: float = 10.0,
) -> None:
    fig, (ax_curve, ax_scale) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    for curve in curves:
        ax_curve.errorbar(
            curve.q,
            curve.i,
            yerr=curve.di,
            fmt="o",
            ms=2,
            lw=0.6,
            capsize=0,
            alpha=0.35,
            label=f"{curve.name} raw",
        )

    for curve in result.scaled_curves:
        ax_curve.loglog(
            curve.q,
            curve.i,
            "-",
            lw=1.0,
            alpha=0.9,
            label=f"{curve.name} scaled",
        )

    ax_curve.loglog(
        result.curve.q,
        result.curve.i * stitched_plot_scale,
        "-",
        color="black",
        lw=1.6,
        label=f"3-config stitched x{stitched_plot_scale:g}",
    )
    for pair_result in result.pair_results:
        ax_curve.axvspan(
            pair_result.retained_overlap[0],
            pair_result.retained_overlap[1],
            color="tab:green",
            alpha=0.08,
        )

    ax_curve.set_xscale("log")
    ax_curve.set_yscale("log")
    ax_curve.set_ylabel("Intensity")
    ax_curve.grid(True, which="both", alpha=0.2)
    ax_curve.legend()

    config_index = np.arange(len(result.cumulative_factors))
    ax_scale.plot(config_index, result.cumulative_factors, "o-", color="black", lw=1.2)
    ax_scale.set_xticks(config_index)
    ax_scale.set_xticklabels([curve.name for curve in curves], rotation=15)
    ax_scale.set_ylabel("Cum. scale")
    ax_scale.set_xlabel("Configuration")
    ax_scale.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def choose_fit_range(
    low: SASCurve,
    high: SASCurve,
    *,
    min_coverage: float,
) -> tuple[float, float]:
    """Infer a fit window from overlap points that remain valid."""
    comparison = common_resolution_overlap(low, high, min_coverage=min_coverage)
    q_valid = comparison.q[comparison.valid]
    if q_valid.size < 3:
        raise ValueError("Not enough valid overlap points to define a fit range.")
    return float(q_valid[0]), float(q_valid[-1])


def safe_stem(name: str) -> str:
    return name.replace(".", "_")


# data_dir = Path(__file__).resolve().parent / "data_D11_AA0"
# curve_paths = (
#     data_dir / "011334_001.dat",
#     data_dir / "011344_002.dat",
#     data_dir / "011361_003.dat",
# )

# curves = tuple(sorted((load_grasp_curve(path) for path in curve_paths), key=lambda curve: curve.q.min()))

data_dir = Path(__file__).resolve().parent / "data_SANS_LLB"
curve_paths = (
    data_dir / "ludox_AM20_config_config_9_detector0.txt",
    data_dir / "ludox_AM20_config_config_10_detector0.txt",
)
curves = tuple(sorted((load_txt_curve(path) for path in curve_paths), key=lambda curve: curve.q.min()))

for curve in curves:
    print(
        f"{curve.name}: q_min={curve.q.min():.6g}, "
        f"q_max={curve.q.max():.6g}, n={curve.q.size}"
    )

print("Testing stitchings from the smallest-Q configurations upward.")

fit_ranges: tuple[tuple[float, float], ...] = tuple(
    choose_fit_range(curves[index], curves[index + 1], min_coverage=0.7)
    for index in range(len(curves) - 1)
)

for pair_index in range(len(curves) - 1):
    low = curves[pair_index]
    high = curves[pair_index + 1]

    print()
    print(f"Pair {pair_index + 1}: {low.name} + {high.name}")

    try:
        result = stitch_pair(
            low,
            high,
            fit_range=fit_ranges[pair_index],
            keep_fraction=0.20,
            overlap_mode="blend",
            min_coverage=0.7,
            outlier_sigma=6.0,
            local_window_points=9,
        )
    except Exception as exc:
        print(f"Stitching failed       : {type(exc).__name__}: {exc}")
        continue

    print(f"Recovered scale factor : {result.scale_factor:.6f}")
    print(f"Approximate scale error: {result.scale_error:.6f}")
    print(f"Reduced chi-square     : {result.chi2_red:.3f}")
    print(f"Local scale stability  : {100 * result.scale_stability:.2f} %")
    print(f"Degraded configuration : {result.degraded_curve}")
    print(f"Fit range              : {result.fit_range}")
    print(f"Retained overlap       : {result.retained_overlap}")

    stem = f"stitched_{safe_stem(low.name)}__{safe_stem(high.name)}"
    output_dat = Path(f"{stem}.dat")
    np.savetxt(
        output_dat,
        result.curve.to_array(),
        header="Q I dI dQ",
    )

    plot_path = Path(f"{stem}.png")
    extra_curves = tuple(curve for curve in curves if curve is not low and curve is not high)
    plot_results(low, high, result, output_path=plot_path, extra_curves=extra_curves)
    print(f"Saved stitched data    : {output_dat.resolve()}")
    print(f"Saved plot             : {plot_path.resolve()}")

print()
print("Three-configuration stitching")

multi_result = stitch_many(
    curves,
    fit_ranges=fit_ranges,
    keep_fraction=0.20,
    overlap_mode="blend",
    min_coverage=0.7,
    outlier_sigma=6.0,
    local_window_points=9,
)

print(f"Cumulative factors     : {multi_result.cumulative_factors}")
for pair_index, pair_result in enumerate(multi_result.pair_results, start=1):
    print(
        f"Pair {pair_index} summary     : "
        f"scale={pair_result.scale_factor:.6f}, "
        f"chi2_red={pair_result.chi2_red:.3f}, "
        f"fit_range={pair_result.fit_range}"
    )

multi_output_dat = Path("stitched_all_3_configs.dat")
np.savetxt(
    multi_output_dat,
    multi_result.curve.to_array(),
    header="Q I dI dQ",
)

multi_plot_path = Path("stitched_all_3_configs.png")
plot_many_results(curves, multi_result, output_path=multi_plot_path)
print(f"Saved stitched data    : {multi_output_dat.resolve()}")
print(f"Saved plot             : {multi_plot_path.resolve()}")
