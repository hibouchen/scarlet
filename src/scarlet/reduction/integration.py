from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import scipp as sc

@dataclass(frozen=True)
class Sector:
    """Angular region used for anisotropic radial integration.

    Angles follow the detector convention returned by
    :func:`scarlet.reduction.geometry.compute_chi_map`: zero lies on the
    positive detector X axis and positive angles turn towards positive Y.
    ``width_deg`` is the full angular width of the sector.
    """

    name: str
    center_deg: float
    width_deg: float
    include_opposite: bool = False

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("sector name must not be empty")

        center = float(self.center_deg)
        width = float(self.width_deg)
        if not np.isfinite(center):
            raise ValueError(f"sector center_deg must be finite, got {self.center_deg!r}")
        if not np.isfinite(width) or width <= 0.0 or width > 360.0:
            raise ValueError(
                f"sector width_deg must be > 0 and <= 360, got {self.width_deg!r}"
            )
        if not isinstance(self.include_opposite, (bool, np.bool_)):
            raise TypeError(
                "sector include_opposite must be a boolean, "
                f"got {self.include_opposite!r}"
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "center_deg", ((center + 180.0) % 360.0) - 180.0)
        object.__setattr__(self, "width_deg", width)
        object.__setattr__(self, "include_opposite", bool(self.include_opposite))


@dataclass(frozen=True)
class AzimuthalAverageResult:
    q: np.ndarray
    intensity: np.ndarray
    intensity_error: np.ndarray | None
    q_error: np.ndarray | None
    counts: np.ndarray
    q_unit: str | None = None
    intensity_unit: str | None = None

    def to_data_array(self) -> "sc.DataArray":
        sc = _require_scipp()

        data_kwargs: dict[str, Any] = {
            "dims": ["q"],
            "values": self.intensity,
        }
        if self.intensity_error is not None:
            data_kwargs["variances"] = np.square(self.intensity_error)
        if self.intensity_unit is not None:
            data_kwargs["unit"] = self.intensity_unit

        q_kwargs: dict[str, Any] = {"dims": ["q"], "values": self.q}
        if self.q_unit is not None:
            q_kwargs["unit"] = self.q_unit

        coords: dict[str, Any] = {
            "q": sc.array(**q_kwargs),
            "counts": sc.array(dims=["q"], values=self.counts),
        }
        if self.q_error is not None:
            q_error_kwargs: dict[str, Any] = {
                "dims": ["q"],
                "values": self.q_error,
            }
            if self.q_unit is not None:
                q_error_kwargs["unit"] = self.q_unit
            coords["q_error"] = sc.array(**q_error_kwargs)

        return sc.DataArray(
            data=sc.array(**data_kwargs),
            coords=coords,
        )


def _require_scipp():
    try:
        import scipp as sc
    except ImportError as exc:
        raise ImportError("scipp is required to use DataArray-based integration helpers") from exc
    return sc


def _maybe_scipp():
    try:
        import scipp as sc
    except ImportError:
        return None
    return sc


def _as_numpy_array(value: Any, *, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
    sc = _maybe_scipp()
    if sc is not None:
        if isinstance(value, sc.DataArray):
            return np.asarray(value.data.values, dtype=dtype)
        if isinstance(value, sc.Variable):
            return np.asarray(value.values, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _get_scipp_dims(value: Any) -> tuple[str, ...] | None:
    sc = _maybe_scipp()
    if sc is None:
        return None
    if isinstance(value, sc.DataArray):
        return tuple(value.dims)
    if isinstance(value, sc.Variable):
        return tuple(value.dims)
    return None


def _get_scipp_unit(value: Any) -> str | None:
    sc = _maybe_scipp()
    if sc is None:
        return None
    if isinstance(value, sc.DataArray):
        return str(value.data.unit)
    if isinstance(value, sc.Variable):
        return str(value.unit)
    return None


def _get_error_from_variances(value: Any, *, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    sc = _maybe_scipp()
    variances = None
    if sc is not None:
        if isinstance(value, sc.DataArray):
            variances = value.data.variances
        elif isinstance(value, sc.Variable):
            variances = value.variances
    if variances is None:
        return None

    error = np.sqrt(np.asarray(variances, dtype=np.float64))
    if error.shape != expected_shape:
        raise ValueError(f"variance shape mismatch: expected {expected_shape}, got {error.shape}")
    return error


def _get_combined_dataarray_mask(image: Any, *, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    sc = _maybe_scipp()
    if sc is None or not isinstance(image, sc.DataArray) or not image.masks:
        return None

    combined_mask = np.zeros(expected_shape, dtype=bool)
    for name, mask in image.masks.items():
        mask_values = np.asarray(mask.values, dtype=bool)
        if mask_values.shape != expected_shape:
            raise ValueError(f"mask {name!r} shape mismatch: expected {expected_shape}, got {mask_values.shape}")
        combined_mask |= mask_values
    return combined_mask


def _validate_scipp_dims(reference: Any, other: Any, *, name: str) -> None:
    reference_dims = _get_scipp_dims(reference)
    other_dims = _get_scipp_dims(other)
    if reference_dims is not None and other_dims is not None and reference_dims != other_dims:
        raise ValueError(f"{name} dims mismatch: expected {reference_dims}, got {other_dims}")


def _compute_bin_edges(values: np.ndarray, n_bins: int, *, scale: str) -> np.ndarray:
    if scale not in {"linear", "log"}:
        raise ValueError(f"scale must be 'linear' or 'log', got {scale!r}")

    if scale == "linear":
        return np.histogram_bin_edges(values, bins=int(n_bins))

    positive_values = values[values > 0.0]
    if positive_values.size == 0:
        raise ValueError("Logarithmic binning requires strictly positive values")
    value_min = float(np.min(positive_values))
    value_max = float(np.max(positive_values))
    if value_max == value_min:
        value_min *= 1.0 - 1e-12
        value_max *= 1.0 + 1e-12
    return np.geomspace(value_min, value_max, int(n_bins) + 1)


def radial_average(
    image,
    mask,
    x0,
    y0,
    x_pixel_size,
    y_pixel_size,
    bins=None,
    error=None,
    *,
    scale: str = "linear",
):
    """Legacy radial-distance integration helper."""
    image_array = _as_numpy_array(image, dtype=np.float64)
    mask_array = _as_numpy_array(mask, dtype=bool)
    if image_array.shape != mask_array.shape:
        raise ValueError(f"mask shape mismatch: expected {image_array.shape}, got {mask_array.shape}")

    if error is None:
        error_array = _get_error_from_variances(image, expected_shape=image_array.shape)
    else:
        error_array = _as_numpy_array(error, dtype=np.float64)
        if error_array.shape != image_array.shape:
            raise ValueError(f"error shape mismatch: expected {image_array.shape}, got {error_array.shape}")

    y, x = np.indices(image_array.shape, dtype="float")
    y = (y - y0) * y_pixel_size
    x = (x - x0) * x_pixel_size
    r_grid = np.ma.masked_array(data=np.sqrt(x**2 + y**2), mask=mask_array).compressed()
    masked_data = np.ma.masked_array(data=image_array, mask=mask_array, dtype="float").compressed()
    if bins is None:
        maxd = np.max(r_grid)
        mind = np.min(r_grid)
        bins = len(np.arange(mind, maxd))
    if scale == "log":
        positive = r_grid > 0.0
        r_grid = r_grid[positive]
        masked_data = masked_data[positive]
        if error_array is not None:
            masked_error = np.ma.masked_array(
                data=error_array,
                mask=mask_array,
                dtype="float",
            ).compressed()[positive]
        else:
            masked_error = None
        if r_grid.size == 0:
            raise ValueError("Logarithmic radial averaging requires strictly positive radii")
    else:
        masked_error = (
            np.ma.masked_array(data=error_array, mask=mask_array, dtype="float").compressed()
            if error_array is not None
            else None
        )
    edges = _compute_bin_edges(r_grid, int(bins), scale=scale)
    indexes = np.digitize(r_grid, edges, right=False)
    indexes = np.clip(indexes, 1, len(edges) - 1)
    counts = np.bincount(indexes, minlength=len(edges))[1:]
    r_mean = np.bincount(indexes, weights=r_grid)[1:] / counts
    r_square = np.bincount(indexes, weights=r_grid**2)[1:] / counts
    intensity = np.bincount(indexes, weights=masked_data)[1:] / counts
    if error_array is None:
        di = np.sqrt(intensity * counts) / counts
    else:
        di = np.sqrt(np.bincount(indexes, weights=masked_error**2)[1:]) / counts
    dr = np.sqrt(r_square - r_mean**2 + 1 / 12)
    return r_mean, intensity, di, dr


@dataclass(frozen=True)
class _IntegrationInputs:
    image: np.ndarray
    q: np.ndarray
    intensity_error: np.ndarray | None
    q_error: np.ndarray | None
    valid: np.ndarray
    q_unit: str | None
    intensity_unit: str | None


def _prepare_integration_inputs(
    image: Any,
    q_map: Any | None,
    *,
    mask: Any | None,
    intensity_error: Any | None,
    q_error: Any | None,
    q_scale: str,
) -> _IntegrationInputs:
    image_array = _as_numpy_array(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {image_array.shape}")

    if q_map is None:
        sc = _maybe_scipp()
        if sc is None or not isinstance(image, sc.DataArray):
            raise ValueError("q_map is required when image is not a scipp.DataArray with a 'q' coord")
        if "q" not in image.coords:
            raise ValueError("q_map is required when image has no 'q' coord")
        q_source: Any = image.coords["q"]
    else:
        q_source = q_map

    _validate_scipp_dims(image, q_source, name="q_map")
    q_array = _as_numpy_array(q_source, dtype=np.float64)
    if q_array.shape != image_array.shape:
        raise ValueError(f"q_map shape mismatch: expected {image_array.shape}, got {q_array.shape}")

    if intensity_error is not None:
        _validate_scipp_dims(image, intensity_error, name="intensity_error")
        intensity_error_array = _as_numpy_array(intensity_error, dtype=np.float64)
        if intensity_error_array.shape != image_array.shape:
            raise ValueError(
                f"intensity_error shape mismatch: expected {image_array.shape}, got {intensity_error_array.shape}"
            )
    else:
        intensity_error_array = _get_error_from_variances(image, expected_shape=image_array.shape)

    if q_error is not None:
        _validate_scipp_dims(image, q_error, name="q_error")
        q_error_array = _as_numpy_array(q_error, dtype=np.float64)
        if q_error_array.shape != image_array.shape:
            raise ValueError(f"q_error shape mismatch: expected {image_array.shape}, got {q_error_array.shape}")
    else:
        q_error_array = _get_error_from_variances(q_source, expected_shape=image_array.shape)

    valid = np.ones(image_array.shape, dtype=bool)
    dataarray_mask = _get_combined_dataarray_mask(image, expected_shape=image_array.shape)
    if dataarray_mask is not None:
        valid &= ~dataarray_mask

    if mask is not None:
        _validate_scipp_dims(image, mask, name="mask")
        mask_array = _as_numpy_array(mask, dtype=bool)
        if mask_array.shape != image_array.shape:
            raise ValueError(f"mask shape mismatch: expected {image_array.shape}, got {mask_array.shape}")
        valid &= ~mask_array.astype(bool)

    if q_scale not in {"linear", "log"}:
        raise ValueError(f"q_scale must be 'linear' or 'log', got {q_scale!r}")

    valid &= np.isfinite(image_array)
    valid &= np.isfinite(q_array)
    if intensity_error_array is not None:
        valid &= np.isfinite(intensity_error_array)
    if q_error_array is not None:
        valid &= np.isfinite(q_error_array)
    if q_scale == "log":
        valid &= q_array > 0.0

    return _IntegrationInputs(
        image=image_array,
        q=q_array,
        intensity_error=intensity_error_array,
        q_error=q_error_array,
        valid=valid,
        q_unit=_get_scipp_unit(q_source),
        intensity_unit=_get_scipp_unit(image),
    )


def _integrate_radially(
    inputs: _IntegrationInputs,
    *,
    n_bins: int,
    q_scale: str,
    empty_selection: str,
) -> AzimuthalAverageResult:
    if not np.any(inputs.valid):
        raise ValueError(empty_selection)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins!r}")

    q_values = inputs.q[inputs.valid].ravel()
    image_values = inputs.image[inputs.valid].ravel()
    bin_edges = _compute_bin_edges(q_values, int(n_bins), scale=q_scale)
    indexes = np.digitize(q_values, bin_edges, right=False)
    indexes = np.clip(indexes, 1, int(n_bins))

    counts = np.bincount(indexes, minlength=int(n_bins) + 1)[1:]
    q_sum = np.bincount(indexes, weights=q_values, minlength=int(n_bins) + 1)[1:]
    intensity_sum = np.bincount(indexes, weights=image_values, minlength=int(n_bins) + 1)[1:]

    intensity = np.full(int(n_bins), np.nan, dtype=np.float64)
    q_binned = np.full(int(n_bins), np.nan, dtype=np.float64)
    non_empty = counts > 0
    intensity[non_empty] = intensity_sum[non_empty] / counts[non_empty]
    q_binned[non_empty] = q_sum[non_empty] / counts[non_empty]

    if inputs.intensity_error is not None:
        error_values = inputs.intensity_error[inputs.valid].ravel()
        variance_sum = np.bincount(
            indexes,
            weights=np.square(error_values),
            minlength=int(n_bins) + 1,
        )[1:]
        binned_intensity_error = np.full(int(n_bins), np.nan, dtype=np.float64)
        binned_intensity_error[non_empty] = np.sqrt(variance_sum[non_empty]) / counts[non_empty]
    else:
        binned_intensity_error = None

    if inputs.q_error is not None:
        q_error_values = inputs.q_error[inputs.valid].ravel()
        q_variance_sum = np.bincount(
            indexes,
            weights=np.square(q_error_values),
            minlength=int(n_bins) + 1,
        )[1:]
        binned_q_error = np.full(int(n_bins), np.nan, dtype=np.float64)
        binned_q_error[non_empty] = np.sqrt(q_variance_sum[non_empty]) / counts[non_empty]
    else:
        binned_q_error = None

    return AzimuthalAverageResult(
        q=q_binned,
        intensity=intensity,
        intensity_error=binned_intensity_error,
        q_error=binned_q_error,
        counts=counts.astype(np.int64, copy=False),
        q_unit=inputs.q_unit,
        intensity_unit=inputs.intensity_unit,
    )


def _angular_distance(angle: np.ndarray, center: float) -> np.ndarray:
    """Return the unsigned shortest angular distance in radians."""
    return np.abs((angle - center + np.pi) % (2.0 * np.pi) - np.pi)


def _sector_selection(chi: np.ndarray, sector: Sector) -> np.ndarray:
    if sector.width_deg == 360.0:
        return np.ones(chi.shape, dtype=bool)

    center = np.deg2rad(sector.center_deg)
    half_width = 0.5 * np.deg2rad(sector.width_deg)
    selected = _angular_distance(chi, center) <= half_width
    if sector.include_opposite:
        selected |= _angular_distance(chi, center + np.pi) <= half_width
    return selected


def radial_average(image, mask, x0, y0, x_pixel_size, y_pixel_size, bins=None, error=None, *, scale: str = "linear"):
    image_array = _as_numpy_array(image, dtype=np.float64)
    mask_array = _as_numpy_array(mask, dtype=bool)
    if image_array.shape != mask_array.shape:
        raise ValueError(f"mask shape mismatch: expected {image_array.shape}, got {mask_array.shape}")

    if error is None:
        error_array = _get_error_from_variances(image, expected_shape=image_array.shape)
    else:
        error_array = _as_numpy_array(error, dtype=np.float64)
        if error_array.shape != image_array.shape:
            raise ValueError(f"error shape mismatch: expected {image_array.shape}, got {error_array.shape}")

    y, x = np.indices(image_array.shape, dtype="float")
    y = (y - y0) * y_pixel_size
    x = (x - x0) * x_pixel_size
    r_grid = np.ma.masked_array(data=np.sqrt(x**2 + y**2), mask=mask_array).compressed()
    masked_data = np.ma.masked_array(data=image_array, mask=mask_array, dtype="float").compressed()
    if bins is None:
        maxd = np.max(r_grid)
        mind = np.min(r_grid)
        bins = len(np.arange(mind, maxd))
    if scale == "log":
        positive = r_grid > 0.0
        r_grid = r_grid[positive]
        masked_data = masked_data[positive]
        if error_array is not None:
            masked_error = np.ma.masked_array(data=error_array, mask=mask_array, dtype="float").compressed()[positive]
        else:
            masked_error = None
        if r_grid.size == 0:
            raise ValueError("Logarithmic radial averaging requires strictly positive radii")
    else:
        masked_error = (
            np.ma.masked_array(data=error_array, mask=mask_array, dtype="float").compressed()
            if error_array is not None
            else None
        )
    edges = _compute_bin_edges(r_grid, int(bins), scale=scale)
    indexes = np.digitize(r_grid, edges, right=False)
    indexes = np.clip(indexes, 1, len(edges) - 1)
    counts = np.bincount(indexes, minlength=len(edges))[1:]
    r_mean = np.bincount(indexes, weights=r_grid)[1:] / counts
    r_square = np.bincount(indexes, weights=r_grid**2)[1:] / counts
    intensity = np.bincount(indexes, weights=masked_data)[1:] / counts
    if error_array is None:
        di = np.sqrt(intensity * counts) / counts
    else:
        di = np.sqrt(np.bincount(indexes, weights=masked_error**2)[1:]) / counts
    dr = np.sqrt(r_square - r_mean**2 + 1 / 12)
    return r_mean, intensity, di, dr


def azimuthal_average(
    image: Any,
    q_map: Any | None = None,
    *,
    mask: Any | None = None,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
    n_bins: int = 200,
    q_scale: str = "linear",
) -> AzimuthalAverageResult:
    """
    Compute a 1D azimuthal average I(q) from a detector image and a q-map.

    Non-finite pixels are ignored. If `mask` is provided, non-zero values are
    treated as masked pixels and excluded from the average. When `image` is a
    `scipp.DataArray`, `q_map` can be omitted if a `q` coord is present, the
    image variances are used as intensity errors, and all masks carried by the
    DataArray are applied automatically.
    """
    inputs = _prepare_integration_inputs(
        image,
        q_map,
        mask=mask,
        intensity_error=intensity_error,
        q_error=q_error,
        q_scale=q_scale,
    )
    return _integrate_radially(
        inputs,
        n_bins=n_bins,
        q_scale=q_scale,
        empty_selection="No valid pixels available for azimuthal averaging",
    )


def sector_average(
    image: Any,
    q_map: Any | None = None,
    chi_map: Any | None = None,
    *,
    sector: Sector,
    mask: Any | None = None,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
    n_bins: int = 200,
    q_scale: str = "linear",
) -> AzimuthalAverageResult:
    """Compute a radial average restricted to one angular sector.

    ``chi_map`` is expressed in radians. It can be omitted when ``image`` is a
    :class:`scipp.DataArray` carrying a two-dimensional ``chi`` coordinate.
    The returned intensity and uncertainty follow the same conventions as
    :func:`azimuthal_average`.
    """
    if not isinstance(sector, Sector):
        raise TypeError(f"sector must be a Sector, got {type(sector).__name__}")

    inputs = _prepare_integration_inputs(
        image,
        q_map,
        mask=mask,
        intensity_error=intensity_error,
        q_error=q_error,
        q_scale=q_scale,
    )

    if chi_map is None:
        sc = _maybe_scipp()
        if sc is None or not isinstance(image, sc.DataArray):
            raise ValueError("chi_map is required when image is not a scipp.DataArray with a 'chi' coord")
        if "chi" not in image.coords:
            raise ValueError("chi_map is required when image has no 'chi' coord")
        chi_source: Any = image.coords["chi"]
    else:
        chi_source = chi_map

    _validate_scipp_dims(image, chi_source, name="chi_map")
    chi_array = _as_numpy_array(chi_source, dtype=np.float64)
    if chi_array.shape != inputs.image.shape:
        raise ValueError(f"chi_map shape mismatch: expected {inputs.image.shape}, got {chi_array.shape}")

    selected = np.isfinite(chi_array) & _sector_selection(chi_array, sector)
    sector_inputs = replace(inputs, valid=inputs.valid & selected)
    return _integrate_radially(
        sector_inputs,
        n_bins=n_bins,
        q_scale=q_scale,
        empty_selection=f"No valid pixels available in sector {sector.name!r}",
    )
