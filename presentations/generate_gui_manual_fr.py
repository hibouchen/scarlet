#!/usr/bin/env python3
"""Generate the French SCARLET GUI manual and its real-UI screenshots."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scarlet-manual-mpl")

import h5py
import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scarlet.gui.application import create_application
from scarlet.gui.main_window import ScarletMainWindow
from scarlet.gui.mask_editor import MaskEditorSource
from scarlet.gui.settings_data import SettingsImagePayload
from scarlet.gui.workflow_controller import AcquisitionPreviewPayload
from scarlet.workflow import RunKey, WorkflowContext
from scarlet.workflow.configuration import Aperture, Collimation, Configuration


ASSET_DIR = ROOT / "presentations" / "manual_gui_assets"
OUTPUT_PATH = ROOT / "presentations" / "SCARLET_manuel_interface_graphique_fr.pptx"
DEMO_DIR = ASSET_DIR / "demo_workspace"
LOGO_PATH = ROOT / "logo_scarlet.png"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BG = RGBColor(243, 246, 245)
WHITE = RGBColor(255, 255, 255)
INK = RGBColor(35, 48, 54)
MUTED = RGBColor(91, 106, 112)
TEAL = RGBColor(0, 123, 115)
TEAL_DARK = RGBColor(0, 92, 86)
TEAL_PALE = RGBColor(224, 243, 240)
BLUE = RGBColor(46, 136, 199)
BLUE_PALE = RGBColor(226, 239, 249)
ORANGE = RGBColor(232, 105, 70)
ORANGE_PALE = RGBColor(252, 235, 228)
GREEN = RGBColor(46, 125, 50)
RED = RGBColor(183, 28, 28)
LINE = RGBColor(211, 221, 220)
FONT = "Aptos"


def _detector_image(size: int, center: tuple[float, float], angle_deg: float) -> np.ndarray:
    y, x = np.indices((size, size), dtype=float)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    angle = np.deg2rad(angle_deg)
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    anisotropic = np.exp(-((along / 22.0) ** 2 + (across / 7.0) ** 2) / 2.0)
    halo = 0.34 * np.exp(-np.sqrt(dx * dx + dy * dy) / 23.0)
    rings = 0.08 * (1.0 + np.cos(np.sqrt(dx * dx + dy * dy) * 0.52))
    beamstop = np.sqrt(dx * dx + dy * dy) < 4.0
    image = 30.0 + 1400.0 * anisotropic + 540.0 * halo + 55.0 * rings
    image[beamstop] *= 0.08
    return image.astype(np.float64)


def _curve(q: np.ndarray, scale: float, slope: float, bump: float = 0.0) -> np.ndarray:
    intensity = scale * (q / 0.02) ** (-slope) * np.exp(-q * 1.8)
    intensity *= 1.0 + bump * np.exp(-((np.log10(q) + 0.75) / 0.20) ** 2)
    error = np.maximum(intensity * 0.055, 1.0e-6)
    dq = q * 0.035
    return np.column_stack((q, intensity, error, dq))


def _write_curve_group(parent: h5py.Group, name: str, data: np.ndarray) -> h5py.Group:
    group = parent.create_group(name)
    group.create_dataset("data", data=np.asarray(data, dtype=float))
    return group


def _write_demo_nexus(
    path: Path,
    *,
    image0: np.ndarray,
    image1: np.ndarray,
    processed: bool,
    curve_scale: float,
    q_shift: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("raw_data")
        raw.attrs["NX_class"] = np.bytes_("NXentry")
        raw.create_dataset("definition", data=np.bytes_("NXsas_raw"))
        control = raw.create_group("control")
        control.create_dataset("integral", data=1_000_000.0)
        control.create_dataset("count_time", data=60.0)
        instrument = raw.create_group("instrument")
        for detector_number, image in enumerate((image0, image1)):
            detector = instrument.create_group(f"detector{detector_number}")
            detector.create_dataset("data", data=image)
            detector.create_dataset("dead_time", data=0.0)
            detector.create_dataset("beam_center_x", data=image.shape[1] / 2.0)
            detector.create_dataset("beam_center_y", data=image.shape[0] / 2.0)
            detector.create_dataset("x_pixel_size", data=5.0e-3)
            detector.create_dataset("y_pixel_size", data=5.0e-3)

        if not processed:
            return
        processed_group = handle.create_group("processed")
        q0 = np.geomspace(0.006 + q_shift, 0.10 + q_shift, 42)
        q1 = np.geomspace(0.055 + q_shift, 0.42 + q_shift, 48)
        for detector_number, q in enumerate((q0, q1)):
            _write_curve_group(
                processed_group,
                f"data{detector_number}",
                _curve(q, curve_scale * (1.0 + 0.025 * detector_number), 2.45, 0.12),
            )
        sector_root = processed_group.create_group("products").create_group("sectors")
        for name, center, factor, bump in (
            ("parallel", 0.0, 1.34, 0.24),
            ("perpendicular", 90.0, 0.76, -0.05),
        ):
            sector_group = sector_root.create_group(name)
            sector_group.attrs["label"] = np.bytes_(name)
            for detector_number, q in enumerate((q0, q1)):
                detector_group = _write_curve_group(
                    sector_group,
                    f"detector{detector_number}",
                    _curve(q, curve_scale * factor, 2.42, bump),
                )
                detector_group.create_dataset("sector_center", data=center)
                detector_group.create_dataset("sector_width", data=24.0)
                detector_group.create_dataset("sector_include_opposite", data=True)


def _write_text_curve(path: Path, data: np.ndarray, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, header=f"SCARLET demo curve: {label}\nQ I dI dQ")


def _make_demo_workflow() -> tuple[WorkflowContext, dict[RunKey, MaskEditorSource]]:
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    raw_dir = DEMO_DIR / "raw"
    output_dir = DEMO_DIR / "processed"
    output_dir.mkdir(parents=True)

    configuration = {
        "config_1": Configuration(
            wavelength=6.0,
            sample_detector_distance=[4.0, 1.45],
            collimation=Collimation(
                Aperture(type="pinhole", diameter=0.012),
                Aperture(type="pinhole", diameter=0.007),
                collimation_distance=5.5,
                last_aperture_to_sample_distance=0.30,
            ),
            config_id="config_1",
            notes="Long distance",
        ),
        "config_2": Configuration(
            wavelength=6.0,
            sample_detector_distance=[1.8, 0.85],
            collimation=Collimation(
                Aperture(type="pinhole", diameter=0.012),
                Aperture(type="pinhole", diameter=0.009),
                collimation_distance=3.5,
                last_aperture_to_sample_distance=0.30,
            ),
            config_id="config_2",
            notes="Short distance",
        ),
    }
    workflow = WorkflowContext(
        experiment_id="demo-anisotrope",
        instrument_name="sansllb",
        root_dir=raw_dir.resolve(),
        output_dir=output_dir.resolve(),
    )
    workflow.configurations.update(configuration)
    sources: dict[RunKey, MaskEditorSource] = {}
    converted: dict[str, str] = {}

    sample_specs = (
        ("Ludox", "config_1", True, 1.00, 0.000),
        ("Ludox", "config_2", True, 0.96, 0.002),
        ("Polymère étiré", "config_1", True, 0.62, 0.000),
        ("Polymère étiré", "config_2", True, 0.60, 0.002),
        ("Protéine A", "config_1", False, 0.38, 0.000),
    )
    for index, (sample, config_id, processed, scale, q_shift) in enumerate(sample_specs, start=1):
        angle = 12.0 if sample == "Ludox" else 28.0 if "Polym" in sample else 72.0
        image0 = _detector_image(128, (64.5, 62.0), angle)
        image1 = _detector_image(96, (48.5, 47.0), angle + 4.0)
        path = raw_dir / f"{index:03d}_{sample.replace(' ', '_')}_{config_id}.nxs"
        _write_demo_nexus(
            path,
            image0=image0,
            image1=image1,
            processed=processed,
            curve_scale=scale,
            q_shift=q_shift,
        )
        key = workflow.add_run(
            RunKey(config_id=config_id, entity="sample", mode="scattering", sample_name=sample),
            path,
        )
        workflow.set_sample_thickness(sample, config_id, 1.0 if sample == "Ludox" else 0.8)
        workflow.set_transmission(sample, config_id, 0.91 - 0.015 * index)
        converted[str(path.resolve())] = str(path.resolve())
        sources[key] = MaskEditorSource(
            file_path=path.resolve(),
            entry_path="/raw_data",
            detector_data={0: image0, 1: image1},
            configuration=configuration[config_id],
            configuration_issues=[],
        )

    reference_index = 20
    for config_id in ("config_1", "config_2"):
        for entity, mode, sample_name in (
            ("empty_beam", "transmission", None),
            ("empty_cell", "transmission", None),
            ("water", "scattering", "H2O"),
            ("water", "transmission", "H2O"),
            ("dark", "scattering", None),
        ):
            reference_index += 1
            image0 = _detector_image(128, (64.5, 62.0), 0.0)
            image1 = _detector_image(96, (48.5, 47.0), 0.0)
            path = raw_dir / f"{reference_index:03d}_{entity}_{mode}_{config_id}.nxs"
            _write_demo_nexus(
                path,
                image0=image0,
                image1=image1,
                processed=False,
                curve_scale=1.0,
                q_shift=0.0,
            )
            key = workflow.add_run(
                RunKey(
                    config_id=config_id,
                    entity=entity,
                    mode=mode,
                    sample_name=sample_name,
                ),
                path,
            )
            converted[str(path.resolve())] = str(path.resolve())
            sources[key] = MaskEditorSource(
                file_path=path.resolve(),
                entry_path="/raw_data",
                detector_data={0: image0, 1: image1},
                configuration=configuration[config_id],
                configuration_issues=[],
            )

    workflow.store["converted_run_paths"] = converted
    for config_id in ("config_1", "config_2"):
        workflow.set_beam_center(config_id, 0, (64.5, 62.0))
        workflow.set_beam_center(config_id, 1, (48.5, 47.0))
        workflow.set_roi(config_id, (52, 78, 50, 76))
        mask0 = np.zeros((128, 128), dtype=np.uint8)
        mask0[:5, :] = 1
        mask0[-5:, :] = 1
        mask0[:, :4] = 1
        mask0[:, -4:] = 1
        mask0[58:67, 60:69] = 1
        mask1 = np.zeros((96, 96), dtype=np.uint8)
        mask1[:4, :] = 1
        mask1[-4:, :] = 1
        workflow.set_masks(config_id, {0: mask0, 1: mask1})
        flatfield = output_dir / f"{config_id}_flatfield.nxs"
        with h5py.File(flatfield, "w") as handle:
            handle.create_dataset("definition", data=np.bytes_("SCARLET_flatfield"))
        workflow.set_flatfield(config_id, flatfield)
        workflow.set_transmission("H2O", config_id, 0.965 if config_id == "config_1" else 0.958)
        workflow.set_empty_cell_transmission(config_id, 0.987 if config_id == "config_1" else 0.982)

    q = np.geomspace(0.006, 0.42, 88)
    outputs = (
        ("Ludox_config_1_detector0.txt", _curve(q[:58], 1.00, 2.45, 0.12)),
        ("Ludox_config_2_detector1.txt", _curve(q[32:], 0.98, 2.45, 0.12)),
        ("Ludox_sector_parallel.txt", _curve(q, 1.34, 2.42, 0.24)),
        ("Ludox_sector_perpendicular.txt", _curve(q, 0.76, 2.42, -0.05)),
        ("Ludox_merged.txt", _curve(q, 1.00, 2.45, 0.12)),
        ("Ludox_sector_parallel_merged.txt", _curve(q, 1.34, 2.42, 0.24)),
        ("Ludox_sector_perpendicular_merged.txt", _curve(q, 0.76, 2.42, -0.05)),
        ("Polymère_étiré_merged.txt", _curve(q, 0.61, 2.15, 0.18)),
    )
    for name, data in outputs:
        _write_text_curve(output_dir / name, data, name)

    workflow.info("Workflow demo loaded", where="manual")
    workflow.info("2 configurations prepared", where="settings")
    workflow.info("Sector integration enabled: parallel, perpendicular", where="processing")
    workflow.info("Sector curves stitched for Ludox", where="stitching")
    return workflow, sources


def _select_source_row(table: QtWidgets.QTableView, source_model, source_row: int) -> None:
    model = table.model()
    selection = table.selectionModel()
    blocker = QtCore.QSignalBlocker(selection)
    selection.clearSelection()
    source_index = source_model.index(source_row, 0)
    if hasattr(model, "mapFromSource"):
        index = model.mapFromSource(source_index)
    else:
        index = source_index
    selection.select(
        index,
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )
    table.scrollTo(index)
    del blocker


def _grab(window: QtWidgets.QWidget, path: Path, app: QtWidgets.QApplication) -> None:
    app.processEvents()
    QtCore.QThread.msleep(80)
    app.processEvents()
    pixmap = window.grab()
    if not pixmap.save(str(path)):
        raise RuntimeError(f"Could not save screenshot: {path}")


def create_screenshots() -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    workflow, sources = _make_demo_workflow()
    app = create_application(["scarlet-manual"])
    app.setStyle("Fusion")
    window = ScarletMainWindow(initial_raw_directory=workflow.root_dir)
    window.controller._workflow = workflow
    window.experiment_page.experiment_id_edit.setText("demo-anisotrope")
    window.experiment_page.raw_directory_edit.setText(str(workflow.root_dir))
    window.experiment_page.output_directory_edit.setText(str(workflow.output_dir))
    window._set_workflow(workflow)
    window.resize(1440, 900)
    window.show()
    app.processEvents()

    paths: dict[str, Path] = {}
    dock = window.findChild(QtWidgets.QDockWidget, "activityDock")

    window.navigation.setCurrentRow(1)
    app.processEvents()
    overview = ASSET_DIR / "01_overview.png"
    _grab(window, overview, app)
    paths["overview"] = overview

    if dock is not None:
        dock.hide()
    app.processEvents()

    window.navigation.setCurrentRow(0)
    experiment = ASSET_DIR / "02_experiment.png"
    _grab(window, experiment, app)
    paths["experiment"] = experiment

    window.navigation.setCurrentRow(1)
    acquisition_row = next(
        row
        for row in range(window.acquisitions_page.model.rowCount())
        if getattr(window.acquisitions_page.model.run_key_at(row), "sample_name", None) == "Ludox"
    )
    acquisition_key = window.acquisitions_page.model.run_key_at(acquisition_row)
    _select_source_row(window.acquisitions_page.table, window.acquisitions_page.model, acquisition_row)
    window.acquisitions_page._requested_preview_run_key = acquisition_key
    acquisition_payload = AcquisitionPreviewPayload(
        run_key=acquisition_key,
        source=sources[acquisition_key],
        masks=workflow.get_masks(acquisition_key.config_id),
        beam_centers=workflow.beam_centers[acquisition_key.config_id],
        converted_for_preview=False,
        display_count_time=60.0,
    )
    window.acquisitions_page.set_preview(acquisition_payload)
    acquisitions = ASSET_DIR / "03_acquisitions.png"
    _grab(window, acquisitions, app)
    paths["acquisitions"] = acquisitions

    window.navigation.setCurrentRow(2)
    _select_source_row(window.settings_page.table, window.settings_page.model, 0)
    window.settings_page._show_row(window.settings_page.model.row_at(0))
    image = sources[acquisition_key].detector_data[0]
    payload = SettingsImagePayload(
        config_id="config_1",
        source_config_id="config_1",
        purpose="center",
        detector_number=0,
        source_path=sources[acquisition_key].file_path,
        image=image,
        center=(64.5, 62.0),
        roi=(52, 78, 50, 76),
    )
    window.settings_page.set_image(payload)
    settings = ASSET_DIR / "04_settings_geometry.png"
    _grab(window, settings, app)
    paths["settings"] = settings

    window.settings_page.settings_tabs.setCurrentIndex(3)
    settings_flatfield = ASSET_DIR / "05_settings_flatfield.png"
    _grab(window, settings_flatfield, app)
    paths["settings_flatfield"] = settings_flatfield

    window.navigation.setCurrentRow(3)
    processing_row = next(
        row
        for row in range(window.processing_page.model.rowCount())
        if getattr(window.processing_page.model.run_key_at(row), "sample_name", None) == "Ludox"
    )
    processing_key = window.processing_page.model.run_key_at(processing_row)
    _select_source_row(window.processing_page.table, window.processing_page.model, processing_row)
    window.processing_page._requested_preview_run_key = processing_key
    window.processing_page.anisotropic_check.setChecked(True)
    window.processing_page.sector_direction_spin.setValue(12.0)
    window.processing_page.sector_width_spin.setValue(24.0)
    window.processing_page.sector_opposite_check.setChecked(True)
    window.processing_page.set_preview(
        AcquisitionPreviewPayload(
            run_key=processing_key,
            source=sources[processing_key],
            masks=workflow.get_masks(processing_key.config_id),
            beam_centers=workflow.beam_centers[processing_key.config_id],
            converted_for_preview=False,
            display_count_time=60.0,
        )
    )
    processing = ASSET_DIR / "06_processing_sectors.png"
    _grab(window, processing, app)
    paths["processing"] = processing

    window.navigation.setCurrentRow(4)
    sector_index = window.stitching_page.curve_group_combo.findData("*")
    window.stitching_page.curve_group_combo.setCurrentIndex(max(0, sector_index))
    if window.stitching_page.model.rowCount():
        window.stitching_page.table.selectRow(0)
    stitching = ASSET_DIR / "07_stitching_sectors.png"
    _grab(window, stitching, app)
    paths["stitching"] = stitching

    window.navigation.setCurrentRow(5)
    selection = window.results_page.table.selectionModel()
    blocker = QtCore.QSignalBlocker(selection)
    selection.clearSelection()
    flags = QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows
    wanted = {"Ludox_merged.txt", "Ludox_sector_parallel_merged.txt", "Ludox_sector_perpendicular_merged.txt"}
    for proxy_row in range(window.results_page.sort_model.rowCount()):
        proxy_index = window.results_page.sort_model.index(proxy_row, 0)
        source_row = window.results_page.sort_model.mapToSource(proxy_index).row()
        path = window.results_page.model.path_at(source_row)
        if path is not None and path.name in wanted:
            selection.select(proxy_index, flags)
    del blocker
    window.results_page._plot_selection()
    results = ASSET_DIR / "08_results.png"
    _grab(window, results, app)
    paths["results"] = results

    window.close()
    app.processEvents()
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    return paths


def _set_cell_text(shape, text: str, font_size: float, color: RGBColor, *, bold: bool = False) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(0.12)
    frame.margin_right = Inches(0.12)
    frame.margin_top = Inches(0.07)
    frame.margin_bottom = Inches(0.07)
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.name = FONT
    paragraph.font.size = Pt(font_size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color


def _add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    color: RGBColor = INK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin: float = 0.02,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = FONT
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    return box


def _add_bullets(slide, items: list[str], x: float, y: float, w: float, h: float, *, size: float = 16, color: RGBColor = INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    frame.margin_right = Inches(0.05)
    frame.margin_top = Inches(0.02)
    for index, item in enumerate(items):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = item
        p.level = 0
        p.text = "•  " + item
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
    return box


def _round_rect(slide, x: float, y: float, w: float, h: float, *, fill: RGBColor = WHITE, line: RGBColor = LINE, radius=True):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.0)
    return shape


def _circle_number(slide, number: str, x: float, y: float, *, color: RGBColor = TEAL):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(0.37), Inches(0.37))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    _set_cell_text(shape, number, 13, WHITE, bold=True)
    shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shape


def _add_picture_contain(slide, path: Path, x: float, y: float, w: float, h: float, *, border=True):
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px, py = x + (w - pw) / 2.0, y + (h - ph) / 2.0
    if border:
        _round_rect(slide, x - 0.03, y - 0.03, w + 0.06, h + 0.06, fill=WHITE, line=LINE)
    slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))


def _add_header(slide, title: str, section: str, number: int):
    _add_text(slide, section.upper(), 0.55, 0.25, 2.6, 0.25, size=10.5, color=TEAL, bold=True)
    _add_text(slide, title, 0.55, 0.52, 11.9, 0.55, size=26, color=INK, bold=True)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.55), Inches(1.12), Inches(12.2), Inches(0.035))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()
    _add_text(slide, f"SCARLET · Manuel de l’interface graphique", 0.58, 7.17, 5.5, 0.20, size=8.5, color=MUTED)
    _add_text(slide, str(number), 12.1, 7.12, 0.6, 0.25, size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)


def _new_slide(prs: Presentation, title: str, section: str, number: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = BG
    _add_header(slide, title, section, number)
    return slide


def _step_card(slide, number: str, title: str, subtitle: str, x: float, y: float, color: RGBColor):
    card = _round_rect(slide, x, y, 3.85, 1.25, fill=WHITE, line=LINE)
    stripe = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(0.10), Inches(1.25))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = color
    stripe.line.fill.background()
    _circle_number(slide, number, x + 0.27, y + 0.22, color=color)
    _add_text(slide, title, x + 0.78, y + 0.17, 2.85, 0.35, size=17, bold=True)
    _add_text(slide, subtitle, x + 0.78, y + 0.58, 2.8, 0.48, size=11.5, color=MUTED)
    return card


def _tip_box(slide, text: str, x: float, y: float, w: float, h: float, *, accent: RGBColor = TEAL, pale: RGBColor = TEAL_PALE, title: str = "CONSEIL"):
    _round_rect(slide, x, y, w, h, fill=pale, line=accent)
    if h < 0.80:
        _add_text(slide, title, x + 0.18, y + 0.10, w - 0.36, 0.17, size=8.5, color=accent, bold=True)
        _add_text(slide, text, x + 0.18, y + 0.29, w - 0.36, h - 0.34, size=9.5, color=INK)
    elif h < 1.05:
        _add_text(slide, title, x + 0.20, y + 0.11, w - 0.40, 0.19, size=9, color=accent, bold=True)
        _add_text(slide, text, x + 0.20, y + 0.34, w - 0.40, h - 0.39, size=10.5, color=INK)
    else:
        _add_text(slide, title, x + 0.22, y + 0.13, w - 0.44, 0.22, size=9.5, color=accent, bold=True)
        _add_text(slide, text, x + 0.22, y + 0.40, w - 0.44, h - 0.47, size=11.2, color=INK)


def build_presentation(shots: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    prs.core_properties.title = "SCARLET — Manuel de l’interface graphique"
    prs.core_properties.subject = "Guide opérateur du workflow SANS, incluant l’intégration et le stitching sectoriels"
    prs.core_properties.author = "SCARLET"
    prs.core_properties.comments = "Généré depuis l’interface SCARLET actuelle."

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    banner = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.18))
    banner.fill.solid()
    banner.fill.fore_color.rgb = TEAL
    banner.line.fill.background()
    if LOGO_PATH.exists():
        slide.shapes.add_picture(str(LOGO_PATH), Inches(9.85), Inches(0.75), Inches(2.55), Inches(2.55))
    _add_text(slide, "SCARLET", 0.72, 0.85, 7.8, 0.7, size=42, color=TEAL_DARK, bold=True)
    _add_text(slide, "Manuel de l’interface graphique", 0.72, 1.62, 8.8, 0.62, size=29, color=INK, bold=True)
    _add_text(slide, "Du dossier brut aux courbes I(q) — avec intégration et stitching sectoriels", 0.75, 2.40, 8.3, 0.75, size=18, color=MUTED)
    _round_rect(slide, 0.75, 3.55, 7.4, 2.25, fill=WHITE, line=LINE)
    _add_text(slide, "Ce guide accompagne l’opérateur page par page :", 1.02, 3.86, 6.8, 0.35, size=16, bold=True)
    _add_bullets(slide, ["préparer et valider le workflow", "réduire en isotrope ou par secteurs", "raccorder puis comparer les courbes"], 1.02, 4.30, 6.65, 1.25, size=15)
    _add_text(slide, "Version 0.4.0 · 17 novembre 2026", 0.75, 6.75, 5.4, 0.3, size=11, color=MUTED)

    slide = _new_slide(prs, "Le parcours en 6 étapes", "Vue d’ensemble", 2)
    cards = (
        ("1", "Experiment", "Choisir les dossiers et créer ou ouvrir un workflow.", TEAL),
        ("2", "Acquisitions", "Vérifier les fichiers, métadonnées et masques.", BLUE),
        ("3", "Settings", "Préparer centres, ROI, transmissions et flatfields.", ORANGE),
        ("4", "Processing", "Réduire en isotrope et, si besoin, par secteurs.", TEAL),
        ("5", "Stitching", "Raccorder les configurations pour chaque famille de courbes.", BLUE),
        ("6", "Results", "Superposer, comparer et contrôler les courbes I(q).", ORANGE),
    )
    positions = ((0.7, 1.55), (4.75, 1.55), (8.8, 1.55), (0.7, 3.15), (4.75, 3.15), (8.8, 3.15))
    for card, pos in zip(cards, positions):
        _step_card(slide, card[0], card[1], card[2], pos[0], pos[1], card[3])
    _tip_box(slide, "L’ordre des pages suit le travail réel. La couleur des statuts indique où intervenir avant de lancer un calcul.", 0.7, 5.18, 11.95, 1.15)

    slide = _new_slide(prs, "Les repères permanents", "Vue d’ensemble", 3)
    _add_picture_contain(slide, shots["overview"], 0.55, 1.38, 9.0, 5.28)
    _round_rect(slide, 9.85, 1.38, 2.9, 5.28, fill=WHITE, line=LINE)
    entries = (
        ("1", "Navigation", "Les six pages restent accessibles à gauche."),
        ("2", "Workflow", "Nom, instrument, nombre d’acquisitions et bouton Save."),
        ("3", "Zone de travail", "Tableaux, visualisations et actions de la page."),
        ("4", "Activity", "Trace des opérations et messages utiles au diagnostic."),
    )
    y = 1.66
    for number, title, text in entries:
        _circle_number(slide, number, 10.10, y, color=TEAL)
        _add_text(slide, title, 10.58, y - 0.02, 1.95, 0.28, size=14, bold=True)
        _add_text(slide, text, 10.10, y + 0.42, 2.35, 0.62, size=11.5, color=MUTED)
        y += 1.17

    slide = _new_slide(prs, "Créer ou ouvrir un workflow", "1 · Experiment", 4)
    _add_picture_contain(slide, shots["experiment"], 0.55, 1.42, 8.35, 4.90)
    _round_rect(slide, 9.18, 1.42, 3.57, 4.90, fill=WHITE, line=LINE)
    _add_text(slide, "À renseigner", 9.45, 1.72, 2.9, 0.35, size=17, bold=True, color=TEAL_DARK)
    _add_bullets(slide, ["Raw data : dossier des acquisitions", "Output : dossier des produits", "Instrument et stratégie de transmission", "Experiment ID : nom court et stable"], 9.42, 2.17, 2.9, 2.55, size=13.5)
    _tip_box(slide, "Scan acquisitions crée le workflow. Open workflow… reprend un fichier sauvegardé.", 9.42, 4.87, 2.95, 1.18)

    slide = _new_slide(prs, "Contrôler les acquisitions", "2 · Acquisitions", 5)
    _add_picture_contain(slide, shots["acquisitions"], 0.55, 1.38, 8.65, 5.35)
    _round_rect(slide, 9.48, 1.38, 3.27, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Avant de poursuivre", 9.74, 1.68, 2.7, 0.35, size=17, bold=True)
    _add_bullets(slide, ["Corriger Sample, Mode, Entity ou Thickness par double-clic.", "Utiliser les filtres pour isoler une configuration ou un statut.", "Exclure uniquement les fichiers qui ne doivent pas entrer dans le traitement.", "Vérifier la ligne verte Ready for processing."], 9.70, 2.12, 2.72, 3.15, size=13)
    _tip_box(slide, "Un clic sur une ligne charge son détecteur à droite.", 9.70, 5.60, 2.70, 0.80, accent=BLUE, pale=BLUE_PALE)

    slide = _new_slide(prs, "Visualiser et enregistrer un masque", "2 · Acquisitions", 6)
    _add_picture_contain(slide, shots["acquisitions"], 0.55, 1.38, 8.65, 5.35)
    _round_rect(slide, 9.48, 1.38, 3.27, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Méthode", 9.74, 1.68, 2.7, 0.35, size=17, bold=True, color=BLUE)
    _add_bullets(slide, ["Sélectionner une acquisition représentative.", "Choisir le détecteur dans la vue.", "Dessiner ou ajuster les zones masquées.", "Cliquer Save and attach mask…"], 9.70, 2.16, 2.72, 2.55, size=13.5)
    _tip_box(slide, "Convention SCARLET : 1 = masqué, 0 = valide. Le masque est associé à la configuration.", 9.70, 5.00, 2.70, 1.18, accent=ORANGE, pale=ORANGE_PALE, title="À RETENIR")

    slide = _new_slide(prs, "Préparer les configurations", "3 · Settings", 7)
    _add_picture_contain(slide, shots["settings"], 0.55, 1.38, 9.0, 5.35)
    _round_rect(slide, 9.85, 1.38, 2.90, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Deux niveaux de contrôle", 10.08, 1.67, 2.42, 0.52, size=16, bold=True)
    _add_text(slide, "Tableau principal", 10.08, 2.35, 2.25, 0.28, size=13, color=TEAL, bold=True)
    _add_text(slide, "Centres, ROI, masques, flatfield et statut global par configuration.", 10.08, 2.70, 2.36, 0.90, size=12.2, color=MUTED)
    _add_text(slide, "Reference transmissions", 10.08, 3.78, 2.35, 0.35, size=13, color=TEAL, bold=True)
    _add_text(slide, "Valeurs effectives de l’eau et de la cellule vide, placées sous le tableau des configurations.", 10.08, 4.18, 2.36, 1.10, size=12.2, color=MUTED)
    _tip_box(slide, "Une configuration est prête quand toutes les colonnes sont complètes.", 10.08, 5.42, 2.40, 1.00)

    slide = _new_slide(prs, "Centre du faisceau et ROI", "3 · Settings", 8)
    _add_picture_contain(slide, shots["settings"], 0.55, 1.38, 8.75, 5.35)
    _round_rect(slide, 9.58, 1.38, 3.17, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Beam center", 9.84, 1.72, 2.55, 0.32, size=16, bold=True, color=TEAL)
    _add_bullets(slide, ["Charger l’empty beam.", "Détecter automatiquement ou ajuster X/Y.", "Sauvegarder pour chaque détecteur."], 9.80, 2.12, 2.62, 1.72, size=13)
    _add_text(slide, "Transmission ROI", 9.84, 4.02, 2.55, 0.32, size=16, bold=True, color=ORANGE)
    _add_bullets(slide, ["Passer Edit à Transmission ROI.", "Détecter ou saisir x0, x1, y0, y1.", "Vérifier la zone sur l’image avant Save."], 9.80, 4.42, 2.62, 1.75, size=13)

    slide = _new_slide(prs, "Masques, flatfields et transmissions", "3 · Settings", 9)
    _add_picture_contain(slide, shots["settings_flatfield"], 0.55, 1.38, 8.65, 5.35)
    _round_rect(slide, 9.48, 1.38, 3.27, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Onglets de préparation", 9.73, 1.70, 2.75, 0.36, size=16, bold=True)
    _add_bullets(slide, ["Transmission : réutiliser une configuration source si nécessaire.", "Mask : attacher un bundle ou rechercher un masque compatible.", "Flatfield : créer, choisir un fichier ou réutiliser une source."], 9.70, 2.17, 2.75, 2.65, size=13.2)
    _tip_box(slide, "Une modification de masque ou de transmission peut rendre un flatfield obsolète : reconstruisez-le si le statut devient stale.", 9.70, 5.18, 2.75, 1.22, accent=ORANGE, pale=ORANGE_PALE, title="ATTENTION")

    slide = _new_slide(prs, "Comprendre les statuts", "3 · Settings", 10)
    columns = (("ready", GREEN, "Tous les prérequis sont présents."), ("incomplete", RED, "Au moins un élément manque."), ("stale", ORANGE, "Un produit doit être reconstruit."))
    x = 0.75
    for label, color, text in columns:
        _round_rect(slide, x, 1.55, 3.78, 1.38, fill=WHITE, line=LINE)
        badge = _round_rect(slide, x + 0.24, 1.82, 1.18, 0.42, fill=color, line=color)
        _set_cell_text(badge, label, 12, WHITE, bold=True)
        badge.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        _add_text(slide, text, x + 1.62, 1.75, 1.78, 0.66, size=13.5)
        x += 4.05
    _round_rect(slide, 0.75, 3.35, 11.9, 2.48, fill=WHITE, line=LINE)
    _add_text(slide, "Checklist avant Processing", 1.03, 3.66, 4.2, 0.38, size=18, bold=True, color=TEAL_DARK)
    _add_bullets(slide, ["un centre de faisceau par détecteur", "une ROI de transmission valide", "des masques assignés", "un flatfield présent et non stale"], 1.02, 4.16, 5.1, 1.35, size=14)
    _add_bullets(slide, ["une transmission pour chaque échantillon", "une épaisseur renseignée si disponible", "des fichiers de référence cohérents", "aucun message bloquant dans Acquisitions"], 6.55, 4.16, 5.45, 1.35, size=14)

    slide = _new_slide(prs, "Réduire les acquisitions", "4 · Processing", 11)
    _add_picture_contain(slide, shots["processing"], 0.55, 1.38, 8.8, 5.35)
    _round_rect(slide, 9.63, 1.38, 3.12, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Traitement standard", 9.88, 1.70, 2.55, 0.35, size=16, bold=True)
    _add_bullets(slide, ["Sélectionner une ou plusieurs acquisitions.", "Régler le nombre de bins par détecteur.", "Choisir l’échelle q : Linear ou Logarithmic.", "Lancer Process selected ou Process pending."], 9.84, 2.15, 2.60, 2.80, size=13)
    _tip_box(slide, "La réduction s’exécute en arrière-plan. Les statuts passent de pending à processed.", 9.84, 5.38, 2.58, 1.02, accent=BLUE, pale=BLUE_PALE)

    slide = _new_slide(prs, "Intégration sectorielle", "4 · Processing", 12)
    _add_picture_contain(slide, shots["processing"], 0.55, 1.38, 8.85, 5.35)
    _round_rect(slide, 9.68, 1.38, 3.07, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Réglages", 9.92, 1.68, 2.55, 0.35, size=16, bold=True, color=TEAL)
    _add_bullets(slide, ["Activer Enable sector integration.", "Preset : parallèle/perpendiculaire ou direction unique.", "Main direction : angle sur le détecteur.", "Sector width : largeur angulaire totale.", "Combine opposite : moyenne la direction à 180°."], 9.88, 2.10, 2.60, 3.10, size=12.8)
    _tip_box(slide, "Rouge : direction principale. Bleu : perpendiculaire. La vue suit la ligne sélectionnée.", 9.88, 5.20, 2.60, 1.22, accent=ORANGE, pale=ORANGE_PALE, title="LECTURE DE LA VUE")

    slide = _new_slide(prs, "Raccorder l’isotrope ou les secteurs", "5 · Stitching", 13)
    _add_picture_contain(slide, shots["stitching"], 0.55, 1.38, 8.9, 5.35)
    _round_rect(slide, 9.73, 1.38, 3.02, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Curves", 9.98, 1.68, 2.45, 0.32, size=16, bold=True, color=TEAL)
    _add_bullets(slide, ["Isotropic : courbe moyenne classique.", "All common sectors : tous les secteurs compatibles.", "Sector: … : un secteur nommé uniquement."], 9.94, 2.10, 2.50, 2.05, size=13)
    _add_text(slide, "Conditions", 9.98, 4.37, 2.45, 0.32, size=16, bold=True, color=ORANGE)
    _add_text(slide, "Toutes les configurations doivent être processed et partager une définition de secteur identique.", 9.94, 4.82, 2.50, 0.78, size=12.2, color=MUTED)
    _tip_box(slide, "…_sector_<name>_merged.txt", 9.94, 5.70, 2.50, 0.82, title="NOM DU FICHIER")

    slide = _new_slide(prs, "Examiner et comparer les résultats", "6 · Results", 14)
    _add_picture_contain(slide, shots["results"], 0.55, 1.38, 9.15, 5.35)
    _round_rect(slide, 9.98, 1.38, 2.77, 5.35, fill=WHITE, line=LINE)
    _add_text(slide, "Utilisation", 10.22, 1.70, 2.28, 0.35, size=16, bold=True)
    _add_bullets(slide, ["Sélection multiple pour superposer les courbes.", "Le type distingue reduced, sector, stitched et sector stitched.", "Utiliser Search pour filtrer les fichiers.", "La légende reprend le nom de chaque sortie."], 10.18, 2.15, 2.35, 2.98, size=12.8)
    _tip_box(slide, "Comparer parallèle et perpendiculaire révèle l’anisotropie.", 10.18, 5.35, 2.34, 1.05, accent=BLUE, pale=BLUE_PALE)

    slide = _new_slide(prs, "Sauvegarde et fichiers produits", "Traçabilité", 15)
    _round_rect(slide, 0.75, 1.48, 5.60, 4.95, fill=WHITE, line=LINE)
    _add_text(slide, "Sauvegarder le workflow", 1.05, 1.80, 4.85, 0.38, size=19, bold=True, color=TEAL_DARK)
    _add_bullets(slide, ["Bouton Save dans la barre latérale", "Ctrl+S pour la sauvegarde courante", "File → Save workflow as… pour créer une copie", "Le workflow conserve acquisitions, réglages et liens vers les produits"], 1.02, 2.35, 4.85, 2.55, size=14)
    _tip_box(slide, "Sauvegarder après chaque étape de préparation importante.", 1.05, 5.35, 4.95, 0.77)
    _round_rect(slide, 6.68, 1.48, 5.90, 4.95, fill=INK, line=INK)
    _add_text(slide, "processed/", 7.03, 1.82, 4.8, 0.33, size=18, color=WHITE, bold=True)
    tree = (
        "├── Ludox_config_1_detector0.txt       reduced\n"
        "├── Ludox_sector_parallel.txt         sector\n"
        "├── Ludox_merged.txt                  stitched\n"
        "├── Ludox_sector_parallel_merged.txt  sector stitched\n"
        "└── config_1_flatfield.nxs            flatfield"
    )
    _add_text(slide, tree, 7.03, 2.38, 4.95, 2.75, size=13.2, color=RGBColor(224, 243, 240))
    _add_text(slide, "Les sorties texte contiennent Q, I, dI et dQ.", 7.03, 5.55, 4.85, 0.45, size=13.2, color=WHITE)

    slide = _new_slide(prs, "Diagnostiquer sans perdre le fil", "Aide", 16)
    rows = (
        ("Vert", GREEN, "Prêt / traité", "Poursuivre vers l’étape suivante."),
        ("Orange", ORANGE, "Avertissement / stale", "Lire le détail et confirmer la cohérence."),
        ("Rouge", RED, "Blocage / missing", "Compléter l’entrée indiquée avant de relancer."),
    )
    y = 1.55
    for color_name, color, state, action in rows:
        _round_rect(slide, 0.80, y, 7.05, 1.18, fill=WHITE, line=LINE)
        swatch = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(1.08), Inches(y + 0.30), Inches(0.50), Inches(0.50))
        swatch.fill.solid(); swatch.fill.fore_color.rgb = color; swatch.line.fill.background()
        _add_text(slide, color_name, 1.82, y + 0.20, 1.1, 0.32, size=14, bold=True)
        _add_text(slide, state, 3.02, y + 0.20, 1.72, 0.32, size=14, color=color, bold=True)
        _add_text(slide, action, 4.86, y + 0.18, 2.55, 0.58, size=12.5, color=MUTED)
        y += 1.42
    _round_rect(slide, 8.18, 1.55, 4.35, 4.02, fill=WHITE, line=LINE)
    _add_text(slide, "Où chercher ?", 8.52, 1.92, 3.55, 0.38, size=19, bold=True, color=TEAL_DARK)
    _add_bullets(slide, ["Le texte sous le résumé de page", "Les infobulles des cellules de statut", "Le panneau Activity en bas de la fenêtre", "Le fichier absent ou incohérent mentionné dans le message"], 8.48, 2.52, 3.52, 2.20, size=13.5)
    _tip_box(slide, "Corriger la première cause signalée, puis rafraîchir la page concernée.", 8.48, 4.78, 3.55, 0.64)

    slide = _new_slide(prs, "Checklist opérateur", "Résumé", 17)
    checks = (
        ("1", "Workflow", "Dossiers, instrument et identifiant vérifiés"),
        ("2", "Acquisitions", "Fichiers classés, épaisseurs et masques contrôlés"),
        ("3", "Settings", "Centres, ROI, transmissions et flatfields ready"),
        ("4", "Processing", "Bins, échelle q et secteurs choisis"),
        ("5", "Stitching", "Famille de courbes, référence et facteur validés"),
        ("6", "Results", "Courbes superposées, cohérentes et sauvegardées"),
    )
    positions = ((0.75, 1.52), (6.80, 1.52), (0.75, 2.87), (6.80, 2.87), (0.75, 4.22), (6.80, 4.22))
    for (number, title, text), (x, y) in zip(checks, positions):
        _round_rect(slide, x, y, 5.75, 1.02, fill=WHITE, line=LINE)
        _circle_number(slide, number, x + 0.25, y + 0.31, color=TEAL if int(number) % 2 else BLUE)
        _add_text(slide, title, x + 0.82, y + 0.17, 1.55, 0.30, size=15.5, bold=True)
        _add_text(slide, text, x + 2.32, y + 0.16, 3.08, 0.56, size=12.3, color=MUTED)
    _tip_box(slide, "Raccourci mental : préparer → vérifier → traiter → raccorder → comparer → sauvegarder.", 0.75, 5.76, 11.80, 0.70, title="BONNE PRATIQUE")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT_PATH)


def main() -> int:
    shots = create_screenshots()
    build_presentation(shots)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
