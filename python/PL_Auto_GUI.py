"""
PL_Auto_GUI.py
──────────────────────────────────────────────────────────────────────────────
GUI for PL analysis with automatic dark / white matching.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import PySide6 as _pyside6
    _plugin_path = str(Path(_pyside6.__file__).parent / "plugins" / "platforms")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _plugin_path)
except Exception:
    pass

import numpy as np
import pandas as pd
import shiboken6
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from scipy.signal import savgol_filter as _savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── Physical constants ──────────────────────────────────────────────────────
HC_EV_NM = 1239.84193
EPSILON   = 1e-30

HALOGEN_FILENAME = "HalogenLamp_Spectrum.txt"
SESSION_FILE     = "PL_Auto_GUI_session.json"

# ── Power-grouping tolerance ─────────────────────────────────────────────────
# Adaptive: 10 % of the larger power value.  This keeps µW-range powers
# separate from one another (e.g. 5 µW ≠ 10 µW) while still tolerating the
# small measurement-to-measurement drift seen at mW-range powers
# (e.g. 10.0 mW ≈ 10.15 mW → same group).  The old fixed 0.2 mW threshold
# was equivalent to ~5 % at 4 mW but became completely wrong below ~1 mW.
_POWER_REL_TOL = 0.10   # 10 % relative tolerance


def _powers_match(p: float, q: float) -> bool:
    """True when two excitation powers belong to the same measurement group."""
    if p <= 0 or q <= 0:
        return p == q
    return abs(p - q) <= _POWER_REL_TOL * max(p, q)


def _lookup_dark_scale(dark_scale_dict: dict, filename: str) -> float:
    """Return the dark scale for a specific file, or 1.0 if not set."""
    return float(dark_scale_dict.get(filename, 1.0)) if dark_scale_dict else 1.0


# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════
def _rx_float(pattern: str, text: str, group: int = 1) -> Optional[float]:
    m = re.search(pattern, str(text))
    return float(m.group(group)) if m else None


def _pick_folder_origin_files(parent, title: str, start_dir: str = "") -> list:
    """Open a folder dialog and return all *.origin files in it (non-recursive)."""
    folder = QFileDialog.getExistingDirectory(parent, title, start_dir)
    if not folder:
        return []
    return sorted(str(p) for p in Path(folder).glob("*.origin"))


# ════════════════════════════════════════════════════════════════════════════
# PL_file — reads the .origin tab-separated format
# ════════════════════════════════════════════════════════════════════════════
class PL_file:
    """
    Reads a single .origin measurement file.

    File layout (1-based line numbers):
      Line 1     : column headers → consumed by pd.read_csv, renamed to
                   ["Energy", "Counts"]
      Lines 2-12 : metadata rows (Temp, int_time, Center_E, Exc_P, …)
      Line 13    : separator / empty row
      Lines 14+  : numeric data (Energy [eV], Counts)
    """

    def __init__(self, file_path: str):
        self.file_path = str(file_path)
        raw = pd.read_csv(self.file_path, sep="\t")
        raw.columns = ["Energy", "Counts"]
        self.header = raw.iloc[:12]
        self.df = raw.iloc[13:].copy().reset_index(drop=True)
        self.df["Energy"] = pd.to_numeric(self.df["Energy"], errors="coerce")
        self.df["Counts"] = pd.to_numeric(self.df["Counts"], errors="coerce")
        self.df = self.df.dropna().reset_index(drop=True)

        h = self.header
        _exc_p_str = str(h.iloc[11, 1])
        _m_mw = re.search(r"(\d+\.?\d*)\s*mW", _exc_p_str)
        _m_uw = re.search(r"(\d+\.?\d*)\s*[uµ]W", _exc_p_str)
        if _m_mw:
            exc_p_raw = float(_m_mw.group(1))
        elif _m_uw:
            exc_p_raw = float(_m_uw.group(1)) * 1e-3  # µW → mW
        else:
            exc_p_raw = None
        
        self.metadata: dict = {
            "Temp":          _rx_float(r"(\d+\.?\d*)\s*K",             str(h.iloc[1, 1])),
            "int_time":      _rx_float(r"(\d+\.?\d*)\s*s",             str(h.iloc[2, 1])),
            "Center_lambda": _rx_float(r"(\d+\.?\d*)\s*nm",            str(h.iloc[4, 1])),
            "Center_E":      _rx_float(r"nm\s*/\s*(\d+\.\d{3})\s*eV", str(h.iloc[4, 1])),
            "Exc_P":         exc_p_raw,
            "filename":      Path(self.file_path).name,
        }
    # Multiplicative scale applied to this file's counts before dark subtraction.
    # For PL files this stays 1.0.  For white files it is set by DarkScalingTab
    # so that  white × scale  matches  dark × dark_scale  at spectral edges.
    scale: float = 1.0

    def subtract_dark_and_normalize(self, dark_dict: dict, dark_scale_dict: dict = None) -> pd.DataFrame:
        """Returns (signal − dark × dark_scale) / int_time.

       
        dark_scale_dict — {Center_E: {int_time: scale}} applied to the dark.
        """
        ce = self.metadata["Center_E"]
        it = self.metadata["int_time"]
        if ce not in dark_dict:
            raise KeyError(f"No dark loaded for Center_E = {ce} eV")
        if it not in dark_dict[ce]:
            raise KeyError(f"No dark loaded for int_time = {it} s at Center_E = {ce} eV")
        dark = dark_dict[ce][it]
        xs = self.df["Energy"].to_numpy(dtype=float)
        ys = self.df["Counts"].to_numpy(dtype=float)
        xd = dark.df["Energy"].to_numpy(dtype=float)
        yd = dark.df["Counts"].to_numpy(dtype=float)
        ds = 1.0
        if dark_scale_dict is not None:
            ds = _lookup_dark_scale(dark_scale_dict, self.metadata["filename"])
        if len(ys) != len(yd):
            ord_d = np.argsort(xd)
            yd = np.interp(xs, xd[ord_d], yd[ord_d])
        return pd.DataFrame({"Energy": xs, "Counts": (ys * self.scale - yd * ds) / it})


# ════════════════════════════════════════════════════════════════════════════
# Bulk file loaders  →  nested dict  { Center_E : { int_time : PL_file } }
# ════════════════════════════════════════════════════════════════════════════
def _load_nested(files) -> tuple[dict, list]:
    store: dict = {}
    errors: list = []
    for fp in files:
        try:
            obj = PL_file(str(fp))
            ce = obj.metadata["Center_E"]
            it = obj.metadata["int_time"]
            if ce is None or it is None:
                errors.append(
                    f"Cannot parse Center_E / int_time from {Path(fp).name}"
                )
                continue
            store.setdefault(ce, {})[it] = obj
        except Exception as exc:
            errors.append(f"{Path(fp).name}: {exc}")
    return store, errors


# ════════════════════════════════════════════════════════════════════════════
# Halogen loader --> converts to energy and sorts
# ════════════════════════════════════════════════════════════════════════════
def load_halogen(path: str, apply_jacobian: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["Wavelength", "Counts"]
    lam = df["Wavelength"].to_numpy(dtype=float)
    i   = df["Counts"].to_numpy(dtype=float)
    e   = HC_EV_NM / lam
    if apply_jacobian:
        i = i * HC_EV_NM / (e ** 2)
    order = np.argsort(e)
    return pd.DataFrame({"Energy": e[order], "Counts": i[order]}).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Spectral correction ratio builder
BAD_POINTS_WARN_THRESHOLD = 0.20   # fraction above which a hard dialog is shown

def build_correction_ratios(
    white_dict: dict,
    dark_dict: dict,
    halogen_df: pd.DataFrame,
    dark_scale_dict: dict = None,
) -> tuple[dict, list, list]:
    """
    For every (Center_E, int_time) entry in white_dict:
      1. Dark-subtract and normalise the white spectrum.
      2. Interpolate the halogen reference onto the same energy grid.
      3. correction_coefficient = halogen / normalised_white

    Returns:
        ratios      – { Center_E : DataFrame{"Energy", "correction_coefficient"} }
        hard_errors – messages that should trigger a blocking dialog
        soft_warns  – messages to show inline (bad-point fraction below threshold)
    """
    ratios: dict = {}
    hard_errors: list = []
    soft_warns:  list = []
    e_halo = halogen_df["Energy"].to_numpy(dtype=float)
    i_halo = halogen_df["Counts"].to_numpy(dtype=float)

    for ce, time_dict in white_dict.items():
        for it, wf in time_dict.items():
            try:
                norm_df = wf.subtract_dark_and_normalize(dark_dict, dark_scale_dict)
                em = norm_df["Energy"].to_numpy(dtype=float)
                im = norm_df["Counts"].to_numpy(dtype=float)

                mask = (e_halo >= em.min() - 1e-12) & (e_halo <= em.max() + 1e-12)
                e_hw, i_hw = e_halo[mask], i_halo[mask]
                if len(e_hw) < 2:
                    hard_errors.append(
                        f"Halogen has <2 points in window Center_E = {ce:.3f} eV"
                    )
                    continue

                # np.interp requires ascending xp; sort both arrays
                ord_h = np.argsort(e_hw)
                e_hw_s, i_hw_s = e_hw[ord_h], i_hw[ord_h]
                ord_m = np.argsort(em)
                em_s  = em[ord_m]
                i_true_s = np.interp(em_s, e_hw_s, i_hw_s)
                # Restore original order
                i_true = i_true_s[np.argsort(ord_m)]
                # Reject non-positive values — they indicate the white
                # spectrum was over-scaled (white*scale < dark), which
                # would produce negative or huge correction coefficients.
                denom = np.where(im > 0, im, np.nan)
                n_bad = int(np.sum(np.isnan(denom)))
                if n_bad > 0:
                    frac = n_bad / len(denom)
                    msg  = (
                        f"Center_E = {ce:.3f} eV: {n_bad}/{len(denom)} "
                        f"({frac*100:.0f}%) points have non-positive normalised "
                        "white — dark scaling may be too aggressive."
                    )
                    if frac >= BAD_POINTS_WARN_THRESHOLD:
                        hard_errors.append(msg)
                    else:
                        soft_warns.append(msg)
                coeff = i_true / denom

                ratios[ce] = pd.DataFrame(
                    {"Energy": em, "correction_coefficient": coeff}
                )
            except Exception as exc:
                hard_errors.append(f"Center_E = {ce}: {exc}")

    return ratios, hard_errors, soft_warns


# ════════════════════════════════════════════════════════════════════════════
# Per-file whitelight correction
# ════════════════════════════════════════════════════════════════════════════
def apply_correction(
    pl: PL_file,
    dark_dict: dict,
    corr_dict: dict,
    dark_scale_dict: dict = None,
) -> pd.DataFrame:
    """Dark-subtracts, normalises, and applies the spectral correction."""
    norm = pl.subtract_dark_and_normalize(dark_dict, dark_scale_dict)
    ce   = pl.metadata["Center_E"]
    if ce not in corr_dict:
        raise KeyError(f"No correction loaded for Center_E = {ce} eV")

    coeff_df = corr_dict[ce]
    xp = norm["Energy"].to_numpy(dtype=float)
    yp = norm["Counts"].to_numpy(dtype=float)
    xc = coeff_df["Energy"].to_numpy(dtype=float)
    yc = coeff_df["correction_coefficient"].to_numpy(dtype=float)

    # # Drop NaN coefficients (from non-positive white values) before interpolating
    # valid = np.isfinite(yc)
    # xc, yc = xc[valid], yc[valid]

    # np.interp requires xc in ascending order; .origin files may be descending
    order_c = np.argsort(xc)
    xc, yc  = xc[order_c], yc[order_c]

    order_p = np.argsort(xp)
    xp_s    = xp[order_p]
    yp_s    = yp[order_p]

    ci = np.interp(xp_s, xc, yc)

    # Restore original point order
    restore = np.argsort(order_p)
    ci = ci[restore]

    return pd.DataFrame({"Energy": xp, "Counts": yp * ci})


def stitch_once(left_df: pd.DataFrame, right_df: pd.DataFrame, x_min: float,x_max:float) -> tuple[pd.DataFrame, float]:
    xl = left_df["Energy"].to_numpy(dtype=float)
    yl = left_df["Counts"].to_numpy(dtype=float)
    xr = right_df["Energy"].to_numpy(dtype=float)
    yr = right_df["Counts"].to_numpy(dtype=float)

    maskl = np.where((x_min < xl) & (xl < x_max))
    maskr = np.where((x_min < xr) & (xr < x_max))
    yl0 = np.mean(yl[maskl])
    yr0 = np.mean(yr[maskr])



    ratio = yl0 / yr0
    # yl_scaled = yl * ratio
    yr_scaled = yr * ratio
    left_keep = xl <= x_min
    right_keep = xr > x_min

    x_out = np.concatenate([xl[left_keep], xr[right_keep]])
    # y_out = np.concatenate([yl_scaled[left_keep], yr[right_keep]])
    y_out = np.concatenate([yl[left_keep], yr_scaled[right_keep]])
    order = np.argsort(x_out)

    out = pd.DataFrame({"Energy": x_out[order], "Counts": y_out[order]}).reset_index(drop=True)
    return out, float(ratio)


# ════════════════════════════════════════════════════════════════════════════
# Plot widget
# ════════════════════════════════════════════════════════════════════════════
class PlotWidget(QWidget):
    def __init__(self, figsize: tuple = (7, 4.5), min_h: int = 320, log_y: bool = False):
        super().__init__()
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(min_h)
        self.ax = self.figure.add_subplot(111)
        self._log_y = log_y
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)

    def _safe_draw(self):
        try:
            if self.canvas is not None and shiboken6.isValid(self.canvas):
                self.canvas.draw()
        except RuntimeError:
            pass

    def clear(self):
        self.ax.clear()
        self._safe_draw()

    def plot_series(
        self,
        series: list,
        title: str = "",
        log_y: Optional[bool] = None,
        xlabel: str = "Energy (eV)",
        ylabel: str = "Counts",
    ):
        """series items: (x_array, y_array, label)  or  (DataFrame_2col, label)"""
        self.ax.clear()
        use_log = self._log_y if log_y is None else log_y

        for item in series:
            if len(item) == 2:
                df, label = item
                x = df.iloc[:, 0].to_numpy(float)
                y = df.iloc[:, 1].to_numpy(float)
            else:
                x, y, label = item

            y_plot = np.where(y > 0, y, np.nan) if use_log else y
            self.ax.plot(x, y_plot, label=label)

        if use_log:
            self.ax.set_yscale("log")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        if series:
            self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self._safe_draw()


# ════════════════════════════════════════════════════════════════════════════
# CheckableTable — table with a checkbox in column 0
# ════════════════════════════════════════════════════════════════════════════
class CheckableTable(QTableWidget):
    """
    A read-only table whose first column contains a check-box.
    All other columns are plain text.

    Usage:
        table = CheckableTable(["Filename", "Center_E", ...])
        table.populate([["file.origin", "1.150", ...], ...])
        checked = table.checked_rows()   # list of row indices
    """

    def __init__(self, cols: list):
        super().__init__(0, len(cols) + 1)           # +1 for the checkbox col
        self.setHorizontalHeaderLabels(["Plot"] + list(cols))
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, self.columnCount()):
            self.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setMaximumHeight(220)

    def populate(self, rows: list, keep_checks: bool = False):
        """
        Refill the table.  When keep_checks=True the checked state of
        existing rows is preserved (used after applying correction to
        update the ✓ columns without losing the user's selection).
        """
        old_checks: list[bool] = []
        if keep_checks:
            old_checks = [self._is_checked(r) for r in range(self.rowCount())]

        self.setRowCount(0)
        for idx, row in enumerate(rows):
            r = self.rowCount()
            self.insertRow(r)

            # Checkbox cell
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            # Restore previous state if available, else default checked
            checked = old_checks[idx] if idx < len(old_checks) else True
            chk.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.setItem(r, 0, chk)

            # Data cells
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val) if val is not None else "—")
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r, c + 1, item)

    def _is_checked(self, row: int) -> bool:
        item = self.item(row, 0)
        return item is not None and item.checkState() == Qt.Checked

    def checked_rows(self) -> list:
        """Return list of row indices that are currently checked."""
        return [r for r in range(self.rowCount()) if self._is_checked(r)]

    def color_column(self, col_name: str, color_map: dict):
        """
        Color cells in the named column according to color_map {cell_text: QColor}.
        col_name matches the header label (excluding the leading "Plot" checkbox col).
        """
        headers = [self.horizontalHeaderItem(c).text()
                   for c in range(self.columnCount())]
        try:
            col = headers.index(col_name)
        except ValueError:
            return
        for r in range(self.rowCount()):
            item = self.item(r, col)
            if item and item.text() in color_map:
                item.setForeground(color_map[item.text()])

    def check_all(self, state: bool = True):
        for r in range(self.rowCount()):
            item = self.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked if state else Qt.Unchecked)


# ════════════════════════════════════════════════════════════════════════════
# Plain read-only MetaTable (still used in Calibration tab)
# ════════════════════════════════════════════════════════════════════════════
class MetaTable(QTableWidget):
    def __init__(self, cols: list):
        super().__init__(0, len(cols))
        self.setHorizontalHeaderLabels(cols)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setMaximumHeight(180)

    def populate(self, rows: list):
        self.setRowCount(0)
        for row in rows:
            r = self.rowCount()
            self.insertRow(r)
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val) if val is not None else "—")
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r, c, item)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Calibration
# ════════════════════════════════════════════════════════════════════════════
class CalibrationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.halogen_df:  Optional[pd.DataFrame] = None
        self.dark_dict:   dict = {}
        self.white_dict:  dict = {}
        self.pl_files:    list = []
        self._dark_row_map:  list = []   # [(ce, it), …] parallel to dark_table rows
        self._white_row_map: list = []   # [(ce, it), …] parallel to white_table rows
        # Set by MainWindow after downstream tabs are created:
        self.on_pl_files_changed = None   # callable() → refresh PLAnalysisTab table
        # Remembered last-used directories — persisted in session JSON.
        self._last_dirs: dict = {"load": "", "save": "", "replay": ""}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Session control ──────────────────────────────────────────────────
        session_row = QHBoxLayout()
        self.new_session_btn = QPushButton("New Session")
        self.new_session_btn.setToolTip(
            "Clears dark and white-light spectra from memory and resets the "
            "correction.  Halogen reference is kept."
        )
        self.replay_btn = QPushButton("Load & Replay JSON …")
        self.replay_btn.setToolTip(
            "Load a PL analysis JSON file and re-apply all analysis steps "
            "(scaling, correction, stitching) exactly as recorded, given that the "
            "same source files are currently loaded."
        )
        session_row.addWidget(self.new_session_btn)
        session_row.addWidget(self.replay_btn)
        session_row.addStretch()
        layout.addLayout(session_row)

        # ── 1. Halogen ──────────────────────────────────────────────────────
        halo_box = QGroupBox("1.  Halogen Lamp Reference")
        hg = QHBoxLayout(halo_box)
        self.halo_load_btn = QPushButton("Load HalogenLamp_Spectrum.txt …")
        self.halo_label = QLabel("Not loaded")
        self.halo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        hg.addWidget(self.halo_load_btn)
        hg.addWidget(self.halo_label)
        layout.addWidget(halo_box)

        # ── 2. Dark spectra ─────────────────────────────────────────────────
        dark_box = QGroupBox("2.  Dark Spectra  (checked files used in analysis)")
        dg = QVBoxLayout(dark_box)
        dark_row = QHBoxLayout()
        self.dark_load_btn   = QPushButton("Load Files …")
        self.dark_folder_btn = QPushButton("Load Folder …")
        self.dark_clear_btn  = QPushButton("Clear")
        self.dark_chk_all    = QPushButton("Check All")
        self.dark_unchk_all  = QPushButton("Uncheck All")
        self.dark_status     = QLabel("0 files")
        for w in [self.dark_load_btn, self.dark_folder_btn, self.dark_clear_btn,
                  self.dark_chk_all, self.dark_unchk_all]:
            dark_row.addWidget(w)
        dark_row.addWidget(self.dark_status, 1)
        dg.addLayout(dark_row)
        self.dark_table = CheckableTable(["Center_E (eV)", "int_time (s)", "Filename"])
        dg.addWidget(self.dark_table)
        layout.addWidget(dark_box)

        # ── 3. White-light spectra ──────────────────────────────────────────
        white_box = QGroupBox("3.  White-Light Spectra  (checked files used in analysis)")
        wg = QVBoxLayout(white_box)
        white_row = QHBoxLayout()
        self.white_load_btn   = QPushButton("Load Files …")
        self.white_folder_btn = QPushButton("Load Folder …")
        self.white_clear_btn  = QPushButton("Clear")
        self.white_chk_all    = QPushButton("Check All")
        self.white_unchk_all  = QPushButton("Uncheck All")
        self.white_status     = QLabel("0 files")
        for w in [self.white_load_btn, self.white_folder_btn, self.white_clear_btn,
                  self.white_chk_all, self.white_unchk_all]:
            white_row.addWidget(w)
        white_row.addWidget(self.white_status, 1)
        wg.addLayout(white_row)
        self.white_table = CheckableTable(["Center_E (eV)", "int_time (s)", "Filename", "Dark"])
        wg.addWidget(self.white_table)
        layout.addWidget(white_box)

        # ── 4. PL Measurement Files ─────────────────────────────────────────
        pl_box = QGroupBox("4.  PL Measurement Files  (checked files used in analysis)")
        pg = QVBoxLayout(pl_box)
        pl_row = QHBoxLayout()
        self.pl_load_btn   = QPushButton("Load Files …")
        self.pl_folder_btn = QPushButton("Load Folder …")
        self.pl_clear_btn  = QPushButton("Clear")
        self.pl_chk_all    = QPushButton("Check All")
        self.pl_unchk_all  = QPushButton("Uncheck All")
        self.pl_status     = QLabel("0 files")
        for w in [self.pl_load_btn, self.pl_folder_btn, self.pl_clear_btn,
                  self.pl_chk_all, self.pl_unchk_all]:
            pl_row.addWidget(w)
        pl_row.addWidget(self.pl_status, 1)
        pg.addLayout(pl_row)
        self.pl_table = CheckableTable([
            "Filename", "Center_E (eV)", "int_time (s)", "Temp (K)", "Power (mW)", "Dark",
        ])
        pg.addWidget(self.pl_table)
        layout.addWidget(pl_box)

        layout.addStretch()

        # Wire up
        self.new_session_btn.clicked.connect(self._new_session)
        self.replay_btn.clicked.connect(self._request_replay)
        self.halo_load_btn.clicked.connect(self._load_halogen)
        self.dark_load_btn.clicked.connect(self._load_dark_files)
        self.dark_folder_btn.clicked.connect(self._load_dark_folder)
        self.dark_clear_btn.clicked.connect(self._clear_dark)
        self.dark_chk_all.clicked.connect(lambda: self.dark_table.check_all(True))
        self.dark_unchk_all.clicked.connect(lambda: self.dark_table.check_all(False))
        self.white_load_btn.clicked.connect(self._load_white_files)
        self.white_folder_btn.clicked.connect(self._load_white_folder)
        self.white_clear_btn.clicked.connect(self._clear_white)
        self.white_chk_all.clicked.connect(lambda: self.white_table.check_all(True))
        self.white_unchk_all.clicked.connect(lambda: self.white_table.check_all(False))
        self.pl_load_btn.clicked.connect(self._load_pl_files)
        self.pl_folder_btn.clicked.connect(self._load_pl_folder)
        self.pl_clear_btn.clicked.connect(self._clear_pl)
        self.pl_chk_all.clicked.connect(lambda: self.pl_table.check_all(True))
        self.pl_unchk_all.clicked.connect(lambda: self.pl_table.check_all(False))

        self._halogen_path: Optional[str] = None
        self._restored_pl_paths: list = []
        # Callbacks set by MainWindow after tabs are created — separate per mode
        self.get_std_pl_done_groups    = None   # () → set[float]
        self.get_std_wl_done_groups    = None   # () → set[float]
        self.get_std_dark_scales       = None   # () → {filename: scale}
        self.get_std_power_stitch_logs = None   # () → {power: [blend_log]}
        self.get_pip_pl_done_groups    = None   # () → set[float]
        self.get_pip_wl_done_groups    = None   # () → set[float]
        self.get_pip_dark_scales       = None   # () → {filename: scale}
        self.get_pip_power_stitch_logs = None   # () → {power: [blend_log]}
        self.get_pip_ce_done               = None   # () → {power: {ce: bool}}
        self.get_pip_correction_applied    = None   # () → bool
        self.on_replay_requested           = None   # (json_path: str) → triggers MainWindow replay
        self._restore_session()

    # ── Directory memory helpers ─────────────────────────────────────────────
    def _get_last_dir(self, category: str) -> str:
        """Return the last-used directory for the given category ("load"/"save"/"replay")."""
        return self._last_dirs.get(category, "")

    def _set_last_dir(self, category: str, path: str):
        """Update the remembered directory from a selected file/folder path and save."""
        p = Path(path)
        new_dir = str(p.parent if p.is_file() else p)
        if new_dir and self._last_dirs.get(category) != new_dir:
            self._last_dirs[category] = new_dir
            self._save_session()

    # ── Session control ──────────────────────────────────────────────────────
    def _new_session(self):
        """Clear all loaded files (dark, white, PL). Halogen reference is kept."""
        self.dark_dict.clear()
        self.white_dict.clear()
        self._dark_row_map.clear()
        self._white_row_map.clear()
        self._refresh_dark_table()
        self._refresh_white_table()
        self.pl_files.clear()
        self.refresh_pl_table()
        self._save_session()
        if self.on_pl_files_changed:
            self.on_pl_files_changed()

    def _request_replay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PL analysis JSON",
            self._get_last_dir("replay"),
            "JSON files (*.json);;All files (*)"
        )
        if path:
            self._set_last_dir("replay", path)
            if self.on_replay_requested:
                self.on_replay_requested(path)

    # ── Session persistence ──────────────────────────────────────────────────
    def _session_path(self) -> Path:
        return Path(__file__).resolve().parent / SESSION_FILE

    def _save_session(self):
        dark_paths  = [v.file_path for td in self.dark_dict.values()  for v in td.values()]
        white_paths = [v.file_path for td in self.white_dict.values() for v in td.values()]
        pl_paths    = [pf.file_path for pf in self.pl_files]

        # White file scales live on pf.scale (shared between modes)
        white_scales = {
            pf.metadata["filename"]: pf.scale
            for td in self.white_dict.values()
            for pf in td.values()
            if abs(pf.scale - 1.0) > 1e-9
        }

        def _serialize_mode(get_pl_done, get_wl_done, get_dark_scales, get_stitch_logs):
            pl_done = sorted(get_pl_done()) if get_pl_done else []
            wl_done = sorted(get_wl_done()) if get_wl_done else []
            raw_ds = get_dark_scales() if get_dark_scales else {}
            # Flat {filename: scale} — values are already plain floats.
            dark_scales = {fname: float(ds) for fname, ds in raw_ds.items()
                           if isinstance(ds, (int, float))}
            power_stitch_logs: dict = {}
            if get_stitch_logs:
                for pw, log in get_stitch_logs().items():
                    if log:
                        power_stitch_logs[f"{pw:.4f}"] = log
            return {
                "dark_scales":             dark_scales,
                "pl_done_groups":          pl_done,
                "wl_done_groups":          wl_done,
                "power_stitch_blend_logs": power_stitch_logs,
            }

        try:
            self._session_path().write_text(json.dumps({
                "halogen_path": self._halogen_path,
                "last_dirs":    self._last_dirs,
                "dark_paths":   dark_paths,
                "white_paths":  white_paths,
                "pl_paths":     pl_paths,
                "white_scales": white_scales,
                "standard": _serialize_mode(
                    self.get_std_pl_done_groups,
                    self.get_std_wl_done_groups,
                    self.get_std_dark_scales,
                    self.get_std_power_stitch_logs,
                ),
                "pipeline": {
                    **_serialize_mode(
                        self.get_pip_pl_done_groups,
                        self.get_pip_wl_done_groups,
                        self.get_pip_dark_scales,
                        self.get_pip_power_stitch_logs,
                    ),
                    "pip_ce_done": {
                        f"{pw:.4f}": {f"{ce:.4f}": bool(done)
                                      for ce, done in ce_dict.items()}
                        for pw, ce_dict in (
                            self.get_pip_ce_done()
                            if self.get_pip_ce_done
                            # Fallback during startup: callback not yet wired by
                            # MainWindow, so use the data loaded from the last
                            # session to avoid overwriting it with an empty dict.
                            else getattr(self, "_restored_pip_ce_done", {})
                        ).items()
                    },
                    "correction_applied": bool(
                        self.get_pip_correction_applied()
                        if self.get_pip_correction_applied
                        else getattr(self, "_restored_pip_correction_applied", False)
                    ),
                },
            }, indent=2))
        except Exception:
            pass

    def _restore_session(self):
        """Called once at startup: reload saved halogen / dark / white paths."""
        p = self._session_path()
        data: dict = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                pass

        # Halogen: prefer saved path, fall back to file-adjacent auto-detect
        hal = data.get("halogen_path")
        if hal and Path(hal).exists():
            self._do_load_halogen(hal, silent=True)
        else:
            here = Path(__file__).resolve().parent
            for candidate in [here / HALOGEN_FILENAME,
                               here / "0000001_Python" / HALOGEN_FILENAME]:
                if candidate.exists():
                    self._do_load_halogen(str(candidate), silent=True)
                    break

        # Dark
        dark_paths = [fp for fp in data.get("dark_paths", []) if Path(fp).exists()]
        if dark_paths:
            self._ingest_dark(dark_paths)

        # White
        white_paths = [fp for fp in data.get("white_paths", []) if Path(fp).exists()]
        if white_paths:
            self._ingest_white(white_paths)

        # White scales (live on pf.scale, shared between modes)
        white_scales = data.get("white_scales", {})
        for td in self.white_dict.values():
            for pf in td.values():
                fname = pf.metadata["filename"]
                if fname in white_scales:
                    pf.scale = float(white_scales[fname])

        # Last-used directories
        saved_dirs = data.get("last_dirs", {})
        for key in ("load", "save", "replay"):
            d = saved_dirs.get(key, "")
            if d and Path(d).is_dir():
                self._last_dirs[key] = d

        # PL files — deferred until MainWindow calls _ingest_pl
        self._restored_pl_paths = [fp for fp in data.get("pl_paths", []) if Path(fp).exists()]

        def _deserialize_mode(mode_data: dict):
            raw = mode_data.get("dark_scales", {})
            # New flat format: {filename: scale}  (values are plain numbers).
            # Old nested formats (2-level or 3-level dicts) can't be mapped to filenames;
            # discard them — the user will need to redo dark scaling.
            is_flat = all(isinstance(v, (int, float)) for v in raw.values()) if raw else True
            if is_flat:
                dark_scales = {k: float(v) for k, v in raw.items()}
            else:
                dark_scales = {}
            pl_done = set(mode_data.get("pl_done_groups", []))
            wl_done = set(mode_data.get("wl_done_groups", []))
            raw_psl = mode_data.get("power_stitch_blend_logs", {})
            psl     = {float(p_str): log for p_str, log in raw_psl.items()}
            return dark_scales, pl_done, wl_done, psl

        # Backward-compatibility: old sessions have flat keys → treat as standard mode
        if "standard" in data:
            std_raw = data["standard"]
            pip_raw = data.get("pipeline", {})
        else:
            std_raw = data   # migrate old flat format to standard mode
            pip_raw = {}

        (self._restored_std_dark_scales,
         self._restored_std_pl_done_groups,
         self._restored_std_wl_done_groups,
         self._restored_std_power_stitch_logs) = _deserialize_mode(std_raw)

        (self._restored_pip_dark_scales,
         self._restored_pip_pl_done_groups,
         self._restored_pip_wl_done_groups,
         self._restored_pip_power_stitch_logs) = _deserialize_mode(pip_raw)

        raw_pcd = pip_raw.get("pip_ce_done", {})
        self._restored_pip_ce_done = {
            float(p_str): {float(ce_str): bool(done)
                           for ce_str, done in ce_dict.items()}
            for p_str, ce_dict in raw_pcd.items()
            if isinstance(ce_dict, dict)
        }
        self._restored_pip_correction_applied = bool(
            pip_raw.get("correction_applied", False)
        )

    # ── Halogen ──────────────────────────────────────────────────────────────
    def _load_halogen(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HalogenLamp_Spectrum.txt",
            self._get_last_dir("load"),
            "Text files (*.txt);;All files (*)"
        )
        if path:
            self._set_last_dir("load", path)
            self._do_load_halogen(path)

    def _do_load_halogen(self, path: str, silent: bool = False):
        try:
            self.halogen_df = load_halogen(path)
            self._halogen_path = path
            self.halo_label.setText(
                f"Loaded:  {Path(path).name}  ({len(self.halogen_df)} pts)"
            )
            self._save_session()
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Halogen Error", str(exc))

    # ── Dark ─────────────────────────────────────────────────────────────────
    def _ingest_dark(self, paths: list):
        if not paths:
            return
        new, errors = _load_nested(paths)
        for ce, td in new.items():
            self.dark_dict.setdefault(ce, {}).update(td)
        self._refresh_dark_table()
        self._save_session()
        if errors:
            QMessageBox.warning(self, "Dark — load warnings", "\n".join(errors))

    def _load_dark_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Dark Files",
            self._get_last_dir("load"),
            "Origin files (*.origin);;All files (*)"
        )
        if paths:
            self._set_last_dir("load", paths[0])
        self._ingest_dark(paths)

    def _load_dark_folder(self):
        files = _pick_folder_origin_files(self, "Select Dark Folder", self._get_last_dir("load"))
        if files:
            self._set_last_dir("load", files[0])
        self._ingest_dark(files)

    def _clear_dark(self):
        self.dark_dict.clear()
        self._refresh_dark_table()
        self._save_session()

    def _refresh_dark_table(self):
        self._dark_row_map = []
        rows = []
        for ce in sorted(self.dark_dict):
            for it in sorted(self.dark_dict[ce]):
                self._dark_row_map.append((ce, it))
                rows.append([
                    f"{ce:.3f}", f"{it:.3f}",
                    self.dark_dict[ce][it].metadata["filename"],
                ])
        self.dark_table.populate(rows)
        n = sum(len(v) for v in self.dark_dict.values())
        self.dark_status.setText(
            f"{n} file(s)  |  {len(self.dark_dict)} Center_E value(s)"
        )
        # Refresh PL and white tables so their Dark-match column updates
        if self.pl_files:
            self.refresh_pl_table()
        if self.white_dict:
            self._refresh_white_table()

    # ── White ────────────────────────────────────────────────────────────────
    def _ingest_white(self, paths: list):
        if not paths:
            return
        new, errors = _load_nested(paths)
        for ce, td in new.items():
            self.white_dict.setdefault(ce, {}).update(td)
        self._refresh_white_table()
        self._save_session()
        if errors:
            QMessageBox.warning(self, "White — load warnings", "\n".join(errors))

    def _load_white_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select White-Light Files",
            self._get_last_dir("load"),
            "Origin files (*.origin);;All files (*)"
        )
        if paths:
            self._set_last_dir("load", paths[0])
        self._ingest_white(paths)

    def _load_white_folder(self):
        files = _pick_folder_origin_files(self, "Select White Folder", self._get_last_dir("load"))
        if files:
            self._set_last_dir("load", files[0])
        self._ingest_white(files)

    def _clear_white(self):
        self.white_dict.clear()
        self._refresh_white_table()
        self._save_session()

    def _refresh_white_table(self):
        self._white_row_map = []
        rows = []
        for ce in sorted(self.white_dict):
            for it in sorted(self.white_dict[ce]):
                self._white_row_map.append((ce, it))
                rows.append([
                    f"{ce:.3f}", f"{it:.3f}",
                    self.white_dict[ce][it].metadata["filename"],
                    self._dark_match_symbol(ce, it),
                ])
        self.white_table.populate(rows)
        self.white_table.color_column("Dark", {
            "✔": QColor("#2e7d32"), "≈": QColor("#e65100"), "✘": QColor("#c62828"),
        })
        n = sum(len(v) for v in self.white_dict.values())
        self.white_status.setText(
            f"{n} file(s)  |  {len(self.white_dict)} Center_E value(s)"
        )

    # ── PL Measurement Files ──────────────────────────────────────────────────
    def _ingest_pl(self, paths: list):
        if not paths:
            return
        self.pl_files.clear()
        errors = []
        for fp in paths:
            try:
                self.pl_files.append(PL_file(str(fp)))
            except Exception as exc:
                errors.append(f"{Path(fp).name}: {exc}")
        self.refresh_pl_table()
        self._save_session()
        if self.on_pl_files_changed:
            self.on_pl_files_changed()
        if errors:
            QMessageBox.warning(self, "PL load warnings", "\n".join(errors))

    def _load_pl_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PL Measurement Files",
            self._get_last_dir("load"),
            "Origin files (*.origin);;All files (*)"
        )
        if paths:
            self._set_last_dir("load", paths[0])
        self._ingest_pl(paths)

    def _load_pl_folder(self):
        files = _pick_folder_origin_files(self, "Select PL Folder", self._get_last_dir("load"))
        if files:
            self._set_last_dir("load", files[0])
        self._ingest_pl(files)

    def _clear_pl(self):
        self.pl_files.clear()
        self.refresh_pl_table()
        self._save_session()
        if self.on_pl_files_changed:
            self.on_pl_files_changed()

    def _dark_match_symbol(self, ce, it) -> str:
        """Return ✔ if an exact (Center_E, int_time) dark match exists, else ✘."""
        if ce is None or it is None:
            return "✘"
        if ce in self.dark_dict and it in self.dark_dict[ce]:
            return "✔"
        return "✘"

    def refresh_pl_table(self):
        """Refresh the PL file table."""
        rows = []
        for pf in self.pl_files:
            m  = pf.metadata
            ce = m["Center_E"]
            it = m["int_time"]
            rows.append([
                m["filename"],
                f"{ce:.3f}" if ce is not None else "—",
                f"{it:.3f}" if it is not None else "—",
                f"{m['Temp']:.1f}" if m["Temp"] is not None else "—",
                f"{m['Exc_P']:.2f}" if m["Exc_P"] is not None else "—",
                self._dark_match_symbol(ce, it),
            ])
        self.pl_table.populate(rows, keep_checks=True)
        self.pl_table.color_column("Dark", {
            "✔": QColor("#2e7d32"), "✘": QColor("#c62828"),
        })
        self.pl_status.setText(f"{len(self.pl_files)} file(s)")

    def checked_pl_files(self) -> list:
        """Return PL_file objects whose table row is checked."""
        checked_idx = self.pl_table.checked_rows()
        return [self.pl_files[i] for i in checked_idx if i < len(self.pl_files)]

    def checked_dark_dict(self) -> dict:
        """Return dark_dict filtered to only the checked rows."""
        result: dict = {}
        for idx in self.dark_table.checked_rows():
            if idx < len(self._dark_row_map):
                ce, it = self._dark_row_map[idx]
                result.setdefault(ce, {})[it] = self.dark_dict[ce][it]
        return result

    def checked_white_dict(self) -> dict:
        """Return white_dict filtered to only the checked rows."""
        result: dict = {}
        for idx in self.white_table.checked_rows():
            if idx < len(self._white_row_map):
                ce, it = self._white_row_map[idx]
                result.setdefault(ce, {})[it] = self.white_dict[ce][it]
        return result


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Apply Corrections
# ════════════════════════════════════════════════════════════════════════════
class CorrectionsTab(QWidget):
    """
    Two-section tab:
      A. Dark Subtraction & Integration-Time Normalisation
         Shows dark-subtracted spectra for all checked PL files.
      B. Spectral Correction Coefficients
         Builds correction ratios from halogen / white / dark and plots them.
    """

    def __init__(self, get_pl_files, get_dark_dict, get_white_dict, get_halogen,
                 get_dark_scale_dict=None):
        super().__init__()
        self.get_pl_files        = get_pl_files
        self.get_dark_dict       = get_dark_dict
        self.get_white_dict      = get_white_dict
        self.get_halogen         = get_halogen
        self.get_dark_scale_dict = get_dark_scale_dict or (lambda: {})

        self.normalized:      dict = {}   # filename → dark-sub / normalised df
        self.correction_dict: dict = {}   # Center_E → correction_coefficient df
        self.on_data_changed  = None      # () → None; called when either step completes

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── A. Dark Subtraction ──────────────────────────────────────────────
        sub_box = QGroupBox("A.  Dark Subtraction  &  Integration-Time Normalisation")
        ag = QVBoxLayout(sub_box)
        sub_row = QHBoxLayout()
        self.sub_btn    = QPushButton("Apply Dark Subtraction to All Checked PL Files")
        self.sub_status = QLabel("Not applied yet")
        sub_row.addWidget(self.sub_btn)
        sub_row.addWidget(self.sub_status, 1)
        ag.addLayout(sub_row)
        self.sub_plot = PlotWidget(min_h=300, log_y=True)
        ag.addWidget(self.sub_plot)
        layout.addWidget(sub_box)

        # ── B. Spectral Correction Coefficients ──────────────────────────────
        corr_box = QGroupBox("B.  Spectral Correction Coefficients")
        cg = QVBoxLayout(corr_box)
        corr_row = QHBoxLayout()
        self.build_btn   = QPushButton("Build Correction Ratios")
        self.corr_status = QLabel("Not built yet")
        corr_row.addWidget(self.build_btn)
        corr_row.addWidget(self.corr_status, 1)
        cg.addLayout(corr_row)
        self.corr_warn_label = QLabel("")
        self.corr_warn_label.setStyleSheet("color: #e65100; font-size: 11px;")
        self.corr_warn_label.setWordWrap(True)
        self.corr_warn_label.setVisible(False)
        cg.addWidget(self.corr_warn_label)
        self.corr_plot = PlotWidget(min_h=300)
        cg.addWidget(self.corr_plot)
        layout.addWidget(corr_box)

        self.sub_btn.clicked.connect(self._apply_dark_sub)
        self.build_btn.clicked.connect(self._build_correction)

    # ── Invalidation (called by pipeline reset callbacks) ────────────────────
    def invalidate_dark_sub(self):
        self.normalized.clear()
        self.sub_status.setText("⚠ PL scaling changed — re-apply dark subtraction.")
        self.sub_plot.clear()

    def invalidate_correction(self):
        self.correction_dict.clear()
        self.corr_status.setText("⚠ White scaling changed — rebuild correction ratios.")
        self.corr_warn_label.setVisible(False)
        self.corr_plot.clear()

    # ── Dark Subtraction ─────────────────────────────────────────────────────
    def _apply_dark_sub(self):
        pl_files  = self.get_pl_files()
        dark_dict = self.get_dark_dict()
        if not pl_files:
            QMessageBox.warning(self, "Missing",
                "Load and check PL files in the Calibration tab first.")
            return
        if not dark_dict:
            QMessageBox.warning(self, "Missing",
                "Load and check dark files in the Calibration tab first.")
            return

        errors = []
        self.normalized.clear()
        dark_scale_dict = self.get_dark_scale_dict()
        for pf in pl_files:
            fname = pf.metadata["filename"]
            try:
                self.normalized[fname] = pf.subtract_dark_and_normalize(dark_dict, dark_scale_dict)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

        n_ok = len(self.normalized)
        self.sub_status.setText(
            f"{n_ok} / {len(pl_files)} file(s) dark-subtracted & normalised."
        )
        series = []
        for pf in pl_files:
            fname = pf.metadata["filename"]
            if fname in self.normalized:
                df  = self.normalized[fname]
                p   = pf.metadata.get("Exc_P")
                lbl = f"{p:.2f} mW" if p is not None else fname
                series.append((df["Energy"].to_numpy(), df["Counts"].to_numpy(), lbl))
        if series:
            self.sub_plot.plot_series(
                series, "Dark-Subtracted & Normalised PL Spectra", log_y=True
            )
        if errors:
            QMessageBox.warning(self, "Dark subtraction warnings", "\n".join(errors))
        if self.normalized and self.on_data_changed:
            self.on_data_changed()

    # ── Build Correction Ratios ───────────────────────────────────────────────
    def _build_correction(self):
        halogen    = self.get_halogen()
        white_dict = self.get_white_dict()
        dark_dict  = self.get_dark_dict()
        if halogen is None:
            QMessageBox.warning(self, "Missing",
                "Load the halogen reference in the Calibration tab first.")
            return
        if not white_dict:
            QMessageBox.warning(self, "Missing",
                "Load and check white-light files in the Calibration tab first.")
            return
        if not dark_dict:
            QMessageBox.warning(self, "Missing",
                "Load and check dark files in the Calibration tab first.")
            return
        try:
            self.correction_dict, hard_errors, soft_warns = build_correction_ratios(
                white_dict, dark_dict, halogen, self.get_dark_scale_dict()
            )
            series = [
                (
                    df["Energy"].to_numpy(),
                    df["correction_coefficient"].to_numpy(),
                    f"Center_E = {ce:.3f} eV",
                )
                for ce, df in sorted(self.correction_dict.items())
            ]
            self.corr_plot.plot_series(
                series, "Spectral Correction Coefficients", log_y=True,
                ylabel="Correction coefficient"
            )
            keys = ", ".join(f"{ce:.3f}" for ce in sorted(self.correction_dict))
            self.corr_status.setText(
                f"Built for {len(self.correction_dict)} window(s):  {keys} eV"
            )
            # Soft warnings (< threshold) → shown inline below the plot button
            if soft_warns:
                self.corr_warn_label.setText("ℹ " + "  |  ".join(soft_warns))
                self.corr_warn_label.setVisible(True)
            else:
                self.corr_warn_label.setVisible(False)
            # Hard warnings (≥ threshold) → blocking dialog
            if hard_errors:
                QMessageBox.warning(self, "Build warnings", "\n".join(hard_errors))
            if self.correction_dict and self.on_data_changed:
                self.on_data_changed()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PL Analysis
# ════════════════════════════════════════════════════════════════════════════
class PLAnalysisTab(QWidget):
    """
    Shows all loaded PL files with checkboxes.  The user selects which spectra
    to apply spectral correction to and plot.  Stitching uses the checked subset.
    """

    def __init__(self, get_dark_dict, get_corr_dict, get_pl_files, get_normalized,
                 get_dark_scale_dict=None):
        super().__init__()
        self.get_dark_dict       = get_dark_dict
        self.get_corr_dict       = get_corr_dict
        self.get_pl_files        = get_pl_files
        self.get_normalized      = get_normalized
        self.get_dark_scale_dict = get_dark_scale_dict or (lambda: {})
        self.corrected: dict = {}   # filename → whitelight-corrected df
        self.on_data_changed = None      # () → None; called when correction is applied

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── File list with checkboxes ────────────────────────────────────────
        list_box = QGroupBox("PL Spectra — select files to correct and plot")
        lg = QVBoxLayout(list_box)
        list_btns = QHBoxLayout()
        self.apply_btn          = QPushButton("Apply Correction to Checked")
        self.plot_raw_btn       = QPushButton("Plot Dark-Subtracted  (checked)")
        self.plot_btn           = QPushButton("Plot Corrected  (checked)")
        self.chk_all            = QPushButton("Check All")
        self.unchk_all          = QPushButton("Uncheck All")
        self.stitch_power_btn   = QPushButton("Stitch Power by Power →")
        self.stitch_power_btn.setToolTip(
            "Stitch each power level separately with its own blend windows.\n"
            "Results accumulate — save collectively at the end."
        )
        self.corr_status  = QLabel("Correction not applied")
        for b in [self.apply_btn, self.plot_raw_btn, self.plot_btn,
                  self.chk_all, self.unchk_all, self.stitch_power_btn]:
            list_btns.addWidget(b)
        list_btns.addWidget(self.corr_status, 1)
        lg.addLayout(list_btns)
        self.pl_table = CheckableTable([
            "Filename", "Center_E (eV)", "int_time (s)", "Temp (K)", "Power (mW)", "Corrected ✓",
        ])
        lg.addWidget(self.pl_table)
        layout.addWidget(list_box)

        # ── Plot ─────────────────────────────────────────────────────────────
        self.pl_plot = PlotWidget(min_h=420, log_y=True)
        layout.addWidget(self.pl_plot)

        self.apply_btn.clicked.connect(self._apply_checked)
        self.plot_raw_btn.clicked.connect(self._plot_raw)
        self.plot_btn.clicked.connect(self._plot_corrected)
        self.chk_all.clicked.connect(lambda: self.pl_table.check_all(True))
        self.unchk_all.clicked.connect(lambda: self.pl_table.check_all(False))

    # ── Table management ─────────────────────────────────────────────────────
    def refresh_table(self):
        """Repopulate the table from get_pl_files().  Preserves check states."""
        pl_files = self.get_pl_files()
        rows = []
        for pf in pl_files:
            m  = pf.metadata
            ce = m["Center_E"]
            it = m["int_time"]
            rows.append([
                m["filename"],
                f"{ce:.3f}" if ce is not None else "—",
                f"{it:.3g}" if it is not None else "—",
                f"{m['Temp']:.1f}" if m["Temp"] is not None else "—",
                f"{m['Exc_P']:.2f}" if m["Exc_P"] is not None else "—",
                "✓" if m["filename"] in self.corrected else "✗",
            ])
        self.pl_table.populate(rows, keep_checks=True)

    def invalidate(self):
        """Clear computed results — called when upstream (dark/white scaling) changes."""
        self.corrected.clear()
        self.corr_status.setText("Correction invalidated — re-apply required.")
        self.refresh_table()

    # ── Apply correction ─────────────────────────────────────────────────────
    def _apply_checked(self):
        dark     = self.get_dark_dict()
        corr     = self.get_corr_dict()
        pl_files = self.get_pl_files()
        checked  = self._checked_files()
        if not dark:
            QMessageBox.warning(self, "Missing",
                "Load and check dark files in the Calibration tab first.")
            return
        if not corr:
            QMessageBox.warning(self, "Missing",
                "Build correction ratios in the Apply Corrections tab first.")
            return
        if not checked:
            QMessageBox.warning(self, "Nothing selected",
                "Check at least one file in the table.")
            return

        dark_scale_dict = self.get_dark_scale_dict()
        errors = []
        for pf in checked:
            fname = pf.metadata["filename"]
            try:
                self.corrected[fname] = apply_correction(pf, dark, corr, dark_scale_dict)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

        n_ok = len(self.corrected)
        self.corr_status.setText(
            f"{n_ok} / {len(pl_files)} file(s) corrected"
        )
        self.refresh_table()
        if errors:
            QMessageBox.warning(self, "Correction warnings", "\n".join(errors))
        self._plot_corrected()
        if self.corrected and self.on_data_changed:
            self.on_data_changed()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _checked_files(self) -> list:
        pl_files    = self.get_pl_files()
        checked_idx = self.pl_table.checked_rows()
        return [pl_files[i] for i in checked_idx if i < len(pl_files)]

    @staticmethod
    def _power_label(pf) -> str:
        p = pf.metadata.get("Exc_P")
        return f"{p:.2f} mW" if p is not None else pf.metadata["filename"]

    def _plot_raw(self):
        normalized = self.get_normalized()
        if not normalized:
            QMessageBox.warning(self, "No data",
                "Apply dark subtraction in the 'Apply Corrections' tab first.")
            return
        series = []
        for pf in sorted(self._checked_files(),
                         key=lambda p: p.metadata["Center_E"] or 0.0):
            fname = pf.metadata["filename"]
            if fname in normalized:
                df = normalized[fname]
                series.append((df["Energy"].to_numpy(), df["Counts"].to_numpy(),
                                self._power_label(pf)))
        if not series:
            QMessageBox.information(self, "Nothing to plot",
                "None of the checked files have been dark-subtracted yet.")
            return
        self.pl_plot.plot_series(
            series, "Dark-Subtracted & Normalised PL  (before whitelight correction)",
            log_y=True
        )

    def _plot_corrected(self):
        if not self.corrected:
            return
        series = []
        for pf in sorted(self._checked_files(),
                         key=lambda p: p.metadata["Center_E"] or 0.0):
            fname = pf.metadata["filename"]
            if fname in self.corrected:
                df = self.corrected[fname]
                series.append((df["Energy"].to_numpy(), df["Counts"].to_numpy(),
                                self._power_label(pf)))
        if not series:
            return
        self.pl_plot.plot_series(series, "Whitelight-Corrected PL Spectra", log_y=True)

    # ── Public accessors for Stitch tab ──────────────────────────────────────
    def get_corrected_grouped(self) -> dict:
        """
        Return {power: [(df, pf), …]} for checked + corrected files.
        Files within 0.05 mW of each other are placed in the same group.
        Each group is sorted by ascending Center_E.
        """
        groups: dict = {}
        for pf in sorted(self._checked_files(),
                         key=lambda p: p.metadata["Center_E"] or 0.0):
            fname = pf.metadata["filename"]
            if fname not in self.corrected:
                continue
            p = pf.metadata.get("Exc_P") or 0.0
            matched_key = None
            for key in groups:
                if _powers_match(key, p):
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = p
                groups[matched_key] = []
            groups[matched_key].append((self.corrected[fname], pf))
        return groups


def _write_dat_with_header(dat_path, result_df: "pd.DataFrame", pf_list: list):
    """
    Write stitched spectrum to dat_path, prefixed with the metadata header taken
    from the first PL file in pf_list (14 lines: col header + 12 metadata + separator).
    Falls back to plain CSV if the source file cannot be read.
    """
    header_text = ""
    if pf_list:
        try:
            with open(pf_list[0].file_path, "r", encoding="utf-8", errors="replace") as fh:
                header_text = "".join(fh.readlines()[:14])
        except Exception:
            header_text = ""
    with open(dat_path, "w", encoding="utf-8") as fh:
        if header_text:
            fh.write(header_text)
            fh.write(result_df.to_csv(index=False, sep="\t", header=False))
        else:
            fh.write(result_df.to_csv(index=False, sep="\t"))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Stitch & Export
# ════════════════════════════════════════════════════════════════════════════
class StitchTab(QWidget):
    """
    Group-based stitching workflow:
      1. "Start Stitching" — groups checked corrected files by excitation power.
         All groups must have the same number of spectral windows.
      2. For each pair of adjacent windows the reference group (lowest power)
         is shown.  Click-and-drag to mark the blend region.
      3. "Do Stitch" — applies that same blend window to EVERY power group
         simultaneously, then shows all stitched results overlaid.
      4. "Next Pair" — advances to the next window pair (all groups again).
      5. "Save All" — saves CSV + PNG + PDF for every power group.
    """

    def __init__(self, get_corrected_grouped,
                 get_scaling_applied_pl, get_scaling_applied_wl,
                 get_correction_dict, get_corrected,
                 get_pl_scaling_meta=None, get_wl_scaling_meta=None,
                 get_save_dir=None, set_save_dir=None):
        super().__init__()
        self.get_corrected_grouped  = get_corrected_grouped
        self.get_scaling_applied_pl = get_scaling_applied_pl
        self.get_scaling_applied_wl = get_scaling_applied_wl
        self.get_correction_dict    = get_correction_dict
        self.get_corrected          = get_corrected
        self.get_pl_scaling_meta    = get_pl_scaling_meta or (lambda: {})
        self.get_wl_scaling_meta    = get_wl_scaling_meta or (lambda: {})
        self.get_save_dir           = get_save_dir or (lambda: "")
        self.set_save_dir           = set_save_dir or (lambda _: None)

        self.on_stitching_done = None   # () → None; called when all stitching is complete

        # ── State ─────────────────────────────────────────────────────────
        self.power_groups:          dict = {}   # {power: [(df, pf), …]}
        self.stitched_results:      dict = {}   # {power: df}
        self._single_window_groups: dict = {}   # {power: [(df, pf)]} — 1-window powers, auto-completed
        self.n_steps:          int  = 0
        self.current_step:     int  = 0
        self._selected_range:  Optional[tuple] = None
        self._span_selector                    = None
        self._blend_log:       list = []   # [{step, left_ce, right_ce, window_eV, ratios}]

        # ── Power-by-power mode state ──────────────────────────────────────
        self._power_mode:         bool = False
        self._power_order:        list = []   # sorted power values
        self._power_idx:          int  = 0
        self._power_done:         dict = {}   # {power: bool}
        self._power_stitched:     dict = {}   # {power: df}  accumulated results
        self._power_blend_logs:   dict = {}   # {power: [blend_log entries]}
        self._power_groups_all:   dict = {}   # {power: [(df, pf)]} for _save filename
        self._power_pills:        dict = {}   # {power: QPushButton}
        self._save_power_progress = None      # () → None, set by MainWindow

        layout = QVBoxLayout(self)

        ctrl_box = QGroupBox(
            "7.  Stitch  &  Export  "
            "(groups checked files by power — same blend window applied to all groups)"
        )
        cg = QVBoxLayout(ctrl_box)

        # ── Button row ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.do_btn      = QPushButton("Do Stitch")
        self.restart_btn = QPushButton("↺ Restart")
        self.restart_btn.setToolTip(
            "Restart stitching from the first step, keeping the current file selection."
        )
        self.save_btn    = QPushButton("Save All")
        self.do_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        for b in [self.do_btn, self.restart_btn, self.save_btn]:
            btn_row.addWidget(b)
        cg.addLayout(btn_row)

        # ── Power-by-power navigation row (hidden in normal mode) ─────────
        power_nav_row = QHBoxLayout()
        power_nav_row.addWidget(QLabel("Powers:"))
        self._prev_power_btn = QPushButton("◀ Prev")
        self._next_power_btn = QPushButton("Next ▶")
        self._redo_power_btn = QPushButton("↺ Redo This Power")
        self._redo_power_btn.setToolTip("Reset and redo stitching for the current power level.")
        self._prev_power_btn.clicked.connect(self._prev_power)
        self._next_power_btn.clicked.connect(self._next_power)
        self._redo_power_btn.clicked.connect(self._redo_current_power)
        power_nav_row.addWidget(self._prev_power_btn)
        power_nav_row.addWidget(self._next_power_btn)
        power_nav_row.addWidget(self._redo_power_btn)
        self._power_pills_widget = QWidget()
        self._power_pills_layout = QHBoxLayout(self._power_pills_widget)
        self._power_pills_layout.setContentsMargins(0, 0, 0, 0)
        self._power_pills_layout.setSpacing(4)
        power_nav_row.addWidget(self._power_pills_widget, 1)
        cg.addLayout(power_nav_row)
        self._prev_power_btn.setVisible(False)
        self._next_power_btn.setVisible(False)
        self._redo_power_btn.setVisible(False)
        self._power_pills_widget.setVisible(False)

        # ── Pipeline checklist ────────────────────────────────────────────
        check_row = QHBoxLayout()
        check_row.addWidget(QLabel("Pipeline:"))
        self._step_labels: list = []
        for text in [
            "① PL Dark Scaling",
            "② White Dark Scaling",
            "③ Correction Ratios",
            "④ Spectra Corrected",
            "⑤ Ready to Stitch",
        ]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "color: #888; border: 1px solid #bbb; "
                "border-radius: 4px; padding: 2px 10px;"
            )
            self._step_labels.append(lbl)
            check_row.addWidget(lbl)
        check_row.addStretch()
        cg.addLayout(check_row)

        # ── Info row ──────────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self.step_label  = QLabel("—")
        self.range_label = QLabel("Click and drag on the plot to select the blend window")
        self.range_label.setStyleSheet("color: grey;")
        info_row.addWidget(self.step_label)
        info_row.addWidget(self.range_label, 1)
        cg.addLayout(info_row)

        # ── Format checkboxes ─────────────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Save formats:"))
        self.chk_csv  = QCheckBox("DAT")
        self.chk_json = QCheckBox("JSON metadata")
        self.chk_csv.setChecked(True)
        self.chk_json.setChecked(True)
        for chk in [self.chk_csv, self.chk_json]:
            fmt_row.addWidget(chk)
        fmt_row.addStretch()
        cg.addLayout(fmt_row)

        self.status = QLabel("Apply correction in the PL Analysis tab first.")
        cg.addWidget(self.status)
        layout.addWidget(ctrl_box)

        self.plot = PlotWidget(min_h=500, log_y=True)
        layout.addWidget(self.plot)

        self.do_btn.clicked.connect(self._do_stitch)
        self.restart_btn.clicked.connect(self._restart)
        self.save_btn.clicked.connect(self._save)

    # ── Pipeline checklist ────────────────────────────────────────────────
    def refresh_checklist(self):
        """Update the pipeline status indicators. Called externally (tab switch)."""
        _OK  = ("color: white; background-color: #43a047; "
                "border-radius: 4px; padding: 2px 10px;")
        _BAD = ("color: white; background-color: #e53935; "
                "border-radius: 4px; padding: 2px 10px;")

        ok_scale_pl  = self.get_scaling_applied_pl()
        ok_scale_wl  = self.get_scaling_applied_wl()
        ok_corr_dict = bool(self.get_correction_dict())
        ok_corrected = bool(self.get_corrected())
        ok_all       = ok_corr_dict and ok_corrected

        states = [ok_scale_pl, ok_scale_wl, ok_corr_dict, ok_corrected, ok_all]
        for lbl, ok in zip(self._step_labels, states):
            lbl.setStyleSheet(_OK if ok else _BAD)

        # Auto-start stitching when all prerequisites are met and groups not yet
        # initialised.  Only applies to standard (non-power) mode — the pipeline
        # calls enter_power_mode() directly from _goto_stitch().
        if ok_all and not self.power_groups and not self._power_mode:
            self._start()

    # ── SpanSelector ──────────────────────────────────────────────────────
    def _install_span(self):
        self._clear_span()
        self._selected_range = None
        self._span_selector = SpanSelector(
            self.plot.ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="steelblue"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def _clear_span(self):
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
                self._span_selector.set_visible(False)
            except Exception:
                pass
            self._span_selector = None

    def _on_span_select(self, xmin: float, xmax: float):
        if xmax - xmin < 1e-6:
            return
        self._selected_range = (xmin, xmax)
        self.range_label.setText(f"Blend window:  {xmin:.4f} – {xmax:.4f} eV")
        self.range_label.setStyleSheet("")
        self.do_btn.setEnabled(True)

    # ── Filename helper ───────────────────────────────────────────────────
    @staticmethod
    def _stitch_name(power: float, pf_list: list) -> str:
        """Build output filename from the first file's metadata."""
        pf   = pf_list[0]
        fp   = pf.file_path.replace("\\", "/")
        m    = re.search(r"/([^/]+)_\d{1,3}\.origin$", fp)
        name = m.group(1) if m else Path(pf.file_path).stem
        temp     = pf.metadata.get("Temp")
        temp_str = f"{temp:.1f}" if temp is not None else "None"
        return f"{name}_{power:.2f}mW_stitched_{temp_str}K"

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self):
        """Fully reset all stitching state so a fresh analysis can start."""
        self._clear_span()
        self.power_groups          = {}
        self.stitched_results      = {}
        self._single_window_groups = {}
        self.n_steps               = 0
        self.current_step          = 0
        self._selected_range       = None
        self._blend_log            = []
        self._power_mode           = False
        self._power_order          = []
        self._power_idx            = 0
        self._power_done           = {}
        self._power_stitched       = {}
        self._power_blend_logs     = {}
        self._power_groups_all     = {}

        self.do_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._prev_power_btn.setVisible(False)
        self._next_power_btn.setVisible(False)
        self._redo_power_btn.setVisible(False)
        self._power_pills_widget.setVisible(False)
        while self._power_pills_layout.count():
            item = self._power_pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._power_pills = {}
        self.plot.clear()
        self.step_label.setText("—")
        self.range_label.setText("")
        self.status.setText("")

    # ── Workflow ──────────────────────────────────────────────────────────
    def _start(self):
        groups = self.get_corrected_grouped()
        if not groups:
            QMessageBox.warning(self, "Not enough data",
                "No corrected files found.")
            return

        # Powers with a single window need no stitching — auto-complete them.
        self._single_window_groups = {p: pfs for p, pfs in groups.items() if len(pfs) == 1}
        groups_ok                  = {p: pfs for p, pfs in groups.items() if len(pfs) >= 2}

        self.stitched_results = {p: pfs[0][0].copy() for p, pfs in self._single_window_groups.items()}

        if not groups_ok:
            # Every power has exactly one spectral window — nothing to stitch interactively.
            self.power_groups = {}
            self.n_steps      = 0
            self.save_btn.setEnabled(True)
            self.restart_btn.setEnabled(False)
            n_g = len(self._single_window_groups)
            self.status.setText(
                f"{n_g} power group(s) — single spectral window each, no stitching needed.  "
                "Press 'Save All' to export."
            )
            if self.on_stitching_done:
                self.on_stitching_done()
            return

        sizes = {p: len(pfs) for p, pfs in groups_ok.items()}
        if len(set(sizes.values())) > 1:
            msg = "\n".join(
                f"  {p:.2f} mW → {n} window(s)"
                for p, n in sorted(sizes.items())
            )
            QMessageBox.warning(self, "Window count mismatch",
                f"Power groups have different numbers of windows:\n{msg}\n\n"
                "All groups must have the same number of spectral windows.")
            return

        self.power_groups = groups_ok
        self.n_steps      = list(set(sizes.values()))[0] - 1
        self.current_step = 0
        self._blend_log   = []
        for p, pfs in groups_ok.items():
            self.stitched_results[p] = pfs[0][0].copy()
        self._show_step()
        self.restart_btn.setEnabled(True)
        n_g   = len(groups_ok)
        extra = (f"  ({len(self._single_window_groups)} single-window power(s) auto-completed.)"
                 if self._single_window_groups else "")
        self.status.setText(
            f"{n_g} power group(s) ready.{extra}  "
            "Click and drag to select blend window, then press 'Do Stitch'."
        )

    def _restart(self):
        """Restart stitching from step 1 without changing the file selection."""
        if not self.power_groups:
            self._start()
            return
        self.current_step     = 0
        self._blend_log       = []
        self.stitched_results = {
            p: pfs[0][0].copy() for p, pfs in self.power_groups.items()
        }
        self.save_btn.setEnabled(False)
        self._show_step()
        self.status.setText(
            "Restarted from step 1.  "
            "Select blend window, then press 'Do Stitch'."
        )

    def _show_step(self):
        """Show all groups' current pair overlaid so the blend window can be judged."""
        series = []
        ovl_mins, ovl_maxs = [], []

        for power, pfs in sorted(self.power_groups.items()):
            left     = self.stitched_results[power]
            right_df = pfs[self.current_step + 1][0]

            xl = left["Energy"].to_numpy(float)
            xr = right_df["Energy"].to_numpy(float)
            ovl_mins.append(float(max(xl.min(), xr.min())))
            ovl_maxs.append(float(min(xl.max(), xr.max())))

            series.append((xl, left["Counts"].to_numpy(),     f"{power:.2f} mW  left"))
            series.append((xr, right_df["Counts"].to_numpy(), f"{power:.2f} mW  right"))

        x_ovl_min = max(ovl_mins)
        x_ovl_max = min(ovl_maxs)
        n_g = len(self.power_groups)

        self.plot.plot_series(
            series,
            f"Step {self.current_step + 1}/{self.n_steps}  —  "
            f"overlap {x_ovl_min:.3f}–{x_ovl_max:.3f} eV  "
            f"({n_g} group(s) — select blend window, then press 'Do Stitch')",
            log_y=True,
        )
        self._install_span()

        self.step_label.setText(
            f"Step {self.current_step + 1}/{self.n_steps}  |  {n_g} group(s)"
        )
        self.range_label.setText("Click and drag on the plot to select the blend window")
        self.range_label.setStyleSheet("color: grey;")
        self.do_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _do_stitch(self):
        if self._selected_range is None:
            QMessageBox.warning(self, "No region selected",
                "Click and drag on the plot to select the blend window first.")
            return
        self._clear_span()
        x_min, x_max = self._selected_range
        errors  = []
        ratios  = {}
        for power, pfs in self.power_groups.items():
            right_df = pfs[self.current_step + 1][0]
            try:
                stitched, ratio = stitch_once(
                    self.stitched_results[power], right_df, x_min, x_max
                )
                self.stitched_results[power] = stitched
                ratios[power] = ratio
            except Exception as exc:
                errors.append(f"{power:.2f} mW: {exc}")

        # Log this stitch step for metadata export
        if self.power_groups:
            sample_pfs = next(iter(self.power_groups.values()))
            left_ce  = sample_pfs[self.current_step][1].metadata.get("Center_E")
            right_ce = sample_pfs[self.current_step + 1][1].metadata.get("Center_E")
            self._blend_log.append({
                "step":           self.current_step + 1,
                "left_center_E":  round(left_ce,  4) if left_ce  is not None else None,
                "right_center_E": round(right_ce, 4) if right_ce is not None else None,
                "window_eV":      [round(x_min, 6), round(x_max, 6)],
                "ratios_by_power": {f"{p:.4f}": round(r, 8) for p, r in ratios.items()},
            })

        is_last = self.current_step >= self.n_steps - 1

        if errors:
            QMessageBox.warning(self, "Stitch warnings", "\n".join(errors))

        if is_last:
            # All steps done for this power/group — show final stitched result
            self.do_btn.setEnabled(False)
            series = [
                (r["Energy"].to_numpy(), r["Counts"].to_numpy(), f"{p:.2f} mW")
                for p, r in sorted(self.stitched_results.items())
            ]
            self.plot.plot_series(series, "Final stitched result", log_y=True)
            self.step_label.setText(f"Step {self.current_step + 1}/{self.n_steps}  ✓  —  Done")

            if self._power_mode:
                # Save this power's result into the accumulator
                power = self._power_order[self._power_idx]
                self._power_stitched[power] = self.stitched_results[power].copy()
                self._power_blend_logs[power] = self._blend_log.copy()
                self._power_done[power] = True
                self._update_power_pills()
                if self._save_power_progress:
                    self._save_power_progress()

                n_done = sum(1 for v in self._power_done.values() if v)
                n_total = len(self._power_order)
                if n_done == n_total:
                    self.save_btn.setEnabled(True)
                    self.status.setText(
                        f"All {n_total} power(s) stitched!  Press 'Save All' to export."
                    )
                    if self.on_stitching_done:
                        self.on_stitching_done()
                else:
                    remaining = [p for p in self._power_order if not self._power_done.get(p)]
                    self.status.setText(
                        f"{power:.2f} mW done  ({n_done}/{n_total})  —  "
                        f"{len(remaining)} power(s) remaining.  "
                        "Use pills or ▶ to navigate."
                    )
                    # Auto-advance to next undone power after a short delay
                    from PySide6.QtCore import QTimer
                    next_undone = next(
                        (p for p in self._power_order if not self._power_done.get(p)), None
                    )
                    if next_undone is not None:
                        QTimer.singleShot(600, lambda p=next_undone: self._goto_power(p))
            else:
                self.save_btn.setEnabled(True)
                if self.on_stitching_done:
                    self.on_stitching_done()
                self.status.setText(
                    f"All {self.n_steps} step(s) done  |  "
                    f"last blend {x_min:.4f}–{x_max:.4f} eV  |  "
                    f"{len(self.stitched_results)} group(s)  |  Press 'Save' to export."
                )
        else:
            # More steps remain — immediately show next pair
            self.current_step += 1
            self._show_step()
            self.status.setText(
                f"Step {self.current_step - 1} done (blend {x_min:.4f}–{x_max:.4f} eV).  "
                "Select blend window for next pair, then press 'Do Stitch'."
            )

    def _next_pair(self):
        self.current_step += 1
        self._show_step()
        self.status.setText(
            "Click and drag to select blend window, then press 'Do Stitch'."
        )

    # ── Power-by-power mode ────────────────────────────────────────────────
    def enter_power_mode(self) -> bool:
        """
        Switch to power-by-power stitching mode.
        Each power is stitched separately with its own blend windows.
        Returns True if there are corrected groups to stitch.
        """
        groups = self.get_corrected_grouped()
        groups_ok = {p: pfs for p, pfs in groups.items() if len(pfs) >= 1}
        if not groups_ok:
            QMessageBox.warning(self, "Not enough data",
                "No corrected files found.")
            return False

        powers = sorted(groups_ok.keys())
        self._power_mode = True
        self._power_order = powers
        self._power_idx = 0
        self._power_done = {p: False for p in powers}
        self._power_stitched = {}
        self._power_blend_logs = {p: [] for p in powers}
        self._power_groups_all = {}

        self._rebuild_power_pills()
        self._prev_power_btn.setVisible(True)
        self._next_power_btn.setVisible(True)
        self._redo_power_btn.setVisible(True)
        self._power_pills_widget.setVisible(True)
        self.restart_btn.setEnabled(False)   # not applicable in power mode
        self.save_btn.setEnabled(False)
        self._start_current_power()
        return True

    def restore_power_mode(self, blend_logs: dict):
        """
        Restore previously saved power stitch state (crash recovery).
        blend_logs: {power_float: [blend_log entries]}
        Re-stitches each power from the saved blend windows.
        """
        if not blend_logs:
            return
        if not self.enter_power_mode():
            return
        groups = self.get_corrected_grouped()
        for json_power, log in sorted(blend_logs.items()):
            # Match the JSON power to the actual loaded power using the same
            # 10 % relative tolerance used everywhere else in the pipeline.
            actual_power = next(
                (p for p in groups if _powers_match(p, json_power)), None
            )
            if actual_power is None or len(groups[actual_power]) < 2 or not log:
                continue
            pfs = groups[actual_power]
            # Point _power_idx at the correct slot so _do_stitch stores
            # the result under the right key in _power_stitched.
            if actual_power in self._power_order:
                self._power_idx = self._power_order.index(actual_power)
            # Set up stitching state for this power
            self._power_groups_all[actual_power] = pfs
            self.power_groups = {actual_power: pfs}
            self.n_steps = len(pfs) - 1
            self.current_step = 0
            self._blend_log = []
            self.stitched_results = {actual_power: pfs[0][0].copy()}
            for entry in sorted(log, key=lambda e: e.get("step", 0)):
                xmin, xmax = entry.get("window_eV", [0, 0])
                self._selected_range = (xmin, xmax)
                try:
                    self._do_stitch()
                except Exception:
                    break

    def _rebuild_power_pills(self):
        while self._power_pills_layout.count():
            item = self._power_pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._power_pills = {}
        for p in self._power_order:
            btn = QPushButton(f"{p:.2f} mW")
            btn.setFlat(True)
            btn.clicked.connect(lambda _checked=False, pw=p: self._goto_power(pw))
            self._power_pills[p] = btn
            self._power_pills_layout.addWidget(btn)
        self._power_pills_layout.addStretch()
        self._update_power_pills()

    def _update_power_pills(self):
        _OK   = ("QPushButton { color: white; background-color: #43a047; "
                 "border-radius: 4px; padding: 2px 8px; border: none; }"
                 "QPushButton:hover { background-color: #388e3c; }")
        _WAIT = ("QPushButton { color: #555; border: 1px solid #bbb; "
                 "border-radius: 4px; padding: 2px 8px; background: transparent; }"
                 "QPushButton:hover { border-color: #888; color: #000; }")
        for p, btn in self._power_pills.items():
            btn.setStyleSheet(_OK if self._power_done.get(p, False) else _WAIT)

    def _start_current_power(self):
        """Start (or resume) the N-step stitching workflow for the current power."""
        self.refresh_checklist()
        groups = self.get_corrected_grouped()
        power = self._power_order[self._power_idx]
        pfs = groups.get(power, [])
        if len(pfs) < 2:
            # Single-window power — auto-complete without any user interaction.
            self._power_groups_all[power] = pfs
            if pfs:
                self._power_stitched[power] = pfs[0][0].copy()
            self._power_done[power] = True
            self._power_blend_logs[power] = []
            self._update_power_pills()
            if self._save_power_progress:
                self._save_power_progress()
            n_done  = sum(1 for v in self._power_done.values() if v)
            n_total = len(self._power_order)
            if n_done == n_total:
                self.save_btn.setEnabled(True)
                self.status.setText(
                    f"All {n_total} power(s) complete!  Press 'Save All' to export."
                )
                if self.on_stitching_done:
                    self.on_stitching_done()
            else:
                next_undone = next((p for p in self._power_order if not self._power_done.get(p)), None)
                self.status.setText(
                    f"{power:.2f} mW: single window, auto-completed  "
                    f"({n_done}/{n_total})."
                )
                if next_undone is not None:
                    self._power_idx = self._power_order.index(next_undone)
                    self._start_current_power()
            return

        self._power_groups_all[power] = pfs
        self.power_groups = {power: pfs}
        self.n_steps = len(pfs) - 1
        n_done = sum(1 for v in self._power_done.values() if v)
        self._update_power_pills()

        if self._power_done.get(power) and power in self._power_stitched:
            # Power already stitched — show the final result immediately.
            self._clear_span()
            self.current_step = self.n_steps - 1
            self._blend_log = list(self._power_blend_logs.get(power, []))
            self.stitched_results = {power: self._power_stitched[power].copy()}
            stitched_df = self._power_stitched[power]
            self.plot.plot_series(
                [(stitched_df["Energy"].to_numpy(), stitched_df["Counts"].to_numpy(),
                  f"{power:.2f} mW  stitched")],
                f"✓ {power:.2f} mW — already stitched  "
                f"({n_done}/{len(self._power_order)} powers done)  "
                "— press '↺ Redo This Power' to redo",
                log_y=True,
            )
            self.step_label.setText(
                f"Step {self.n_steps}/{self.n_steps}  ✓  —  Done"
            )
            self.do_btn.setEnabled(False)
            self.save_btn.setEnabled(n_done == len(self._power_order))
            self.status.setText(
                f"✓ {power:.2f} mW already stitched  "
                f"({n_done}/{len(self._power_order)} powers done).  "
                "Use pills or ▶ to navigate, or '↺ Redo This Power' to redo."
            )
        else:
            self.current_step = 0
            self._blend_log = []
            self.stitched_results = {power: pfs[0][0].copy()}
            self._show_step()
            self.status.setText(
                f"Power {self._power_idx + 1}/{len(self._power_order)}: {power:.2f} mW  "
                f"|  {n_done}/{len(self._power_order)} powers complete."
            )

    def _goto_power(self, power: float):
        if power in self._power_order:
            self._power_idx = self._power_order.index(power)
            self._start_current_power()

    def _prev_power(self):
        if self._power_idx > 0:
            self._power_idx -= 1
            self._start_current_power()

    def _next_power(self):
        if self._power_idx < len(self._power_order) - 1:
            self._power_idx += 1
            self._start_current_power()

    def _redo_current_power(self):
        """Reset done-state for current power and restart its stitching from step 1."""
        if not self._power_order:
            return
        power = self._power_order[self._power_idx]
        self._power_done[power] = False
        self._power_stitched.pop(power, None)
        self._power_blend_logs[power] = []
        self._update_power_pills()
        self.save_btn.setEnabled(False)
        self._start_current_power()

    def _build_metadata(self) -> dict:
        """Collect all analysis parameters into a dict suitable for JSON export."""
        meta: dict = {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        # ── PL dark scaling ───────────────────────────────────────────────
        pl_meta = self.get_pl_scaling_meta()
        meta["pl_dark_scaling"] = pl_meta

        # ── White dark scaling ────────────────────────────────────────────
        wl_meta = self.get_wl_scaling_meta()
        meta["white_dark_scaling"] = wl_meta

        # ── Correction coefficients ───────────────────────────────────────
        corr_dict = self.get_correction_dict()
        corr_info = {}
        for ce, df in sorted(corr_dict.items()):
            energies = df["Energy"].to_numpy(float)
            coeffs   = df["correction_coefficient"].to_numpy(float)
            valid    = np.isfinite(coeffs)
            corr_info[f"{ce:.4f}"] = {
                "energy_range_eV": [round(float(energies.min()), 6),
                                    round(float(energies.max()), 6)],
                "n_points":        int(len(coeffs)),
                "n_nan_points":    int(np.sum(~valid)),
            }
        meta["correction_coefficients"] = corr_info

        # ── Stitching ─────────────────────────────────────────────────────
        if self._power_mode:
            # Per-power blend logs
            stitch_info = {}
            all_blend = []
            for power in sorted(self._power_groups_all):
                pfs = self._power_groups_all[power]
                center_Es = [round(pf.metadata["Center_E"], 4) for _, pf in pfs
                             if pf.metadata.get("Center_E") is not None]
                stitch_info[f"{power:.4f}"] = {
                    "power_mW": round(power, 4),
                    "center_energies_eV": center_Es,
                    "blend_windows": self._power_blend_logs.get(power, []),
                }
                all_blend.extend(self._power_blend_logs.get(power, []))
            meta["stitching"] = {
                "blend_windows": all_blend,
                "power_groups":  stitch_info,
            }
        else:
            stitch_info = {}
            for power, pfs in sorted(self.power_groups.items()):
                center_Es = [round(pf.metadata["Center_E"], 4) for _, pf in pfs
                             if pf.metadata.get("Center_E") is not None]
                stitch_info[f"{power:.4f}"] = {
                    "power_mW":        round(power, 4),
                    "center_energies_eV": center_Es,
                }
            meta["stitching"] = {
                "blend_windows": self._blend_log,
                "power_groups":  stitch_info,
            }

        return meta

    def _save(self):
        # Use power-accumulated results in power mode, normal results otherwise
        results = (
            self._power_stitched
            if (self._power_mode and self._power_stitched)
            else self.stitched_results
        )
        if not results:
            return

        save_csv  = self.chk_csv.isChecked()
        save_json = self.chk_json.isChecked()
        if not any([save_csv, save_json]):
            QMessageBox.warning(self, "No format selected",
                "Check at least one output format (DAT / JSON).")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.get_save_dir()
        )
        if not folder:
            return
        self.set_save_dir(folder)
        out = Path(folder)

        # Create per-format subfolders up-front
        if save_csv:
            (out / "dat").mkdir(exist_ok=True)

        saved, errors = [], []

        for power, result_df in sorted(results.items()):
            # In power mode, pf_list comes from _power_groups_all; else from power_groups
            if self._power_mode and power in self._power_groups_all:
                pf_list = [pf for _, pf in self._power_groups_all[power]]
            elif power in self.power_groups:
                pf_list = [pf for _, pf in self.power_groups[power]]
            elif power in self._single_window_groups:
                pf_list = [pf for _, pf in self._single_window_groups[power]]
            else:
                pf_list = []
            if not pf_list:
                continue
            base = self._stitch_name(power, pf_list)

            # ── CSV ───────────────────────────────────────────────────────
            if save_csv:
                try:
                    dat_path = out / "dat" / f"{base}.dat"
                    _write_dat_with_header(dat_path, result_df, pf_list)
                    saved.append(f"dat/{base}.dat")
                except Exception as exc:
                    errors.append(f"{base}.dat: {exc}")

        # ── JSON metadata ─────────────────────────────────────────────────
        if save_json:
            try:
                meta = self._build_metadata()
                # Build name from sample + date: e.g. GaAs_sample_PL_analysis_2026-04-16.json
                sample_name = "unknown"
                ref_groups = (self.power_groups or self._power_groups_all
                              or self._single_window_groups)
                if ref_groups:
                    first_pfs = next(iter(ref_groups.values()))
                    fp = first_pfs[0][1].file_path.replace("\\", "/")
                    m  = re.search(r"/([^/]+)_\d{1,3}\.origin$", fp)
                    sample_name = m.group(1) if m else Path(first_pfs[0][1].file_path).stem
                date_str  = datetime.date.today().isoformat()
                json_name = f"{sample_name}_PL_analysis_{date_str}.json"
                json_path = out / json_name
                json_path.write_text(json.dumps(meta, indent=2))
                saved.append(json_name)
            except Exception as exc:
                errors.append(f"metadata JSON: {exc}")

        msg = f"Saved {len(self.stitched_results)} group(s) ({len(saved)} file(s)) → {out}"
        if errors:
            msg += "\n\nWarnings:\n" + "\n".join(errors)
        QMessageBox.information(self, "Saved", msg)
        self.status.setText(f"Saved to  {out.name}/")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2b / 3b — Dark Scaling (pre-processing before dark subtraction)
# ════════════════════════════════════════════════════════════════════════════
class DarkScalingTab(QWidget):
    """
    mode="pl" / mode="white"  — Both tabs now use the same logic:
      Computes dark_scale = mean(spectrum_edge) / mean(dark_edge) per file and
      stores it in self.dark_scale_dict {filename: scale}.  Each spectrum —
      whether PL or white — gets its own independently scaled dark so that
      different y-offsets between files at the same (center_E, int_time) are
      handled individually.  The dark is always shifted to match the spectrum;
      the spectrum itself is never scaled (pf.scale stays 1.0).

    Pill colours reflect only the current in-memory done-state:
      green  → user has applied scaling or confirmed this group
      grey   → not yet reviewed / reset by user
    """

    def __init__(self, get_files_fn, get_dark_dict, file_label: str = "PL",
                 on_data_changed=None,
                 get_partner_dark_scales=None, dark_scale_dict: Optional[dict] = None):
        super().__init__()
        self.get_files_fn           = get_files_fn
        self.get_dark_dict          = get_dark_dict
        self.file_label             = file_label
        self.on_data_changed        = on_data_changed
        self.on_group_done          = None   # (ce: float) → None; called after each group is applied/skipped
        self.get_partner_dark_scales = get_partner_dark_scales  # callable → {filename: scale}

        self._groups:           list          = []
        self._group_idx:        int           = 0
        self._selected_range:   Optional[tuple] = None
        self._span_selector                     = None
        self._group_done:       dict          = {}   # {ce: bool} — single source of truth for pills
        self._group_pills:      dict          = {}   # {ce: QPushButton}
        self._externally_done:  set           = set()  # CEs marked done across tab-switches / session
        self._scaling_windows:  dict          = {}   # {ce: (x_min, x_max)}
        self._power_filter:     Optional[float] = None  # pipeline: show only CEs for this power
        self._manual_scaling_in_progress: bool  = False  # suppresses power auto-advance

        # dark_scale_dict: optionally shared externally so multiple tabs write to the same dict.
        # Structure: {filename: dark_scale} — one entry per individual spectrum file.
        self.dark_scale_dict: dict = dark_scale_dict if dark_scale_dict is not None else {}

        layout = QVBoxLayout(self)

        _title = (f"Pre-Processing:  Scale Dark to match {file_label} at edges  "
                  "(dark_scale applied to dark before subtraction)")
        ctrl_box = QGroupBox(_title)
        cg = QVBoxLayout(ctrl_box)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Scaling")
        self.all_btn   = QPushButton("Next Group")
        self.skip_btn  = QPushButton("No Scaling Needed")
        self.skip_btn.setToolTip(
            "Mark this group as done without applying any scaling.\n"
            "Advance through all groups to enable the stitch tab indicator."
        )
        self.reset_btn = QPushButton("Reset Scales")
        self.apply_btn.setEnabled(False)
        self.all_btn.setEnabled(False)
        for b in [self.apply_btn, self.all_btn, self.skip_btn, self.reset_btn]:
            btn_row.addWidget(b)
        cg.addLayout(btn_row)

        # ── Per-group pipeline pills ──────────────────────────────────────
        pills_row = QHBoxLayout()
        pills_row.addWidget(QLabel("Groups:"))
        self._pills_container = QWidget()
        self._pills_layout    = QHBoxLayout(self._pills_container)
        self._pills_layout.setContentsMargins(0, 0, 0, 0)
        self._pills_layout.setSpacing(4)
        pills_row.addWidget(self._pills_container, 1)
        cg.addLayout(pills_row)

        # ── Info row ──────────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self.group_label = QLabel("—")
        self.range_label = QLabel(
            "Click and drag on the LEFT plot to select an edge window"
        )
        self.range_label.setStyleSheet("color: grey;")
        info_row.addWidget(self.group_label)
        info_row.addWidget(self.range_label, 1)
        cg.addLayout(info_row)

        self.status = QLabel(
            f"Load {file_label} files and dark spectra, then navigate here to scale."
        )
        cg.addWidget(self.status)

        # ── Manual scale override ─────────────────────────────────────────
        self.manual_scale_spin: Optional[QDoubleSpinBox] = None
        self.manual_scale_btn:  Optional[QPushButton]    = None
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Manual dark scale:"))
        self.manual_scale_spin = QDoubleSpinBox()
        self.manual_scale_spin.setRange(0.001, 1000.0)
        self.manual_scale_spin.setDecimals(4)
        self.manual_scale_spin.setSingleStep(0.01)
        self.manual_scale_spin.setValue(1.0)
        self.manual_scale_spin.setMaximumWidth(120)
        self.manual_scale_btn = QPushButton("Apply Manual Scale")
        self.manual_scale_btn.setToolTip(
            "Enter a dark scale factor and click to apply it directly to the current group.\n"
            "The plot updates immediately and the group is confirmed —\n"
            "adjust the value and click again to refine."
        )
        manual_row.addWidget(self.manual_scale_spin)
        manual_row.addWidget(self.manual_scale_btn)
        manual_row.addStretch()
        cg.addLayout(manual_row)

        layout.addWidget(ctrl_box)

        # ── Two-subplot figure ────────────────────────────────────────────
        self.figure   = Figure(figsize=(14, 5))
        self.ax_before = self.figure.add_subplot(121)
        self.ax_after  = self.figure.add_subplot(122)
        self.canvas    = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(420)
        layout.addWidget(self.canvas)

        self.apply_btn.clicked.connect(self._apply_group)
        self.all_btn.clicked.connect(self._next_group)
        self.skip_btn.clicked.connect(self._skip)
        self.reset_btn.clicked.connect(self._reset)
        if self.manual_scale_btn is not None:
            self.manual_scale_btn.clicked.connect(self._apply_manual_scale)

    # ── Per-group done tracking ───────────────────────────────────────────
    @property
    def scaling_applied(self) -> bool:
        """True only when every group has been either scaled or skipped."""
        return bool(self._groups) and all(
            self._group_done.get(ce, False) for ce, _ in self._groups
        )

    def _update_pills(self):
        _OK   = ("QPushButton { color: white; background-color: #43a047; "
                 "border-radius: 4px; padding: 2px 8px; border: none; }"
                 "QPushButton:hover { background-color: #388e3c; }")
        _WAIT = ("QPushButton { color: #555; border: 1px solid #bbb; "
                 "border-radius: 4px; padding: 2px 8px; background: transparent; }"
                 "QPushButton:hover { border-color: #888; color: #000; }")
        for ce, btn in self._group_pills.items():
            btn.setStyleSheet(_OK if self._group_done.get(ce, False) else _WAIT)

    def _goto_group(self, ce: float):
        """Navigate to the group for the given Center_E without touching scales."""
        for i, (c, _) in enumerate(self._groups):
            if c == ce:
                self._group_idx = i
                self._draw_group()
                break

    # ── Canvas helpers ────────────────────────────────────────────────────
    def _safe_draw(self):
        try:
            if self.canvas is not None and shiboken6.isValid(self.canvas):
                self.canvas.draw_idle()
        except RuntimeError:
            pass

    # ── SpanSelector on left subplot ──────────────────────────────────────
    def _install_span(self):
        self._clear_span()
        self._selected_range = None
        self._span_selector = SpanSelector(
            self.ax_before,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="steelblue"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def _clear_span(self):
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
                self._span_selector.set_visible(False)
            except Exception:
                pass
            self._span_selector = None

    def _on_span_select(self, xmin: float, xmax: float):
        if xmax - xmin < 1e-6:
            return
        self._selected_range = (xmin, xmax)
        self.range_label.setText(f"Edge window:  {xmin:.4f} – {xmax:.4f} eV")
        self.range_label.setStyleSheet("")
        self.apply_btn.setEnabled(True)

    # ── Scale computation ─────────────────────────────────────────────────
    def _compute_dark_scale(self, pf, dark_dict: dict,
                            x_min: float, x_max: float) -> float:
        """
        Returns dark_scale so that  dark × dark_scale ≈ spectrum  at [x_min, x_max].
        Applies to both PL and white files — the spectrum itself is never scaled.
        Falls back to 1.0 if the window has no data or denominator is ~0.
        """
        ce = pf.metadata["Center_E"]
        it = pf.metadata["int_time"]
        dark = dark_dict.get(ce, {}).get(it)
        if dark is None:
            return 1.0

        xp = pf.df["Energy"].to_numpy(float)
        yp = pf.df["Counts"].to_numpy(float)
        xd = dark.df["Energy"].to_numpy(float)
        yd = dark.df["Counts"].to_numpy(float)

        mask = (xp >= x_min) & (xp <= x_max)
        if not np.any(mask):
            return 1.0

        mean_signal = float(np.mean(yp[mask]))
        mean_dark   = float(np.mean(yd[mask]))

        # Both PL and white: scale dark so that dark × dark_scale ≈ spectrum at edge.
        if abs(mean_dark) < EPSILON:
            return 1.0
        return mean_signal / mean_dark

    # ── Plot helpers ──────────────────────────────────────────────────────
    def _draw_group(self):
        """Redraw both subplots for the current group."""
        ce, pf_list = self._groups[self._group_idx]
        dark_dict   = self.get_dark_dict()

        for ax, title, apply_scale in [
            (self.ax_before, "Before scaling", False),
            (self.ax_after,  "After scaling",  True),
        ]:
            ax.clear()
            # Dark spectra (one per unique int_time): raw in the before plot, scaled in the after plot.
            seen_it: set = set()
            for pf in pf_list:
                it = pf.metadata["int_time"]
                if it in seen_it:
                    continue
                seen_it.add(it)
                dark_pf = dark_dict.get(ce, {}).get(it)
                if dark_pf is not None:
                    xd = dark_pf.df["Energy"].to_numpy(float)
                    yd = dark_pf.df["Counts"].to_numpy(float)
                    if apply_scale:
                        ds   = self._get_dark_scale(ce, it)
                        yd   = yd * ds
                        dlbl = f"dark ×{ds:.4f}  {it:.3g} s"
                    else:
                        dlbl = f"dark  {it:.3g} s"
                    ax.semilogy(xd, np.where(yd > 0, yd, np.nan),
                                color="black", lw=1.5, ls="--", label=dlbl)

            # Spectra — spectrum itself is never scaled; dark is always shifted to match it.
            for pf in pf_list:
                xp  = pf.df["Energy"].to_numpy(float)
                yp  = pf.df["Counts"].to_numpy(float)
                p   = pf.metadata.get("Exc_P")
                it  = pf.metadata.get("int_time")
                lbl = (f"{p:.2f} mW" if p is not None
                       else f"{self.file_label}  {it:.3g} s" if it is not None
                       else pf.metadata["filename"])
                ax.semilogy(xp, np.where(yp > 0, yp, np.nan), lw=1, label=lbl)

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Counts")
            ax.set_title(
                f"{title}  —  Center_E = {ce:.3f} eV  [{self.file_label}]"
            )
            ax.legend(fontsize=7)

        self.figure.tight_layout()
        self._safe_draw()
        # Both modes use SpanSelector for edge-window selection
        self._install_span()

        n_total = len(self._groups)
        self.group_label.setText(
            f"Group {self._group_idx + 1}/{n_total}  |  "
            f"Center_E = {ce:.3f} eV  |  {len(pf_list)} file(s)"
        )
        self.range_label.setText(
            "Click and drag on the LEFT plot to select an edge window"
        )
        self.range_label.setStyleSheet("color: grey;")
        self.apply_btn.setEnabled(False)

        # Restore manual scale spinbox: show the saved scale if one exists for
        # the first file in this group, otherwise reset to 1.0.
        if self.manual_scale_spin is not None and pf_list:
            fname = pf_list[0].metadata["filename"]
            saved_scale = self.dark_scale_dict.get(fname, 1.0)
            self.manual_scale_spin.blockSignals(True)
            self.manual_scale_spin.setValue(float(saved_scale))
            self.manual_scale_spin.blockSignals(False)
        # "Next Group" is available whenever there is a following group
        self.all_btn.setEnabled(self._group_idx < len(self._groups) - 1)

    def set_power_filter(self, power: Optional[float]):
        """
        Restrict groups to Center_E values that have at least one file at the
        given power level (±tol).  Pass None to clear the filter.
        Automatically re-prepares groups if files and darks are already loaded.
        """
        self._power_filter = power
        files     = self.get_files_fn()
        dark_dict = self.get_dark_dict()
        if not files or not dark_dict:
            return
        groups = self._build_groups(files, dark_dict)
        if not groups:
            if power is not None:
                self.status.setText(
                    f"No groups match power ≈ {power:.2f} mW — "
                    "load files or check dark matching."
                )
            return
        self._apply_groups(groups)
        self._draw_group()
        n = sum(len(lst) for _, lst in self._groups)
        suffix = f"  (filtered to ≈ {power:.2f} mW)" if power is not None else ""
        self.status.setText(
            f"{len(self._groups)} group(s) ready  ({n} {self.file_label} file(s)){suffix}."
        )

    # ── Workflow ──────────────────────────────────────────────────────────
    def _build_groups(self, files, dark_dict) -> dict:
        """Group files by Center_E, filtering to those with a dark match.
        If _power_filter is set, only includes CEs with at least one file at that power."""
        groups: dict = {}
        for pf in files:
            ce = pf.metadata.get("Center_E")
            it = pf.metadata.get("int_time")
            if ce is None or it is None:
                continue
            if ce not in dark_dict or it not in dark_dict[ce]:
                continue
            if self._power_filter is not None:
                exc_p = pf.metadata.get("Exc_P")
                if exc_p is None or not _powers_match(exc_p, self._power_filter):
                    continue
            groups.setdefault(ce, []).append(pf)
        return groups

    def _apply_groups(self, groups: dict):
        """Install sorted groups, rebuild pills, preserve existing done state."""
        new_groups = sorted(groups.items())

        prev_done = self._group_done.copy()
        self._groups    = new_groups
        self._group_idx = min(self._group_idx, max(0, len(new_groups) - 1))

        # Done = previously confirmed in this session OR restored from session.
        # The dark_scale_dict is intentionally NOT used here: any apply/skip/manual
        # action (including scale=1.0) writes filenames to that dict, so dict-based
        # detection produces false positives on restart.  _externally_done and
        # _pip_ce_done (restored from session) are the only authoritative sources.
        self._group_done = {
            ce: (prev_done.get(ce, False) or (ce in self._externally_done))
            for ce, pf_list in new_groups
        }

        # Rebuild pills
        while self._pills_layout.count():
            item = self._pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._group_pills = {}
        for ce, _ in self._groups:
            btn = QPushButton(f"{ce:.3f} eV")
            btn.setFlat(True)
            btn.setStyleSheet(
                "QPushButton { color: #555; border: 1px solid #bbb; "
                "border-radius: 4px; padding: 2px 8px; background: transparent; }"
                "QPushButton:hover { border-color: #888; color: #000; }"
            )
            btn.clicked.connect(lambda _checked=False, c=ce: self._goto_group(c))
            self._group_pills[ce] = btn
            self._pills_layout.addWidget(btn)
        self._pills_layout.addStretch()
        self._update_pills()

    def get_dark_scale_dict(self) -> dict:
        """Return the current dark_scale_dict {Center_E: {int_time: float}}."""
        return self.dark_scale_dict

    def _get_dark_scale(self, ce: float, it: float) -> float:
        """Average dark scale for files in the current group that share the given it; for visualisation."""
        if not self._groups:
            return 1.0
        _, pf_list = self._groups[self._group_idx]
        vals = [
            self.dark_scale_dict[pf.metadata["filename"]]
            for pf in pf_list
            if pf.metadata.get("int_time") == it
            and pf.metadata["filename"] in self.dark_scale_dict
        ]
        return float(np.mean(vals)) if vals else 1.0

    def set_done_groups(self, ces):
        """Mark Center_E values as done externally (e.g. restored from session)."""
        self._externally_done = set(ces)

    def get_scaling_meta(self) -> dict:
        """Return per-group scaling metadata for JSON export."""
        result = {}
        for ce, pf_list in self._groups:
            win = self._scaling_windows.get(ce)
            result[f"{ce:.4f}"] = {
                "center_E_eV":        round(ce, 4),
                "edge_window_eV":     [round(win[0], 6), round(win[1], 6)] if win else None,
                "skipped":            win is None and self._group_done.get(ce, False),
                "dark_scale_by_file": {
                    pf.metadata["filename"]: round(
                        self.dark_scale_dict.get(pf.metadata["filename"], 1.0), 8
                    )
                    for pf in pf_list
                },
            }
        return result

    def refresh_if_needed(self):
        """
        Called when this tab becomes visible.  If files and darks are already
        loaded, silently rebuild groups/pills (no dialogs) so that previously
        applied scales are reflected as green pills immediately.
        """
        files     = self.get_files_fn()
        dark_dict = self.get_dark_dict()
        if not files or not dark_dict:
            return
        groups = self._build_groups(files, dark_dict)
        if not groups:
            return
        self._apply_groups(groups)
        if self._groups:
            self._draw_group()
            n = sum(len(lst) for _, lst in self._groups)
            self.status.setText(
                f"{len(self._groups)} group(s)  ({n} {self.file_label} file(s))."
            )

    def reset_groups(self):
        """
        Called when the source files are replaced.  Clears all groups, pills,
        and scales so the tab starts fresh for the new file set.
        """
        self.dark_scale_dict.clear()
        for _, pf_list in self._groups:
            for pf in pf_list:
                pf.scale = 1.0
        self._groups           = []
        self._group_idx        = 0
        self._group_done       = {}
        self._externally_done  = set()
        self._scaling_windows  = {}
        while self._pills_layout.count():
            item = self._pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._group_pills = {}
        self.status.setText("PL files changed — press 'Prepare Groups' to reload.")
        self.group_label.setText("—")

    def _prepare(self):
        files     = self.get_files_fn()
        dark_dict = self.get_dark_dict()
        if not files:
            QMessageBox.warning(self, "No files",
                f"Load {self.file_label} files first.")
            return
        if not dark_dict:
            QMessageBox.warning(self, "No dark files",
                "Load dark files in the Calibration tab first.")
            return

        groups = self._build_groups(files, dark_dict)
        if not groups:
            QMessageBox.warning(self, "No matches",
                f"No {self.file_label} file has a matching dark "
                "(Center_E + int_time).")
            return

        self._apply_groups(groups)
        self._draw_group()
        n = sum(len(lst) for _, lst in self._groups)
        self.status.setText(
            f"{len(self._groups)} group(s) ready  ({n} {self.file_label} file(s)).  "
            "Select an edge window on the left plot, then press "
            "'Apply to This Group'."
        )

    def _apply_group(self):
        if self._selected_range is None:
            return
        x_min, x_max = self._selected_range
        ce, pf_list  = self._groups[self._group_idx]
        dark_dict    = self.get_dark_dict()

        # Both modes: compute an individual dark_scale per file and store by filename.
        for pf in pf_list:
            s = self._compute_dark_scale(pf, dark_dict, x_min, x_max)
            self.dark_scale_dict[pf.metadata["filename"]] = s
        action_msg = f"Dark scaled for {len(pf_list)} file(s) at Center_E = {ce:.3f} eV."

        self._scaling_windows[ce] = (x_min, x_max)
        self._group_done[ce] = True
        self._externally_done.add(ce)
        self._update_pills()
        if self.on_data_changed:
            self.on_data_changed()
        self._draw_group()
        self.status.setText(action_msg)
        if self.on_group_done:
            self.on_group_done(ce)

    def _apply_all(self):
        """Apply current window to every group and mark all as done."""
        if self._selected_range is None:
            QMessageBox.warning(self, "No window selected",
                "Click and drag on the left plot first.")
            return
        x_min, x_max = self._selected_range
        dark_dict    = self.get_dark_dict()
        n = 0

        for ce, pf_list in self._groups:
            for pf in pf_list:
                s = self._compute_dark_scale(pf, dark_dict, x_min, x_max)
                self.dark_scale_dict[pf.metadata["filename"]] = s
                n += 1
            self._scaling_windows[ce] = (x_min, x_max)
            self._group_done[ce] = True
            self._externally_done.add(ce)
        action_msg = f"Dark scaled {n} file(s) across all {len(self._groups)} group(s)."

        self._update_pills()
        if self.on_data_changed:
            self.on_data_changed()
        self._draw_group()
        self.status.setText(action_msg)

    def _next_group(self):
        if self._group_idx < len(self._groups) - 1:
            self._group_idx += 1
            self._draw_group()
            self.status.setText(
                "Select an edge window on the left plot, then apply."
            )

    def _skip(self):
        """Mark the current group as done without changing scale factors."""
        if not self._groups:
            return
        ce, pf_list = self._groups[self._group_idx]

        # Write sentinel 1.0 for every file in the group so the skipped state
        # survives session restore via the dark_scale_dict lookup.
        # setdefault: don't overwrite a scale the user already set.
        for pf in pf_list:
            self.dark_scale_dict.setdefault(pf.metadata["filename"], 1.0)

        self._group_done[ce] = True
        self._externally_done.add(ce)
        self._update_pills()
        if self.on_data_changed:
            self.on_data_changed()
        remaining = [c for c, _ in self._groups if not self._group_done.get(c, False)]
        if remaining:
            self.status.setText(
                f"Group {ce:.3f} eV: no scaling applied.  "
                f"{len(remaining)} group(s) still pending."
            )
        else:
            self.status.setText(
                f"All groups confirmed — {self.file_label} dark scaling step complete."
            )
        if self.on_group_done:
            self.on_group_done(ce)

    def _apply_manual_scale(self):
        """
        Apply the manually entered dark scale to all files in the current CE
        group.  Works in both PL and white modes.  The plot refreshes and the
        group is confirmed immediately.
        """
        if not self._groups:
            return
        if self.manual_scale_spin is None:
            return
        ce, pf_list = self._groups[self._group_idx]
        scale = self.manual_scale_spin.value()

        for pf in pf_list:
            self.dark_scale_dict[pf.metadata["filename"]] = scale

        # Mark group done so the pill turns green and the state is persisted
        self._group_done[ce] = True
        self._externally_done.add(ce)
        self._update_pills()

        self._draw_group()
        self.status.setText(
            f"Manual scale {scale:.4f} applied to Center_E = {ce:.3f} eV."
        )
        self._manual_scaling_in_progress = True
        try:
            if self.on_data_changed:
                self.on_data_changed()
        finally:
            self._manual_scaling_in_progress = False

    def _reset(self):
        if not self._groups:
            return
        ce, pf_list = self._groups[self._group_idx]
        for pf in pf_list:
            self.dark_scale_dict.pop(pf.metadata["filename"], None)
        self._scaling_windows.pop(ce, None)
        # Clear done state from ALL sources so tab-switch cannot re-mark as done
        self._group_done[ce] = False
        self._externally_done.discard(ce)
        self._update_pills()
        if self.on_data_changed:
            self.on_data_changed()
        self._draw_group()
        self.status.setText(f"Scaling reset for Center_E = {ce:.3f} eV group.")


# ════════════════════════════════════════════════════════════════════════════
# Export Tab — multi-format export with optional Savitzky-Golay smoothing
# ════════════════════════════════════════════════════════════════════════════
class ExportTab(QWidget):
    """
    Export raw, dark-subtracted, whitelight-corrected, and stitched spectra.

    File format: one Energy column + one column per selected format, all
    interpolated onto a shared x-axis that is the union of ALL energy points
    across all formats for the given power (full spectrum, no truncation).
    Smoothing parameters are written into the file header.

    UI state is intentionally NOT saved to the session JSON.
    """

    _COLOURS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    # (linestyle, linewidth, alpha) per column name keyword
    _LINE_STYLE = {
        "Raw":            (":",  1.0, 0.7),
        "DarkSub":        ("--", 1.0, 0.8),
        "WL_corrected":   ("-.", 1.0, 0.8),
        "Stitched_smooth":("-",  2.0, 1.0),
        "Stitched":       ("--", 1.5, 0.55),  # dashed so smooth stands out
    }

    def __init__(self, get_pl_files_fn, get_normalized_fn, get_corrected_fn,
                 get_stitched_fn, get_save_dir=None, set_save_dir=None):
        super().__init__()
        self.get_pl_files_fn   = get_pl_files_fn
        self.get_normalized_fn = get_normalized_fn
        self.get_corrected_fn  = get_corrected_fn
        self.get_stitched_fn   = get_stitched_fn
        self.get_save_dir      = get_save_dir or (lambda: "")
        self.set_save_dir      = set_save_dir or (lambda _: None)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ── Left control panel ────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(290)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)

        # Power selector (multi-select)
        pw_box = QGroupBox("Power levels")
        pb = QVBoxLayout(pw_box)
        pw_btns = QHBoxLayout()
        sel_all_btn = QPushButton("All")
        sel_none_btn = QPushButton("None")
        sel_all_btn.setMaximumWidth(50)
        sel_none_btn.setMaximumWidth(50)
        sel_all_btn.clicked.connect(self._select_all_powers)
        sel_none_btn.clicked.connect(self._deselect_all_powers)
        pw_btns.addWidget(QLabel("Select:"))
        pw_btns.addWidget(sel_all_btn)
        pw_btns.addWidget(sel_none_btn)
        pw_btns.addStretch()
        pb.addLayout(pw_btns)
        self._power_list = QListWidget()
        self._power_list.setMaximumHeight(110)
        self._power_list.itemChanged.connect(self._refresh_preview)
        pb.addWidget(self._power_list)
        lv.addWidget(pw_box)

        # Format checkboxes
        fmt_box = QGroupBox("Data to include")
        fb = QVBoxLayout(fmt_box)
        self._chk_raw     = QCheckBox("Raw PL data")
        self._chk_darksub = QCheckBox("Dark-subtracted & normalised")
        self._chk_corr    = QCheckBox("Whitelight corrected")
        self._chk_stitch  = QCheckBox("Stitched data")
        self._chk_smooth  = QCheckBox("Smoothed stitched data")
        self._chk_stitch.setChecked(True)
        self._chk_smooth.setChecked(True)
        for chk in [self._chk_raw, self._chk_darksub, self._chk_corr,
                    self._chk_stitch, self._chk_smooth]:
            fb.addWidget(chk)
            chk.toggled.connect(self._refresh_preview)
        lv.addWidget(fmt_box)

        # Smoothing controls
        sm_box = QGroupBox("Smoothing (Savitzky-Golay)")
        sv = QVBoxLayout(sm_box)
        if not _SCIPY_AVAILABLE:
            sv.addWidget(QLabel("⚠  scipy not installed — using\nmoving-average fallback."))

        sv.addWidget(QLabel("Window length (points):"))
        win_row = QHBoxLayout()
        self._win_slider = QSlider(Qt.Horizontal)
        self._win_slider.setRange(1, 50)   # slider n → window 2n+1
        self._win_slider.setValue(5)       # default: 11 pts
        self._win_label = QLabel("11")
        self._win_label.setFixedWidth(28)
        win_row.addWidget(self._win_slider)
        win_row.addWidget(self._win_label)
        sv.addLayout(win_row)

        sv.addWidget(QLabel("Polynomial order:"))
        poly_row = QHBoxLayout()
        self._poly_slider = QSlider(Qt.Horizontal)
        self._poly_slider.setRange(1, 5)
        self._poly_slider.setValue(3)
        self._poly_label = QLabel("3")
        self._poly_label.setFixedWidth(20)
        poly_row.addWidget(self._poly_slider)
        poly_row.addWidget(self._poly_label)
        sv.addLayout(poly_row)

        self._win_slider.valueChanged.connect(self._on_slider_changed)
        self._poly_slider.valueChanged.connect(self._on_slider_changed)
        lv.addWidget(sm_box)

        # Save scope
        scope_box = QGroupBox("Save scope")
        scb = QVBoxLayout(scope_box)

        scb.addWidget(QLabel("Powers:"))
        self._save_selected_rb = QCheckBox("Selected powers only")
        self._save_all_rb      = QCheckBox("All powers")
        self._save_all_rb.setChecked(True)
        scb.addWidget(self._save_selected_rb)
        scb.addWidget(self._save_all_rb)
        self._save_selected_rb.toggled.connect(
            lambda c: self._save_all_rb.setChecked(not c) if c else None)
        self._save_all_rb.toggled.connect(
            lambda c: self._save_selected_rb.setChecked(not c) if c else None)

        scb.addWidget(QLabel("Formats:"))
        self._save_fmt_preview_rb = QCheckBox("Shown in preview")
        self._save_fmt_all_rb     = QCheckBox("All available")
        self._save_fmt_preview_rb.setChecked(True)
        scb.addWidget(self._save_fmt_preview_rb)
        scb.addWidget(self._save_fmt_all_rb)
        self._save_fmt_preview_rb.toggled.connect(
            lambda c: self._save_fmt_all_rb.setChecked(not c) if c else None)
        self._save_fmt_all_rb.toggled.connect(
            lambda c: self._save_fmt_preview_rb.setChecked(not c) if c else None)

        lv.addWidget(scope_box)

        self._save_btn = QPushButton("Save …")
        self._save_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 12px; padding: 5px; }"
        )
        self._save_btn.clicked.connect(self._save)
        lv.addWidget(self._save_btn)

        # Axis limits
        lim_box = QGroupBox("Axis limits  (empty = auto)")
        lb = QVBoxLayout(lim_box)
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("x:"))
        self._xmin_edit = QLineEdit(); self._xmin_edit.setPlaceholderText("min")
        self._xmax_edit = QLineEdit(); self._xmax_edit.setPlaceholderText("max")
        self._xmin_edit.setMaximumWidth(70); self._xmax_edit.setMaximumWidth(70)
        x_row.addWidget(self._xmin_edit); x_row.addWidget(QLabel("–")); x_row.addWidget(self._xmax_edit)
        lb.addLayout(x_row)
        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("y:"))
        self._ymin_edit = QLineEdit(); self._ymin_edit.setPlaceholderText("min")
        self._ymax_edit = QLineEdit(); self._ymax_edit.setPlaceholderText("max")
        self._ymin_edit.setMaximumWidth(70); self._ymax_edit.setMaximumWidth(70)
        y_row.addWidget(self._ymin_edit); y_row.addWidget(QLabel("–")); y_row.addWidget(self._ymax_edit)
        lb.addLayout(y_row)
        apply_lim_btn = QPushButton("Apply limits")
        apply_lim_btn.clicked.connect(self._refresh_preview)
        lb.addWidget(apply_lim_btn)
        lv.addWidget(lim_box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #555;")
        lv.addWidget(self._status)
        lv.addStretch()
        root.addWidget(left)

        # ── Right: preview plot ───────────────────────────────────────
        self._plot = PlotWidget(min_h=420, log_y=True)
        root.addWidget(self._plot, 1)

    # ── Properties ────────────────────────────────────────────────────

    def _window_length(self) -> int:
        return self._win_slider.value() * 2 + 1

    def _on_slider_changed(self):
        wl = self._window_length()
        po = self._poly_slider.value()
        self._win_label.setText(str(wl))
        self._poly_label.setText(str(po))
        if po >= wl:
            self._poly_slider.blockSignals(True)
            self._poly_slider.setValue(max(1, wl - 1))
            self._poly_slider.blockSignals(False)
            self._poly_label.setText(str(max(1, wl - 1)))
        if self._chk_smooth.isChecked():
            self._refresh_preview()

    # ── Power list helpers ────────────────────────────────────────────

    def _available_powers(self) -> list:
        powers: list = []
        for p in (self.get_stitched_fn() or {}):
            if not any(_powers_match(p, q) for q in powers):
                powers.append(p)
        if not powers:
            for pf in (self.get_pl_files_fn() or []):
                p = pf.metadata.get("Exc_P")
                if p is not None and not any(_powers_match(p, q) for q in powers):
                    powers.append(p)
        return sorted(powers)

    def _selected_powers(self) -> list:
        result = []
        for i in range(self._power_list.count()):
            item = self._power_list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.data(Qt.UserRole))
        return result

    def _select_all_powers(self):
        self._power_list.blockSignals(True)
        for i in range(self._power_list.count()):
            self._power_list.item(i).setCheckState(Qt.Checked)
        self._power_list.blockSignals(False)
        self._refresh_preview()

    def _deselect_all_powers(self):
        self._power_list.blockSignals(True)
        for i in range(self._power_list.count()):
            self._power_list.item(i).setCheckState(Qt.Unchecked)
        self._power_list.blockSignals(False)
        self._refresh_preview()

    # ── Data collection ───────────────────────────────────────────────

    def _build_columns(self, power: float, force_all: bool = False) -> tuple:
        """
        Return (x_ref, col_dict) for a single power.

        x_ref is the union of ALL energy points from all selected formats
        (raw, dark-sub, corrected, stitched) — the full spectrum from every
        window, not just the stitched portion.

        force_all=True ignores the preview checkboxes and builds every format
        that has data (used when "All available" is selected for saving).
        """
        pl_files = self.get_pl_files_fn() or []
        meta = {pf.metadata["filename"]: pf.metadata for pf in pl_files}

        stitched  = self.get_stitched_fn() or {}
        stitch_key = next((p for p in stitched if _powers_match(p, power)), None)
        stitch_df  = stitched.get(stitch_key) if stitch_key is not None else None

        want_raw    = force_all or self._chk_raw.isChecked()
        want_darksub = force_all or self._chk_darksub.isChecked()
        want_corr   = force_all or self._chk_corr.isChecked()
        want_stitch = force_all or self._chk_stitch.isChecked()
        want_smooth = force_all or self._chk_smooth.isChecked()

        # Build full x-axis: union of ALL relevant data points
        x_parts: list = []
        if want_raw or want_darksub or want_corr:
            for pf in pl_files:
                if pf.metadata.get("Exc_P") is not None and _powers_match(pf.metadata["Exc_P"], power):
                    x_parts.append(pf.df["Energy"].to_numpy(float))
        if (want_stitch or want_smooth) and stitch_df is not None:
            x_parts.append(stitch_df["Energy"].to_numpy(float))
        if not x_parts:
            return np.array([]), {}
        x_ref = np.sort(np.unique(np.concatenate(x_parts)))
        if len(x_ref) < 2:
            return x_ref, {}

        def _cat_interp(data_iter) -> Optional[np.ndarray]:
            xs_all, ys_all = [], []
            for x, y in data_iter:
                xs_all.append(x); ys_all.append(y)
            if not xs_all:
                return None
            xc = np.concatenate(xs_all); yc = np.concatenate(ys_all)
            ord_ = np.argsort(xc)
            return np.interp(x_ref, xc[ord_], yc[ord_])

        def _file_pairs(src_dict):
            for fname, df in (src_dict or {}).items():
                m = meta.get(fname, {})
                p = m.get("Exc_P")
                if p is not None and _powers_match(p, power):
                    yield (df["Energy"].to_numpy(float), df.iloc[:, 1].to_numpy(float))

        cols: dict = {}

        if want_raw:
            y = _cat_interp(
                (pf.df["Energy"].to_numpy(float), pf.df["Counts"].to_numpy(float))
                for pf in pl_files
                if pf.metadata.get("Exc_P") is not None
                and _powers_match(pf.metadata["Exc_P"], power)
            )
            if y is not None:
                cols["Raw_counts"] = y

        if want_darksub:
            y = _cat_interp(_file_pairs(self.get_normalized_fn()))
            if y is not None:
                cols["DarkSub_norm"] = y

        if want_corr:
            y = _cat_interp(_file_pairs(self.get_corrected_fn()))
            if y is not None:
                cols["WL_corrected"] = y

        if want_stitch and stitch_df is not None:
            xs = stitch_df["Energy"].to_numpy(float); ys = stitch_df["Counts"].to_numpy(float)
            cols["Stitched"] = np.interp(x_ref, xs, ys)

        if want_smooth and stitch_df is not None:
            xs = stitch_df["Energy"].to_numpy(float); ys = stitch_df["Counts"].to_numpy(float)
            y_s = np.interp(x_ref, xs, ys)
            wl = self._window_length(); po = self._poly_slider.value()
            if _SCIPY_AVAILABLE and wl <= len(y_s) and po < wl:
                y_s = _savgol_filter(y_s, wl, po)
            else:
                kernel = np.ones(min(wl, len(y_s))) / min(wl, len(y_s))
                y_s = np.convolve(y_s, kernel, mode="same")
            cols["Stitched_smooth"] = y_s

        return x_ref, cols

    # ── UI updates ────────────────────────────────────────────────────

    def refresh(self):
        """Rebuild the power list (preserve checked state) and refresh preview."""
        powers = self._available_powers()
        # Remember which powers were checked
        prev_checked = set()
        for i in range(self._power_list.count()):
            item = self._power_list.item(i)
            if item.checkState() == Qt.Checked:
                prev_checked.add(item.data(Qt.UserRole))

        self._power_list.blockSignals(True)
        self._power_list.clear()
        for p in powers:
            item = QListWidgetItem(f"{p:.2f} mW")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            # Default: check all; restore previous state if available
            was_checked = not prev_checked or any(_powers_match(p, q) for q in prev_checked)
            item.setCheckState(Qt.Checked if was_checked else Qt.Unchecked)
            item.setData(Qt.UserRole, p)
            self._power_list.addItem(item)
        self._power_list.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self):
        sel = self._selected_powers()
        if not sel:
            self._plot.clear()
            self._status.setText("No powers selected.")
            return

        ax = self._plot.ax
        ax.clear()
        any_data = False
        for p_idx, power in enumerate(sel):
            colour = self._COLOURS[p_idx % len(self._COLOURS)]
            x_ref, cols = self._build_columns(power)
            if not cols:
                continue
            any_data = True
            for col_name, y in cols.items():
                ls, lw, alpha = next(
                    (v for k, v in self._LINE_STYLE.items() if col_name.startswith(k)),
                    ("-", 1.0, 0.9),
                )
                y_plot = np.where(y > 0, y, np.nan)
                ax.semilogy(x_ref, y_plot, color=colour, ls=ls, lw=lw, alpha=alpha,
                            label=f"{power:.2f} mW — {col_name}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity")
        title = "Export preview"
        if len(sel) == 1:
            title += f" — {sel[0]:.2f} mW"
        ax.set_title(title)
        if any_data:
            ax.legend(fontsize=7)

        # Apply axis limits
        def _parse(edit) -> Optional[float]:
            txt = edit.text().strip()
            try:
                return float(txt) if txt else None
            except ValueError:
                return None
        xmin, xmax = _parse(self._xmin_edit), _parse(self._xmax_edit)
        ymin, ymax = _parse(self._ymin_edit), _parse(self._ymax_edit)
        if xmin is not None or xmax is not None:
            ax.set_xlim(
                xmin if xmin is not None else ax.get_xlim()[0],
                xmax if xmax is not None else ax.get_xlim()[1],
            )
        if ymin is not None or ymax is not None:
            ax.set_ylim(
                ymin if ymin is not None else ax.get_ylim()[0],
                ymax if ymax is not None else ax.get_ylim()[1],
            )

        self._plot.figure.tight_layout()
        self._plot._safe_draw()

        if not any_data:
            self._status.setText("No data for selected powers — run correction/stitching first.")
        else:
            self._status.setText(f"{len(sel)} power(s) selected")

    # ── Save ──────────────────────────────────────────────────────────

    def _save(self):
        if self._save_selected_rb.isChecked():
            powers_to_save = self._selected_powers()
        else:
            powers_to_save = self._available_powers()
        force_all_formats = self._save_fmt_all_rb.isChecked()

        if not powers_to_save:
            QMessageBox.warning(self, "No data", "No data available to save.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.get_save_dir()
        )
        if not folder:
            return
        self.set_save_dir(folder)
        out = Path(folder)

        wl = self._window_length()
        po = self._poly_slider.value()
        saved, errors = [], []

        for power in powers_to_save:
            x_ref, cols = self._build_columns(power, force_all=force_all_formats)
            if not cols:
                continue

            pl_files = self.get_pl_files_fn() or []
            pf0 = next(
                (pf for pf in pl_files
                 if pf.metadata.get("Exc_P") is not None
                 and _powers_match(pf.metadata["Exc_P"], power)),
                None,
            )
            if pf0:
                fp     = pf0.file_path.replace("\\", "/")
                m_name = re.search(r"/([^/]+)_\d{1,3}\.origin$", fp)
                sample = m_name.group(1) if m_name else Path(pf0.file_path).stem
                temp   = pf0.metadata.get("Temp")
                temp_s = f"{temp:.1f}" if temp is not None else "?"
            else:
                sample, temp_s = "sample", "?"

            fname = f"{sample}_{power:.2f}mW_export_{temp_s}K.dat"
            fpath = out / fname

            col_names = ["Energy(eV)"] + list(cols.keys())
            try:
                with open(fpath, "w", encoding="utf-8") as fh:
                    fh.write(f"# Sample: {sample}\n")
                    fh.write(f"# Temperature: {temp_s} K\n")
                    fh.write(f"# Power: {power:.4f} mW\n")
                    if "Stitched_smooth" in cols:
                        method = "Savitzky-Golay" if _SCIPY_AVAILABLE else "moving-average"
                        fh.write(
                            f"# Smoothing: {method}, window={wl} pts, poly_order={po}\n"
                        )
                    fh.write("# Columns: " + " | ".join(col_names) + "\n")
                    fh.write("\t".join(col_names) + "\n")
                    ys_list = list(cols.values())
                    for i, x in enumerate(x_ref):
                        row = [f"{x:.6f}"] + [
                            f"{y[i]:.6e}" if np.isfinite(y[i]) else "NaN"
                            for y in ys_list
                        ]
                        fh.write("\t".join(row) + "\n")
                saved.append(fname)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

        msg = f"Saved {len(saved)} file(s) to {out.name}/"
        if saved:
            msg += "\n\n" + "\n".join(saved)
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        QMessageBox.information(self, "Export complete", msg)
        self._status.setText(f"Saved {len(saved)} file(s).")


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Power-by-Power Pipeline
# ════════════════════════════════════════════════════════════════════════════
class PowerPipelineTab(QWidget):
    """
    Tab ⑧ — fully self-contained guided pipeline with embedded sub-widgets.

    All four phases live inside a QStackedWidget so the user never leaves
    this tab:
      Page 0  Overview    — start button, jump-to-phase shortcuts
      Page 1  PL Dark     — embedded DarkScalingTab(mode="pl"), one power at a time
      Page 2  White Dark  — embedded DarkScalingTab(mode="white"), all windows
      Page 3  Correction  — auto-build + apply, then advance
      Page 4  Stitch      — embedded StitchTab in power mode
    """

    PAGE_OVERVIEW   = 0
    PAGE_PL_DARK    = 1
    PAGE_WL_DARK    = 2
    PAGE_CORRECTION = 3
    PAGE_STITCH     = 4
    PAGE_POWER_PLOT = 5
    PAGE_EXPORT     = 6

    def __init__(self,
                 embedded_pl_dark_tab,            # DarkScalingTab(mode="pl")
                 embedded_wl_dark_tab,            # DarkScalingTab(mode="white")
                 embedded_stitch_tab,             # StitchTab
                 auto_correct_fn,                 # () → None
                 get_pl_powers_fn,                # () → sorted list[float]
                 get_corrected_fn,                # () → dict {filename: df}
                 reset_scaling_fn,                # () → None
                 get_pl_dark_done_for_power_fn,   # (power: float) → bool
                 check_window_counts_fn=None,     # () → Optional[str] — error msg or None
                 replay_fn=None,                  # (json_path: str) → None
                 embedded_power_plot_tab=None,    # PowerSeriesPlotTab
                 get_correction_dict_fn=None,     # () → {ce: df} correction coefficients
                 get_pl_files_fn=None,            # () → list[PL_file] raw PL files
                 get_normalized_fn=None,          # () → {filename: df} dark-sub normalised
                 silent_auto_correct_fn=None,     # () → None, no dialogs (for restore)
                 get_replay_dir=None,             # () → str
                 set_replay_dir=None,             # (path: str) → None
                 embedded_export_tab=None,        # ExportTab
                 ):
        super().__init__()

        self.embedded_pl_dark_tab           = embedded_pl_dark_tab
        self.embedded_wl_dark_tab           = embedded_wl_dark_tab
        self.embedded_stitch_tab            = embedded_stitch_tab
        self.auto_correct_fn                = auto_correct_fn
        self.get_pl_powers_fn               = get_pl_powers_fn
        self.get_corrected_fn               = get_corrected_fn
        self.reset_scaling_fn               = reset_scaling_fn
        self.get_pl_dark_done_for_power_fn  = get_pl_dark_done_for_power_fn
        self.check_window_counts_fn         = check_window_counts_fn
        self.replay_fn                      = replay_fn
        self.embedded_power_plot_tab        = embedded_power_plot_tab
        self.get_correction_dict_fn         = get_correction_dict_fn
        self.get_replay_dir                 = get_replay_dir or (lambda: "")
        self.set_replay_dir                 = set_replay_dir or (lambda _: None)
        self.embedded_export_tab            = embedded_export_tab
        self.get_pl_files_fn                = get_pl_files_fn
        self.get_normalized_fn              = get_normalized_fn
        self.silent_auto_correct_fn         = silent_auto_correct_fn
        self.on_correction_applied          = None   # () → None; wired to _save_session
        self._corr_view_mode: int           = 0   # 0=coefficients 1=raw 2=dark-sub 3=corrected
        self._correction_applied: bool      = False  # persisted in session JSON

        self._pl_dark_powers: list = []
        self._pl_dark_idx:    int  = 0
        self._pl_dark_pills:  dict = {}   # {power: QPushButton}
        self._pl_done:        dict = {}   # {power: bool}
        self._pip_ce_done:    dict = {}   # {power: {ce: bool}} — authoritative CE pill state per power

        # ── Root layout ───────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Persistent header bar (always visible) ────────────────────────
        hdr_box = QGroupBox("⑧  Power-by-Power Pipeline")
        hdr_lay = QVBoxLayout(hdr_box)
        hdr_lay.setSpacing(4)

        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Phase:"))
        self._phase_labels: list = []
        _phase_nav = [
            ("① PL Dark",    lambda: self._goto_pl_dark()),
            ("② White Dark", lambda: self._goto_wl_dark()),
            ("③ Correction", lambda: self._goto_correction()),
            ("④ Stitching",  lambda: self._goto_stitch()),
            ("⑤ Power Plot", lambda: self._goto_power_plot()),
            ("⑥ Export",     lambda: self._goto_export()),
        ]
        for txt, nav_fn in _phase_nav:
            btn = QPushButton(txt)
            btn.setStyleSheet(
                "QPushButton { color: #888; border: 1px solid #bbb; "
                "border-radius: 4px; padding: 2px 10px; background: transparent; }"
                "QPushButton:hover { border-color: #555; color: #000; }"
            )
            btn.clicked.connect(nav_fn)
            self._phase_labels.append(btn)
            phase_row.addWidget(btn)
        phase_row.addStretch()
        self._overview_btn = QPushButton("◀ Overview")
        self._overview_btn.setVisible(False)
        self._overview_btn.clicked.connect(
            lambda: self._stack.setCurrentIndex(self.PAGE_OVERVIEW)
        )
        phase_row.addWidget(self._overview_btn)
        self._reset_btn = QPushButton("Reset All")
        self._reset_btn.setStyleSheet("QPushButton { color: #c62828; }")
        self._reset_btn.setToolTip(
            "Clear all dark and white scaling progress and return to the overview."
        )
        self._reset_btn.clicked.connect(self._reset_scaling)
        phase_row.addWidget(self._reset_btn)
        hdr_lay.addLayout(phase_row)
        root.addWidget(hdr_box)

        # ── Stacked pages ─────────────────────────────────────────────────
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._stack.addWidget(self._make_overview_page())   # 0
        self._stack.addWidget(self._make_pl_dark_page())    # 1
        self._stack.addWidget(self._make_wl_dark_page())    # 2
        self._stack.addWidget(self._make_correction_page()) # 3
        self._stack.addWidget(self._make_stitch_page())     # 4
        if self.embedded_power_plot_tab is not None:
            self._stack.addWidget(self._make_power_plot_page())  # 5
        if self.embedded_export_tab is not None:
            self._stack.addWidget(self._make_export_page())      # 6

        # ── Hook into embedded tab callbacks for auto-advance ─────────────
        _orig_pl_cb = self.embedded_pl_dark_tab.on_data_changed
        def _pl_hook():
            # Sync current power's CE done states into _pip_ce_done BEFORE calling
            # the original callback so that _save_session() sees the updated state.
            if self._pl_dark_powers and self._pl_dark_idx < len(self._pl_dark_powers):
                power = self._pl_dark_powers[self._pl_dark_idx]
                pd_entry = self._pip_ce_done.setdefault(power, {})
                for ce, done in self.embedded_pl_dark_tab._group_done.items():
                    pd_entry[ce] = done
            if _orig_pl_cb:
                _orig_pl_cb()
            self._check_pl_power_done()
        self.embedded_pl_dark_tab.on_data_changed = _pl_hook

        _orig_wl_cb = self.embedded_wl_dark_tab.on_data_changed
        def _wl_hook():
            if _orig_wl_cb:
                _orig_wl_cb()
            self._check_wl_done()
        self.embedded_wl_dark_tab.on_data_changed = _wl_hook

        # Auto-advance to the next undone CE group after apply/skip (600 ms delay).
        # Searches forward from the current index, wrapping around if needed.
        def _make_group_done_hook(emb_tab, page_idx):
            def _hook(done_ce: float):
                if self._stack.currentIndex() != page_idx:
                    return
                groups     = emb_tab._groups
                group_done = emb_tab._group_done
                n = len(groups)
                if not n:
                    return
                current = emb_tab._group_idx
                next_ce = None
                for i in range(1, n + 1):
                    c, _ = groups[(current + i) % n]
                    if not group_done.get(c, False):
                        next_ce = c
                        break
                if next_ce is not None:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(
                        600, lambda c=next_ce: emb_tab._goto_group(c)
                    )
            return _hook

        self.embedded_pl_dark_tab.on_group_done = _make_group_done_hook(
            self.embedded_pl_dark_tab, self.PAGE_PL_DARK
        )
        self.embedded_wl_dark_tab.on_group_done = _make_group_done_hook(
            self.embedded_wl_dark_tab, self.PAGE_WL_DARK
        )

    # ══ Page builders ══════════════════════════════════════════════════════

    def _make_overview_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignTop)

        title = QLabel("<b>Power-by-Power Pipeline</b>")
        title.setStyleSheet("font-size: 14px; padding: 8px 0;")
        lay.addWidget(title)

        desc = QLabel(
            "Guided analysis — all steps happen here, nothing to navigate away:\n"
            "  ①  Scale PL dark spectra for each excitation power independently\n"
            "  ②  Scale white dark spectra (once, for all spectral windows)\n"
            "  ③  Auto-compute and apply whitelight correction\n"
            "  ④  Stitch spectra power by power\n"
            "  ⑤  Plot and export the power series\n\n"
            "Load PL, dark, white, and halogen files in Tab ① before starting."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 4px 0 12px 0;")
        lay.addWidget(desc)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Pipeline")
        self._start_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 13px; padding: 6px 20px; }"
        )
        self._start_btn.clicked.connect(self._start_pipeline)
        btn_row.addWidget(self._start_btn)

        self._replay_btn = QPushButton("Load & Replay JSON …")
        self._replay_btn.setToolTip(
            "Load a previously saved pipeline analysis JSON and re-apply all "
            "recorded dark scaling and stitching steps automatically."
        )
        self._replay_btn.clicked.connect(self._request_replay)
        btn_row.addWidget(self._replay_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._overview_status = QLabel("")
        self._overview_status.setWordWrap(True)
        self._overview_status.setStyleSheet("color: #c62828; padding-top: 8px;")
        lay.addWidget(self._overview_status)
        lay.addStretch()
        return page

    def _make_pl_dark_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr_lay = QVBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 4, 4, 4)
        hdr_lay.setSpacing(3)

        pills_row = QHBoxLayout()
        pills_row.addWidget(QLabel("<b>① PL Dark — Power:</b>"))
        self._pl_pills_container = QWidget()
        self._pl_pills_layout    = QHBoxLayout(self._pl_pills_container)
        self._pl_pills_layout.setContentsMargins(0, 0, 0, 0)
        self._pl_pills_layout.setSpacing(4)
        pills_row.addWidget(self._pl_pills_container, 1)
        self._pl_next_btn = QPushButton("Next Power ▶")
        self._pl_next_btn.setVisible(False)
        self._pl_next_btn.clicked.connect(lambda: self._advance_pl_power())
        pills_row.addWidget(self._pl_next_btn)
        self._pl_to_wl_btn = QPushButton("All Powers Done — ② White Dark ▶")
        self._pl_to_wl_btn.setVisible(False)
        self._pl_to_wl_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; "
            "background-color: #1565c0; border-radius: 4px; padding: 3px 10px; }"
        )
        self._pl_to_wl_btn.clicked.connect(self._goto_wl_dark)
        pills_row.addWidget(self._pl_to_wl_btn)
        hdr_lay.addLayout(pills_row)

        self._pl_dark_status = QLabel("")
        self._pl_dark_status.setStyleSheet("color: #555; padding: 2px 0;")
        hdr_lay.addWidget(self._pl_dark_status)

        lay.addWidget(hdr)
        lay.addWidget(self.embedded_pl_dark_tab, 1)
        return page

    def _make_wl_dark_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 4, 4, 4)
        hdr_lay.addWidget(
            QLabel("<b>② White Dark Scaling</b>  — Scale all windows, then advance.")
        )
        hdr_lay.addStretch()
        self._wl_to_corr_btn = QPushButton("Done — ③ Auto-Correct ▶")
        self._wl_to_corr_btn.setVisible(False)
        self._wl_to_corr_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; "
            "background-color: #1565c0; border-radius: 4px; padding: 3px 10px; }"
        )
        self._wl_to_corr_btn.clicked.connect(self._goto_correction)
        hdr_lay.addWidget(self._wl_to_corr_btn)

        lay.addWidget(hdr)
        lay.addWidget(self.embedded_wl_dark_tab, 1)
        return page

    def _make_correction_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(12, 12, 12, 12)

        lay.addWidget(QLabel("<b>③ Auto-Compute &amp; Apply Correction</b>"))

        self._corr_status = QLabel(
            "Press the button below to compute and apply whitelight correction."
        )
        self._corr_status.setWordWrap(True)
        lay.addWidget(self._corr_status)

        btn_row = QHBoxLayout()
        self._run_corr_btn = QPushButton("Run Correction")
        self._run_corr_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 12px; padding: 5px 18px; }"
        )
        self._run_corr_btn.clicked.connect(self._run_correction)
        btn_row.addWidget(self._run_corr_btn)

        self._corr_to_stitch_btn = QPushButton("✓ Correction Done — ④ Stitch ▶")
        self._corr_to_stitch_btn.setVisible(False)
        self._corr_to_stitch_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; "
            "background-color: #1565c0; border-radius: 4px; padding: 5px 18px; }"
        )
        self._corr_to_stitch_btn.clicked.connect(self._goto_stitch)
        btn_row.addWidget(self._corr_to_stitch_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # ── View selector ────────────────────────────────────────────────
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self._corr_view_btns: list = []
        _views = [
            "Correction Coefficients",
            "Raw PL Data",
            "Dark-Subtracted & Normalised",
            "Fully Corrected",
        ]
        for i, label in enumerate(_views):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda _=False, idx=i: self._set_corr_view(idx))
            self._corr_view_btns.append(btn)
            view_row.addWidget(btn)
        view_row.addStretch()
        lay.addLayout(view_row)
        self._update_corr_view_btn_styles()

        self._corr_plot = PlotWidget(min_h=380, log_y=True)
        lay.addWidget(self._corr_plot, 1)
        return page

    def _make_stitch_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 4, 4, 4)
        hdr_lay.addWidget(QLabel("<b>④ Stitch — Power by Power</b>"))
        hdr_lay.addStretch()
        if self.embedded_power_plot_tab is not None:
            self._stitch_to_plot_btn = QPushButton("⑤ Power Series Plot →")
            self._stitch_to_plot_btn.setStyleSheet(
                "QPushButton { font-weight: bold; color: white; "
                "background-color: #6a1b9a; border-radius: 4px; padding: 3px 10px; }"
            )
            self._stitch_to_plot_btn.clicked.connect(self._goto_power_plot)
            hdr_lay.addWidget(self._stitch_to_plot_btn)
        lay.addWidget(hdr)
        lay.addWidget(self.embedded_stitch_tab, 1)
        return page

    # ══ Pipeline start / navigation ════════════════════════════════════════

    def _start_pipeline(self):
        powers = self.get_pl_powers_fn()
        if not powers:
            self._overview_status.setText(
                "⚠  No PL files loaded — load them in Tab ① first."
            )
            return

        if self.check_window_counts_fn:
            err = self.check_window_counts_fn()
            if err:
                self._overview_status.setText("⚠  " + err)
                return

        self._overview_status.setText("")

        if self._correction_applied or self.get_corrected_fn():
            # Ensure correction is computed (may need re-applying after restart)
            if not self.get_corrected_fn() and self.silent_auto_correct_fn:
                try:
                    self.silent_auto_correct_fn()
                except Exception:
                    pass
            # If stitching was already done (restored from session), jump to stitch
            stitched = (self.embedded_stitch_tab.stitched_results
                        or self.embedded_stitch_tab._power_stitched)
            if stitched:
                self._goto_stitch()
            else:
                self._goto_correction()
            return

        pl_done_all = all(self.get_pl_dark_done_for_power_fn(p) for p in powers)
        if pl_done_all:
            self._goto_wl_dark()
        else:
            self._start_pl_dark_phase(powers)

    # ── Phase 1: PL Dark ─────────────────────────────────────────────────

    def _start_pl_dark_phase(self, powers: list):
        self._pl_dark_powers = list(powers)
        self._pl_dark_idx    = 0
        self._pl_done        = {p: self.get_pl_dark_done_for_power_fn(p)
                                for p in powers}
        self._rebuild_pl_pills()
        self._update_phase_labels(0)
        self._overview_btn.setVisible(True)
        self._stack.setCurrentIndex(self.PAGE_PL_DARK)
        first_undone = next((p for p in powers if not self._pl_done[p]), None)
        if first_undone is not None:
            self._select_pl_power(first_undone)
        else:
            self._show_pl_all_done()

    def _rebuild_pl_pills(self):
        while self._pl_pills_layout.count():
            item = self._pl_pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._pl_dark_pills = {}
        for p in self._pl_dark_powers:
            btn = QPushButton(f"{p:.2f} mW")
            btn.setFlat(True)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, pw=p: self._select_pl_power(pw))
            self._pl_dark_pills[p] = btn
            self._pl_pills_layout.addWidget(btn)
        self._pl_pills_layout.addStretch()
        self._update_pl_pill_colors()

    def _update_pl_pill_colors(self):
        _OK   = ("QPushButton { color: white; background-color: #43a047; "
                 "border-radius: 4px; padding: 2px 8px; border: none; }"
                 "QPushButton:checked { background-color: #1b5e20; }"
                 "QPushButton:hover { background-color: #388e3c; }")
        _CURR = ("QPushButton { color: white; background-color: #1565c0; "
                 "border-radius: 4px; padding: 2px 8px; border: none; }"
                 "QPushButton:checked { background-color: #0d47a1; }"
                 "QPushButton:hover { background-color: #1976d2; }")
        _WAIT = ("QPushButton { color: #555; border: 1px solid #bbb; "
                 "border-radius: 4px; padding: 2px 8px; background: transparent; }"
                 "QPushButton:checked { border-color: #1565c0; color: #1565c0; "
                 "font-weight: bold; }"
                 "QPushButton:hover { border-color: #888; color: #000; }")
        current_p = (self._pl_dark_powers[self._pl_dark_idx]
                     if self._pl_dark_powers else None)
        for p, btn in self._pl_dark_pills.items():
            done       = self._pl_done.get(p, False)
            is_current = (p == current_p)
            if done:
                btn.setStyleSheet(_OK)
            elif is_current:
                btn.setStyleSheet(_CURR)
            else:
                btn.setStyleSheet(_WAIT)

    def _select_pl_power(self, power: float):
        if power in self._pl_dark_powers:
            self._pl_dark_idx = self._pl_dark_powers.index(power)
        for p, btn in self._pl_dark_pills.items():
            btn.setChecked(p == power)
        self._update_pl_pill_colors()
        # Reset _group_done and _externally_done BEFORE set_power_filter so that
        # _apply_groups() starts with a blank slate (prev_done all False) and
        # cannot carry done state from the previous power.
        for ce in list(self.embedded_pl_dark_tab._group_done.keys()):
            self.embedded_pl_dark_tab._group_done[ce] = False
        self.embedded_pl_dark_tab._externally_done.clear()

        self.embedded_pl_dark_tab.set_power_filter(power)

        # Restore this power's CE done states from the per-power tracker.
        for ce, done in self._pip_ce_done.get(power, {}).items():
            if ce in self.embedded_pl_dark_tab._group_done:
                self.embedded_pl_dark_tab._group_done[ce] = done
        self.embedded_pl_dark_tab._update_pills()

        # Always start at the first CE group for the new power.
        self.embedded_pl_dark_tab._group_idx = 0
        if self.embedded_pl_dark_tab._groups:
            self.embedded_pl_dark_tab._draw_group()
        n_done  = sum(1 for v in self._pl_done.values() if v)
        n_total = len(self._pl_dark_powers)
        self._pl_dark_status.setText(
            f"Power {power:.2f} mW  ({n_done}/{n_total} powers done)  "
            "— Scale all CE groups for this power, then advance."
        )
        self._pl_next_btn.setVisible(False)
        self._pl_to_wl_btn.setVisible(False)

    def _check_pl_power_done(self):
        """Called after every on_data_changed on the embedded PL dark tab."""
        if not self._pl_dark_powers:
            return
        if self._stack.currentIndex() != self.PAGE_PL_DARK:
            return
        if self.embedded_pl_dark_tab.scaling_applied:
            power = self._pl_dark_powers[self._pl_dark_idx]
            self._pl_done[power] = True
            self._update_pl_pill_colors()
            if all(self._pl_done.values()):
                self._show_pl_all_done()
            elif not self.embedded_pl_dark_tab._manual_scaling_in_progress:
                next_p = next(
                    (p for p in self._pl_dark_powers if not self._pl_done.get(p)),
                    None
                )
                self._pl_dark_status.setText(
                    f"✓ Power {power:.2f} mW done — advancing to next power…"
                )
                self._pl_next_btn.setVisible(True)
                self._pl_to_wl_btn.setVisible(False)
                if next_p is not None:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(600, lambda p=next_p: self._select_pl_power(p))

    def _show_pl_all_done(self):
        self._pl_next_btn.setVisible(False)
        self._pl_to_wl_btn.setVisible(True)
        self._pl_dark_status.setText(
            "✓ All powers done — press '② White Dark ▶' to continue."
        )

    def _advance_pl_power(self):
        next_p = next(
            (p for p in self._pl_dark_powers if not self._pl_done.get(p, False)),
            None
        )
        if next_p is not None:
            self._select_pl_power(next_p)
        else:
            self._show_pl_all_done()

    # ── Phase 2: White Dark ───────────────────────────────────────────────

    def _goto_pl_dark(self):
        """Navigate back to the PL dark phase, preserving all existing done state."""
        if not self._pl_dark_powers:
            return
        self._update_phase_labels(0)
        self._overview_btn.setVisible(True)
        self._stack.setCurrentIndex(self.PAGE_PL_DARK)
        # Re-select the current power so CE pills and the power filter are restored.
        self._select_pl_power(self._pl_dark_powers[self._pl_dark_idx])

    def _goto_wl_dark(self):
        self.embedded_pl_dark_tab.set_power_filter(None)
        self._update_phase_labels(1)
        self._overview_btn.setVisible(True)
        self._stack.setCurrentIndex(self.PAGE_WL_DARK)
        self.embedded_wl_dark_tab.refresh_if_needed()
        self._wl_to_corr_btn.setVisible(self.embedded_wl_dark_tab.scaling_applied)

    def _check_wl_done(self):
        """Called after every on_data_changed on the embedded white dark tab."""
        if self._stack.currentIndex() != self.PAGE_WL_DARK:
            return
        if self.embedded_wl_dark_tab.scaling_applied:
            self._wl_to_corr_btn.setVisible(True)

    # ── Phase 3: Correction ───────────────────────────────────────────────

    def _goto_correction(self):
        self._update_phase_labels(2)
        self._overview_btn.setVisible(True)
        self._stack.setCurrentIndex(self.PAGE_CORRECTION)
        already_done = bool(self.get_corrected_fn())
        self._corr_to_stitch_btn.setVisible(already_done)
        self._run_corr_btn.setEnabled(not already_done)
        if already_done:
            self._corr_status.setText(
                "✓ Correction already applied.  Select a view or press ④ Stitch to continue."
            )
            self._update_corr_plot()
        else:
            self._corr_status.setText(
                "Press 'Run Correction' to compute and apply whitelight correction."
            )

    def _run_correction(self):
        try:
            self.auto_correct_fn()
            if self.get_corrected_fn():
                self._correction_applied = True
                if self.on_correction_applied:
                    self.on_correction_applied()   # persists flag to session JSON
                self._corr_status.setText("✓ Correction applied successfully.")
                self._corr_to_stitch_btn.setVisible(True)
                self._run_corr_btn.setEnabled(False)
                self._update_corr_plot()
            else:
                self._corr_status.setText(
                    "⚠  Correction failed — check that white files are loaded "
                    "and all scaling is complete."
                )
        except Exception as exc:
            self._corr_status.setText(f"⚠  Error: {exc}")

    # ── Correction view helpers ───────────────────────────────────────────

    _CORR_VIEW_STYLE_ACTIVE = (
        "QPushButton { color: white; background-color: #1565c0; "
        "border-radius: 4px; padding: 2px 10px; font-weight: bold; border: none; }"
    )
    _CORR_VIEW_STYLE_IDLE = (
        "QPushButton { color: #555; border: 1px solid #bbb; "
        "border-radius: 4px; padding: 2px 10px; background: transparent; }"
        "QPushButton:hover { border-color: #888; color: #000; }"
    )

    def _update_corr_view_btn_styles(self):
        for i, btn in enumerate(self._corr_view_btns):
            btn.setStyleSheet(
                self._CORR_VIEW_STYLE_ACTIVE if i == self._corr_view_mode
                else self._CORR_VIEW_STYLE_IDLE
            )
            btn.setChecked(i == self._corr_view_mode)

    def _set_corr_view(self, mode: int):
        self._corr_view_mode = mode
        self._update_corr_view_btn_styles()
        self._update_corr_plot()

    def _update_corr_plot(self):
        """Refresh the correction-page plot for the current view mode."""
        mode = self._corr_view_mode

        if mode == 0:
            # ── Correction coefficients (one colour per CE) ───────────────
            if self.get_correction_dict_fn is None:
                return
            corr_dict = self.get_correction_dict_fn()
            if not corr_dict:
                self._corr_plot.clear()
                return
            series = [
                (df["Energy"].to_numpy(),
                 df["correction_coefficient"].to_numpy(),
                 f"Center_E = {ce:.3f} eV")
                for ce, df in sorted(corr_dict.items())
            ]
            self._corr_plot.plot_series(
                series, title="Correction Coefficients",
                log_y=True, ylabel="Correction coefficient",
            )

        else:
            # ── Modes 1/2/3: all windows of one power get the same colour ─
            title_map = {
                1: "Raw PL Data",
                2: "Dark-Subtracted & Int.-Time Normalised",
                3: "Fully Corrected (ready for stitching)",
            }
            ylabel_map = {1: "Counts", 2: "Counts / s", 3: "Counts (corrected)"}

            pl_files = self.get_pl_files_fn() if self.get_pl_files_fn else []
            meta = {pf.metadata["filename"]: pf.metadata for pf in pl_files}

            # Collect (x, y, filename) triples for the chosen data source
            items: list = []
            if mode == 1:
                if not pl_files:
                    self._corr_plot.clear()
                    return
                for pf in pl_files:
                    items.append((
                        pf.df["Energy"].to_numpy(float),
                        pf.df["Counts"].to_numpy(float),
                        pf.metadata["filename"],
                    ))
            elif mode == 2:
                if self.get_normalized_fn is None:
                    self._corr_plot.clear()
                    return
                normalized = self.get_normalized_fn()
                if not normalized:
                    self._corr_plot.clear()
                    return
                for fname, df in normalized.items():
                    items.append((
                        df["Energy"].to_numpy(float),
                        df["Counts"].to_numpy(float),
                        fname,
                    ))
            elif mode == 3:
                corrected = self.get_corrected_fn()
                if not corrected:
                    self._corr_plot.clear()
                    return
                for fname, df in corrected.items():
                    items.append((
                        df["Energy"].to_numpy(float),
                        df["Counts"].to_numpy(float),
                        fname,
                    ))

            # Build power groups using the same 10 % relative tolerance as
            # the rest of the pipeline, then assign one colour per group.
            raw_powers = [
                meta.get(fname, {}).get("Exc_P")
                for _, _, fname in items
            ]
            canonical: list = []   # one representative power per group
            for p in raw_powers:
                if p is None:
                    continue
                if not any(_powers_match(p, c) for c in canonical):
                    canonical.append(p)
            canonical.sort()

            _COLOURS = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf",
            ]
            # Map every raw power to its group's colour via _powers_match
            def _colour_for(p):
                if p is None:
                    return "#333333"
                for i, c in enumerate(canonical):
                    if _powers_match(p, c):
                        return _COLOURS[i % len(_COLOURS)]
                return "#333333"

            def _label_for(p):
                if p is None:
                    return None
                for c in canonical:
                    if _powers_match(p, c):
                        return f"{c:.2f} mW"
                return f"{p:.2f} mW"

            ax = self._corr_plot.ax
            ax.clear()
            seen_labels: set = set()
            for x, y, fname in items:
                m     = meta.get(fname, {})
                p     = m.get("Exc_P")
                ce    = m.get("Center_E")
                colour    = _colour_for(p)
                lbl_power = _label_for(p) or fname
                # Only the first curve of each power goes into the legend
                legend_label = lbl_power if lbl_power not in seen_labels else ""
                seen_labels.add(lbl_power)
                y_plot = np.where(y > 0, y, np.nan)
                ax.semilogy(x, y_plot, color=colour, lw=1, label=legend_label)

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel(ylabel_map[mode])
            ax.set_title(title_map[mode])
            if seen_labels:
                ax.legend(fontsize=8)
            self._corr_plot.figure.tight_layout()
            self._corr_plot._safe_draw()

    # ── Phase 4: Stitch ───────────────────────────────────────────────────

    def _goto_stitch(self):
        self._update_phase_labels(3)
        self._overview_btn.setVisible(True)
        self._stack.setCurrentIndex(self.PAGE_STITCH)
        if not self.embedded_stitch_tab._power_mode:
            ok = self.embedded_stitch_tab.enter_power_mode()
            if not ok:
                self._stack.setCurrentIndex(self.PAGE_CORRECTION)
                self._corr_status.setText(
                    "⚠  Could not start stitching — apply correction first."
                )

    def _goto_power_plot(self):
        if self.embedded_power_plot_tab is not None:
            self._update_phase_labels(4)
            self._overview_btn.setVisible(True)
            self._stack.setCurrentIndex(self.PAGE_POWER_PLOT)

    def _make_power_plot_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 4, 4, 4)
        hdr_lay.addWidget(QLabel("<b>⑤ Power Series Plot</b>"))
        hdr_lay.addStretch()
        back_btn = QPushButton("◀ Back to Stitch")
        back_btn.clicked.connect(lambda: self._stack.setCurrentIndex(self.PAGE_STITCH))
        hdr_lay.addWidget(back_btn)
        if self.embedded_export_tab is not None:
            to_export_btn = QPushButton("⑥ Export →")
            to_export_btn.setStyleSheet(
                "QPushButton { font-weight: bold; color: white; "
                "background-color: #00695c; border-radius: 4px; padding: 3px 10px; }"
                "QPushButton:hover { background-color: #00796b; }"
            )
            to_export_btn.clicked.connect(self._goto_export)
            hdr_lay.addWidget(to_export_btn)
        lay.addWidget(hdr)
        lay.addWidget(self.embedded_power_plot_tab, 1)
        return page

    def _goto_export(self):
        if self.embedded_export_tab is not None:
            self._update_phase_labels(5)
            self._overview_btn.setVisible(True)
            self._stack.setCurrentIndex(self.PAGE_EXPORT)
            self.embedded_export_tab.refresh()

    def _make_export_page(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 4, 4, 4)
        hdr_lay.addWidget(QLabel("<b>⑥ Export</b>"))
        hdr_lay.addStretch()
        back_btn = QPushButton("◀ Back to Power Plot")
        back_btn.clicked.connect(self._goto_power_plot)
        hdr_lay.addWidget(back_btn)
        lay.addWidget(hdr)
        lay.addWidget(self.embedded_export_tab, 1)
        return page

    # ══ Phase indicator helpers ════════════════════════════════════════════

    def _phase_done_states(self) -> list:
        """Return per-phase completion booleans (index matches _phase_labels)."""
        stitched = (self.embedded_stitch_tab.stitched_results
                    or self.embedded_stitch_tab._power_stitched)
        return [
            bool(self._pl_done) and all(self._pl_done.values()),       # ① PL Dark
            self.embedded_wl_dark_tab.scaling_applied,                  # ② White Dark
            self._correction_applied or bool(self.get_corrected_fn()),  # ③ Correction
            bool(stitched),                                             # ④ Stitching
            False,                                                      # ⑤ Power Plot
            False,                                                      # ⑥ Export
        ]

    def _update_phase_labels(self, active: int):
        _ACTIVE = ("color: white; background-color: #1565c0; "
                   "border-radius: 4px; padding: 2px 10px; font-weight: bold;")
        _DONE   = ("color: white; background-color: #43a047; "
                   "border-radius: 4px; padding: 2px 10px;")
        _WAIT   = ("color: #888; border: 1px solid #bbb; "
                   "border-radius: 4px; padding: 2px 10px;")
        done = self._phase_done_states()
        for i, lbl in enumerate(self._phase_labels):
            if i == active:
                lbl.setStyleSheet(_ACTIVE)
            elif i < len(done) and done[i]:
                lbl.setStyleSheet(_DONE)
            else:
                lbl.setStyleSheet(_WAIT)

    # ══ Reset ══════════════════════════════════════════════════════════════

    def _reset_scaling(self):
        reply = QMessageBox.question(
            self, "Reset All Scaling",
            "This will erase all PL and white dark scaling progress.\n"
            "Correction ratios and corrected spectra will also be cleared.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.reset_scaling_fn()
            self._pl_done              = {}
            self._pip_ce_done          = {}
            self._correction_applied   = False
            if self._pl_dark_pills:
                self._update_pl_pill_colors()
            self.embedded_stitch_tab.reset()
            self._stack.setCurrentIndex(self.PAGE_OVERVIEW)
            self._overview_btn.setVisible(False)
            self._update_phase_labels(-1)

    # ══ JSON replay ════════════════════════════════════════════════════════

    def _request_replay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pipeline analysis JSON",
            self.get_replay_dir(),
            "JSON files (*.json);;All files (*)"
        )
        if path:
            self.set_replay_dir(path)
            if self.replay_fn:
                self.replay_fn(path)

    # ══ Called when this tab becomes visible ═══════════════════════════════

    def refresh_status(self):
        powers = self.get_pl_powers_fn()
        if powers:
            self._pl_done = {p: self.get_pl_dark_done_for_power_fn(p) for p in powers}
            if set(powers) != set(self._pl_dark_pills.keys()):
                self._pl_dark_powers = list(powers)
                self._rebuild_pl_pills()
            else:
                self._update_pl_pill_colors()
        # Refresh phase-label colours to reflect restored done state.
        # Pass the currently active page index so the active label stays blue;
        # pass -1 when on the overview so no label is highlighted.
        current_page = self._stack.currentIndex()
        active_phase = current_page - 1 if current_page > 0 else -1
        self._update_phase_labels(active_phase)


# ════════════════════════════════════════════════════════════════════════════
# Standard Mode container — inner tab widget for the manual analysis workflow
# ════════════════════════════════════════════════════════════════════════════
class StandardModeTab(QWidget):
    """
    Wraps the manual analysis workflow in an inner QTabWidget.

    All inner tab instances are created in MainWindow and passed in here so
    that MainWindow can still reference them directly for wiring.
    """

    _TAB_LABELS = [
        "① Dark Scaling (PL)",
        "② Dark Scaling (White)",
        "③ Apply Corrections",
        "④ PL Analysis",
        "⑤ Stitch & Export",
        "⑥ Power Series Plot",
    ]
    # Tab index → label shown on the advance button
    _NEXT_LABELS = {
        0: "② Dark Scaling (White)",
        1: "③ Apply Corrections",
        2: "④ PL Analysis",
        3: "⑤ Stitch & Export",
        4: "⑥ Power Series Plot",
    }

    def __init__(self,
                 dark_scaling_pl_tab,
                 dark_scaling_wl_tab,
                 corrections_tab,
                 pl_tab,
                 stitch_tab,
                 power_plot_tab):
        super().__init__()
        self._dark_scaling_pl_tab = dark_scaling_pl_tab
        self._dark_scaling_wl_tab = dark_scaling_wl_tab
        self._corrections_tab     = corrections_tab
        self._pl_tab              = pl_tab
        self._stitch_tab          = stitch_tab

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar: blue "continue" button on the right ───────────────
        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 3, 4, 3)
        hdr_lay.addStretch()
        self._next_btn = QPushButton("")
        self._next_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; "
            "background-color: #1565c0; border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        self._next_btn.setVisible(False)
        self._next_btn.clicked.connect(self._advance_tab)
        hdr_lay.addWidget(self._next_btn)
        layout.addWidget(hdr)

        self.inner_tabs = QTabWidget()
        for label, tab in zip(self._TAB_LABELS, [
            dark_scaling_pl_tab, dark_scaling_wl_tab, corrections_tab,
            pl_tab, stitch_tab, power_plot_tab,
        ]):
            self.inner_tabs.addTab(tab, label)
        layout.addWidget(self.inner_tabs, 1)

        self.inner_tabs.currentChanged.connect(self.refresh_next_btn)

    def _is_complete(self, idx: int) -> bool:
        if idx == 0:
            return self._dark_scaling_pl_tab.scaling_applied
        if idx == 1:
            return self._dark_scaling_wl_tab.scaling_applied
        if idx == 2:
            return bool(self._corrections_tab.correction_dict)
        if idx == 3:
            return bool(self._pl_tab.corrected)
        if idx == 4:
            return bool(
                self._stitch_tab.stitched_results
                or self._stitch_tab._power_stitched
            )
        return False  # last tab — no next

    def refresh_next_btn(self):
        idx = self.inner_tabs.currentIndex()
        tab_bar = self.inner_tabs.tabBar()

        # Colour completed tab labels green, reset others to default
        for i in range(self.inner_tabs.count()):
            if self._is_complete(i):
                tab_bar.setTabTextColor(i, QColor("#2e7d32"))
            else:
                tab_bar.setTabTextColor(i, QColor())   # default (inherits palette)

        # Show / hide the advance button
        if idx >= self.inner_tabs.count() - 1:
            self._next_btn.setVisible(False)
            return
        if self._is_complete(idx):
            self._next_btn.setText(
                f"▶  Continue to {self._NEXT_LABELS.get(idx, 'Next Step')}"
            )
            self._next_btn.setVisible(True)
        else:
            self._next_btn.setVisible(False)

    def _advance_tab(self):
        idx = self.inner_tabs.currentIndex()
        if idx < self.inner_tabs.count() - 1:
            self.inner_tabs.setCurrentIndex(idx + 1)


# ════════════════════════════════════════════════════════════════════════════
# Main window
# ════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL Auto Analysis")
        self.resize(1200, 940)

        self.tabs = QTabWidget()
        tabs = self.tabs
        self.calib_tab = CalibrationTab()

        # ── Separate dark-scale dicts per mode ───────────────────────────────
        # Standard mode and pipeline mode each keep their own dark_scale_dict
        # so that scaling applied in one mode never bleeds into the other.
        self._std_dark_dict:      dict = {}
        self._pipeline_dark_dict: dict = {}

        # ── Standard-mode invalidation callbacks ─────────────────────────────
        def _on_std_pl_scaling_changed():
            self.calib_tab._save_session()
            self.std_corrections_tab.invalidate_dark_sub()
            self.std_pl_tab.invalidate()
            self.standard_mode_tab.refresh_next_btn()

        def _on_std_wl_scaling_changed():
            self.calib_tab._save_session()
            self.std_corrections_tab.invalidate_correction()
            self.std_pl_tab.invalidate()
            self.standard_mode_tab.refresh_next_btn()

        # ── Pipeline-mode invalidation callbacks ─────────────────────────────
        def _on_pip_pl_scaling_changed():
            self.calib_tab._save_session()
            self.pipeline_corrections_tab.invalidate_dark_sub()
            self.pipeline_pl_tab.invalidate()

        def _on_pip_wl_scaling_changed():
            self.calib_tab._save_session()
            self.pipeline_corrections_tab.invalidate_correction()
            self.pipeline_pl_tab.invalidate()

        # ── PL files reloaded → reset all mode scaling tabs ──────────────────
        def _on_pl_files_changed():
            self.std_dark_scaling_pl_tab.reset_groups()
            self.pipeline_pl_dark_tab.reset_groups()
            self.std_corrections_tab.invalidate_dark_sub()
            self.pipeline_corrections_tab.invalidate_dark_sub()
            self.std_pl_tab.invalidate()
            self.pipeline_pl_tab.invalidate()

        self.calib_tab.on_pl_files_changed = _on_pl_files_changed
        self.calib_tab.on_replay_requested = self._replay_from_json

        # ════════════════════════════════════════════════════════════════════
        # STANDARD MODE TABS
        # ════════════════════════════════════════════════════════════════════
        self.std_dark_scaling_pl_tab = DarkScalingTab(
            get_files_fn=lambda: self.calib_tab.checked_pl_files(),
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            file_label="PL",
            on_data_changed=_on_std_pl_scaling_changed,
            dark_scale_dict=self._std_dark_dict,
        )
        self.std_dark_scaling_wl_tab = DarkScalingTab(
            get_files_fn=lambda: [
                pf for d in self.calib_tab.checked_white_dict().values()
                for pf in d.values()
            ],
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            file_label="White",
            on_data_changed=_on_std_wl_scaling_changed,
            get_partner_dark_scales=lambda: self._std_dark_dict,
            dark_scale_dict=self._std_dark_dict,
        )
        self.std_corrections_tab = CorrectionsTab(
            get_pl_files=lambda: self.calib_tab.checked_pl_files(),
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            get_white_dict=lambda: self.calib_tab.checked_white_dict(),
            get_halogen=lambda: self.calib_tab.halogen_df,
            get_dark_scale_dict=lambda: self._std_dark_dict,
        )
        self.std_pl_tab = PLAnalysisTab(
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            get_corr_dict=lambda: self.std_corrections_tab.correction_dict,
            get_pl_files=lambda: self.calib_tab.checked_pl_files(),
            get_normalized=lambda: self.std_corrections_tab.normalized,
            get_dark_scale_dict=lambda: self._std_dark_dict,
        )
        self.std_stitch_tab = StitchTab(
            get_corrected_grouped=self.std_pl_tab.get_corrected_grouped,
            get_scaling_applied_pl=lambda: self.std_dark_scaling_pl_tab.scaling_applied,
            get_scaling_applied_wl=lambda: self.std_dark_scaling_wl_tab.scaling_applied,
            get_correction_dict=lambda: self.std_corrections_tab.correction_dict,
            get_corrected=lambda: self.std_pl_tab.corrected,
            get_pl_scaling_meta=lambda: self.std_dark_scaling_pl_tab.get_scaling_meta(),
            get_wl_scaling_meta=lambda: self.std_dark_scaling_wl_tab.get_scaling_meta(),
            get_save_dir=lambda: self.calib_tab._get_last_dir("save"),
            set_save_dir=lambda p: self.calib_tab._set_last_dir("save", p),
        )

        def _std_get_stitched_for_plot():
            if self.std_stitch_tab._power_mode and self.std_stitch_tab._power_stitched:
                return self.std_stitch_tab._power_stitched
            return self.std_stitch_tab.stitched_results

        self.std_power_plot_tab = PowerSeriesPlotTab(
            get_stitched_results=_std_get_stitched_for_plot,
            get_save_dir=lambda: self.calib_tab._get_last_dir("save"),
            set_save_dir=lambda p: self.calib_tab._set_last_dir("save", p),
        )

        # Container that nests all standard-mode tabs in an inner QTabWidget
        self.standard_mode_tab = StandardModeTab(
            dark_scaling_pl_tab = self.std_dark_scaling_pl_tab,
            dark_scaling_wl_tab = self.std_dark_scaling_wl_tab,
            corrections_tab     = self.std_corrections_tab,
            pl_tab              = self.std_pl_tab,
            stitch_tab          = self.std_stitch_tab,
            power_plot_tab      = self.std_power_plot_tab,
        )
        self.std_corrections_tab.on_data_changed = self.standard_mode_tab.refresh_next_btn
        self.std_pl_tab.on_data_changed          = self.standard_mode_tab.refresh_next_btn
        self.std_stitch_tab.on_stitching_done    = self.standard_mode_tab.refresh_next_btn

        # ════════════════════════════════════════════════════════════════════
        # PIPELINE MODE TABS  (all embedded inside PowerPipelineTab)
        # ════════════════════════════════════════════════════════════════════
        self.pipeline_pl_dark_tab = DarkScalingTab(
            get_files_fn=lambda: self.calib_tab.checked_pl_files(),
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            file_label="PL",
            on_data_changed=_on_pip_pl_scaling_changed,
            dark_scale_dict=self._pipeline_dark_dict,
        )
        self.pipeline_wl_dark_tab = DarkScalingTab(
            get_files_fn=lambda: [
                pf for d in self.calib_tab.checked_white_dict().values()
                for pf in d.values()
            ],
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            file_label="White",
            on_data_changed=_on_pip_wl_scaling_changed,
            get_partner_dark_scales=lambda: self._pipeline_dark_dict,
            dark_scale_dict=self._pipeline_dark_dict,
        )
        # Pipeline embedded tabs: review each CE individually
        self.pipeline_pl_dark_tab.all_btn.setVisible(False)
        self.pipeline_wl_dark_tab.all_btn.setVisible(False)

        self.pipeline_corrections_tab = CorrectionsTab(
            get_pl_files=lambda: self.calib_tab.checked_pl_files(),
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            get_white_dict=lambda: self.calib_tab.checked_white_dict(),
            get_halogen=lambda: self.calib_tab.halogen_df,
            get_dark_scale_dict=lambda: self._pipeline_dark_dict,
        )
        self.pipeline_pl_tab = PLAnalysisTab(
            get_dark_dict=lambda: self.calib_tab.checked_dark_dict(),
            get_corr_dict=lambda: self.pipeline_corrections_tab.correction_dict,
            get_pl_files=lambda: self.calib_tab.checked_pl_files(),
            get_normalized=lambda: self.pipeline_corrections_tab.normalized,
            get_dark_scale_dict=lambda: self._pipeline_dark_dict,
        )
        self.pipeline_stitch_tab = StitchTab(
            get_corrected_grouped=self.pipeline_pl_tab.get_corrected_grouped,
            get_scaling_applied_pl=lambda: self.pipeline_pl_dark_tab.scaling_applied,
            get_scaling_applied_wl=lambda: self.pipeline_wl_dark_tab.scaling_applied,
            get_correction_dict=lambda: self.pipeline_corrections_tab.correction_dict,
            get_corrected=lambda: self.pipeline_pl_tab.corrected,
            get_pl_scaling_meta=lambda: self.pipeline_pl_dark_tab.get_scaling_meta(),
            get_wl_scaling_meta=lambda: self.pipeline_wl_dark_tab.get_scaling_meta(),
            get_save_dir=lambda: self.calib_tab._get_last_dir("save"),
            set_save_dir=lambda p: self.calib_tab._set_last_dir("save", p),
        )

        def _pip_get_stitched_for_plot():
            if self.pipeline_stitch_tab._power_mode and self.pipeline_stitch_tab._power_stitched:
                return self.pipeline_stitch_tab._power_stitched
            return self.pipeline_stitch_tab.stitched_results

        self.pipeline_power_plot_tab = PowerSeriesPlotTab(
            get_stitched_results=_pip_get_stitched_for_plot,
            get_save_dir=lambda: self.calib_tab._get_last_dir("save"),
            set_save_dir=lambda p: self.calib_tab._set_last_dir("save", p),
        )

        self.pipeline_export_tab = ExportTab(
            get_pl_files_fn   = lambda: self.calib_tab.checked_pl_files(),
            get_normalized_fn = lambda: self.pipeline_corrections_tab.normalized,
            get_corrected_fn  = lambda: self.pipeline_pl_tab.corrected,
            get_stitched_fn   = _pip_get_stitched_for_plot,
            get_save_dir      = lambda: self.calib_tab._get_last_dir("save"),
            set_save_dir      = lambda p: self.calib_tab._set_last_dir("save", p),
        )

        # ── Helper closures for the pipeline ─────────────────────────────────
        def _pipeline_auto_correct_and_apply():
            """User-triggered: uses the standard tab methods (with their UI feedback)."""
            self.pipeline_corrections_tab._build_correction()
            if not self.pipeline_corrections_tab.correction_dict:
                return
            self.pipeline_corrections_tab._apply_dark_sub()
            self.pipeline_pl_tab.refresh_table()
            self.pipeline_pl_tab.pl_table.check_all(True)
            self.pipeline_pl_tab._apply_checked()

        def _pipeline_auto_correct_silent():
            """Session-restore: same computation but no QMessageBox dialogs."""
            pl_files  = self.calib_tab.checked_pl_files()
            dark_dict = self.calib_tab.checked_dark_dict()
            halogen   = self.calib_tab.halogen_df
            white_dict = self.calib_tab.checked_white_dict()
            if not pl_files or not dark_dict or halogen is None or not white_dict:
                return
            dsd = self._pipeline_dark_dict
            # Dark subtraction
            self.pipeline_corrections_tab.normalized.clear()
            for pf in pl_files:
                try:
                    self.pipeline_corrections_tab.normalized[pf.metadata["filename"]] = \
                        pf.subtract_dark_and_normalize(dark_dict, dsd)
                except Exception:
                    pass
            # Correction ratios
            ratios, hard_errors, _ = build_correction_ratios(
                white_dict, dark_dict, halogen, dsd)
            if hard_errors or not ratios:
                return
            self.pipeline_corrections_tab.correction_dict.clear()
            self.pipeline_corrections_tab.correction_dict.update(ratios)
            # Apply correction
            self.pipeline_pl_tab.corrected.clear()
            for pf in pl_files:
                try:
                    self.pipeline_pl_tab.corrected[pf.metadata["filename"]] = \
                        apply_correction(pf, dark_dict,
                                         self.pipeline_corrections_tab.correction_dict, dsd)
                except Exception:
                    pass

        def _get_pl_powers():
            pl_files = self.calib_tab.checked_pl_files()
            powers: list = []
            for pf in pl_files:
                p = pf.metadata.get("Exc_P")
                if p is None:
                    continue
                if not any(_powers_match(p, q) for q in powers):
                    powers.append(p)
            return sorted(powers)

        def _pl_dark_done_for_power(power: float) -> bool:
            """True if all CE groups for this power are confirmed in _pip_ce_done."""
            pl_files = self.calib_tab.checked_pl_files()
            ces = {
                pf.metadata["Center_E"]
                for pf in pl_files
                if (pf.metadata.get("Exc_P") is not None
                    and _powers_match(pf.metadata["Exc_P"], power)
                    and pf.metadata.get("Center_E") is not None)
            }
            if not ces:
                return False
            pip_ce = self.pipeline_tab._pip_ce_done.get(power, {})
            # Try also with a power key found via _powers_match (handles float imprecision)
            if not pip_ce:
                for p_key, p_dict in self.pipeline_tab._pip_ce_done.items():
                    if _powers_match(p_key, power):
                        pip_ce = p_dict
                        break
            return all(pip_ce.get(ce, False) for ce in ces)

        def _reset_pipeline_scaling():
            self.pipeline_pl_dark_tab.reset_groups()
            self.pipeline_wl_dark_tab.reset_groups()
            self.pipeline_corrections_tab.invalidate_dark_sub()
            self.pipeline_corrections_tab.invalidate_correction()
            self.pipeline_pl_tab.invalidate()
            self.calib_tab._save_session()

        def _check_window_counts():
            """
            Return an error string if powers have inconsistent measurement files.
            Catches two problems:
              1. Different sets of Center_E windows across powers.
              2. Duplicate files at the same CE for one power (same CE set, but
                 more files than the reference — the set-based check misses this
                 because duplicates collapse to one entry in a set).
            """
            pl_files = self.calib_tab.checked_pl_files()
            if not pl_files:
                return None

            powers_files: dict = {}
            for pf in pl_files:
                p  = pf.metadata.get("Exc_P")
                ce = pf.metadata.get("Center_E")
                if p is None or ce is None:
                    continue
                bucket = next((q for q in powers_files if _powers_match(q, p)), p)
                powers_files.setdefault(bucket, []).append(pf)
            if not powers_files:
                return None

            powers_ces    = {p: {pf.metadata["Center_E"] for pf in files}
                             for p, files in powers_files.items()}
            powers_counts = {p: len(files) for p, files in powers_files.items()}
            ref_power  = next(iter(powers_files))
            ref_ces    = powers_ces[ref_power]
            ref_count  = powers_counts[ref_power]

            problems: list = []
            for p, ces in sorted(powers_ces.items()):
                n       = powers_counts[p]
                missing = ref_ces - ces
                extra   = ces - ref_ces
                issues: list = []
                if missing:
                    issues.append(
                        "missing: " + ", ".join(f"{c:.3f} eV" for c in sorted(missing))
                    )
                if extra:
                    issues.append(
                        "extra: " + ", ".join(f"{c:.3f} eV" for c in sorted(extra))
                    )
                if not missing and not extra and n != ref_count:
                    diff = n - ref_count
                    issues.append(
                        f"{abs(diff)} {'extra' if diff > 0 else 'fewer'} file(s) "
                        f"({n} vs {ref_count}) — likely duplicate measurement(s)"
                    )
                if issues:
                    problems.append(
                        f"  {p:.2f} mW ({n} file(s)) — " + "; ".join(issues)
                    )

            if problems:
                return (
                    f"Inconsistent measurements across powers "
                    f"(reference: {ref_power:.2f} mW — {ref_count} file(s), "
                    f"{', '.join(f'{c:.3f} eV' for c in sorted(ref_ces))}):\n"
                    + "\n".join(problems)
                    + "\n\nUncheck the duplicate or load missing files in Tab ① "
                    "before starting."
                )
            return None

        # ── PowerPipelineTab ──────────────────────────────────────────────────
        self.pipeline_tab = PowerPipelineTab(
            embedded_pl_dark_tab           = self.pipeline_pl_dark_tab,
            embedded_wl_dark_tab           = self.pipeline_wl_dark_tab,
            embedded_stitch_tab            = self.pipeline_stitch_tab,
            auto_correct_fn                = _pipeline_auto_correct_and_apply,
            get_pl_powers_fn               = _get_pl_powers,
            get_corrected_fn               = lambda: self.pipeline_pl_tab.corrected,
            reset_scaling_fn               = _reset_pipeline_scaling,
            get_pl_dark_done_for_power_fn  = _pl_dark_done_for_power,
            check_window_counts_fn         = _check_window_counts,
            replay_fn                      = self._pipeline_replay_from_json,
            embedded_power_plot_tab        = self.pipeline_power_plot_tab,
            get_correction_dict_fn         = lambda: self.pipeline_corrections_tab.correction_dict,
            get_pl_files_fn                = lambda: self.calib_tab.checked_pl_files(),
            get_normalized_fn              = lambda: self.pipeline_corrections_tab.normalized,
            silent_auto_correct_fn         = _pipeline_auto_correct_silent,
            get_replay_dir                 = lambda: self.calib_tab._get_last_dir("replay"),
            set_replay_dir                 = lambda p: self.calib_tab._set_last_dir("replay", p),
            embedded_export_tab            = self.pipeline_export_tab,
        )

        # ── Three top-level tabs ──────────────────────────────────────────────
        tabs.addTab(self.calib_tab,          "① Load Files")
        tabs.addTab(self.pipeline_tab,       "② Power-by-Power Pipeline")
        tabs.addTab(self.standard_mode_tab,  "③ Automatic Analysis")

        self.setCentralWidget(tabs)

        # ── Inner-tab refresh for standard mode ───────────────────────────────
        inner = self.standard_mode_tab.inner_tabs
        std_pl_scale_idx = inner.indexOf(self.std_dark_scaling_pl_tab)
        std_wl_scale_idx = inner.indexOf(self.std_dark_scaling_wl_tab)
        std_pl_idx       = inner.indexOf(self.std_pl_tab)
        std_stitch_idx   = inner.indexOf(self.std_stitch_tab)

        inner.currentChanged.connect(lambda idx: (
            self.std_dark_scaling_pl_tab.refresh_if_needed() if idx == std_pl_scale_idx else None
        ))
        inner.currentChanged.connect(lambda idx: (
            self.std_dark_scaling_wl_tab.refresh_if_needed() if idx == std_wl_scale_idx else None
        ))
        inner.currentChanged.connect(
            lambda idx: self.std_pl_tab.refresh_table() if idx == std_pl_idx else None
        )
        inner.currentChanged.connect(
            lambda idx: self.std_stitch_tab.refresh_checklist() if idx == std_stitch_idx else None
        )

        # ── Standard mode: "Stitch Power by Power" button in PL Analysis tab ──
        def _std_enter_power_stitch():
            if not self.std_pl_tab.corrected:
                QMessageBox.warning(
                    self, "Not ready",
                    "Apply correction first — use 'Apply Correction to Checked' in "
                    "the PL Analysis tab of Standard Analysis."
                )
                return
            ok = self.std_stitch_tab.enter_power_mode()
            if ok:
                inner.setCurrentWidget(self.std_stitch_tab)

        self.std_pl_tab.stitch_power_btn.clicked.connect(_std_enter_power_stitch)

        # ── Pipeline tab refresh when switching to it ─────────────────────────
        pipeline_idx = tabs.indexOf(self.pipeline_tab)
        tabs.currentChanged.connect(
            lambda idx: self.pipeline_tab.refresh_status() if idx == pipeline_idx else None
        )

        # ── Session persistence callbacks ─────────────────────────────────────
        self.std_stitch_tab._save_power_progress      = self.calib_tab._save_session
        self.pipeline_stitch_tab._save_power_progress = self.calib_tab._save_session

        self.calib_tab.get_std_pl_done_groups    = lambda: self.std_dark_scaling_pl_tab._externally_done
        self.calib_tab.get_std_wl_done_groups    = lambda: self.std_dark_scaling_wl_tab._externally_done
        self.calib_tab.get_std_dark_scales       = lambda: self._std_dark_dict
        self.calib_tab.get_std_power_stitch_logs = lambda: self.std_stitch_tab._power_blend_logs
        self.calib_tab.get_pip_pl_done_groups    = lambda: self.pipeline_pl_dark_tab._externally_done
        self.calib_tab.get_pip_wl_done_groups    = lambda: self.pipeline_wl_dark_tab._externally_done
        self.calib_tab.get_pip_dark_scales       = lambda: self._pipeline_dark_dict
        self.calib_tab.get_pip_power_stitch_logs = lambda: self.pipeline_stitch_tab._power_blend_logs
        self.calib_tab.get_pip_ce_done              = lambda: self.pipeline_tab._pip_ce_done
        self.calib_tab.get_pip_correction_applied   = lambda: self.pipeline_tab._correction_applied

        # ── Session restore ───────────────────────────────────────────────────
        # WL done-groups are restored BEFORE _ingest_pl so that the _save_session
        # call inside _ingest_pl captures the correct wl_done_groups.
        # PL done-groups must come AFTER _ingest_pl because _ingest_pl triggers
        # on_pl_files_changed → reset_groups(), which clears _externally_done on
        # the PL dark tabs.  WL tabs are unaffected by reset_groups, so their
        # done state can be set first.
        if self.calib_tab._restored_std_wl_done_groups:
            self.std_dark_scaling_wl_tab.set_done_groups(
                self.calib_tab._restored_std_wl_done_groups)
        if self.calib_tab._restored_pip_wl_done_groups:
            self.pipeline_wl_dark_tab.set_done_groups(
                self.calib_tab._restored_pip_wl_done_groups)

        if self.calib_tab._restored_pl_paths:
            self.calib_tab._ingest_pl(self.calib_tab._restored_pl_paths)

        # Standard mode restore
        if self.calib_tab._restored_std_dark_scales:
            self._std_dark_dict.update(self.calib_tab._restored_std_dark_scales)
        if self.calib_tab._restored_std_pl_done_groups:
            self.std_dark_scaling_pl_tab.set_done_groups(
                self.calib_tab._restored_std_pl_done_groups)
        self.std_dark_scaling_pl_tab.refresh_if_needed()
        self.std_dark_scaling_wl_tab.refresh_if_needed()

        # Pipeline mode restore
        if self.calib_tab._restored_pip_dark_scales:
            self._pipeline_dark_dict.update(self.calib_tab._restored_pip_dark_scales)
        if getattr(self.calib_tab, "_restored_pip_ce_done", {}):
            self.pipeline_tab._pip_ce_done = self.calib_tab._restored_pip_ce_done
        if self.calib_tab._restored_pip_pl_done_groups:
            self.pipeline_pl_dark_tab.set_done_groups(
                self.calib_tab._restored_pip_pl_done_groups)
        self.pipeline_pl_dark_tab.refresh_if_needed()
        self.pipeline_wl_dark_tab.refresh_if_needed()

        # ── Pipeline: restore correction + stitch state ───────────────────────
        # If blend logs exist, run correction eagerly so restore_power_mode has
        # corrected data to work with.  If only correction was done (no stitch),
        # the flag is set immediately and correction is deferred to Start Pipeline.
        restored_pip_psl = getattr(self.calib_tab, "_restored_pip_power_stitch_logs", {})
        pip_corr_applied  = getattr(self.calib_tab, "_restored_pip_correction_applied", False)

        if restored_pip_psl and self.calib_tab.pl_files:
            try:
                _pipeline_auto_correct_silent()
                if self.pipeline_pl_tab.corrected:
                    self.pipeline_tab._correction_applied = True
                    self.pipeline_stitch_tab.restore_power_mode(restored_pip_psl)
                    # on_correction_applied saves the session (correction flag)
                    if self.pipeline_tab.on_correction_applied:
                        self.pipeline_tab.on_correction_applied()
                    # Refresh phase labels so ①–④ show green on first tab switch
                    self.pipeline_tab.refresh_status()
            except Exception:
                pass
        elif pip_corr_applied:
            # Correction was done but no stitching — defer to Start Pipeline.
            self.pipeline_tab._correction_applied = True

        # Re-save once so the session file reflects the fully-restored state.
        self.calib_tab._save_session()

        # ── Standard mode: restore stitch blend logs ──────────────────────────
        restored_std_psl = getattr(self.calib_tab, "_restored_std_power_stitch_logs", {})
        if restored_std_psl and self.calib_tab.pl_files:
            try:
                self.std_corrections_tab._build_correction()
                if self.std_corrections_tab.correction_dict:
                    self.std_corrections_tab._apply_dark_sub()
                    self.std_pl_tab.pl_table.check_all(True)
                    self.std_pl_tab._apply_checked()
                    if self.std_pl_tab.corrected:
                        self.std_stitch_tab.restore_power_mode(restored_std_psl)
            except Exception:
                pass

    # ── JSON Replay ──────────────────────────────────────────────────────────
    def _replay_from_json(self, json_path: str):
        """
        Load analysis_metadata.json and re-apply every recorded analysis step:
          1. Validate that all required files are currently loaded.
          2. Apply PL and white dark-scaling factors.
          3. Build correction ratios (re-computed, deterministic).
          4. Apply dark subtraction + whitelight correction to all PL files.
          5. Run stitching with the recorded blend windows.
          6. Switch to the Stitch tab with results ready.
        """
        try:
            meta = json.loads(Path(json_path).read_text())
        except Exception as exc:
            QMessageBox.critical(self, "Replay — load error",
                f"Could not read JSON:\n{exc}")
            return

        # Replay targets Standard Analysis mode tabs.
        # ── 1. Validate files ────────────────────────────────────────────────
        missing: list[str] = []

        if self.calib_tab.halogen_df is None:
            missing.append("• Halogen lamp reference not loaded")

        # Validate by filename (path-independent — works across computers).
        # The JSON stores every measured file's filename in dark_scale_by_file.
        expected_pl_fnames: set = set()
        for grp in meta.get("pl_dark_scaling", {}).values():
            expected_pl_fnames.update(grp.get("dark_scale_by_file", {}).keys())
        loaded_pl_fnames = {pf.metadata["filename"] for pf in self.calib_tab.pl_files}
        for fname in sorted(expected_pl_fnames - loaded_pl_fnames):
            missing.append(f"• PL file not loaded: {fname}")

        expected_wl_fnames: set = set()
        for grp in meta.get("white_dark_scaling", {}).values():
            expected_wl_fnames.update(grp.get("dark_scale_by_file", {}).keys())
        loaded_wl_fnames = {
            pf.metadata["filename"]
            for td in self.calib_tab.white_dict.values()
            for pf in td.values()
        }
        for fname in sorted(expected_wl_fnames - loaded_wl_fnames):
            missing.append(f"• White file not loaded: {fname}")

        dark_dict = self.calib_tab.checked_dark_dict()
        for ce_str in meta.get("pl_dark_scaling", {}):
            ce = float(ce_str)
            if ce not in dark_dict:
                missing.append(f"• No dark file loaded for Center_E = {ce:.4f} eV (needed for PL)")
        for ce_str in meta.get("white_dark_scaling", {}):
            ce = float(ce_str)
            if ce not in dark_dict:
                missing.append(f"• No dark file loaded for Center_E = {ce:.4f} eV (needed for white)")

        if missing:
            QMessageBox.warning(
                self, "Replay — missing files",
                "Cannot replay: the following files or calibration data are missing.\n"
                "Please load them in the Load Files tab first.\n\n"
                + "\n".join(missing)
            )
            return

        # ── 2. Apply PL dark-scaling factors (into standard mode dict) ────────
        for ce_str, grp in meta.get("pl_dark_scaling", {}).items():
            ce = float(ce_str)
            for fname, ds in grp.get("dark_scale_by_file", {}).items():
                self.std_dark_scaling_pl_tab.dark_scale_dict[fname] = float(ds)
            win = grp.get("edge_window_eV")
            if win:
                self.std_dark_scaling_pl_tab._scaling_windows[ce] = tuple(win)

        pl_done_ces = set(float(ce) for ce in meta.get("pl_dark_scaling", {}))
        self.std_dark_scaling_pl_tab.set_done_groups(pl_done_ces)
        self.std_dark_scaling_pl_tab.refresh_if_needed()
        self.calib_tab._save_session()
        self.std_corrections_tab.invalidate_dark_sub()
        self.std_pl_tab.invalidate()

        # Also restore white file dark scales from the JSON
        for ce_str, grp in meta.get("white_dark_scaling", {}).items():
            for fname, ds in grp.get("dark_scale_by_file", {}).items():
                self.std_dark_scaling_wl_tab.dark_scale_dict[fname] = float(ds)

        # ── 3. Restore white confirmation state ───────────────────────────────
        wl_done_ces = set(float(ce) for ce in meta.get("white_dark_scaling", {}))
        self.std_dark_scaling_wl_tab.set_done_groups(wl_done_ces)
        self.std_dark_scaling_wl_tab.refresh_if_needed()
        self.calib_tab._save_session()
        self.std_corrections_tab.invalidate_correction()
        self.std_pl_tab.invalidate()

        # ── 4. Build correction ratios ───────────────────────────────────────
        self.std_corrections_tab._build_correction()
        if not self.std_corrections_tab.correction_dict:
            QMessageBox.critical(self, "Replay — correction failed",
                "Could not build correction ratios. Check that halogen, white, "
                "and dark files are correctly loaded and checked.")
            return

        # ── 5. Apply dark subtraction + whitelight correction ────────────────
        self.std_corrections_tab._apply_dark_sub()
        if not self.std_corrections_tab.normalized:
            QMessageBox.critical(self, "Replay — dark subtraction failed",
                "Dark subtraction produced no results.")
            return

        self.std_pl_tab.refresh_table()
        self.std_pl_tab.pl_table.check_all(True)
        self.std_pl_tab._apply_checked()
        if not self.std_pl_tab.corrected:
            QMessageBox.critical(self, "Replay — correction failed",
                "Whitelight correction produced no results.")
            return

        # ── 6. Stitch with recorded blend windows ────────────────────────────
        blend_windows = meta.get("stitching", {}).get("blend_windows", [])
        if not blend_windows:
            QMessageBox.information(self, "Replay — done (no stitching)",
                "Scaling and correction applied successfully.\n"
                "No stitching data found in the JSON — skipping stitch step.")
            self.tabs.setCurrentWidget(self.standard_mode_tab)
            self.standard_mode_tab.inner_tabs.setCurrentWidget(self.std_pl_tab)
            return

        self.std_stitch_tab._start()
        if not self.std_stitch_tab.power_groups:
            QMessageBox.warning(self, "Replay — stitching skipped",
                "Scaling and correction applied, but stitching could not start "
                "(no corrected groups with ≥ 2 windows).")
            self.tabs.setCurrentWidget(self.standard_mode_tab)
            self.standard_mode_tab.inner_tabs.setCurrentWidget(self.std_pl_tab)
            return

        stitch_errors = []
        for entry in sorted(blend_windows, key=lambda e: e["step"]):
            x_min, x_max = entry["window_eV"]
            self.std_stitch_tab._selected_range = (x_min, x_max)
            try:
                self.std_stitch_tab._do_stitch()
            except Exception as exc:
                stitch_errors.append(f"Step {entry['step']}: {exc}")
                break

        if stitch_errors:
            QMessageBox.warning(self, "Replay — stitch warnings",
                "\n".join(stitch_errors))

        # ── 7. Navigate to stitch tab ─────────────────────────────────────────
        self.tabs.setCurrentWidget(self.standard_mode_tab)
        self.standard_mode_tab.inner_tabs.setCurrentWidget(self.std_stitch_tab)
        QMessageBox.information(self, "Replay complete",
            f"All {len(blend_windows)} stitch step(s) replayed successfully.\n"
            "Review the result and press 'Save All' to export.")

    # ── Pipeline JSON replay ─────────────────────────────────────────────────
    def _pipeline_replay_from_json(self, json_path: str):
        """
        Load an analysis JSON and re-apply all recorded steps into the
        Power-by-Power Pipeline (pipeline-specific tabs and dark dict).

        Steps mirror _replay_from_json but target pipeline_ instances.
        """
        try:
            meta = json.loads(Path(json_path).read_text())
        except Exception as exc:
            QMessageBox.critical(self, "Replay — load error",
                f"Could not read JSON:\n{exc}")
            return

        # ── 1. Validate files ────────────────────────────────────────────────
        missing: list[str] = []

        if self.calib_tab.halogen_df is None:
            missing.append("• Halogen lamp reference not loaded")

        # Validate by filename (path-independent — works across computers).
        expected_pl_fnames: set = set()
        for grp in meta.get("pl_dark_scaling", {}).values():
            expected_pl_fnames.update(grp.get("dark_scale_by_file", {}).keys())
        loaded_pl_fnames = {pf.metadata["filename"] for pf in self.calib_tab.pl_files}
        for fname in sorted(expected_pl_fnames - loaded_pl_fnames):
            missing.append(f"• PL file not loaded: {fname}")

        expected_wl_fnames: set = set()
        for grp in meta.get("white_dark_scaling", {}).values():
            expected_wl_fnames.update(grp.get("dark_scale_by_file", {}).keys())
        loaded_wl_fnames = {
            pf.metadata["filename"]
            for td in self.calib_tab.white_dict.values()
            for pf in td.values()
        }
        for fname in sorted(expected_wl_fnames - loaded_wl_fnames):
            missing.append(f"• White file not loaded: {fname}")

        dark_dict = self.calib_tab.checked_dark_dict()
        for ce_str in meta.get("pl_dark_scaling", {}):
            ce = float(ce_str)
            if ce not in dark_dict:
                missing.append(
                    f"• No dark file loaded for Center_E = {ce:.4f} eV (needed for PL)")
        for ce_str in meta.get("white_dark_scaling", {}):
            ce = float(ce_str)
            if ce not in dark_dict:
                missing.append(
                    f"• No dark file loaded for Center_E = {ce:.4f} eV (needed for white)")

        if missing:
            QMessageBox.warning(
                self, "Replay — missing files",
                "Cannot replay: the following files or calibration data are missing.\n"
                "Please load them in the Load Files tab first.\n\n"
                + "\n".join(missing)
            )
            return

        # ── 2. Apply PL dark-scaling factors (into pipeline dark dict) ────────
        for ce_str, grp in meta.get("pl_dark_scaling", {}).items():
            ce = float(ce_str)
            for fname, ds in grp.get("dark_scale_by_file", {}).items():
                self.pipeline_pl_dark_tab.dark_scale_dict[fname] = float(ds)
            win = grp.get("edge_window_eV")
            if win:
                self.pipeline_pl_dark_tab._scaling_windows[ce] = tuple(win)

        pl_done_ces = set(float(ce) for ce in meta.get("pl_dark_scaling", {}))
        self.pipeline_pl_dark_tab.set_done_groups(pl_done_ces)
        self.pipeline_pl_dark_tab.refresh_if_needed()
        self.calib_tab._save_session()
        self.pipeline_corrections_tab.invalidate_dark_sub()
        self.pipeline_pl_tab.invalidate()

        for ce_str, grp in meta.get("white_dark_scaling", {}).items():
            for fname, ds in grp.get("dark_scale_by_file", {}).items():
                self.pipeline_wl_dark_tab.dark_scale_dict[fname] = float(ds)

        # ── 3. Restore white confirmation state ───────────────────────────────
        wl_done_ces = set(float(ce) for ce in meta.get("white_dark_scaling", {}))
        self.pipeline_wl_dark_tab.set_done_groups(wl_done_ces)
        self.pipeline_wl_dark_tab.refresh_if_needed()
        self.calib_tab._save_session()
        self.pipeline_corrections_tab.invalidate_correction()
        self.pipeline_pl_tab.invalidate()

        # ── 4. Build correction ratios ───────────────────────────────────────
        self.pipeline_corrections_tab._build_correction()
        if not self.pipeline_corrections_tab.correction_dict:
            QMessageBox.critical(self, "Replay — correction failed",
                "Could not build correction ratios. Check that halogen, white, "
                "and dark files are correctly loaded and checked.")
            return

        # ── 5. Apply dark subtraction + whitelight correction ────────────────
        self.pipeline_corrections_tab._apply_dark_sub()
        if not self.pipeline_corrections_tab.normalized:
            QMessageBox.critical(self, "Replay — dark subtraction failed",
                "Dark subtraction produced no results.")
            return

        self.pipeline_pl_tab.refresh_table()
        self.pipeline_pl_tab.pl_table.check_all(True)
        self.pipeline_pl_tab._apply_checked()
        if not self.pipeline_pl_tab.corrected:
            QMessageBox.critical(self, "Replay — correction failed",
                "Whitelight correction produced no results.")
            return

        # Mark correction as applied so phase ③ shows green and state persists.
        self.pipeline_tab._correction_applied = True
        if self.pipeline_tab.on_correction_applied:
            self.pipeline_tab.on_correction_applied()   # saves session

        # ── 6. Stitch with recorded blend windows ─────────────────────────────
        # Prefer per-power blend logs (power-mode JSON); fall back to flat list.
        power_groups_data = meta.get("stitching", {}).get("power_groups", {})
        blend_logs: dict = {}
        for power_str, grp in power_groups_data.items():
            bw = grp.get("blend_windows", [])
            if bw:
                blend_logs[float(power_str)] = bw

        if blend_logs:
            # Power-mode JSON — restore each power individually
            self.pipeline_stitch_tab.restore_power_mode(blend_logs)
            n_restored = sum(1 for v in self.pipeline_stitch_tab._power_done.values() if v)
            n_total    = len(self.pipeline_stitch_tab._power_order)
            if n_restored == 0:
                QMessageBox.warning(self, "Replay — stitching skipped",
                    "Correction applied, but no stitching steps could be replayed.\n"
                    "Power keys in the JSON may not match the loaded files.")
                self.tabs.setCurrentWidget(self.pipeline_tab)
                return
            n_steps = sum(len(v) for v in blend_logs.values())
            QMessageBox.information(self, "Replay complete",
                f"{n_restored}/{n_total} power(s) re-stitched "
                f"({n_steps} blend step(s) total).\n"
                "Review the result and press 'Save All' to export.")
        else:
            # Flat blend-window JSON (e.g. from standard non-power stitch)
            flat_bw = meta.get("stitching", {}).get("blend_windows", [])
            if not flat_bw:
                QMessageBox.information(self, "Replay — done (no stitching)",
                    "Scaling and correction applied successfully.\n"
                    "No stitching data found in the JSON — skipping stitch step.")
                self.tabs.setCurrentWidget(self.pipeline_tab)
                return
            self.pipeline_stitch_tab._start()
            if not self.pipeline_stitch_tab.power_groups:
                QMessageBox.warning(self, "Replay — stitching skipped",
                    "Correction applied, but stitching could not start "
                    "(no corrected groups with ≥ 2 windows).")
                self.tabs.setCurrentWidget(self.pipeline_tab)
                return
            stitch_errors = []
            for entry in sorted(flat_bw, key=lambda e: e["step"]):
                x_min, x_max = entry["window_eV"]
                self.pipeline_stitch_tab._selected_range = (x_min, x_max)
                try:
                    self.pipeline_stitch_tab._do_stitch()
                except Exception as exc:
                    stitch_errors.append(f"Step {entry['step']}: {exc}")
                    break
            if stitch_errors:
                QMessageBox.warning(self, "Replay — stitch warnings",
                    "\n".join(stitch_errors))
            else:
                QMessageBox.information(self, "Replay complete",
                    f"All {len(flat_bw)} stitch step(s) replayed successfully.\n"
                    "Review the result and press 'Save All' to export.")

        # ── 7. Update pipeline phase indicators ──────────────────────────────
        # Build the list of unique power levels from loaded PL files.
        pl_files = self.calib_tab.checked_pl_files()
        powers_list: list = []
        for pf in pl_files:
            p = pf.metadata.get("Exc_P")
            if p is not None and not any(_powers_match(p, q) for q in powers_list):
                powers_list.append(p)
        powers_list.sort()

        # Populate _pip_ce_done so get_pl_dark_done_for_power_fn returns True.
        # All CEs present in pl_dark_scaling were scaled; mark them done for
        # every power so that refresh_status() correctly sets _pl_done to True.
        scaled_ces = {float(ce_str) for ce_str in meta.get("pl_dark_scaling", {})}
        if powers_list and scaled_ces:
            self.pipeline_tab._pip_ce_done = {
                p: {ce: True for ce in scaled_ces} for p in powers_list
            }

        # Now refresh_status can evaluate _pl_done correctly.
        self.pipeline_tab.refresh_status()
        # Save final state so the replay is fully persisted.
        self.calib_tab._save_session()

        # ── 8. Navigate pipeline to stitch page ──────────────────────────────
        self.tabs.setCurrentWidget(self.pipeline_tab)
        self.pipeline_tab._goto_stitch()


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Power Series Plot
# ════════════════════════════════════════════════════════════════════════════
_POWER_SERIES_COLORS = (
    "#000000", "#ff0000", "#00ff00", "#0000ff",
    "#00ffff", "#ff00ff", "#ffff00", "#808000",
    "#000080", "#800080", "#800000", "#008000",
)


class PowerSeriesPlotTab(QWidget):
    """
    Displays all stitched power-series spectra overlaid on a single axes,
    reproducing the style from the plotter_nice.ipynb notebook.

    The user can tweak xlim, ylim, title, and an optional annotation text,
    then save the figure as PNG and/or SVG at 300 dpi.
    """

    def __init__(self, get_stitched_results, get_save_dir=None, set_save_dir=None):
        super().__init__()
        self.get_stitched_results = get_stitched_results  # () -> dict {power: df}
        self.get_save_dir         = get_save_dir or (lambda: "")
        self.set_save_dir         = set_save_dir or (lambda _: None)

        # ── Figure ────────────────────────────────────────────────────────
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = QGroupBox("⑦  Power Series Plot  —  adjust settings and press 'Plot'")
        cg   = QVBoxLayout(ctrl)

        # Row 1: xlim / ylim
        lim_row = QHBoxLayout()
        lim_row.addWidget(QLabel("x min:"))
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-10, 10); self.xmin_spin.setDecimals(3)
        self.xmin_spin.setSingleStep(0.05); self.xmin_spin.setValue(1.05)
        lim_row.addWidget(self.xmin_spin)

        lim_row.addWidget(QLabel("x max:"))
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(-10, 10); self.xmax_spin.setDecimals(3)
        self.xmax_spin.setSingleStep(0.05); self.xmax_spin.setValue(1.42)
        lim_row.addWidget(self.xmax_spin)

        lim_row.addSpacing(16)
        lim_row.addWidget(QLabel("y min:"))
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(1e-6, 1e12); self.ymin_spin.setDecimals(2)
        self.ymin_spin.setSingleStep(1.0); self.ymin_spin.setValue(1.0)
        lim_row.addWidget(self.ymin_spin)

        lim_row.addWidget(QLabel("y max:"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(1e-6, 1e12); self.ymax_spin.setDecimals(0)
        self.ymax_spin.setSingleStep(1000); self.ymax_spin.setValue(10000)
        lim_row.addWidget(self.ymax_spin)
        lim_row.addStretch()
        cg.addLayout(lim_row)

        # Row 2: title
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Power series — sample name")
        title_row.addWidget(self.title_edit, 1)
        cg.addLayout(title_row)

        # Row 3: annotation text + x/y position
        ann_row = QHBoxLayout()
        self.ann_chk = QCheckBox("Show annotation")
        self.ann_chk.setChecked(True)
        ann_row.addWidget(self.ann_chk)

        self.ann_edit = QTextEdit()
        self.ann_edit.setPlaceholderText(
            r"$\mathbf{T_{Lattice}}$= 10K" + "\n" + " Excitation $\\lambda$ = 740nm"
        )
        self.ann_edit.setPlainText(
            "$\\mathbf{T_{Lattice}}$= 10K\n Excitation $\\lambda$ = 740nm"
        )
        self.ann_edit.setMaximumHeight(56)
        self.ann_edit.setMaximumWidth(300)
        ann_row.addWidget(self.ann_edit)

        ann_row.addSpacing(8)
        ann_row.addWidget(QLabel("x:"))
        self.ann_x_spin = QDoubleSpinBox()
        self.ann_x_spin.setRange(-10, 10); self.ann_x_spin.setDecimals(3)
        self.ann_x_spin.setSingleStep(0.01)
        self.ann_x_spin.setValue(
            (self.xmin_spin.value() + self.xmax_spin.value()) / 2
        )
        ann_row.addWidget(self.ann_x_spin)

        ann_row.addWidget(QLabel("y:"))
        self.ann_y_spin = QDoubleSpinBox()
        self.ann_y_spin.setRange(1e-6, 1e12); self.ann_y_spin.setDecimals(2)
        self.ann_y_spin.setSingleStep(1.0); self.ann_y_spin.setValue(1.5)
        ann_row.addWidget(self.ann_y_spin)

        ann_row.addStretch()
        cg.addLayout(ann_row)

        # Row 4: format checkboxes + save button + plot button
        btn_row = QHBoxLayout()
        self.plot_btn  = QPushButton("Plot")
        btn_row.addWidget(self.plot_btn)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("Save formats:"))
        self.chk_png = QCheckBox("PNG")
        self.chk_svg = QCheckBox("SVG")
        self.chk_png.setChecked(True)
        self.chk_svg.setChecked(True)
        btn_row.addWidget(self.chk_png)
        btn_row.addWidget(self.chk_svg)
        self.save_btn = QPushButton("Save Figure")
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: grey;")
        btn_row.addWidget(self.status_lbl)
        cg.addLayout(btn_row)

        # ── Layout ────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.addWidget(ctrl)
        layout.addWidget(self.canvas, 1)

        # ── Signals ───────────────────────────────────────────────────────
        self.plot_btn.clicked.connect(self._plot)
        self.save_btn.clicked.connect(self._save)

    # ── Internal helpers ──────────────────────────────────────────────────
    def _build_figure(self):
        """Render the power series onto self.figure; returns True on success."""
        stitched = self.get_stitched_results()
        if not stitched:
            QMessageBox.warning(self, "No data",
                "No stitched results found.\n"
                "Run stitching in the '⑥ Stitch & Export' tab first.")
            return False

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for idx, (power, df) in enumerate(sorted(stitched.items())):
            color = _POWER_SERIES_COLORS[idx % len(_POWER_SERIES_COLORS)]
            x = df["Energy"].to_numpy(float)
            y = df["Counts"].to_numpy(float)
            ax.plot(x, y, label=f"{power:.2f} mW",
                    linewidth=2, color=color)

        ax.set_yscale("log")
        ax.set_xlim(self.xmin_spin.value(), self.xmax_spin.value())
        ax.set_ylim(self.ymin_spin.value(), self.ymax_spin.value())

        ax.set_ylabel("PL Intensitdy (a.u.)", fontweight="bold")
        ax.set_xlabel("Energy (eV)", fontweight="bold")

        title_text = self.title_edit.text().strip()
        if title_text:
            ax.set_title(title_text, fontsize=16, fontweight="bold")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize="small",
                  loc="center left", bbox_to_anchor=(1, 0.5))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

        if self.ann_chk.isChecked():
            ann_text = self.ann_edit.toPlainText()
            if ann_text.strip():
                ax.text(self.ann_x_spin.value(), self.ann_y_spin.value(),
                        ann_text, fontweight="bold")

        self.figure.tight_layout()
        self.canvas.draw()
        return True

    def _plot(self):
        ok = self._build_figure()
        self.save_btn.setEnabled(ok)
        if ok:
            n = len(self.get_stitched_results())
            self.status_lbl.setText(f"{n} group(s) plotted")

    def _save(self):
        save_png = self.chk_png.isChecked()
        save_svg = self.chk_svg.isChecked()
        if not save_png and not save_svg:
            QMessageBox.warning(self, "No format selected",
                "Select at least one format (PNG or SVG) before saving.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder",
            self.get_save_dir() or str(Path.cwd())
        )
        if not out_dir:
            return
        self.set_save_dir(out_dir)

        # Re-render fresh so the saved file matches what's on screen
        ok = self._build_figure()
        if not ok:
            return

        out = Path(out_dir)
        stem = "power_series"
        errors, saved = [], []
        for fmt, enabled in [("png", save_png), ("svg", save_svg)]:
            if not enabled:
                continue
            path = out / f"{stem}.{fmt}"
            try:
                self.figure.savefig(str(path), dpi=300, bbox_inches="tight")
                saved.append(path.name)
            except Exception as exc:
                errors.append(f"{fmt.upper()}: {exc}")

        msg = f"Saved {len(saved)} file(s) → {out}"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        QMessageBox.information(self, "Saved", msg)
        self.status_lbl.setText(f"Saved → {', '.join(saved)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
