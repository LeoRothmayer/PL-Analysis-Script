# PL_Auto_GUI — Workflow Description
Note that this description is also AI generated and therefore may contain errors.
## Purpose

`PL_Auto_GUI.py` is a graphical tool for processing photoluminescence (PL) spectra recorded with a grating spectrometer.  Each measurement window covers only part of the energy range of interest, so a full spectrum is built from several overlapping sub-spectra ("spectral windows") that must be:

1. **Dark-subtracted** — the detector dark signal is removed.
2. **Normalised** by integration time — so different windows measured with different exposure times are on the same scale.
3. **Spectrally corrected** — the wavelength-dependent throughput of the optics and detector sensitivity is divided out using a white-light (halogen) reference measurement.
4. **Stitched** — the corrected sub-spectra from each window are joined into one continuous spectrum per excitation power level.

The script replaces an earlier manual workflow (`PL_Software.py`) by automatically matching dark and white-light spectra to their corresponding PL measurement using metadata embedded in each file's header (`Center_E` and `int_time`), so no manual ruler-based pairing is needed.

---

## File Format (`.origin`)

All measurement files share the same tab-separated format:

| Rows | Content |
|------|---------|
| 1 | Column headers (`Energy`, `Counts`) |
| 2–12 | Metadata: temperature (K), integration time (s), centre wavelength / energy (nm / eV), excitation power (mW or µW), … |
| 13 | Separator / empty row |
| 14+ | Numeric data: Energy (eV, **descending**), Counts |

The energy axis is always descending as recorded; all internal interpolation steps sort it ascending first before calling `np.interp`.

The halogen lamp reference is a plain two-column tab-separated file (`HalogenLamp_Spectrum.txt`) with wavelength (nm) and counts.

Excitation powers are parsed from the header and converted to mW internally (µW values are divided by 1000).  Two measurement files are considered to belong to the **same power group** when their powers agree within **10 % of the larger value** (adaptive relative tolerance).  This keeps µW-range powers correctly separated while tolerating small measurement-to-measurement drift at mW-range powers.

---

## Top-Level Tab Structure

The main window has three outer tabs:

| Tab | Name | Purpose |
|-----|------|---------|
| ① | **Load Files** | Load all raw data; configure the session |
| ② | **Power-by-Power Pipeline** | Guided single-tab pipeline — recommended workflow |
| ③ | **Automatic Analysis** | Step-by-step workflow with guided progression (inner tabs ①–⑥) |

The Power-by-Power Pipeline and Automatic Analysis each maintain their **own independent dark-scale dictionaries**, so scaling applied in one mode never affects the other.

---

## Tab ① — Load Files

**Goal:** get all raw data files into memory and verify that every PL and white-light measurement has a matching dark spectrum.

**Steps:**

1. **Halogen lamp reference** — loaded once per session (auto-detected next to the script if present).  Converted from wavelength to energy and sorted ascending.

2. **Dark spectra** — one or more `.origin` files measured with the shutter closed, at every `(Center_E, int_time)` combination used during the session.  Loaded in bulk (individual files or an entire folder).  The table shows `Center_E`, `int_time`, and filename; a coloured indicator next to each PL and white file shows whether a matching dark exists.

3. **White-light spectra** — halogen-lamp measurements through the full optical path (but without PL excitation), one per `(Center_E, int_time)`.

4. **PL measurement files** — the actual sample spectra, one file per excitation power × spectral window.

All four sections have "Load Files …", "Load Folder …", and "Clear" buttons.  Checkboxes in every table control which files participate in the analysis — unchecked rows are ignored by all downstream steps.

Additional controls on this tab:
- **New Session** — clears dark and white files from memory; halogen reference is kept.
- **Load & Replay JSON …** — replays a previously exported analysis JSON into the **Automatic Analysis** mode (see JSON Replay section below).

**Session persistence:** every change is auto-saved to `PL_Auto_GUI_session.json` next to the script so the loaded files and all scaling state survive an application restart.

---

## Tab ② — Power-by-Power Pipeline  *(main recommended workflow)*

This tab provides a fully guided, self-contained analysis pipeline.  All five phases live inside a stacked widget — the user never needs to navigate away.  A persistent header bar shows phase-navigation buttons (① PL Dark → ② White Dark → ③ Correction → ④ Stitching → ⑤ Power Plot) that light up **green** as each phase is completed and remain green after a restart.

### Overview page

Press **▶ Start Pipeline** to begin.  The pipeline automatically skips to the furthest incomplete phase — if correction was already run in a previous session it is re-applied silently and the user lands directly on the Correction tab with results shown.  **Reset All** clears all scaling progress and returns to this overview.  **Load & Replay JSON …** replays a previously exported JSON into the pipeline mode.

---

### Phase ① — PL Dark Scaling  *(per-power)*

**Goal:** for each excitation power level, scale the dark spectrum so that it matches the PL signal at spectral edges where no real PL emission is expected.

Files are processed **one power level at a time**.  A row of pill buttons represents every power level; pills are colour-coded:
- **Grey** — not yet processed.
- **Blue** — currently selected power.
- **Green** — all `Center_E` groups for this power are confirmed.

**Steps for each power:**

1. The embedded dark-scaling widget shows all PL spectra at the selected power grouped by `Center_E`.  A pill row shows each spectral window with the same grey/blue/green coding.

2. **Drag a region** on the left plot to select a spectral edge window where PL counts ≈ dark counts.

3. Press **Apply Scaling** — for each PL file in the group:
   ```
   dark_scale(file) = mean(PL counts in window) / mean(dark counts in window)
   ```
   `dark_scale` is stored keyed by **filename**.

4. After all `Center_E` groups for the current power are confirmed, the tab auto-advances to the next undone power after a short delay.

5. Once all powers are done a **"② White Dark ▶"** button appears to advance.

Additional actions per group:
- **No Scaling Needed** — marks the group done without computing a scale.
- **Apply Manual Scale** — enter a scale factor directly; the plot updates immediately but the group stays on screen for verification before confirming.
- **Reset Scales** — clears the `dark_scale` for the current group.

---

### Phase ② — White Dark Scaling

**Goal:** scale each white-light spectrum so that its dark subtraction is consistent with the PL dark subtraction from Phase ①.

The dark is scaled to match the white spectrum at spectral edges:

```
dark_scale(white_file) = mean(white counts in window) / mean(dark counts in window)
```

**Steps:** identical interaction to Phase ① (drag edge window, apply per group), but all spectral windows are processed at once — no per-power separation.  The same manual scale and reset options are available.  When all `Center_E` groups are done a **"③ Auto-Correct ▶"** button appears.

---

### Phase ③ — Auto-Correction

**Goal:** compute spectral correction coefficients and apply them to all PL files in one click, with interactive plot views for verification.

Press **Run Correction**.  Internally:

**Step A — Build correction coefficients.**  For each white-light file at a given `Center_E`:

1. Dark-subtract and normalise the white spectrum:
   ```
   normalised_white(E) = (white(E) − dark(E) × dark_scale) / int_time
   ```
2. Interpolate the halogen lamp reference onto the white spectrum's energy grid.
3. Compute:
   ```
   correction_coefficient(E) = halogen(E) / normalised_white(E)
   ```

A **soft inline warning** is shown if fewer than 20 % of points have non-positive normalised white values.  A **blocking dialog** appears if 20 % or more are affected.

**Step B — Apply correction to all checked PL files.**  For each PL file:
```
corrected(E) = [(PL(E) − dark(E) × dark_scale) / int_time] × correction_coefficient(E)
```

After a successful run the **"✓ Correction Done — ④ Stitch ▶"** button appears.  The correction state is saved to the session so that on the next restart the correction is re-applied automatically and phase ③ immediately shows green.

#### Plot view selector

Four view buttons let the user inspect the data at different processing stages without leaving the correction tab:

| View | Content |
|------|---------|
| **Correction Coefficients** | `halogen / normalised_white` per `Center_E` (log-y) |
| **Raw PL Data** | Unprocessed counts from the original files |
| **Dark-Subtracted & Normalised** | After dark subtraction and integration-time normalisation |
| **Fully Corrected** | After spectral correction — ready for stitching |

In views 2–4, all spectral windows belonging to the same power level share the same colour.

---

### Phase ④ — Stitch  *(power-by-power)*

**Goal:** join corrected sub-spectra from adjacent spectral windows into a single continuous spectrum, independently for each excitation power level.

A row of power-level pills tracks which powers are stitched (green = done).  Navigating back to an already-stitched power shows the **final stitched result directly** rather than restarting the step workflow; the user can still press **↺ Redo This Power** to redo it.

**Stitching algorithm (one step per adjacent window pair):**

1. The plot shows the current left and right spectra overlaid.
2. **Drag a blend window** in the overlap region.
3. Press **Do Stitch**:
   - Compute `ratio = mean(left counts in window) / mean(right counts in window)`.
   - Scale the **right** spectrum by `ratio`.
   - Concatenate: left points up to the blend start, right (scaled) points from the blend start onward.
4. The same blend window is applied across all power groups in that step, preserving relative intensities.
5. After the final step the accumulated stitched spectra are shown overlaid.

After all powers are stitched, press **Save All** to export DAT files and a JSON metadata file.

**Save formats:** DAT (tab-separated, Energy / Counts) and JSON metadata.  DAT files go into a `dat/` subfolder of the chosen output directory.

**Output filename convention:**
```
{sample_name}_{power:.2f}mW_stitched_{temperature:.1f}K.dat
```

---

### Phase ⑤ — Power Series Plot

**Goal:** produce a publication-ready overlay of all stitched power-series spectra.

Controls:
- **x min / x max** — energy axis limits (eV).
- **y min / y max** — intensity axis limits (log scale).
- **Title** — figure title.
- **Annotation** — optional text box (supports LaTeX math), with adjustable x / y position.

The legend lists power levels in **descending order** (highest power at the top, lowest at the bottom).

Save as PNG and/or SVG at 300 dpi.

---

## Tab ③ — Automatic Analysis  *(step-by-step)*

This tab wraps the step-by-step analysis workflow in an inner tab widget with six sub-tabs (① – ⑥).  It shares the same PL, dark, and white files from Tab ① but uses its own independent dark-scale dictionary.

**Guided progression:** when a step is complete, a blue **"▶ Continue to …"** button appears in the upper right.  Completed tab labels turn **green** to show overall progress.

### Inner tab ① — Dark Scaling (PL)

Identical interaction to Pipeline Phase ① but without per-power separation — all power levels and `Center_E` groups are presented together.  A **Next Group** button navigates to the next CE group without applying the current window.

### Inner tab ② — Dark Scaling (White)

Identical to Pipeline Phase ②.

### Inner tab ③ — Apply Corrections

Two sections:

**A. Dark Subtraction & Normalisation** — Press **Apply Dark Subtraction**.  For each PL file:
```
normalised(E) = (PL(E) − dark(E) × dark_scale) / int_time
```
Result is plotted on a log-y scale grouped by excitation power.

**B. Spectral Correction Coefficients** — Press **Build Correction Ratios** to compute `correction_coefficient(E) = halogen(E) / normalised_white(E)` per `Center_E`, with the same bad-point warnings as Pipeline Phase ③.

### Inner tab ④ — PL Analysis

Press **Apply Correction to Checked** to apply the spectral correction to all selected PL files.  Two plot modes: **Plot Dark-Subtracted** and **Plot Corrected**.  A `✓`/`✗` column tracks which files have been corrected.

Press **Stitch Power by Power →** to enter power-by-power stitching mode (identical to Pipeline Phase ④).

### Inner tab ⑤ — Stitch & Export

**Standard mode**: all power groups are stitched simultaneously with the same blend window.

**Power-by-power mode** (entered via the button in tab ④): each power is stitched independently.

**Save formats:** DAT (tab-separated) and JSON metadata.

**JSON metadata** (`{sample}_PL_analysis_{date}.json`) records:
- Per-group dark scaling windows and `dark_scale` factors (keyed by filename).
- Per-group white scaling windows and `white_scale` factors.
- Correction coefficient energy ranges and quality statistics.
- Stitching blend windows and per-power scaling ratios.

### Inner tab ⑥ — Power Series Plot

Same controls and output as Pipeline Phase ⑤.

---

## Session Persistence

Every file-loading and scaling action is auto-saved to `PL_Auto_GUI_session.json` in the same folder as the script.  The session JSON has two top-level sections — `"standard"` and `"pipeline"` — so both modes survive independently across restarts.

Each section stores:
- Dark-scale factors per filename (both PL and white files share the same dict).
- Which `Center_E` groups were marked done on each scaling tab.
- Stitch blend logs per power.

The pipeline section additionally stores:
- `pip_ce_done`: `{power: {Center_E: bool}}` — per-power CE done state for Phase ①.
- `correction_applied`: whether Phase ③ was completed — triggers silent re-application on the next startup so phase ③ shows green immediately.

On startup, PL files are loaded first, then all scale values and done-group states are restored so pills and phase indicators immediately reflect the saved state.

---

## JSON Replay

A previously exported analysis JSON can be loaded to reproduce the exact analysis automatically.

- **Automatic Analysis replay**: **Load & Replay JSON …** in Tab ①.  Validates that all required files are loaded, then applies dark scales, white scales, rebuilds correction coefficients, applies correction to PL files, and runs stitching.
- **Pipeline replay**: **Load & Replay JSON …** on the Pipeline overview page.  Same steps targeting the pipeline-mode tabs.

If any required file is missing, a detailed list is shown before anything is applied.

---

## Summary of the Correction Chain

```
Raw PL counts
    │
    ├─ subtract  dark × dark_scale(file)       (Phase ①/Auto ①: per-file scale)
    ├─ divide by int_time                      (normalise to counts/second)
    ├─ multiply by correction_coefficient(E)   (Phase ③/Auto ③: halogen / norm_white)
    │                                           norm_white uses same dark_scale
    │                                           and white dark_scale (Phase ②/Auto ②)
    └─ stitch adjacent windows                 (Phase ④/Auto ⑤: ratio scaling,
         │                                      left unchanged, right scaled by ratio,
         │                                      concatenated at the blend point)
         └─ corrected PL spectrum  [a.u., proportional to photon emission rate / eV]
```
