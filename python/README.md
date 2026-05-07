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
| ③ | **Standard Analysis** | Manual step-by-step workflow (inner tabs ②–⑦) |

Standard Analysis and the Power-by-Power Pipeline each maintain their **own independent dark-scale dictionaries**, so scaling applied in one mode never affects the other.

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
- **Load & Replay JSON …** — replays a previously exported analysis JSON into the **Standard Analysis** mode (see JSON Replay section below).

**Session persistence:** every change is auto-saved to `PL_Auto_GUI_session.json` next to the script so the loaded files and all scaling state survive an application restart.

---

## Tab ② — Power-by-Power Pipeline  *(main recommended workflow)*

This tab provides a fully guided, self-contained analysis pipeline.  All five phases live inside a stacked widget — the user never needs to navigate away.  A persistent header bar shows phase-navigation buttons (① PL Dark → ② White Dark → ③ Correction → ④ Stitching → ⑤ Power Plot) that light up green as each phase is completed.

### Overview page

Press **▶ Start Pipeline** to begin.  The pipeline automatically skips to the furthest incomplete phase (e.g. jumps straight to Correction if all scaling is already done).  **Reset All** clears all scaling progress and returns to this overview.  **Load & Replay JSON …** replays a previously exported JSON into the pipeline mode.

---

### Phase ① — PL Dark Scaling  *(per-power)*

**Goal:** for each excitation power level, scale the dark spectrum so that it matches the PL signal at spectral edges where no PL emission is expected.

This phase differs from the standard mode's Dark Scaling: files are processed **one power level at a time**.  A row of pill buttons at the top of the page represents every power level found in the loaded PL files.  Pills are colour-coded:
- **Grey** — not yet processed.
- **Blue** — currently selected power.
- **Green** — all `Center_E` groups for this power are done.

**Steps for each power:**

1. The embedded dark-scaling widget shows all PL spectra at the selected power grouped by `Center_E`.  A pill row inside the widget shows each spectral window (`Center_E`) with the same grey/blue/green colour coding.

2. **Drag a region** on the left plot to select a spectral edge window where PL counts ≈ dark counts.

3. Press **Apply to This Group** — for each PL file in the group:
   ```
   dark_scale(file) = mean(PL counts in window) / mean(dark counts in window)
   ```
   `dark_scale` is stored in the pipeline dark-scale dictionary keyed by **filename**.

4. After all `Center_E` groups for the current power are marked done, the tab auto-advances to the next undone power after a short delay.

5. Once all powers are done, a **"② White Dark ▶"** button appears to advance to the next phase.

Additional actions per group:
- **Apply to All Groups** — applies the same edge window to every `Center_E` group at once.
- **No Scaling Needed** — marks a group done without computing a scale (when dark and PL already agree at edges).
- **Reset Scales** — clears the `dark_scale` for the current group and resets its pill to grey.

---

### Phase ② — White Dark Scaling

**Goal:** scale each white-light spectrum so that, after dark subtraction, it is on the same scale as the dark-corrected PL spectra.

Because the dark was already matched to the PL signal in Phase ①, the white spectrum is now matched to the already-shifted dark:

```
white_scale(file) = mean(dark × dark_scale in window) / mean(white counts in window)
```

The subtraction formula for white then becomes:

```
(white × white_scale − dark × dark_scale) / int_time
```

**Steps:** identical interaction model to Phase ① (drag edge window, apply per group or to all), but all spectral windows are processed at once — no per-power separation.  When all `Center_E` groups are done a **"③ Auto-Correct ▶"** button appears.

---

### Phase ③ — Auto-Correction

**Goal:** compute spectral correction coefficients and apply them to all PL files in one click.

Press **Run Correction**.  Internally this calls two steps in sequence:

**Step A — Build correction coefficients.**  For each white-light file at a given `Center_E`:

1. Dark-subtract and normalise the white spectrum:
   ```
   normalised_white(E) = (white(E) × white_scale − dark(E) × dark_scale) / int_time
   ```
2. Interpolate the halogen lamp reference onto the white spectrum's energy grid.
3. Compute:
   ```
   correction_coefficient(E) = halogen(E) / normalised_white(E)
   ```

A **soft inline warning** is shown if fewer than 20 % of points in a window have non-positive normalised white values.  A **blocking dialog** appears if 20 % or more are affected (indicates over-aggressive dark scaling).

**Step B — Apply correction to all checked PL files.**  For each PL file:

```
corrected(E) = [(PL(E) − dark(E) × dark_scale) / int_time] × correction_coefficient(E)
```

After a successful run, the correction coefficients are plotted (log-y) and a **"✓ Correction Done — ④ Stitch ▶"** button appears.

---

### Phase ④ — Stitch  *(power-by-power)*

**Goal:** join corrected sub-spectra from adjacent spectral windows into a single continuous spectrum, independently for each excitation power level.

The stitch tab operates in **power-by-power mode**: a row of power-level pills tracks which powers are stitched.  Each power is stitched through the same number of steps as there are adjacent window pairs.

**Stitching algorithm (one step per adjacent window pair):**

1. The plot shows the current left and right spectra overlaid for all power groups.
2. **Drag a blend window** in the overlap region of the two spectra.
3. Press **Do Stitch**:
   - Compute `ratio = mean(left counts in window) / mean(right counts in window)`.
   - Scale the **right** spectrum by `ratio`.
   - Concatenate: all left-spectrum points up to the blend start, then all (scaled) right-spectrum points from the blend start onward.
4. The same blend window is used for every power group in that step, preserving relative intensities between power levels.
5. After the final step the accumulated stitched spectra are shown overlaid.

Results and blend parameters are saved per-power so that individual powers can be revisited.  After stitching is complete, a **"⑤ Power Series Plot →"** button appears.

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

## Tab ③ — Standard Analysis  *(manual step-by-step)*

This tab wraps the manual analysis workflow in an inner tab widget with six sub-tabs.  It shares the same PL, dark, and white files from Tab ① but uses its own independent dark-scale dictionary.

### Inner tab ② — Dark Scaling (PL)

Identical interaction to Pipeline Phase ① but without per-power separation — all power levels and `Center_E` groups are presented together.

### Inner tab ③ — Dark Scaling (White)

Identical to Pipeline Phase ②.

### Inner tab ④ — Apply Corrections

Two sections:

**A. Dark Subtraction & Normalisation** — Press **Apply Dark Subtraction to All Checked PL Files**.  For each PL file:
```
normalised(E) = (PL(E) − dark(E) × dark_scale) / int_time
```
Result is plotted on a log-y scale grouped by excitation power.

**B. Spectral Correction Coefficients** — Press **Build Correction Ratios** to compute `correction_coefficient(E) = halogen(E) / normalised_white(E)` per `Center_E`, with the same bad-point warnings as Phase ③ of the pipeline.

### Inner tab ⑤ — PL Analysis

Press **Apply Correction to Checked** to apply the spectral correction to all selected PL files.  Two plot modes: **Plot Dark-Subtracted** (normalised, uncorrected) and **Plot Corrected** (fully corrected, grouped by `Center_E`).  A `✓`/`✗` column tracks which files have been corrected.

Press **Stitch Power by Power →** to hand the corrected files to the stitch tab in power-by-power mode, identical to Pipeline Phase ④.

### Inner tab ⑥ — Stitch & Export

**Standard mode** (without power-by-power): files are grouped by excitation power, all groups are stitched with the same blend window applied simultaneously.

**Power-by-power mode** (entered via the button in tab ⑤): each power is stitched independently, accumulating results.

**Save formats:** DAT (tab-separated, Energy / Counts), PNG, PDF, and a JSON metadata file.  DAT, PNG, and PDF go into `dat/`, `png/`, `pdf/` subfolders of the chosen output directory.

**Output filename convention:**
```
{sample_name}_{power:.2f}mW_stitched_{temperature:.1f}K.{ext}
```
The sample name is extracted from the first file's path using the pattern `…/{sample}_NNN.origin`.

**JSON metadata** (`{sample}_PL_analysis_{date}.json`) records every analysis parameter:
- Per-group dark scaling windows and `dark_scale` factors (keyed by filename).
- Per-group white scaling windows and `white_scale` factors.
- Correction coefficient energy ranges and quality statistics.
- Stitching blend windows and per-power scaling ratios.

### Inner tab ⑦ — Power Series Plot

Same controls and output as Pipeline Phase ⑤.

---

## Session Persistence

Every file-loading and scaling action is auto-saved to `PL_Auto_GUI_session.json` in the same folder as the script.  The session JSON has two top-level sections — `"standard"` and `"pipeline"` — so both modes survive independently across restarts.

Each section stores:
- Dark-scale factors per filename.
- White-scale factors per white file.
- Which `Center_E` groups were marked done on the scaling tab.
- Stitch blend logs per power.

The pipeline section additionally stores `pip_ce_done`: `{power: {Center_E: bool}}`, which tracks the done state of each spectral window independently for every excitation power.

On startup, PL files are loaded first (which resets scaling tabs), then all scale values and done-group states are restored so the pills immediately reflect the saved state.

---

## JSON Replay

A previously exported analysis JSON can be loaded to reproduce the exact analysis automatically.

- **Standard mode replay**: **Load & Replay JSON …** in Tab ①.  Validates that all required files are loaded, then applies dark scales, white scales, rebuilds correction coefficients, applies correction to PL files, and runs stitching.  Navigates to the Stitch tab when done.
- **Pipeline mode replay**: **Load & Replay JSON …** on the Pipeline overview page.  Same steps, but all actions target the pipeline-mode tabs and dark-scale dictionary.

If any required file is missing, a detailed list of what is absent is shown before anything is applied.

---

## Summary of the Correction Chain

```
Raw PL counts
    │
    ├─ subtract  dark × dark_scale(file)       (Phase ①/Standard ②: per-file scale)
    ├─ divide by int_time                      (normalise to counts/second)
    ├─ multiply by correction_coefficient(E)   (Phase ③/Standard ④: halogen / norm_white)
    │                                           norm_white uses same dark_scale
    │                                           and white × white_scale (Phase ②/Standard ③)
    └─ stitch adjacent windows                 (Phase ④/Standard ⑥: ratio scaling,
         │                                      left unchanged, right scaled by ratio,
         │                                      then concatenated at the blend point)
         └─ corrected PL spectrum  [a.u., proportional to photon emission rate / eV]
```
