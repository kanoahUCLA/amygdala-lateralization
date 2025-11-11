# Amygdala Asymmetry and Trait Anxiety

This project examines how trait anxiety (STAI-T) relates to left–right amygdala activation asymmetry during anxiety-inducing stimuli (e.g., fear > neutral).

## Quick Start
1. **Create and activate environment (optional):**
   ```bash
   conda env create -f environment.yml
   conda activate amygdala-asymmetry
   ```
2. **Place your data:**
   - Preprocessed subject-level contrast maps (z- or t-maps) into `data/fmri/` named like:
     `sub-01_task-emotion_contrast-fear_vs_neutral_zmap.nii.gz`
   - Participant metadata with STAI-T into `data/behavioral/participants.tsv` (BIDS-style).
   - A Harvard-Oxford atlas NIfTI in `data/atlas/` or let the scripts auto-fetch from Nilearn.

3. **Run the pipeline:**
   ```bash
   cd scripts
   python 01_extract_amygdala_activation.py
   python 02_compute_asymmetry_vs_anxiety.py
   python 03_generate_figures.py
   ```

## Outputs
- `results/roi_values.csv` — left/right amygdala activation per subject.
- `results/asymmetry_stats.csv` — per-subject asymmetry + STAI-T.
- `results/correlations.txt` — correlation summary.
- `figures/fig2_left_vs_right_boxplot.png` — ROI comparison by hemisphere.
- `figures/fig3_asymmetry_vs_anxiety_scatter.png` — Asymmetry vs STAI-T.

## Notes
- The scripts default to the Harvard-Oxford cortical+subcortical atlas via Nilearn. If you provide your own atlas, update the `ATLAS_PATH` in the scripts.
- Rename your contrast files or adapt the filename pattern at the top of `01_extract_amygdala_activation.py` to match your dataset.
- Make sure `participants.tsv` contains at least the columns: `participant_id`, `STAI_T` (and optionally `age`, `sex`, `handedness`).
