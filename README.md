# Project Dialogue-Systems: DAIC-WOZ ADR
This repository deals with automatic depression recognition on the DAIC-WOZ data set. It is a project at *Ulm University*, part of the *Dialogue Systems* Course at the *Institute of Communications Engineering*.

The DAIC-WOZ Data Set was collected by the University of Southern California and can be requested for scientific use [on the university website](https://dcapswoz.ict.usc.edu).

# Goal
The Goal of this project is to use audio data from the DAIC-WOZ data set to construct machine learning models, able to predict depression (Major Depressive Disorder) in human individuals.

# Recreating results
The repository consists of standalone python scripts that are meant to be executed one after another.
In order to flawlessly recreate results, the structure of the input data as well as the order in which to execute the python scripts needs to be followed closely. 
For obvious reasons, the interview data is NOT part of the repository. Only the data processing / machine learning code is. 
Beware that following these results will take up a significant amount of storage on your machine due to the generation of audio chunks as well as features in a multitude of lengths (sum of ~100GB).

## Virtual environment
Before running any code in this repository, make sure to create a virtual environment, activate it and install all requirements. 
### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

## Data structure 
The unprocessed data needs to be placed at the directory root as follows:

```text
data/
├── daic_woz/
│   ├── 300_P/
│   │   └── diarisation_participant/
│   │       ├── part_0_62.328_63.178.wav
│   │       ├── part_1_68.978_70.288.wav
│   │       ├── part_2_75.028_78.128.wav
│   │       └── ...
│   ├── 301_P/
│   │   └── diarisation_participant/
│   │       └── ...
│   ├── meta_info.csv
```

where `meta_info.csv` contains the columns *participant, target, subset* and contains the ground truth labels corresponding to the patients.

## Audio concatenation
Once the data is provided in the structure outlined above, run the following script to generate a long audio file per patient. Make sure the venv is activated.
```bash
python src/preprocessing/audio_concatenator.py
```
The results will be saved to `data/created_data/concatenated_diarisation/300_P/300_P_concatenated.wav` and so on. The `data/created_data` directory will be the target directory for all further created data.

## Unified trainer (single-model pipeline)
Use the unified trainer to train a single classifier on one feature set with a clean, repeatable pipeline.

### Quick start
1. Open `src/machine_learning/unified_trainer.py` and edit the config block near the top:
   - `FEATURE_TYPE` (e.g., `"paper_fused_features"`, `"egemap"`, `"hubert"`, `"paper_covarep"`, `"paper_hosf"`)
   - `CHUNK_LENGTH` and `OVERLAP` (e.g., `"5.0s"` and `"2.5s"`)
   - `MODELS_TO_TRAIN` (choose one: `['svm']`, `['rf']`, `['xgb']`, `['lgb']`, or `['ensemble']`)
   - Optional: `EVALUATE_TEST_SET`, `BALANCED_SAMPLING`, `OVERSAMPLE_MINORITY`, `RANDOM_SEED`
2. Run:
```bash
python src/machine_learning/unified_trainer.py
```
3. Artifacts and results are saved under:
```
data/results/{FEATURE_TYPE}/{FEATURE_TYPE}_{MODEL}_{CHUNK_LENGTH}_{OVERLAP}_overlap_{TIMESTAMP}/
```
This folder contains `model.pkl`, `scaler.pkl`, and `results.txt` (combined dev + test metrics and patient-level summaries).

### Models and key defaults
- **SVM (`'svm'`)**: RBF kernel (`C=0.7`), `class_weight='balanced'`.
- **Random Forest (`'rf'`)**: `n_estimators=500`, `max_depth=10`, `max_features='sqrt'`, `min_samples_split=10`, `min_samples_leaf=5`, class weights `'balanced'` (or disabled if undersampling is used).
- **XGBoost (`'xgb'`)**: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight` auto-computed from class ratio.
- **LightGBM (`'lgb'`)**: Similar tuned defaults with `class_weight` from class ratio.
- **Ensemble (`'ensemble'`)**: Soft-vote of SVM + LightGBM with equal weights.

### Class imbalance handling
- Loader supports patient-level balancing: `BALANCED_SAMPLING` and `OVERSAMPLE_MINORITY` (train-only by default for non-SVM models).
- Per-model:
  - SVM uses `class_weight='balanced'` (no undersampling).
  - RF/XGB can auto-enable undersampling for the majority class; when undersampling is active, class weighting is disabled.
  - LGB uses `class_weight` derived from inverse class frequency.

### Inputs expected by the trainer
Features must exist at `data/features/{FEATURE_TYPE}/{CHUNK_LENGTH}_{OVERLAP}_overlap/{participant}/...csv`.
If you see "Feature directory not found", ensure you generated the corresponding features first.

## Extended evaluator (chunk length sweep + thresholding)
For multi-model, multi-chunk-length experiments with probability-based threshold tuning and patient-level aggregation, use:

```bash
python src/eval/extended_chunk_length_evaluator.py
```
Configure `FEATURE_TYPE`, `MODEL_TYPES`, and (optionally) the list of `CHUNK_LENGTHS` inside the script. Results are written to `data/results/{FEATURE_TYPE}` with per-run directories and CSV summaries.
