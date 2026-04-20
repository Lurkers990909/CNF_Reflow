# Additional Testing Notes

This document provides a quick overview of the three main experiment pipelines currently used in this repository for comparison with the paper:

- `train_cnf.py`: trains the teacher FFJORD/CNF model
- `train_rf_pipeline.py`: trains and evaluates `Teacher / RF / RF Student`
- `train_fort.py`: trains and evaluates the `FORT student`

The following data types have been verified to work:

- image: `mnist`, `svhn`, `cifar10`
- tabular: `power`, `gas`, `hepmass`, `miniboone`, `bsds300`
- peptide: `aldp`, `al3`, `al4`

Notes:

- `aldp` = alanine dipeptide
- `al3` = alanine tripeptide
- `al4` = alanine tetrapeptide
- The code still supports the legacy name `tetra`, but it is internally normalized to `al4`

## 1. Recommended Experiment Workflow

For paper-level comparisons, the recommended workflow is:

1. Train a teacher checkpoint with `train_cnf.py`
2. Run `Teacher / RF / RF Student` with `train_rf_pipeline.py`
3. Run `FORT` with `train_fort.py`
4. Compare `NLL`, sampling time, and for peptide datasets also `ESS / E-W1 / T-W2`

## 2. Data Preparation

### 2.1 Tabular

Tabular datasets follow the existing loader conventions in the repository and are usually placed under `data/`. Common directory examples include:

- `data/miniboone/data.npy`
- `data/power/data.npy`
- `data/gas/ethylene_CO.pickle`
- `data/hepmass/...`
- `data/BSDS300/BSDS300.hdf5`

### 2.2 Peptide: ALDP / AL3 / AL4

The current peptide download script supports the official dataset names `aldp`, `al3`, and `al4`:

```powershell
.\.venv\Scripts\python.exe scripts\download_fort_peptides.py --datasets aldp al3 al4 --data-root data
```

To force overwrite local files:

```powershell
.\.venv\Scripts\python.exe scripts\download_fort_peptides.py --datasets al3 al4 --data-root data --force
```

After preparation, the directory structure will typically look like this:

```text
data/
  aldp/
    train.npy
    val.npy
    test.npy
    topology.pdb
    metadata.json
    metrics_cache/
  al3/
    train.npy
    val.npy
    test.npy
    topology.pdb
    metadata.json
    metrics_cache/
  al4/
    train.npy
    val.npy
    test.npy
    topology.pdb
    metadata.json
    metrics_cache/
```

The corresponding dataset names in the public mirror are:

- `aldp` -> `Ace-A-Nme_300K`
- `al3` -> `AAA_310K`
- `al4` -> `Ace-AAA-Nme_300K`

`metadata.json` stores:

- `num_atoms`
- `torsion_atom_indices`
- topology information
- OpenMM force field configuration
- split information

## 3. Dependencies

It is recommended to use the project virtual environment directly:

```powershell
.\.venv\Scripts\python.exe --version
```

Common dependencies used in the current workflow:

- General: `torch`, `torchdiffeq`
- Tabular: `pandas`, `h5py`
- Peptide: `openmm`, `pot`

## 4. Train the Teacher First

### 4.1 Tabular teacher

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data miniboone `
  --data-type tabular `
  --conv False `
  --dims 100,100 `
  --save experiments\tabular_compare\miniboone\teacher_ffjord
```

### 4.2 ALDP teacher

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data aldp `
  --data-type peptide `
  --conv False `
  --dims 100,100 `
  --save experiments\aldp_teacher
```

### 4.3 AL3 teacher

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data al3 `
  --data-type peptide `
  --conv False `
  --dims 100,100 `
  --save experiments\al3_teacher
```

### 4.4 AL4 teacher

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data al4 `
  --data-type peptide `
  --conv False `
  --dims 100,100 `
  --save experiments\al4_teacher
```

After that:

- `--cnf-path` in `train_rf_pipeline.py`
- `--reflow_model` in `train_fort.py`

should both point to the generated `checkpt.pth`.

## 5. `train_rf_pipeline.py`

### 5.1 Tabular example

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\tabular_compare\miniboone\teacher_ffjord\checkpt.pth `
  --data miniboone `
  --data-type tabular `
  --eval-cnf `
  --eval-nll `
  --rf-model-path experiments\tabular_compare\miniboone\rf_pipeline\miniboone_rf_model_final.pth `
  --rf-ckpt-path experiments\tabular_compare\miniboone\rf_pipeline\miniboone_rf_ckpt.pth `
  --student-model-path experiments\tabular_compare\miniboone\rf_pipeline\miniboone_student_model_final.pth `
  --student-ckpt-path experiments\tabular_compare\miniboone\rf_pipeline\miniboone_student_ckpt.pth
```

### 5.2 Official ALDP command

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\aldp_teacher\checkpt.pth `
  --data aldp `
  --data-type peptide `
  --eval-cnf `
  --eval-nll `
  --max-train-samples 100000 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --rf-model-path experiments\aldp_rf\aldp_rf_model_final.pth `
  --rf-ckpt-path experiments\aldp_rf\aldp_rf_ckpt.pth `
  --student-model-path experiments\aldp_rf\aldp_student_model_final.pth `
  --student-ckpt-path experiments\aldp_rf\aldp_student_ckpt.pth
```

### 5.3 Official AL3 command

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\al3_teacher\checkpt.pth `
  --data al3 `
  --data-type peptide `
  --eval-cnf `
  --eval-nll `
  --max-train-samples 100000 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --rf-model-path experiments\al3_rf\al3_rf_model_final.pth `
  --rf-ckpt-path experiments\al3_rf\al3_rf_ckpt.pth `
  --student-model-path experiments\al3_rf\al3_student_model_final.pth `
  --student-ckpt-path experiments\al3_rf\al3_student_ckpt.pth
```

### 5.4 Official AL4 command

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\al4_teacher\checkpt.pth `
  --data al4 `
  --data-type peptide `
  --eval-cnf `
  --eval-nll `
  --max-train-samples 100000 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --rf-model-path experiments\al4_rf\al4_rf_model_final.pth `
  --rf-ckpt-path experiments\al4_rf\al4_rf_ckpt.pth `
  --student-model-path experiments\al4_rf\al4_student_model_final.pth `
  --student-ckpt-path experiments\al4_rf\al4_student_ckpt.pth
```

### 5.5 ALDP quick validation command

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\cnf_smoke_aldp\checkpt.pth `
  --data aldp `
  --data-type peptide `
  --eval-cnf `
  --rf-epochs 1 `
  --student-epochs 1 `
  --student-batch-size 8 `
  --max-train-samples 32 `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --n-samples 8 `
  --openmm-platform cpu `
  --rf-model-path experiments\usability_aldp_rf\aldp_rf_model_final.pth `
  --rf-ckpt-path experiments\usability_aldp_rf\aldp_rf_ckpt.pth `
  --student-model-path experiments\usability_aldp_rf\aldp_student_model_final.pth `
  --student-ckpt-path experiments\usability_aldp_rf\aldp_student_ckpt.pth
```

### 5.6 AL4 quick validation command

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\smoke_al4_cnf_20260331\checkpt.pth `
  --data al4 `
  --data-type peptide `
  --data-root tmp\al4_smoke_data `
  --eval-cnf `
  --rf-epochs 1 `
  --student-epochs 1 `
  --student-batch-size 8 `
  --max-train-samples 32 `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --n-samples 8 `
  --openmm-platform cpu `
  --rf-model-path experiments\quickcheck_tetra_rf_20260330\tetra_rf_model_final.pth `
  --rf-ckpt-path experiments\quickcheck_tetra_rf_20260330\tetra_rf_ckpt.pth `
  --student-model-path experiments\quickcheck_tetra_rf_20260330\tetra_student_model_final.pth `
  --student-ckpt-path experiments\quickcheck_tetra_rf_20260330\tetra_student_ckpt.pth `
  --eval-only
```

Notes:

- `tmp\al4_smoke_data` is a locally trimmed smoke dataset and should only be used for quick validation
- The current `--eval-only` branch in `train_rf_pipeline.py` requires existing RF/Student weights
- If `--eval-only` is provided without valid `rf_model_path` / `student_model_path`, the main flow falls back to the training branch and fails because `train_loader` is missing

## 6. `train_fort.py`

### 6.1 Tabular example

For tabular experiments, the current recommended fixed settings are:

- `--flow realnvp`
- `--target reflow`

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data miniboone `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\tabular_compare\miniboone\teacher_ffjord\checkpt.pth `
  --save experiments\tabular_compare\miniboone\fort_realnvp
```

### 6.2 ALDP example

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data aldp `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\aldp_teacher\checkpt.pth `
  --tabular-depth 10 `
  --tabular-hidden-dims 100-100 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --save experiments\aldp_fort
```

Quick validation command:

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data aldp `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\cnf_smoke_aldp\checkpt.pth `
  --num_epochs 1 `
  --batch_size 16 `
  --test_batch_size 64 `
  --tabular-depth 1 `
  --tabular-hidden-dims 16-16 `
  --val_freq 1 `
  --log_freq 1 `
  --early_stop 0 `
  --max_train_samples 32 `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --openmm-platform cpu `
  --save experiments\usability_aldp_fort_quick
```

### 6.3 AL3 example

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data al3 `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\al3_teacher\checkpt.pth `
  --tabular-depth 10 `
  --tabular-hidden-dims 100-100 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --save experiments\al3_fort
```

### 6.4 AL4 example

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data al4 `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\al4_teacher\checkpt.pth `
  --tabular-depth 10 `
  --tabular-hidden-dims 100-100 `
  --metric-samples 250000 `
  --tw2-subsample 4096 `
  --openmm-platform cuda `
  --save experiments\al4_fort
```

### 6.5 AL4 quick validation command

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data al4 `
  --data-root tmp\al4_smoke_data `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\smoke_al4_cnf_20260331\checkpt.pth `
  --num_epochs 1 `
  --batch_size 16 `
  --test_batch_size 16 `
  --tabular-depth 1 `
  --tabular-hidden-dims 16-16 `
  --val_freq 1 `
  --log_freq 1 `
  --early_stop 0 `
  --max_train_samples 32 `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --openmm-platform cpu `
  --save experiments\smoke_al4_fort_20260331
```

## 7. Key Argument Explanations

### 7.1 `train_cnf.py`

| Argument | Meaning |
| --- | --- |
| `--data` | dataset name |
| `--data-type` | data type, commonly `image / tabular / peptide` |
| `--conv False` | convolutional models must be disabled for vector data; this is required for both tabular and peptide teacher training |
| `--dims` | CNF hidden layer widths, e.g. `100,100` |
| `--max-train-samples` | maximum number of training samples for vector data; useful for debugging or small smoke tests |
| `--save` | output directory for the teacher model |

### 7.2 `train_rf_pipeline.py`

| Argument | Meaning |
| --- | --- |
| `--cnf-path` | path to the teacher checkpoint |
| `--data` | dataset name |
| `--data-type` | `auto / image / tabular / peptide` |
| `--data-root` | dataset root directory, default is `data/` |
| `--eval-cnf` | also evaluate the teacher model |
| `--eval-only` | evaluation only; do not train RF or Student |
| `--eval-nll` | additionally compute test NLL; for image data this also reports BPD |
| `--rf-epochs` | number of RF training epochs |
| `--student-epochs` | number of Student training epochs |
| `--max-train-samples` | maximum number of training samples for vector data; very useful for debugging |
| `--metric-samples` | number of proposal samples for peptide metrics; larger values are more stable |
| `--tw2-subsample` | subsample size used for peptide `T-W2` |
| `--openmm-platform` | `auto / cpu / cuda / opencl / reference` |

### 7.3 `train_fort.py`

| Argument | Meaning |
| --- | --- |
| `--data` | dataset name |
| `--data-root` | dataset root directory, default is `data/` |
| `--flow` | currently `realnvp` is recommended for both tabular and peptide experiments |
| `--target reflow` | use the teacher model to generate targets; this is currently the recommended setting |
| `--reflow_model` | path to the teacher checkpoint |
| `--tabular-depth` | number of coupling layers; despite the name, peptide also reuses this vector student architecture |
| `--tabular-hidden-dims` | hidden dimensions of the vector student, e.g. `100-100` |
| `--max_train_samples` | maximum number of training samples for vector data |
| `--metric-samples` | number of proposal samples used in final peptide evaluation |
| `--tw2-subsample` | subsample size for peptide `T-W2` |
| `--openmm-platform` | OpenMM platform used for peptide energy evaluation |
| `--save` | output directory |

## 8. Output Description

### 8.1 `train_cnf.py`

Typical outputs:

- `checkpt.pth`: best checkpoint
- `latest.pth`: most recent checkpoint
- `logs_*.txt`: training logs
- `figs/` for image mode

### 8.2 `train_rf_pipeline.py`

Typical outputs:

- `*_rf_model_final.pth`
- `*_rf_ckpt.pth`
- `*_student_model_final.pth`
- `*_student_ckpt.pth`
- evaluation logs
- sample grids in image mode
- energy histograms and Ramachandran plots in peptide mode

Note: in the current peptide pipeline, some visualizations are written directly to the repository root, for example:

- `al4_cnf_teacher_energy_hist.png`
- `al4_rf_ramachandran.png`

### 8.3 `train_fort.py`

Typical outputs:

- `checkpt.pth`
- `latest.pth`
- `logs_*.txt`
- `test_metrics.json`
- `figs/student_final_energy_hist.png`
- `figs/student_final_ramachandran.png`

## 9. What the Metrics Mean

### 9.1 General metrics

- `NLL`: negative log-likelihood, lower is better
- `Time/Sample`: average generation time per sample, lower is faster
- `ODE Steps`: number of ODE solver steps, usually fewer means faster

### 9.2 Image metrics

- `BPD`: bits per dimension, a standard metric for image density models; lower is better
- `FID`: sample quality metric; lower is better

### 9.3 Auxiliary tabular metrics

- `Mean L1`: L1 difference between generated and real feature means; lower is better
- `Std L1`: L1 difference between generated and real feature standard deviations; lower is better

### 9.4 Peptide metrics

- `ESS`: effective sample size ratio, higher is better
- `E-W1`: 1D Wasserstein distance between the reweighted energy distribution and the test energy distribution, lower is better
- `T-W2`: approximate torus Wasserstein-2 distance on the torsion-angle distribution, lower is better

These peptide metrics use FORT-style post-processing:

- `log w = -u(x) - log q(x)`
- `ESS` is computed using the Kish formula
- `E-W1` compares energy distributions
- `T-W2` compares distributions in torsion space

## 10. Smoke Tests Conducted in This Round

### 10.1 Re-downloading data

On `2026-03-31`, the following command was executed:

```powershell
.\.venv\Scripts\python.exe scripts\download_fort_peptides.py --datasets al3 al4 --data-root data --force
```

Results:

- `data/al3`: `AAA_310K`, `33` atoms, `4` torsions
- `data/al4`: `Ace-AAA-Nme_300K`, `42` atoms, `6` torsions

### 10.2 Local smoke datasets

For quick validation, two additional small local datasets were created:

- `tmp/al3_smoke_data/al3`
- `tmp/al4_smoke_data/al4`

Each split keeps:

- train = `32`
- val = `8`
- test = `16`

### 10.3 `train_cnf.py` smoke test

Command:

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data al3 `
  --data-type peptide `
  --data-root tmp\al3_smoke_data `
  --conv False `
  --dims 8,8 `
  --num_blocks 1 `
  --num_epochs 1 `
  --batch_size 8 `
  --test_batch_size 16 `
  --max-train-samples 32 `
  --save experiments\smoke_al3_cnf_20260331 `
  --log_freq 1
```

Results:

- completed successfully for 1 epoch
- output directory: `experiments/smoke_al3_cnf_20260331`
- validation NLL: `150.5510`

AL4 smoke command:

```powershell
.\.venv\Scripts\python.exe train_cnf.py `
  --data al4 `
  --data-type peptide `
  --data-root tmp\al4_smoke_data `
  --conv False `
  --dims 8,8 `
  --num_blocks 1 `
  --num_epochs 1 `
  --batch_size 8 `
  --test_batch_size 16 `
  --max-train-samples 32 `
  --save experiments\smoke_al4_cnf_20260331 `
  --log_freq 1
```

Results:

- completed successfully for 1 epoch
- output directory: `experiments/smoke_al4_cnf_20260331`
- validation NLL: `202.8381`

### 10.4 `train_fort.py` smoke test

Command:

```powershell
.\.venv\Scripts\python.exe train_fort.py `
  --data al4 `
  --data-root tmp\al4_smoke_data `
  --flow realnvp `
  --target reflow `
  --reflow_model experiments\smoke_al4_cnf_20260331\checkpt.pth `
  --num_epochs 1 `
  --batch_size 16 `
  --test_batch_size 16 `
  --tabular-depth 1 `
  --tabular-hidden-dims 16-16 `
  --max_train_samples 32 `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --openmm-platform cpu `
  --save experiments\smoke_al4_fort_20260331 `
  --log_freq 1 `
  --early_stop 0
```

Results:

- completed successfully for 1 epoch
- output directory: `experiments/smoke_al4_fort_20260331`
- final test NLL: `166.6971`
- ESS: `1.0`
- E-W1: `5913184.0444`
- T-W2: `16.1458`

### 10.5 `train_rf_pipeline.py` smoke test

Evaluation command:

```powershell
.\.venv\Scripts\python.exe train_rf_pipeline.py `
  --cnf-path experiments\smoke_al4_cnf_20260331\checkpt.pth `
  --data al4 `
  --data-type peptide `
  --data-root tmp\al4_smoke_data `
  --eval-only `
  --eval-cnf `
  --rf-model-path experiments\quickcheck_tetra_rf_20260330\tetra_rf_model_final.pth `
  --student-model-path experiments\quickcheck_tetra_rf_20260330\tetra_student_model_final.pth `
  --metric-samples 8 `
  --tw2-subsample 4 `
  --openmm-platform cpu
```

Results:

- all three evaluation stages for `CNF / RF / RF Student` ran successfully
- figures `al4_cnf_teacher_*`, `al4_rf_*`, and `al4_rf_student_*` were generated

Additional notes:

- This `train_rf_pipeline.py` run was successfully executed under the `al4` name
- The old checkpoint filenames still use the `tetra_*` prefix, but loading is unaffected
- The current `--eval-only` branch still has a control-flow issue: if RF/Student weights are not already available, the code attempts to fall back to the training branch and then fails

## 11. Verified Working Pipelines

The following pipelines have been verified in practice:

- `train_cnf.py -> miniboone`
- `train_cnf.py -> aldp`
- `train_cnf.py -> al3`
- `train_cnf.py -> al4`
- `train_cnf.py teacher -> train_rf_pipeline.py -> miniboone`
- `train_cnf.py teacher -> train_rf_pipeline.py -> aldp`
- `train_cnf.py teacher -> train_rf_pipeline.py -> al4`
- `train_cnf.py teacher -> train_fort.py -> miniboone`
- `train_cnf.py teacher -> train_fort.py -> aldp`
- `train_cnf.py teacher -> train_fort.py -> al4`
- the original image pipeline `train_cnf.py -> mnist` has also been smoke-tested and is not affected by the added vector-data support

## 12. Common Notes and Caveats

- The teacher model used in both `train_rf_pipeline.py` and `train_fort.py` must come from the `checkpt.pth` produced by `train_cnf.py`
- For tabular and peptide teacher training, always add `--conv False`
- During the first formal peptide evaluation, if `metrics_cache/` is empty, the code will first compute the reference observables, so it may be slow
- If `openmm` or `pot` is not installed correctly, the peptide pipeline cannot be fully evaluated
- `tmp/al3_smoke_data` and `tmp/al4_smoke_data` are only for local smoke tests and should not be used for formal paper results
- `tetra` should now be treated only as a compatibility alias; all new commands should consistently use `al4`

## 13. Suggested Columns for the Paper

If the paper compares the two student routes, it is recommended to keep at least the following columns:

- `Dataset`
- `Teacher NLL`
- `RF Student NLL`
- `FORT Student NLL`
- `Sampling Time`

For peptide datasets, also add:

- `ESS`
- `E-W1`
- `T-W2`

Such a table can simultaneously cover:

- teacher accuracy
- student accuracy
- sampling efficiency
- molecular distribution consistency
