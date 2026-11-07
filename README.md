# Generative Modeling of Cell-State Transitions in Psoriasis

**Shailja Dhanuka, Nikita Rajesh, Mrunmayee Wankhede**
CMU Generative AI in Biomedicine — Course Project

---

## Project Overview

This project applies generative representation learning to longitudinal single-cell RNA-sequencing (scRNA-seq) data from psoriasis patients undergoing treatment. The goal is to learn a low-dimensional latent representation of cells and model how cell populations move through that space across disease and treatment time points, without paired measurements of the same cells across time.

The model is evaluated on three research questions:
- **RQ1**: Do predicted post-treatment cells match the real cell distribution (Wasserstein-2, MMD)?
- **RQ2**: Do predicted cells recover treatment-associated genes and pathways?
- **RQ3**: Do predicted cells move toward a healthier (non-lesional) state?

---

## Dataset

**GEO accession: GSE228421** (Francis et al. 2024, *Nature Communications* 15:913)

- 5 psoriasis patients treated with risankizumab (IL-23 inhibitor)
- Lesional skin biopsies at **Day 0, Day 3, and Day 14** post-treatment; non-lesional skin at Day 0
- Dermis and epidermis mixed 1:1 before sequencing
- All 5 patients were confirmed responders (PASI reduced to ~25–50% of baseline by week 16)

**After QC** (empty droplet filtering, low-quality cell removal, doublet detection via Scrublet, MAD outlier filters):
- 180,754 cells retained across 20 samples (~6,000–12,400 per sample)
- 12 annotated cell types + KC_other (Leiden clustering, 26 clusters)
- Median mitochondrial fraction 3.8–8.5% per sample

**Data integrity note:** GEO's Patient 3 Day-0 files (lesional and non-lesional) are a re-sequencing of Patient 2's Day-0 library (100% barcode overlap, r=1.000 per-cell UMI correlation, pseudobulk r=0.9997). Both P3 Day-0 files are excluded. P3 contributes Day 3 and Day 14 as training data only and has no leave-one-patient-out fold (4 folds, not 5).

**Processed data location (Bridges-2):**
```
/ocean/projects/cis260266p/wankhede/psoriasis/data/processed/
  gse228421_labelled.h5ad   # clean, labelled cells; P3 Day-0 excluded
  splits.csv                # train/start/test role for every cell, every split
  scnode_input.h5ad         # data formatted for scNODE input
  latent/pca_*.npz          # PCA latents per split
  latent/scvi_*.npz         # scVI latents per split
```

---

## Evaluation Splits

| Split | Train on | Predict | Task |
|---|---|---|---|
| `tp_holdout_d3` | Day 0 + Day 14, all patients | Day 3 | Filling a missing time point (main task) |
| `forecast_d14` | Day 0 + Day 3 | Day 14 | Forecasting |
| `lopo_P1/P2/P4/P5` | Other patients, all days | Held-out patient's Day 3 and Day 14 from their Day 0 | New patient generalization |

Primary metric: **PCA-space Wasserstein-2** (lower is better), reported alongside a noise floor (two halves of real test cells compared to each other). Gene-space W2 is secondary as it rewards over-smoothed predictions.

---

## Baseline Results

| Task | Noise floor | "No change" | Best simple method | Gap closed |
|---|---|---|---|---|
| Day-3 holdout | 6.84 | 8.15 | 7.88 (OT interpolation, PCA) | 21% |
| Day-14 forecast | 6.46 | 9.90 | 8.62 (use Day-3 cells) | 37% |
| New patient, Day 3 | 6.37 | 10.18 | 10.13 (average shift, scVI) | 1% |
| New patient, Day 14 | 6.10 | 13.22 | none beats "no change" | 0% |

No simple method predicts a new patient's Day 14 — this is the gap the trained model needs to close.

---

## Modified Files

### `scNODE/optim/loss_func.py`

**What changed:** Two new loss functions added after the original `MSELoss` and `SinkhornLoss`.

**Why:** The original scNODE uses MSE for VAE reconstruction (pre-training) and Sinkhorn/Wasserstein for ODE distribution matching (main training). Both are ablation targets for this psoriasis application.

| Function | Purpose |
|---|---|
| `NegBinLoss(true_obs, mu, theta)` | Negative Binomial NLL as a replacement reconstruction loss. scRNA-seq data is count-based and overdispersed; NB is the generative model used by scVI and is more appropriate than Gaussian/MSE. Uses the lgamma formulation so it also works on log-normalized expression. `theta` is a learned per-gene dispersion parameter. |
| `NonLesionalAlignmentLoss(latent_seq, nonlesional_centroid)` | Auxiliary loss pulling predicted latents at the final time point toward the non-lesional Day-0 centroid in latent space. Directly optimizes for RQ3. Non-lesional cells are held out of training (no data leakage) and used only as a healthy reference direction. |

---

### `scNODE/optim/running.py`

**What changed:** `scNODETrainWithPreTrain` accepts four new optional parameters; all default to the original scNODE behavior so existing calls are unaffected.

**Why:** To run the three ablation experiments without duplicating the training loop.

| Parameter | Default | Ablation values | Effect |
|---|---|---|---|
| `recon_loss` | `"mse"` | `"nb"` | Switches VAE pre-training reconstruction loss from MSE to Negative Binomial NLL; creates a learnable `log_theta` per-gene dispersion and adds it to the pre-training optimizer |
| `blur` | `0.05` | `0.01`, `0.5` | Controls the Sinkhorn smoothing bandwidth in both the OT loss and dynamic regularization. Lower = sharper distribution matching; higher = more smoothing. Previously hardcoded to 0.05. |
| `nonlesional_ref` | `None` | `torch.FloatTensor` of non-lesional Day-0 gene expression | At each training step, encodes the reference cells through the current encoder to get a live centroid, then adds `align_coeff * NonLesionalAlignmentLoss` to the total loss |
| `align_coeff` | `0.1` | tunable | Weight for the non-lesional alignment loss |

The per-iteration loss log is extended from 3 entries to 4: `[total, ot, dynamic_reg, align]`.

---

## Ablation Summary

| Ablation | Parameter to change | Biological motivation |
|---|---|---|
| NB reconstruction loss | `recon_loss="nb"` | scRNA-seq counts are overdispersed; NB is the correct generative model |
| Sinkhorn blur (less smoothing) | `blur=0.01` | Tests whether sharper OT matching improves cell distribution prediction |
| Sinkhorn blur (more smoothing) | `blur=0.5` | Tests whether looser matching prevents overfitting on the small dataset (5 patients) |
| Non-lesional alignment | `nonlesional_ref=<tensor>` | Encodes the biological prior that treated cells should move toward healthy states |

---

## References

1. Francis et al. 2024, *Nature Communications* 15:913
2. scNODE: Jiaqi Zhang, rsinghlab/scNODE
3. scVI: Lopez et al. 2018, *Nature Methods*
4. Optimal-transport analysis of single-cell gene expression — Schiebinger et al. 2019, *Cell*
