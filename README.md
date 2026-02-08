# Analysis Scripts: Plasma GFAP, Externalizing, and Resting Brain Connectivity

Analysis code for the manuscript:

**Plasma Glial Fibrillary Acidic Protein (GFAP) Shows Age-Dependent Associations with Externalizing Psychopathology and Atypical Brain Connectivity**

---

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_run_3dnetcorr.py` | Compute between-network connectivity (Yeo 17-networks) with AFNI 3dNetCorr; optional parallel runs. |
| 2 | `02_extract_FZ.py` | Extract Fisher-Z connectivity values and build edge-wise datasets. |
| 3 | `03_apply_normative_models.py` | Apply pre-trained Yeo-17 normative models; export deviation (Z) scores and transfer metrics (EXPV, RMSE, Rho, pRho, SMSE, Z). |
| 4 | `04_run_clinical.R` | Clinical models: Gamma GLM (log link) for GFAP and NfL ~ Age×EXT + covariates; interaction plots, Johnson–Neyman, regression tables. |
| 5 | `05_run_sPLS.R` | Sparse PLS: connectivity deviation (Z) ~ GFAP/NfL; cross-validation, bootstrap CIs, multiple-comparison correction (Bonferroni/BH). |
| 6 | `06_run_fMRI.R` | Per-edge lm(connectivity ~ Age×EXT + covariates); bootstrap BCa CIs, partial ω²; heatmap of EXT effect. |

---

## Requirements

- **Python 3:** numpy, pandas, scipy; pcntoolkit for normative modeling (script 03).  
- **R:** ggplot2, ggpubr, interactions, gtsummary (04); spls (05); parameters, effectsize, tidyverse (06).  
- **AFNI:** for 3dNetCorr (01; optional if connectivity matrices are precomputed).

Paths, column names, and options are set at the top of each script; edit as needed for your data paths and variable names.

---

## Normative model (script 03)

Script 03 requires the pre-trained Yeo-17 normative model from pcntoolkit. Download from Dropbox and extract into `models/`:

```bash
cd scripts
wget -O BLR_yeonetworks_22K_45sites.zip "https://www.dropbox.com/sh/6pks34rtt9dg7x7/AAD0B1zyrq9NbQQxdzUcEHAHa?dl=1"
unzip BLR_yeonetworks_22K_45sites.zip -d models/
```

---

## Citation

If you use these scripts, please cite the manuscript:

*Niveditha BS, Holla B, Subramanian S, Gagana N, Bhargavi KM, Sharma E, Mahadevan J, Purushottam M, Viswanath B, Benegal V, Arunachal G, Heron J, Hickman M, Basu D, Subodh BN, Singh L, Singh R, Kumaran K, Kuriyan R, Kurpad SS, Kartik K, Kalyanram K, Desrivieres S, Barker G, Papadopoulos Orfanos D, Toledano M, Murthy P, Vaidya N, Krishnaveni G, Schumann G, Sharma KK, BinuKumar B, Thennarasu K, Kashyap R, Bharath RD, Chakrabarti A, Chetan GK, Srinivas Bharath MM. Plasma Glial Fibrillary Acidic Protein (GFAP) Shows Age-Dependent Associations with Externalizing Psychopathology and Atypical Brain Connectivity.*
