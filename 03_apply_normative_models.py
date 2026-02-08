# -*- coding: utf-8 -*-
"""
Apply pre-trained Yeo-network normative models to cVEDA between-network
connectivity data.

Adapted from the braincharts example apply_normative_models_yeo17.ipynb.
  1) Loads pre-trained BLR Yeo-17 network models
  2) Uses training/adaptation data to estimate site effects (if needed)
  3) Applies the model to test data
  4) Saves deviation (Z) scores and transfer metrics (EXPV, RMSE, Rho, etc.)

Usage:
  python 03_apply_normative_models.py --test-csv data/te.csv --adapt-csv data/ad.csv
  python 03_apply_normative_models.py --help
"""

# ---------------------------------------------------------------------
# 0. Compatibility: scipy.signal.gaussian was removed in newer scipy
# ---------------------------------------------------------------------
import scipy.signal
if not hasattr(scipy.signal, "gaussian"):
    try:
        from scipy.signal.windows import gaussian as _gaussian
        scipy.signal.gaussian = _gaussian
    except ImportError:
        import numpy as _np
        def _gaussian(M, std, sym=True):
            n = _np.arange(0, M) - (M - 1.0) / 2.0
            return _np.exp(-(n ** 2) / (2 * std ** 2))
        scipy.signal.gaussian = _gaussian

# ---------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------
import argparse
import os
import re
import shutil
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

def load_2d(fpath):
    """Load a text file as (n_samples, n_features) using pcntoolkit.dataio.fileio."""
    from pcntoolkit.dataio import fileio
    x = np.asarray(fileio.load(fpath))
    if x.ndim == 1:
        return x[:, np.newaxis]  # (N,) -> (N, 1)
    if x.ndim == 2 and x.shape[0] == 1:
        return x.T  # (1, N) -> (N, 1)
    return x

class _FilterStdout:
    def __init__(self, skip_substring):
        self.skip_substring = skip_substring
        self._real_stdout = None
    def write(self, data):
        if self.skip_substring not in data:
            self._real_stdout.write(data)
    def flush(self):
        self._real_stdout.flush()
    def __enter__(self):
        self._real_stdout = sys.stdout
        sys.stdout = self
        return self
    def __exit__(self, *args):
        sys.stdout = self._real_stdout
        return False


# ---------------------------------------------------------------------
# 2. Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_models = os.path.join(_script_dir, "models")
    _default_data = os.path.join(_script_dir, "data")

    p = argparse.ArgumentParser(
        description="Apply pre-trained Yeo-17 normative models; output Z scores and transfer metrics."
    )
    p.add_argument("--model-dir", type=str, default=_default_models,
                  help="Directory containing model folder and optional Models/, site_ids.txt, idp_ids.txt")
    p.add_argument("--model-name", type=str, default="BLR_yeonetworks_22K_45sites",
                  help="Name of model folder under --model-dir")
    p.add_argument("--test-csv", type=str, default=None,
                  help="Path to test dataset CSV (default: <model-dir>/../data/cveda_z_yeonetworks_te.csv)")
    p.add_argument("--adapt-csv", type=str, default=None,
                  help="Path to adaptation/calibration dataset CSV")
    p.add_argument("--site-ids", type=str, default=None,
                  help="Path to site IDs text file (default: model-dir/site_ids.txt)")
    p.add_argument("--idp-ids", type=str, default=None,
                  help="Path to IDP/edge list text file (default: model-dir/idp_ids.txt)")
    p.add_argument("--covariates", type=str, default="age,sex,mean_FD",
                  help="Comma-separated covariate column names (default: age,sex,mean_FD)")
    p.add_argument("--xmin", type=float, default=-5.0, help="B-spline lower bound")
    p.add_argument("--xmax", type=float, default=110.0, help="B-spline upper bound")
    return p.parse_args()


# ---------------------------------------------------------------------
# 3. Model and data configuration (from args + defaults)
# ---------------------------------------------------------------------

args = parse_args()

from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives

_script_dir = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.abspath(args.model_dir)
model_name = args.model_name
out_dir = os.path.join(_model_dir, model_name)

# Test and adaptation CSVs
if args.test_csv:
    test_data = os.path.abspath(args.test_csv)
else:
    test_data = os.path.join(_script_dir, "data", "cveda_z_yeonetworks_te.csv")
if args.adapt_csv:
    adaptation_data = os.path.abspath(args.adapt_csv)
else:
    adaptation_data = os.path.join(_script_dir, "data", "cveda_z_yeonetworks_tr.csv")

# Site IDs
_site_candidates = [
    args.site_ids,
    os.path.join(_model_dir, "site_ids.txt"),
    os.path.join(_model_dir, "site_ids.txt."),
    os.path.join(_script_dir, "docs", "site_ids.txt"),
]
site_ids_path = next((p for p in _site_candidates if p and os.path.isfile(p)), None)
if site_ids_path is None:
    raise FileNotFoundError("site_ids.txt not found. Use --site-ids or put site_ids.txt in model-dir.")
with open(site_ids_path) as f:
    site_ids_tr = [line.strip() for line in f if line.strip()]

# ---------------------------------------------------------------------
# 4. Load test and adaptation datasets
# ---------------------------------------------------------------------

df_te = pd.read_csv(test_data)
df_ad = pd.read_csv(adaptation_data)

# Ensure 'site' column exists
for name, df in [('test', df_te), ('adaptation', df_ad)]:
    if 'site' not in df.columns:
        raise ValueError(f"{name} CSV must have a 'site' column.")

# Derive sitenum from site if missing (1-based index: training sites first, then any new sites)
if 'sitenum' not in df_te.columns or 'sitenum' not in df_ad.columns:
    site_to_num = {s: i + 1 for i, s in enumerate(site_ids_tr)}
    next_num = len(site_ids_tr) + 1
    for s in sorted(set(df_te['site'].dropna()) | set(df_ad['site'].dropna())):
        if s not in site_to_num:
            site_to_num[s] = next_num
            next_num += 1
    if 'sitenum' not in df_te.columns:
        df_te['sitenum'] = df_te['site'].map(site_to_num)
    if 'sitenum' not in df_ad.columns:
        df_ad['sitenum'] = df_ad['site'].map(site_to_num)

# unique site IDs in test and adaptation sets
site_ids_te = sorted(set(df_te['site'].to_list()))
site_ids_ad = sorted(set(df_ad['site'].to_list()))

if not all(elem in site_ids_ad for elem in site_ids_te):
    print('Warning: some test sites are not present in the adaptation data.')

# ---------------------------------------------------------------------
# 5. Configure IDPs (between-network connectivity features)
# ---------------------------------------------------------------------

_idp_candidates = [
    args.idp_ids,
    os.path.join(_model_dir, "idp_ids.txt"),
    os.path.join(_model_dir, "phenotypes_yeonetworks.txt"),
    os.path.join(_script_dir, "docs", "phenotypes_yeonetworks.txt"),
]
idp_ids_path = next((p for p in _idp_candidates if p and os.path.isfile(p)), None)
if idp_ids_path is None:
    raise FileNotFoundError("idp_ids.txt not found. Use --idp-ids or put idp_ids.txt in model-dir.")
with open(idp_ids_path) as f:
    idp_ids = [line.strip() for line in f if line.strip()]

# ---------------------------------------------------------------------
# 6. Flat model layout: one shared Models/ with NM_0_*_fit.pkl (or estimate.pkl) per edge
# ---------------------------------------------------------------------
_shared_models = os.path.join(_model_dir, "Models")
_flat_pkls = None
if os.path.isdir(_shared_models):
    _all_pkl = [f for f in os.listdir(_shared_models) if f.endswith(".pkl")]
    # Sort by number in filename (e.g. NM_0_0_fit.pkl, NM_0_100_fit.pkl -> 0, 100, ...)
    _num = re.compile(r"NM_0_(\d+)_(?:fit|estimate)\.pkl")
    _flat_pkls = sorted(
        [f for f in _all_pkl if _num.match(f)],
        key=lambda f: int(_num.match(f).group(1))
    )
if _flat_pkls is not None and len(_flat_pkls) >= len(idp_ids):
    _flat_pkls = _flat_pkls[:len(idp_ids)]
    _meta_src = os.path.join(_shared_models, "meta_data.md")
    for i, idp in enumerate(idp_ids):
        idp_dir = os.path.join(out_dir, idp)
        models_sub = os.path.join(idp_dir, "Models")
        os.makedirs(models_sub, exist_ok=True)
        target = os.path.join(models_sub, "NM_0_0_estimate.pkl")
        src = os.path.join(_shared_models, _flat_pkls[i])
        if not os.path.lexists(target):
            os.symlink(os.path.abspath(src), target)
        if os.path.isfile(_meta_src):
            meta_target = os.path.join(models_sub, "meta_data.md")
            if not os.path.lexists(meta_target):
                os.symlink(os.path.abspath(_meta_src), meta_target)
    print("Using flat model layout: linked models/Models/*.pkl into per-edge Models/.")

# ---------------------------------------------------------------------
# 7. Covariates and spline basis
# ---------------------------------------------------------------------

cols_cov = [c.strip() for c in args.covariates.split(",")]
xmin = args.xmin
xmax = args.xmax
outlier_thresh = 7

# ---------------------------------------------------------------------
# 8. Make predictions for each BNFC feature
# ---------------------------------------------------------------------

for idp_num, idp in enumerate(idp_ids):
    print('Running IDP', idp_num, idp, ':')

    idp_dir = os.path.join(out_dir, idp)
    os.chdir(idp_dir)

    # response variable (BNFC) for test set
    y_te = df_te[idp].to_numpy()
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt')
    np.savetxt(resp_file_te, y_te)

    # design matrix for test set
    cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
    X_te = create_design_matrix(df_te[cols_cov],
                                site_ids=df_te['site'],
                                all_sites=site_ids_tr,
                                basis='bspline',
                                xmin=xmin,
                                xmax=xmax)
    np.savetxt(cov_file_te, X_te)

    # decide whether we can use training-site offsets directly or need adaptation
    if all(elem in site_ids_tr for elem in site_ids_te):
        print('All test sites are present in the training data (no adaptation).')

        # direct prediction using pre-trained model
        with _FilterStdout("No meta-data file is found!"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in sqrt", RuntimeWarning, module="pcntoolkit.util.utils")
                yhat_te, s2_te, Z = predict(
                    cov_file_te,
                    alg='blr',
                    respfile=resp_file_te,
                    model_path=os.path.join(idp_dir, 'Models')
                )

    else:
        print('Some test sites missing from training data. Adapting model.')

        # design matrix for adaptation data
        X_ad = create_design_matrix(df_ad[cols_cov],
                                    site_ids=df_ad['site'],
                                    all_sites=site_ids_tr,
                                    basis='bspline',
                                    xmin=xmin,
                                    xmax=xmax)
        cov_file_ad = os.path.join(idp_dir, 'cov_bspline_ad.txt')
        np.savetxt(cov_file_ad, X_ad)

        # responses for adaptation data
        resp_file_ad = os.path.join(idp_dir, 'resp_ad.txt')
        y_ad = df_ad[idp].to_numpy()
        np.savetxt(resp_file_ad, y_ad)

        # site numbers for adaptation and test data
        sitenum_file_ad = os.path.join(idp_dir, 'sitenum_ad.txt')
        site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
        np.savetxt(sitenum_file_ad, site_num_ad)

        sitenum_file_te = os.path.join(idp_dir, 'sitenum_te.txt')
        site_num_te = df_te['sitenum'].to_numpy(dtype=int)
        np.savetxt(sitenum_file_te, site_num_te)

        # prediction with adaptation
        with _FilterStdout("No meta-data file is found!"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in sqrt", RuntimeWarning, module="pcntoolkit.util.utils")
                yhat_te, s2_te, Z = predict(
                    cov_file_te,
                    alg='blr',
                    respfile=resp_file_te,
                    model_path=os.path.join(idp_dir, 'Models'),
                    adaptrespfile=resp_file_ad,
                    adaptcovfile=cov_file_ad,
                    adaptvargroupfile=sitenum_file_ad,
                    testvargroupfile=sitenum_file_te
                )

# ---------------------------------------------------------------------
# 6. Summarize model performance across IDPs and collect transfer metrics
# ---------------------------------------------------------------------

suffix = 'predict'
warp = 'WarpSinArcsinh'

blr_metrics = pd.DataFrame(columns=['eid', 'NLL', 'EV', 'Skew', 'Kurtosis'])

# Per-edge vectors for transfer metric CSVs (one value per IDP / edge)
expv_list = []
rmse_list = []
rho_list = []
prho_list = []
smse_list = []
# Full Z matrix: one column per edge, one row per subject (for Z_transfer.csv)
z_cols = []

for idp_num, idp in enumerate(idp_ids):
    idp_dir = os.path.join(out_dir, idp)

    # load predictions and true data (load_2d returns shape (n_samples, 1))
    yhat_te = load_2d(os.path.join(idp_dir, f'yhat_{suffix}.txt'))
    s2_te = load_2d(os.path.join(idp_dir, f'ys2_{suffix}.txt'))
    y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))

    with open(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle)

    # evaluate in input space (undo warping)
    warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]
    W = nm.blr.warp

    med_te = W.warp_predictions(np.squeeze(yhat_te),
                                np.squeeze(s2_te),
                                warp_param)[0]
    med_te = med_te[:, np.newaxis]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in sqrt", RuntimeWarning)
        metrics = evaluate(y_te, med_te)

    # calibration metrics for Z scores
    Z = np.loadtxt(os.path.join(idp_dir, f'Z_{suffix}.txt'))
    cd = calibration_descriptives(Z)  # from pcntoolkit.util.utils: [skew, sdskew, kurtosis, sdkurtosis, semean, sesd]
    skew = float(np.asarray(cd[0]).flat[0])
    kurtosis = float(np.asarray(cd[2]).flat[0])

    blr_metrics.loc[len(blr_metrics)] = [
        idp, nm.neg_log_lik, metrics['EXPV'][0], skew, kurtosis
    ]

    # --- Per-edge transfer metrics for export ---
    y_flat = np.asarray(y_te).ravel()
    med_flat = np.asarray(med_te).ravel()
    s2_flat = np.asarray(s2_te).ravel()

    ex = metrics['EXPV'][0]
    expv_list.append(float(ex) if np.isfinite(ex) else ex)  # keep NaN for failed edges

    rmse = np.sqrt(np.mean((y_flat - med_flat) ** 2))
    rmse_list.append(rmse)

    rho, p_rho = stats.pearsonr(y_flat, med_flat)
    rho_list.append(rho)
    prho_list.append(p_rho)

    # Standardized MSE: mean of (y - yhat)^2 / s2
    smse = np.mean((y_flat - med_flat) ** 2 / np.clip(s2_flat, 1e-10, None))
    smse_list.append(smse)

    # Z scores for this edge (one column of the full Z_transfer matrix)
    z_cols.append(np.asarray(Z).ravel())

blr_metrics.to_csv(os.path.join(out_dir, 'blr_cVEDA_transfer_metrics.csv'),
                   index=False)

# ---------------------------------------------------------------------
# 6b. Export transfer metric CSVs (one value per edge, for heatmaps / R)
# ---------------------------------------------------------------------

def save_transfer_vector(values, fname):
    """Save a 1D vector as CSV with index column (matches EvaluationFigures / R)."""
    df = pd.DataFrame({'index': range(len(values)), 'value': values})
    df.to_csv(os.path.join(out_dir, fname), index=False)

save_transfer_vector(expv_list, 'EXPV_transfer.csv')
save_transfer_vector(rmse_list, 'RMSE_transfer.csv')
save_transfer_vector(rho_list, 'Rho_transfer.csv')
save_transfer_vector(prho_list, 'pRho_transfer.csv')
save_transfer_vector(smse_list, 'SMSE_transfer.csv')

# Z_transfer.csv: full matrix (subjects x edges), one column per edge
Z_matrix = np.column_stack(z_cols)
pd.DataFrame(Z_matrix).to_csv(os.path.join(out_dir, 'Z_transfer.csv'), index=False)

print('Exported transfer metrics to', out_dir, ':',
      'EXPV_transfer.csv, RMSE_transfer.csv, Rho_transfer.csv, pRho_transfer.csv, SMSE_transfer.csv, Z_transfer.csv')

# ---------------------------------------------------------------------
# 7. Collect deviation (Z) scores across IDPs and save one merged CSV
# ---------------------------------------------------------------------

z_dir = os.path.join(out_dir, "deviation_scores")
os.makedirs(z_dir, exist_ok=True)

# Gather Z_predict.txt from each IDP folder into deviation_scores
for idp in idp_ids:
    src = os.path.join(out_dir, idp, "Z_predict.txt")
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(z_dir, f"{idp}_Z_predict.txt"))

filelist = [f for f in os.listdir(z_dir) if f.endswith("_Z_predict.txt")]
filelist.sort()

Z_cols = {
    f.replace(".txt", ""): pd.read_csv(os.path.join(z_dir, f), header=None).squeeze()
    for f in filelist
}
Z_df = pd.DataFrame(Z_cols)

df_te.reset_index(inplace=True)
Z_df = Z_df.assign(sub_id=df_te["sub_id"].values)

df_te_Z = pd.merge(df_te, Z_df, on="sub_id", how="inner")
_test_stem = os.path.splitext(os.path.basename(test_data))[0]
deviation_scores_csv = os.path.join(out_dir, f"{_test_stem}_deviation_scores.csv")
df_te_Z.to_csv(deviation_scores_csv, index=False)
