"""
=============================================================
 BACHELOR THESIS — ALL FIGURES (Fig. 1 to Fig. 8)
 Longevity Risk vs. Structural Labor Market Shocks:
 A Comparative Mathematical Analysis of PAYG Pension
 Sustainability in Germany and South Africa

 Author: Ngnabeyeu Mbiada Merveille Ruth (562732)
 RheinAhrCampus Remagen | University of KwaZulu-Natal

 USAGE:
   python figures_all.py

 OUTPUT:
   fig1.png  — German log-mortality heatmap
   fig2.png  — Log-mortality age profiles + improvement heatmap
   fig3.png  — SVD scree plot + residual heatmap
   fig4.png  — alpha_x and beta_x profiles
   fig5.png  — kappa_t historical fit + RWD forecast
   fig6.png  — Survival probabilities s_x
   fig7.png  — SA LFPR + German improvement rates
   fig8.png  — OLG tau* trajectories + hyperbolic sensitivity

 DATA FILES REQUIRED (see DATA_GUIDE.md):
   hmd_germany_lt.xlsx  — HMD Germany period life tables
   sa_lfpr_wb.csv       — World Bank WDI SA LFPR

 DEPENDENCIES:
   pip install numpy pandas scipy matplotlib openpyxl
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy import linalg
import warnings
import os

warnings.filterwarnings("ignore")

# ─── GLOBAL STYLE ───────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette matching the thesis
NAVY   = "#12285A"
TEAL   = "#028090"
GOLD   = "#CAA014"
SA_GREEN = "#007749"
DE_BLUE  = "#003296"
MID_GRAY = "#64748B"

# ─────────────────────────────────────────────────────────────
# SECTION A: DATA LOADING UTILITIES
# ─────────────────────────────────────────────────────────────

def build_mortality_matrix(filepath: str,
                           year_range: tuple = (1960, 2022),
                           max_age: int = 89) -> pd.DataFrame:
    """
    Load HMD Germany life tables and build sex-averaged mortality matrix.

    The HMD Excel file (hmd_germany_lt.xlsx) has columns:
      Year1, AgeInt, TypeLT, Age, Sex, m(x), qx, lx, dx, Lx, Tx, ex

    We sex-average m(x) = (male_rate + female_rate) / 2.

    Returns
    -------
    M : pd.DataFrame, shape (n_ages, n_years), values = m_{x,t}
    """
    print("Loading HMD Germany life tables...")
    df = pd.read_excel(filepath, sheet_name="DEU")

    # Keep only single-year age intervals, both sexes, target years, ages <= 89
    mask = (
        (df["Year1"] >= year_range[0]) &
        (df["Year1"] <= year_range[1]) &
        (df["Age"] <= max_age)
    )
    df_f = df[mask].copy()

    # Pivot: rows = Age (0..89), columns = Year (1960..2022)
    M = df_f.pivot(index="Age", columns="Year1", values="m(x)")
    M = M.dropna(axis=0, how="any")
    print(f"  Mortality matrix shape: {M.shape} (ages x years)")
    return M


def load_sa_lfpr(filepath: str) -> pd.DataFrame:
    """
    Load South Africa LFPR from the cleaned data file.
    Handles format ambiguity by checking both true Excel and text-based engines.
    """
    print("Loading South Africa LFPR data from clean data sheet...")
    
    try:
        # First, attempt to read as a standard openpyxl Excel file
        df = pd.read_excel(filepath, sheet_name="LFPR", engine="openpyxl")
    except Exception:
        try:
            # If that fails, fallback to standard Excel parsing
            df = pd.read_excel(filepath, sheet_name="LFPR")
        except Exception:
            # If Excel format completely fails, it's actually a text/CSV structure under the hood
            print("  Note: File structure appears text-based. Reading as delimited data...")
            df = pd.read_csv(filepath)

    # Strip any hidden whitespaces from column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Dynamically rename the first two columns to guarantee exact matching (Year, LFPR)
    df = df.rename(columns={df.columns[0]: "Year", df.columns[1]: "LFPR"})
    
    # Convert data types to numeric to prevent parsing issues
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["LFPR"] = pd.to_numeric(df["LFPR"], errors="coerce")
    
    # Drop missing rows, sort by year chronologically, and reset the index
    df = df.dropna(subset=["Year", "LFPR"]).sort_values("Year").reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)
    
    print(f"  SA LFPR data: {df['Year'].min()}–{df['Year'].max()}, "
          f"range [{df['LFPR'].min():.1f}%, {df['LFPR'].max():.1f}%]")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION B: LEE-CARTER ESTIMATION
# ─────────────────────────────────────────────────────────────

def lee_carter_svd(M: pd.DataFrame):
    """
    Estimate Lee-Carter parameters via Singular Value Decomposition.

    Steps (following Neidhardt 2024, Section 1.2.3):
      A. alpha_x = row means of ln(M)
      B. Z = ln(M) - alpha_x  (centered residual matrix)
      C. Z = U * Sigma * V^T  (SVD)
      D. Rank-1 approximation: beta_x * kappa_t = sigma_1 * u1 * v1^T
      E. Normalize: sum(beta_x) = 1, sum(kappa_t) = 0

    Identification constraints (Lee & Carter 1992):
      sum_x(beta_x) = 1   and   sum_t(kappa_t) = 0

    Returns
    -------
    alpha_x, beta_x, kappa_t : np.ndarray
    s_vals                   : all singular values (for scree plot)
    residuals                : Z - beta_x * kappa_t (for diagnostics)
    log_M                    : ln(M) matrix
    Z                        : centered residual matrix
    """
    log_M = np.log(M.values.astype(float))  # shape (n_ages, n_years)

    # Step A: time-averaged log-mortality at each age
    alpha_x = np.mean(log_M, axis=1)        # shape (n_ages,)

    # Step B: remove age means -> centered matrix
    Z = log_M - alpha_x[:, np.newaxis]      # shape (n_ages, n_years)

    # Step C: full SVD
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    # U: (n_ages, k), s: (k,), Vt: (k, n_years)

    # Step D: rank-1 approximation
    u1 = U[:, 0]   # first left singular vector
    v1 = Vt[0, :]  # first right singular vector
    sigma1 = s[0]  # largest singular value

    # Step E: Lee-Carter normalization
    beta_sum = np.sum(u1)
    beta_x   = u1 / beta_sum                     # sum(beta_x) = 1
    kappa_t  = sigma1 * beta_sum * v1            # sum(kappa_t) = 0 by SVD property

    # Enforce declining kappa_t (mortality improves over time = kappa decreasing)
    if np.polyfit(np.arange(len(kappa_t)), kappa_t, 1)[0] > 0:
        beta_x  = -beta_x
        kappa_t = -kappa_t

    # Diagnostics
    r2 = s[0]**2 / np.sum(s**2) * 100
    residuals = Z - np.outer(beta_x, kappa_t)
    print(f"  Lee-Carter rank-1 variance explained: {r2:.2f}%")
    print(f"  sum(kappa_t) = {np.sum(kappa_t):.6f} [should be ~0]")
    print(f"  sum(beta_x)  = {np.sum(beta_x):.6f} [should be 1]")

    return alpha_x, beta_x, kappa_t, s, residuals, log_M, Z


# ─────────────────────────────────────────────────────────────
# SECTION C: RANDOM WALK WITH DRIFT FORECAST
# ─────────────────────────────────────────────────────────────

def forecast_kappa_rwd(kappa_t: np.ndarray,
                       n_forecast: int = 28,
                       n_sims: int = 1000,
                       alpha: float = 0.10,
                       seed: int = 42):
    """
    Forecast kappa_t using a Random Walk with Drift (RWD).

    Drift estimator (telescoping sum derivation, eq. 4.16):
      theta_hat = (kappa_tn - kappa_t0) / (tn - t0)

    Innovation variance (eq. 4.17):
      sigma_rw^2 = mean((Delta_kappa_i - theta_hat)^2)

    h-step forecast (eq. 4.18-4.19):
      E[kappa_{t+h}] = kappa_t + h * theta_hat
      PI_90%(h) = kappa_{t+h} +/- 1.645 * sigma_rw * sqrt(h)

    Returns
    -------
    theta_hat, sigma_rw, fc_mean, fc_low, fc_high, all_sims
    """
    n = len(kappa_t)
    theta_hat = (kappa_t[-1] - kappa_t[0]) / (n - 1)

    diffs = np.diff(kappa_t)
    sigma_rw = np.sqrt(np.mean((diffs - theta_hat)**2))

    print(f"  RWD drift theta_hat = {theta_hat:.4f} per year")
    print(f"  RWD innovation sigma = {sigma_rw:.4f}")

    np.random.seed(seed)
    sims = np.zeros((n_sims, n_forecast))
    for i in range(n_sims):
        sims[i, 0] = kappa_t[-1] + theta_hat + np.random.normal(0, sigma_rw)
        for t in range(1, n_forecast):
            sims[i, t] = sims[i, t-1] + theta_hat + np.random.normal(0, sigma_rw)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    return (theta_hat, sigma_rw,
            sims.mean(axis=0),
            np.percentile(sims, lo, axis=0),
            np.percentile(sims, hi, axis=0),
            sims)


# ─────────────────────────────────────────────────────────────
# SECTION D: SURVIVAL PROBABILITIES & OLG
# ─────────────────────────────────────────────────────────────

def compute_sx(age: int, kappa_val: float,
               alpha_x: np.ndarray,
               beta_x: np.ndarray,
               ages: np.ndarray) -> float:
    """
    Compute s_x = P(surviving from birth to age x) given kappa_t.

    Chain-product formula (eq. 4.22):
      ln(m_hat_a) = alpha_a + beta_a * kappa_val      [eq. 4.20]
      q_hat_a    = 1 - exp(-m_hat_a)                  [eq. 4.21, Poisson approx.]
      s_x         = prod_{a=0}^{x-1} (1 - q_hat_a)   [eq. 4.22]
    """
    idx = np.where(ages <= age)[0]
    log_mx = alpha_x[idx] + beta_x[idx] * kappa_val
    mx     = np.exp(log_mx)
    qx_hat = np.clip(1 - np.exp(-mx), 0.0, 1.0)
    return float(np.prod(1.0 - qx_hat))


def compute_survival_table(kappa_series: np.ndarray,
                           alpha_x: np.ndarray,
                           beta_x: np.ndarray,
                           ages: np.ndarray,
                           retirement_ages: list) -> dict:
    """
    Build a lookup table of survival probabilities for all
    retirement ages and all kappa values in kappa_series.
    """
    table = {xr: [] for xr in retirement_ages}
    for kappa_val in kappa_series:
        for xr in retirement_ages:
            table[xr].append(compute_sx(xr, kappa_val, alpha_x, beta_x, ages))
    return {xr: np.array(v) for xr, v in table.items()}


def calculate_tau(alpha_L: float, L: float, R: float,
                  b: float, w: float) -> float:
    """
    Equilibrium PAYG contribution rate (eq. 4.2):
      tau* = (R * b) / (alpha_L * L * w)

    Analytical properties (Proposition 4.1):
      d(tau*)/d(alpha_L) < 0  (strictly decreasing)
      d^2(tau*)/d(alpha_L)^2 > 0  (strictly convex)
    """
    denom = alpha_L * L * w
    if denom <= 0.0:
        return float("nan")
    return (R * b) / denom


# ─────────────────────────────────────────────────────────────
# SECTION E: FIGURE GENERATION
# ─────────────────────────────────────────────────────────────

def fig1_mortality_heatmap(M: pd.DataFrame, output_path: str = "fig1.png"):
    """
    Figure 1 — German log-mortality heatmap ln(m_{x,t}).

    Reproduces Figure 3.1 of the thesis.
    Color: lighter = lower mortality (improvement visible left->right).
    Annotations: age 65 (white dashed), age 67 (yellow dotted), COVID 2020.
    """
    print("Generating Figure 1: Mortality heatmap...")
    log_M = np.log(M.values)
    years = M.columns.astype(int)
    ages  = M.index.astype(int)

    fig, ax = plt.subplots(figsize=(11, 5))

    im = ax.imshow(
        log_M,
        aspect="auto",
        cmap="plasma",
        origin="upper",
        extent=[years[0], years[-1], ages[-1], ages[0]],
        vmin=log_M.min(),
        vmax=log_M.max()
    )

    # Retirement age lines
    ax.axhline(y=65, color="white",  linestyle="--", linewidth=1.2,
               label="Age 65 (statutory retirement)")
    ax.axhline(y=67, color="#FFD700", linestyle=":", linewidth=1.2,
               label="Age 67 (post-2012 reform)")

    # COVID annotation
    ax.axvline(x=2020, color="red", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.text(2020.3, 85, "COVID-19\n(2020)", color="red",
            fontsize=7.5, va="top")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\ln(m_{x,t})$", fontsize=9)

    ax.set_xlabel("Calendar Year $t$")
    ax.set_ylabel("Age $x$")
    ax.set_title(
        r"Figure 1: Mortality Matrix — Heatmap of $\ln(m_{x,t})$, Germany 1960–2022"
        "\n(Lighter colours = lower mortality; secular improvement visible left→right at all ages)",
        fontsize=9, pad=8
    )
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig2_log_mortality_profiles(M: pd.DataFrame, output_path: str = "fig2.png"):
    """
    Figure 2 — Log-mortality age profiles + annual improvement heatmap.

    Left panel: ln(m_{x,t}) profiles for selected years (near-parallel = Lee-Carter ok).
    Right panel: Annual improvement heatmap -Delta ln(m_{x,t}) following Neidhardt (2024).
    Reproduces Figure 3.2 of the thesis.
    """
    print("Generating Figure 2: Log-mortality profiles...")
    log_M = np.log(M.values)
    years = M.columns.astype(int)
    ages  = M.index.astype(int)

    selected_years = [1960, 1970, 1980, 1990, 2000, 2010, 2022]
    colors_sel = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_years)))

    # Annual improvement: -Delta ln(m) = -(ln(m_{x,t+1}) - ln(m_{x,t}))
    improvement = -np.diff(log_M, axis=1)  # shape (n_ages, n_years-1)
    imp_years   = years[:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── LEFT: Age profiles ──
    for yr, clr in zip(selected_years, colors_sel):
        if yr in M.columns:
            idx = list(M.columns).index(yr)
            ax1.plot(ages, log_M[:, idx], color=clr, linewidth=1.5, label=str(yr))

    ax1.axvline(x=20, color=GOLD, linestyle=":", linewidth=1, alpha=0.8)
    ax1.text(21, -9.5, "Accident\nHump\n(≈20)", fontsize=7, color=GOLD)
    ax1.axvline(x=65, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax1.text(65.5, -4.5, "Age 65", fontsize=7, color="red")

    ax1.set_xlabel("Age $x$")
    ax1.set_ylabel(r"$\ln(m_{x,t})$ — Log Death Rate")
    ax1.set_title("Figure 6a: Log-Mortality Profiles by Calendar Year\n"
                  "(Downward shift = secular improvement; accident hump at ~20)",
                  fontsize=8)
    ax1.legend(title="Year $t$", fontsize=7, title_fontsize=7)

    # ── RIGHT: Improvement heatmap ──
    # Diverging colormap: green = improvement (positive), red = deterioration (negative)
    vmax = np.percentile(np.abs(improvement), 97)
    im2 = ax2.imshow(
        improvement,
        aspect="auto",
        cmap="RdYlGn",
        origin="upper",
        extent=[imp_years[0], imp_years[-1], ages[-1], ages[0]],
        vmin=-vmax, vmax=vmax
    )
    ax2.axvline(x=2020, color="black", linestyle="--", linewidth=0.8)
    ax2.text(2020.3, 82, "COVID-19", color="darkred", fontsize=7)

    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label(r"$-\Delta\ln(m_{x,t})$ p.a.", fontsize=9)

    ax2.set_xlabel("Calendar Year $t$")
    ax2.set_ylabel("Age $x$")
    ax2.set_title("Figure 6b: Neidhardt Heat-Map — Annual Mortality Improvements\n"
                  "(Green = improvement; red = deterioration; calendar-year effects visible)",
                  fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig3_svd_scree_residuals(M: pd.DataFrame,
                              alpha_x, beta_x, kappa_t, s_vals, residuals, Z,
                              output_path: str = "fig3.png"):
    """
    Figure 3 — SVD scree plot + residual heatmap.

    Left: Scree plot of singular values sigma_i^2 / sum(sigma_i^2).
          First value accounts for 91.56% of variance.
    Right: Residual heatmap e_{x,t} = Z_{x,t} - beta_x * kappa_t.
           Near-random pattern confirms rank-1 adequacy (no cohort diagonals).
    Reproduces Figure 4.1 of the thesis.
    """
    print("Generating Figure 3: SVD scree plot + residuals...")
    years = M.columns.astype(int)
    ages  = M.index.astype(int)
    n_sv  = len(s_vals)

    variance_explained = s_vals**2 / np.sum(s_vals**2) * 100
    cumulative         = np.cumsum(variance_explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── LEFT: Scree plot ──
    sv_indices = np.arange(n_sv)
    bar_colors = [TEAL if i == 0 else MID_GRAY for i in sv_indices[:16]]
    ax1.bar(sv_indices[:16], variance_explained[:16], color=bar_colors, alpha=0.85)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(sv_indices[:16], cumulative[:16],
                  color=NAVY, marker="o", markersize=3, linewidth=1.5,
                  label="Cumulative %")
    ax1_twin.set_ylabel("Cumulative Variance (%)", color=NAVY)
    ax1_twin.tick_params(axis="y", labelcolor=NAVY)

    ax1.text(0, variance_explained[0] * 0.98,
             f"$\\sigma_1$ explains\n{variance_explained[0]:.1f}%",
             ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    ax1.set_xlabel("Singular Value Rank $i$")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Figure 3a: SVD Scree Plot\n"
                  "Rank-1 approximation explains 91.6% of variance in Z",
                  fontsize=8)

    # ── RIGHT: Residual heatmap ──
    vmax = np.percentile(np.abs(residuals), 97)
    im = ax2.imshow(
        residuals,
        aspect="auto",
        cmap="RdBu_r",
        origin="upper",
        extent=[years[0], years[-1], ages[-1], ages[0]],
        vmin=-vmax, vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r"Residual $e_{x,t}$", fontsize=9)

    ax2.set_xlabel("Calendar Year $t$")
    ax2.set_ylabel("Age $x$")
    ax2.set_title(r"Figure 3b: Residuals $e_{x,t} = Z_{x,t} - \hat{\beta}_x\hat{\kappa}_t$"
                  "\n(Random pattern confirms rank-1 adequacy; no strong cohort diagonals)",
                  fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig4_alpha_beta_profiles(M: pd.DataFrame,
                              alpha_x, beta_x,
                              output_path: str = "fig4.png"):
    """
    Figure 4 — Estimated alpha_x and beta_x profiles.

    Left:  alpha_x = time-averaged log-mortality.
           Canonical Gompertz shape above age 30; accident hump at ~20.
    Right: beta_x = age-specific sensitivity to kappa_t.
           Highest at infants (post-war neonatal gains); secondary peak near 65.
    Reproduces Figure 4.2 of the thesis.
    """
    print("Generating Figure 4: alpha_x and beta_x profiles...")
    ages = M.index.astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── LEFT: alpha_x ──
    ax1.plot(ages, alpha_x, color=NAVY, linewidth=2)
    ax1.axvline(x=20, color=GOLD, linestyle=":", linewidth=1.2, alpha=0.9,
                label="Accident hump (~20)")
    ax1.axvline(x=65, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                label="Age 65 (retirement)")

    ax1.annotate("Infant\nmortality", xy=(0, alpha_x[0]),
                 xytext=(5, alpha_x[0]+1.5), fontsize=7.5, color=TEAL,
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))
    ax1.annotate("Accident\nhump", xy=(20, alpha_x[20]),
                 xytext=(28, alpha_x[20]+0.8), fontsize=7.5, color=GOLD,
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8))

    ax1.set_xlabel("Age $x$")
    ax1.set_ylabel(r"$\hat{\alpha}_x$")
    ax1.set_title(r"Figure 2a: $\hat{\alpha}_x$ — Time-Averaged Log-Mortality"
                  "\n(Gompertz shape; near-linear increase after age 30)",
                  fontsize=8)
    ax1.legend(fontsize=8)

    # ── RIGHT: beta_x ──
    bar_colors = [TEAL if x <= 5 else (SA_GREEN if 60 <= x <= 75 else MID_GRAY)
                  for x in ages]
    ax2.bar(ages, beta_x, color=bar_colors, alpha=0.85, width=0.9)
    ax2.axvline(x=65, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                label="Age 65")

    ax2.text(1, beta_x[0] * 0.97, "Infants:\nhighest\nsensitivity",
             fontsize=7.5, color=TEAL, ha="left", va="top")

    ax2.set_xlabel("Age $x$")
    ax2.set_ylabel(r"$\hat{\beta}_x$ (Neidhardt: $|\hat{b}_{x} - 1|$)")
    ax2.set_title(r"Figure 2b: $\hat{\beta}_x$ — Age-Specific Sensitivity to $\hat{\kappa}_t$"
                  "\n(High at young ages & oldest-old; lower for accident-hump ages)",
                  fontsize=8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig5_kappa_forecast(M: pd.DataFrame,
                         kappa_t, theta_hat, sigma_rw,
                         fc_mean, fc_low, fc_high, all_sims,
                         output_path: str = "fig5.png"):
    """
    Figure 5 — kappa_t historical fit and RWD forecast to 2050.

    Historical kappa_t (1960-2022) + deterministic drift + 50% and 90%
    Monte Carlo prediction bands from 1000 simulated paths.
    Reproduces Figure 4.3 of the thesis.
    """
    print("Generating Figure 5: kappa_t forecast...")
    years_hist  = M.columns.astype(int)
    t0, tn      = int(years_hist[0]), int(years_hist[-1])
    n_forecast  = len(fc_mean)
    years_fc    = np.arange(tn + 1, tn + 1 + n_forecast)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Historical kappa
    ax.plot(years_hist, kappa_t, color=NAVY, linewidth=2,
            label=r"Historical $\hat{\kappa}_t$ (1960–2022)")

    # Monte Carlo paths (plot a sample)
    for i in range(min(100, all_sims.shape[0])):
        ax.plot(years_fc, all_sims[i], color=TEAL, alpha=0.04, linewidth=0.5)

    # 90% prediction interval
    ax.fill_between(years_fc, fc_low, fc_high,
                    color=TEAL, alpha=0.25, label="90% Prediction Interval")

    # 50% prediction interval (recompute)
    fc_low50  = np.percentile(all_sims, 25, axis=0)
    fc_high50 = np.percentile(all_sims, 75, axis=0)
    ax.fill_between(years_fc, fc_low50, fc_high50,
                    color=TEAL, alpha=0.40, label="50% Prediction Interval")

    # RWD mean forecast
    ax.plot(years_fc, fc_mean, color=TEAL, linewidth=2, linestyle="--",
            label=r"RWD Forecast $\hat{\kappa}_t$ (mean)")

    # Deterministic drift line
    drift_line = kappa_t[-1] + theta_hat * np.arange(1, n_forecast + 1)
    ax.plot(years_fc, drift_line, color=GOLD, linewidth=1.5, linestyle="-.",
            label=fr"Deterministic drift ($\hat{{\theta}}={theta_hat:.2f}$/yr)")

    # Divider between historical and forecast
    ax.axvline(x=tn, color="gray", linestyle=":", linewidth=1)
    ax.text(tn + 0.5, ax.get_ylim()[0] * 0.9, "Forecast →",
            fontsize=8, color="gray", va="bottom")

    # COVID-19 annotation
    covid_idx = list(years_hist).index(2020)
    ax.annotate("COVID-19\nmortality spike",
                xy=(2020, kappa_t[covid_idx]),
                xytext=(2010, kappa_t[covid_idx] + 20),
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax.set_xlabel("Calendar Year $t$")
    ax.set_ylabel(r"$\hat{\kappa}_t$ — Mortality Time Index")
    ax.set_title(
        r"Figure 4: Lee-Carter Mortality Index $\hat{\kappa}_t$ — "
        "Historical Fit and RWD Forecast to 2050\n"
        r"(Declining $\hat{\kappa}_t$ = mortality improvement = Longevity Risk for PAYG)",
        fontsize=9
    )
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig6_survival_probabilities(M: pd.DataFrame,
                                  kappa_t, fc_mean,
                                  alpha_x, beta_x,
                                  output_path: str = "fig6.png"):
    """
    Figure 6 — Survival probabilities s_x(kappa_t) for retirement ages 60–80.

    Left:  Time series of s_x for ages 60, 65, 67, 70, 75, 80 (1960-2050).
           Shaded region = forecast period.
    Right: Bar chart comparing s_x at 2000, 2022, 2050 (annotated +7.2 pp at 65).
    Reproduces Figure 4.4 of the thesis.
    """
    print("Generating Figure 6: Survival probabilities...")
    years_hist = M.columns.astype(int)
    t0, tn     = int(years_hist[0]), int(years_hist[-1])
    n_fc       = len(fc_mean)
    years_fc   = np.arange(tn + 1, tn + 1 + n_fc)
    years_all  = np.concatenate([years_hist, years_fc])
    kappa_all  = np.concatenate([kappa_t, fc_mean])
    ages       = M.index.astype(int)

    ret_ages   = [60, 65, 67, 70, 75, 80]
    colors_ret = plt.cm.cool(np.linspace(0.1, 0.9, len(ret_ages)))

    # Compute survival probabilities for all years
    sx_table = {}
    for xr in ret_ages:
        sx_table[xr] = [compute_sx(xr, kappa_val, alpha_x, beta_x, ages)
                        for kappa_val in kappa_all]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── LEFT: Time series ──
    for xr, clr in zip(ret_ages, colors_ret):
        ax1.plot(years_all, [v * 100 for v in sx_table[xr]],
                 color=clr, linewidth=1.8, label=f"Age {xr}")
        # Dashed for forecast period
        fc_vals = [v * 100 for v in sx_table[xr][len(years_hist):]]
        ax1.plot(years_fc, fc_vals, color=clr, linewidth=1.8, linestyle="--")

    ax1.axvspan(tn + 1, years_all[-1], alpha=0.06, color="gray")
    ax1.text(tn + 2, 20, "Forecast\nperiod", fontsize=8, color="gray")

    ax1.set_xlabel("Calendar Year $t$")
    ax1.set_ylabel(r"$\hat{s}_x(\hat{\kappa}_t)$ (%)")
    ax1.set_title(r"Figure 5a: Survival Probabilities $\hat{s}_x(\hat{\kappa}_t)$ by Retirement Age"
                  "\n(Germany 1960–2050; dashed line = post-2022 forecast)",
                  fontsize=8)
    ax1.legend(title="Retire age", fontsize=7.5, title_fontsize=8)
    ax1.set_ylim(0, 105)

    # ── RIGHT: Bar chart at 2000, 2022, 2050 ──
    benchmark_years = [2000, 2022, 2050]
    benchmark_kappas = {}
    for yr in benchmark_years:
        if yr <= tn:
            idx = list(years_hist).index(yr)
            benchmark_kappas[yr] = kappa_t[idx]
        else:
            fc_idx = yr - (tn + 1)
            benchmark_kappas[yr] = fc_mean[fc_idx]

    bar_colors_bm = [NAVY, TEAL, GOLD]
    x_pos = np.arange(len(ret_ages))
    width = 0.25

    for i, (yr, clr) in enumerate(zip(benchmark_years, bar_colors_bm)):
        vals = [compute_sx(xr, benchmark_kappas[yr], alpha_x, beta_x, ages) * 100
                for xr in ret_ages]
        ax2.bar(x_pos + i * width, vals, width=width, color=clr,
                alpha=0.85, label=str(yr))

    # Annotate +7.2 pp at age 65
    age65_idx = ret_ages.index(65)
    val_2000  = compute_sx(65, benchmark_kappas[2000], alpha_x, beta_x, ages) * 100
    val_2050  = compute_sx(65, benchmark_kappas[2050], alpha_x, beta_x, ages) * 100
    ax2.annotate(f"$\\Delta = +${val_2050 - val_2000:.1f} pp",
                 xy=(age65_idx + width, val_2050),
                 xytext=(age65_idx + width + 0.3, val_2050 + 3),
                 fontsize=8, color=GOLD, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8))

    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels([f"Age {xr}" for xr in ret_ages], fontsize=8)
    ax2.set_ylabel(r"$\hat{s}_x$ (%)")
    ax2.set_title(r"Figure 5b: Survival Probabilities $\hat{s}_x$ — 2000 vs 2022 vs 2050"
                  "\n(Longevity Risk quantified: each bar increase expands R in OLG)",
                  fontsize=8)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig7_sa_lfpr_improvement_rates(M: pd.DataFrame,
                                     sa_lfpr: pd.DataFrame,
                                     output_path: str = "fig7.png"):
    """
    Figure 7 — South Africa LFPR (1990-2024) + German improvement rates.

    Left:  SA LFPR time series with OECD avg. (65%) and Germany (80%) benchmarks.
           Structural ceiling below 56% and COVID-19 shock annotated.
    Right: Empirical annual log-mortality improvement rates rho_x for Germany
           computed as rho_x = [ln(m_{x,1990}) - ln(m_{x,2022})] / (2022-1990) * 100%.
    Reproduces Figure 5.1 of the thesis.
    """
    print("Generating Figure 7: SA LFPR + German improvement rates...")
    ages = M.index.astype(int)
    log_M = np.log(M.values)

    # German improvement rates 1990-2022 (Table 5.1)
    if 1990 in M.columns and 2022 in M.columns:
        idx1990 = list(M.columns).index(1990)
        idx2022 = list(M.columns).index(2022)
        rho_x = (log_M[:, idx1990] - log_M[:, idx2022]) / (2022 - 1990) * 100
    else:
        # Fallback values from Table 5.1
        rho_x = np.array([2.27, 2.27, 2.27, 2.27, 2.27,   # 0-4
                          2.45, 2.45, 2.45, 2.45, 2.45,   # 5-9 (approx)
                          2.45, 2.45, 2.45, 2.45, 2.45,   # 10-14
                          1.82, 1.82, 1.82, 1.82, 1.82,   # 15-19
                          1.82, 1.82, 1.82, 1.82, 1.82,   # 20-24
                          2.47, 2.47, 2.47, 2.47, 2.47,   # 25-29
                          2.47, 2.47, 2.47, 2.47, 2.47,   # 30-34
                          2.05, 2.05, 2.05, 2.05, 2.05,   # 35-39
                          2.05, 2.05, 2.05, 2.05, 2.05,   # 40-44
                          1.59, 1.59, 1.59, 1.59, 1.59,   # 45-49
                          1.59, 1.59, 1.59, 1.59, 1.59,   # 50-54
                          1.34, 1.34, 1.34, 1.34, 1.34,   # 55-59
                          1.34, 1.34, 1.34, 1.34, 1.34,   # 60-64
                          1.19, 1.19, 1.19, 1.19, 1.19,   # 65-69
                          1.19, 1.19, 1.19, 1.19, 1.19,   # 70-74
                          1.44, 1.44, 1.44, 1.44, 1.44,   # 75-79
                          1.44, 1.44, 1.44, 1.44, 1.44,   # 80-84
                          0.97, 0.97, 0.97, 0.97, 0.97])  # 85-89

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── LEFT: SA LFPR ──
    sa_plot = sa_lfpr[sa_lfpr["Year"] >= 1990]
    ax1.plot(sa_plot["Year"], sa_plot["LFPR"],
             color=SA_GREEN, linewidth=2.2, label=r"South Africa LFPR $\alpha_L$")

    ax1.axhline(y=65, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
                label="OECD avg. (~65%)")
    ax1.axhline(y=80, color=DE_BLUE, linestyle=":", linewidth=1.2, alpha=0.8,
                label="Germany (~80%)")

    # Structural ceiling annotation
    ax1.axhline(y=56, color=GOLD, linestyle="-.", linewidth=1, alpha=0.7,
                label="Structural ceiling (<56%)")
    ax1.fill_between(sa_plot["Year"], 0, 56, alpha=0.04, color=GOLD)

    # COVID annotation
    ax1.annotate("COVID-19\nshock",
                 xy=(2020, sa_plot.loc[sa_plot["Year"] == 2020, "LFPR"].values[0]
                     if 2020 in sa_plot["Year"].values else 50.5),
                 xytext=(2015, 45),
                 fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax1.set_xlim(1989, 2026)
    ax1.set_ylim(30, 90)
    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"LFPR $\alpha_L$ (%)")
    ax1.set_title(r"Figure 8a: South Africa LFPR $\alpha_L$, 1990–2024"
                  "\n(Structural ceiling below 56%; COVID-19 lockdown shock annotated)",
                  fontsize=8)
    ax1.legend(fontsize=7.5, loc="upper left")

    # ── RIGHT: German improvement rates ──
    bar_colors = [SA_GREEN if r > 2.0 else (TEAL if r > 1.4 else MID_GRAY)
                  for r in rho_x[:90]]
    ax2.bar(ages[:90], rho_x[:90], color=bar_colors, alpha=0.85, width=0.9)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axvline(x=65, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Age 65")

    ax2.set_xlabel("Age $x$")
    ax2.set_ylabel(r"$\hat{\rho}_x$ (% p.a.)")
    ax2.set_title(r"Figure 8b: Empirical Improvement Rates $\hat{\rho}_x$ for Germany, 1990–2022"
                  "\n(Validates back-projection in Section 3.2)",
                  fontsize=8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig8_olg_tau_trajectories(M: pd.DataFrame,
                               kappa_t, fc_mean,
                               alpha_x, beta_x,
                               output_path: str = "fig8.png"):
    """
    Figure 8 — OLG equilibrium contribution rate tau* trajectories.

    Left:  tau*(t) for Germany (longevity channel) and South Africa under
           three scenarios (Baseline, Recovery, Stress), 2000-2050.
           Germany crosses 20% sustainability threshold ~2028.
    Right: Hyperbolic sensitivity tau*(alpha_L) for South Africa.
           Steep region left of current alpha_L = 53.5% illustrates
           the convex-decreasing relationship of Proposition 4.1.
    Reproduces Figure 5.2 of the thesis.
    """
    print("Generating Figure 8: OLG tau* trajectories...")
    ages       = M.index.astype(int)
    years_hist = M.columns.astype(int)
    t0, tn     = int(years_hist[0]), int(years_hist[-1])
    n_fc       = len(fc_mean)
    years_fc   = np.arange(tn + 1, tn + 1 + n_fc)

    # Combine historical + forecast kappa
    kappa_all  = np.concatenate([kappa_t, fc_mean])
    years_all  = np.concatenate([years_hist, years_fc])

    # 2022 baseline calibration (Table 4.1)
    params_de = dict(alpha_L=0.80, L=45.0e6, w=45000, b=18000,
                     R_base=18.0e6)
    params_sa = dict(alpha_L=0.535, L=40.0e6, w=8000,  b=2400,
                     R_base=4.5e6)

    # Compute s_65 at 2022 baseline
    kappa_2022_idx = list(years_hist).index(2022)
    s65_2022 = compute_sx(65, kappa_t[kappa_2022_idx], alpha_x, beta_x, ages)

    # ── GERMANY tau*(t) ──
    # R(t) = R_2022 * [s_65(kappa_t) / s_65(kappa_2022)] * delta_t
    # delta_t: linear baby-boom factor 1.0 at 2020, 1.08 at 2040, then stable
    tau_de = []
    for t, kappa_val in zip(years_all, kappa_all):
        if t < 2000:
            tau_de.append(None)
            continue
        s65_t = compute_sx(65, kappa_val, alpha_x, beta_x, ages)
        delta = 1.0 + max(0, min(0.08, 0.08 * (t - 2020) / 20)) if t >= 2020 else 1.0
        R_t   = params_de["R_base"] * (s65_t / s65_2022) * delta
        tau   = calculate_tau(params_de["alpha_L"], params_de["L"],
                              R_t, params_de["b"], params_de["w"])
        tau_de.append(tau * 100)

    # ── SOUTH AFRICA: three scenarios ──
    sa_scenarios = {
        "SA Baseline": 0.535,
        "SA Recovery ($\\alpha_L \\to 65\\%$)": None,  # linearly rising
        "SA Stress ($\\alpha_L - 5$ pp)": 0.485,
    }
    tau_sa_base     = []
    tau_sa_recovery = []
    tau_sa_stress   = []

    for t in years_all:
        if t < 2000:
            tau_sa_base.append(None)
            tau_sa_recovery.append(None)
            tau_sa_stress.append(None)
            continue

        # Baseline: alpha_L fixed at current level
        tau_b = calculate_tau(0.535, params_sa["L"], params_sa["R_base"],
                              params_sa["b"], params_sa["w"]) * 100

        # Recovery: linearly from 55.6% (2022) to 60.7% (2050)
        if t <= 2022:
            alpha_rec = 0.535
        else:
            alpha_rec = 0.535 + (0.607 - 0.535) * (t - 2022) / (2050 - 2022)
        tau_r = calculate_tau(alpha_rec, params_sa["L"], params_sa["R_base"],
                              params_sa["b"], params_sa["w"]) * 100

        # Stress: alpha_L drops 5 pp permanently
        tau_s = calculate_tau(0.485, params_sa["L"], params_sa["R_base"],
                              params_sa["b"], params_sa["w"]) * 100

        tau_sa_base.append(tau_b)
        tau_sa_recovery.append(tau_r)
        tau_sa_stress.append(tau_s)

    # Filter from year 2000 onward
    mask2000 = years_all >= 2000
    years_plot = years_all[mask2000]
    tau_de_plot        = [v for v, m in zip(tau_de, mask2000) if m]
    tau_base_plot      = [v for v, m in zip(tau_sa_base, mask2000) if m]
    tau_recovery_plot  = [v for v, m in zip(tau_sa_recovery, mask2000) if m]
    tau_stress_plot    = [v for v, m in zip(tau_sa_stress, mask2000) if m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── LEFT: tau* time series ──
    ax1.plot(years_plot, tau_de_plot,
             color=DE_BLUE, linewidth=2.5, label=r"Germany ($\uparrow s_{65}$)")
    ax1.plot(years_plot, tau_base_plot,
             color=SA_GREEN, linewidth=2, label="SA Baseline")
    ax1.plot(years_plot, tau_recovery_plot,
             color=SA_GREEN, linewidth=1.8, linestyle="--",
             label=r"SA Recovery ($\alpha_L \to 65\%$)")
    ax1.plot(years_plot, tau_stress_plot,
             color=SA_GREEN, linewidth=1.8, linestyle=":",
             label=r"SA Stress ($\alpha_L - 5$ pp)")

    ax1.axhline(y=20, color="red", linestyle="-.", linewidth=1.2, alpha=0.7,
                label=r"$\tau^*$ = 20% threshold")

    # Annotate threshold crossing ~2028
    ax1.axvline(x=2028, color="red", linewidth=0.8, alpha=0.5)
    ax1.text(2028.5, 15.5, "~2028", fontsize=8, color="red")

    # Forecast shading
    ax1.axvspan(tn + 1, years_plot[-1], alpha=0.05, color="gray")

    ax1.set_xlim(1998, 2052)
    ax1.set_ylim(0, 36)
    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"$\tau^*$ (%)")
    ax1.set_title(r"Figure 7a: OLG Equilibrium Contribution Rate $\tau^*$"
                  "\nGermany vs. South Africa (2000–2050)",
                  fontsize=8)
    ax1.legend(fontsize=7.5, loc="upper left")

    # ── RIGHT: Hyperbolic sensitivity ──
    alpha_range = np.linspace(0.30, 0.72, 300)
    tau_sensitivity = [calculate_tau(a, params_sa["L"], params_sa["R_base"],
                                     params_sa["b"], params_sa["w"]) * 100
                       for a in alpha_range]

    ax2.plot(alpha_range * 100, tau_sensitivity, color=SA_GREEN, linewidth=2.5)

    # Current alpha_L
    ax2.axvline(x=53.5, color=NAVY, linestyle="--", linewidth=1.5,
                label=r"Current $\alpha_L = 53.5\%$")
    # Stress scenario
    ax2.axvline(x=40, color="red", linestyle=":", linewidth=1.2,
                label=r"Stress: 40%")
    # Recovery scenario
    ax2.axvline(x=65, color=TEAL, linestyle=":", linewidth=1.2,
                label=r"Recovery: 65%")

    # Shade steep zone
    steep_mask = (alpha_range * 100) <= 55
    ax2.fill_between(alpha_range[steep_mask] * 100,
                     np.array(tau_sensitivity)[steep_mask],
                     alpha=0.15, color="red", label="Convex fragility zone")

    ax2.set_xlim(28, 72)
    ax2.set_ylim(0, 14)
    ax2.set_xlabel(r"Labor Force Participation Rate $\alpha_L$ (%)")
    ax2.set_ylabel(r"$\tau^* = Rb/(\alpha_L Lw)$")
    ax2.set_title(r"Figure 7b: Sensitivity of $\tau^*$ to $\alpha_L$"
                  "\n(Hyperbolic: convex-decreasing)",
                  fontsize=8)
    ax2.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# SECTION F: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("THESIS FIGURES — FULL PIPELINE")
    print("=" * 60)

    # ── LOAD DATA ──
    hmd_path  = "hmd_germany_lt.xlsx"
    sa_path   = "sa_lfpr_wb.xlsx"

    if not os.path.exists(hmd_path):
        raise FileNotFoundError(
            f"Missing '{hmd_path}'.\n"
            "Download Germany complete period life tables from:\n"
            "  https://www.mortality.org/\n"
            "Login > Data > Germany (DEU) > Period Life Tables > Download Excel\n"
            "Rename the downloaded file to 'hmd_germany_lt.xlsx'.\n"
            "See DATA_GUIDE.md for full instructions."
        )
    if not os.path.exists(sa_path):
        raise FileNotFoundError(
            f"Missing '{sa_path}'. Please ensure your clean Excel file "
            "is saved in this directory with the sheet named 'LFPR'."
        )
        

    # ── BUILD MORTALITY MATRIX ──
    M = build_mortality_matrix(hmd_path, year_range=(1960, 2022), max_age=89)

    # ── LEE-CARTER ESTIMATION ──
    print("\nRunning Lee-Carter SVD estimation...")
    alpha_x, beta_x, kappa_t, s_vals, residuals, log_M, Z = lee_carter_svd(M)

    # ── KAPPA FORECAST ──
    print("\nForecasting kappa_t via Random Walk with Drift...")
    theta_hat, sigma_rw, fc_mean, fc_low, fc_high, all_sims = forecast_kappa_rwd(
        kappa_t, n_forecast=28, n_sims=1000, alpha=0.10, seed=42
    )

    # ── SA LFPR ──
    sa_lfpr = load_sa_lfpr(sa_path)

    # ── GENERATE FIGURES ──
    print("\nGenerating all figures...")
    fig1_mortality_heatmap(M)
    fig2_log_mortality_profiles(M)
    fig3_svd_scree_residuals(M, alpha_x, beta_x, kappa_t, s_vals, residuals, Z)
    fig4_alpha_beta_profiles(M, alpha_x, beta_x)
    fig5_kappa_forecast(M, kappa_t, theta_hat, sigma_rw, fc_mean, fc_low, fc_high, all_sims)
    fig6_survival_probabilities(M, kappa_t, fc_mean, alpha_x, beta_x)
    fig7_sa_lfpr_improvement_rates(M, sa_lfpr)
    fig8_olg_tau_trajectories(M, kappa_t, fc_mean, alpha_x, beta_x)

    print("\n" + "=" * 60)
    print("All 8 figures saved successfully.")
    print("Copy fig1.png through fig8.png into the same folder as")
    print("presentation.tex and thesis_final_v2.tex to compile LaTeX.")
    print("=" * 60)


if __name__ == "__main__":
    main()
