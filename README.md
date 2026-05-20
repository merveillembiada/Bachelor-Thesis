# Longevity Risk vs. Structural Labor Market Shocks: A Comparative Mathematical Analysis of PAYG Pension Sustainability

This repository contains the complete Python implementation and reproducible data pipeline for the empirical framework presented in the Bachelor Thesis in Business Mathematics 
(Wirtschaftsmathematik) submitted to **Hochschule Koblenz (RheinAhrCampus)** in collaboration with the **University of KwaZulu-Natal (UKZN)**.

## Project Overview
The research establishes a unified Overlapping Generations (OLG) macro-actuarial framework to analyze and compare two distinct systemic pressures threatening Pay-As-You-Go (PAYG) pension sustainability through 2050:
1. **Longevity Risk (Germany):** The biometric risk arising from secular improvements in survival probabilities ($\hat{s}_x$), which continuously expands the beneficiary population ($R$).
2. **Structural Labor Market Shocks (South Africa):** An economic shock characterized by a structurally depressed Labor Force Participation Rate ($\alpha_L$), compressing the institutional contribution base despite a young demographic profile.

---

## Repository Structure
```text
├── figures_all.py            # Main execution script containing estimation, forecasting, and plotting pipelines (SVD, RWD, and OLG paths)
├── Germany_LifeTables.xlsx   # Harmonized mortality statistics (HMD + Destatis)
├── Germany_LifeTables.xlsx   # South Africa LFPR dataset (World Bank)
└── README.md                 # Project documentation and execution guide
