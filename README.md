# An Explainable Decision Intelligence Framework for Labour Market Analytics and Skill Gap Analysis

A research prototype that integrates **O*NET**, **ESCO**, and a historical **New York City (NYC) job-postings snapshot** to explore occupation–skill relationships, prioritize candidate skill gaps, evaluate machine-learning models, and explain predictions through an interactive Streamlit dashboard.

> **Interpretation:** An occupation–skill relationship that was not observed in this processed NYC dataset is **not** proof of a labour-market shortage, and the model does **not** measure an individual's proficiency. The framework is an academic decision-support prototype, not a live labour-market forecasting service.

## Architecture

| Layer | Implementation |
|---|---|
| 1. Data sources | O*NET occupational attributes, ESCO taxonomy and occupation–skill links, NYC job-posting snapshot |
| 2. Processing and integration | Cleaning, occupation alignment, definition-reviewed skill equivalences, approved-only crosswalk, NYC evidence join |
| 3. Analytics | Exploratory analysis, occupation-level demand, priority rules and sensitivity tests, grouped-CV classification |
| 4. Explainability | Global and local SHAP explanations for the saved TF-IDF + Logistic Regression classifier |
| 5. Decision support | Streamlit dashboard, including SHAP Explainability |

## Data sources

- **O*NET:** occupation and skill metadata, including importance and level measurements. Original files are organized under `data/raw/onet/`; later experiments also use `data/raw/content_model_reference.csv` for official skill definitions.
- **ESCO:** occupations, skills, and occupation–skill relations under `data/raw/esco/`.
- **NYC job postings:** historical anonymized snapshot dated **2025-07-13**, under `data/raw/job_postings/`.

Before publicly distributing any raw file, confirm its license/redistribution rights and ensure the job-posting snapshot has been reviewed for disclosure risks. A local file appearing in this repository does not automatically mean it can be republished.

## Core workflow and key scripts

The repository evolved through multiple research stages. **Keep the original preprocessing, modeling, validation and prioritization scripts:** they are needed to reproduce earlier results even when the dashboard does not import them directly.

| Stage | Scripts / artifacts |
|---|---|
| Original preparation and analysis | `src/preprocessing.py`, `src/skill_mapping.py`, `src/skill_demand_matrix.py`, `src/skill_gap.py`, `src/skill_gap_prioritisation_final.py` and other `src/` analysis/validation scripts |
| Original model evaluation | `src/modelling_occupation_aware_grouped_cv.py`, `src/modelling_occupation_aware_baseline_comparison.py`, `src/experiment_nested_threshold.py` and related evaluation scripts |
| Occupation crosswalk (later addition) | `src/build_validated_occupation_crosswalk_v2.py` |
| O*NET skill attributes (later addition) | `src/build_onet_integrated_attributes_v2.py` |
| Semantic candidate retrieval | `src/propose_onet_esco_matches_v2.py` |
| Definition enrichment and review | `src/enrich_onet_esco_review_v3.py`; reviewed `definition_based_skill_review_v4.csv` |
| Approved skill crosswalk | `src/build_approved_skill_crosswalk_v5.py` |
| NYC demand join | `src/integrate_onet_esco_nyc_demand_v6.py` |
| SHAP analysis | `src/explain_model_shap_v7.py` |
| Lexical versus semantic experiment | `src/compare_lexical_semantic_v1.py` |
| Exploratory threshold analysis | Preserve and document the threshold-sensitivity script from the final experiment package; check its exact filename locally |
| Dashboard | Root `app.py` (use the latest tested version with corrected 50% prediction label) |

**Note:** Some later scripts may not be present in the *original* repository ZIP. Copy the working versions from the later experiment packages before relying on the corresponding commands below. Do not replace the working `app.py` with the older seven-page file from the original archive.

## Installation (Windows, macOS and Linux)

**Python:** This project was developed and tested on macOS using Python 3.13. Windows instructions are provided for running the same Python code, but a full Windows end-to-end run has **not yet been verified**. Use **Python 3.13, 64-bit** where compatible with the package versions installed on your machine. In particular, saved `joblib` scikit-learn models may require the versions used during training; if a model fails to load, check its original environment rather than retraining or replacing it.

### Prerequisites (all platforms)

- Python and `pip` (make sure Python is available from your shell).
- An internet connection for the initial package installation and optional MiniLM model download.
- The project's required raw/processed files under `data/`; the repository may exclude datasets that cannot be redistributed.
- From a terminal, change directory to the repository root (the folder containing `app.py` and `requirements.txt`).

### Windows — PowerShell

Open PowerShell **inside the project folder** and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `py -3.13` is unavailable but `python` invokes the correct version, use `python -m venv .venv` instead. If PowerShell prevents activation, you can avoid changing script-execution policy entirely by calling the environment's executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To permit activation **only in the current PowerShell process**, if you prefer activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows — Command Prompt (CMD)

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux — Terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If using several installed Python versions, create the virtual environment with the intended interpreter (for example, `python3.13 -m venv .venv`).

When Streamlit starts, open the **Local URL** printed in the terminal (typically `http://localhost:8501`). End the server using **Ctrl+C**. To deactivate an activated environment, run `deactivate`.

### Expected dashboard and input files

The tested current dashboard has eight pages: Overview; Occupation Explorer; Skill Gap Analysis; Priority Recommendations; ML Evaluation; ML-Assisted Decision Support; Personalised Skill Gaps; and SHAP Explainability. It loads **precomputed artifacts** from `data/processed/`, including the saved classifier and SHAP outputs. If these files are not available, the corresponding pages cannot display their full results; merely installing the packages does not regenerate the data.

**Cross-platform notes:** Use forward slashes (`data/processed/...`) in Python scripts via `pathlib` wherever possible; Windows terminal commands above use native backslashes. Avoid relying on shell-specific `cp`, `ls`, `source`, or `curl` commands in Windows instructions. If Windows package installation fails because a compatible binary wheel is unavailable, check the package's supported Python versions and use an appropriate compatible environment instead of changing the project's saved model files.

### Running research scripts (optional)

After the environment is activated and the corresponding input files exist, these commands work in PowerShell, CMD, and macOS/Linux terminals:

```text
python src/explain_model_shap_v7.py
python src/compare_lexical_semantic_v1.py
```

The SHAP script uses the saved model and dataset; the semantic comparison can download the pretrained `sentence-transformers/all-MiniLM-L6-v2` model on first run. Neither operation should overwrite validated original nested-CV metrics. For the full script-by-script data workflow, see the Core workflow table above.

## Validated integration results

| Measure | Recorded result |
|---|---:|
| Target O*NET occupations | 5 |
| O*NET transferable skill concepts | 35 |
| O*NET target occupation–skill observations | 175 |
| Definition-reviewed ESCO candidate relationships | 350 |
| Approved skill-concept equivalences | 8 |
| Approved occupation–skill links | 40 |
| Approved links observed in the processed NYC matrix | 10 |
| ESCO–NYC demand matrix rows | 8,734 |
| Prioritized candidate skill gaps | 35 (4 High, 10 Moderate, 21 Emerging) |

Only **8 of 35** reviewed O*NET skill concepts were approved as direct equivalents. The other mappings remain partial, rejected, or unresolved. The **10/40** observed match rate describes coverage in the processed NYC evidence matrix, **not** the prevalence of real-world shortages.

## Machine learning

The original task predicts whether an occupation–skill relationship was **observed in the processed NYC job-posting dataset**. The model is a TF-IDF + Logistic Regression classifier. The recorded occupation-grouped nested cross-validation results were:

| Metric | Recorded score |
|---|---:|
| F1 | 0.4872 |
| ROC-AUC | 0.7011 |
| Average precision | 0.4806 |
| Precision | 0.3811 |
| Recall | 0.6751 |

These are from the **earlier nested grouped-CV study**, **not** from the separate fixed-parameter semantic experiment. The latter should not be presented as a direct replication of these scores.

### Additional controlled representation comparison

Four representations were compared using five shared occupation-grouped folds (2,656 rows, 190 occupations) and a fixed 0.5 classification threshold:

| Configuration | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|
| TF-IDF, occupation + skill | 0.2320 | 0.6970 | 0.4796 |
| **TF-IDF, skill only** | **0.2503** | **0.7090** | **0.4927** |
| Frozen MiniLM embeddings, occupation + skill | 0.1058 | 0.6629 | 0.4168 |
| Frozen MiniLM embeddings, skill only | 0.1748 | 0.6928 | 0.4463 |

In this tested configuration, TF-IDF outperformed the frozen sentence-embedding alternative; removing occupation text improved the reported scores. This does not establish that semantic representations always underperform.

### Threshold sensitivity

A further **exploratory** sweep over the saved out-of-fold probabilities found larger F1 scores at thresholds below 0.5 (approximately 0.20–0.25, depending on model). These maxima were chosen using the evaluation labels themselves and **must not be represented as unbiased held-out estimates**. The 0.5-threshold table above remains the primary controlled comparison.

To reproduce the comparison, when the later scripts and inputs are present:

```bash
python src/compare_lexical_semantic_v1.py
```

Its outputs are saved in `data/processed/semantic_comparison_v1/`, including `comparison_summary.csv`, `fold_metrics.csv`, `oof_predictions.csv`, and `experiment_manifest.json`.

## SHAP explainability

The saved classifier was explained without retraining. The recorded SHAP run used **100 examples**, **150 background examples** and **3,588 TF-IDF features**, with a maximum additivity error of approximately **8.88 × 10⁻¹⁶** in log-odds space.

```bash
python src/explain_model_shap_v7.py
```

Expected files in `data/processed/shap_v7/`:

- `global_feature_importance.csv`
- `global_shap_importance.png`
- `local_predictions.csv`
- `local_feature_contributions.csv`
- `shap_validation.json`

Global explanations showed relatively high importance for occupation-related terms such as `civil`, `manager`, and `environmental`. This is a potential feature-reliance limitation, not proof of causality. The dashboard displays predicted probabilities, the class using a documented **50% threshold**, recorded labels, and signed local contributions.

## Repository layout

```text
.
├── app.py
├── README.md
├── requirements.txt
├── src/                 # Processing, analytics, models, mapping and explanation scripts
├── data/
│   ├── raw/             # Original sources and later O*NET definition reference
│   └── processed/       # Analytical outputs, reviewed crosswalks, models and results
└── figures/             # Evaluation and paper figures
```

This layout reflects the original repository. Later experiments add versioned folders beneath `data/processed/` and scripts under `src/`. Keep these paths unchanged unless code references are also updated and tested.

## Reproducibility and publication notes

1. Preserve raw data, approved review files, model artifacts, fold-level predictions, SHAP exports, validation JSONs and figures. They establish the provenance of the reported results.
2. Do not publish `.venv/`, `__pycache__/`, `.DS_Store`, macOS ZIP metadata or a zipped `.git/` directory. These are environment or packaging artifacts, not research evidence.
3. Review all datasets for redistribution permission and privacy before pushing them publicly. If a dataset is excluded, document how an authorized user can obtain it and which derived artifacts are required to run the dashboard.
4. Avoid publishing passwords, access tokens, `.env` files or Streamlit secrets.
5. Validate that the working `app.py`, all required model paths, and the reported metrics agree with your final local checkout.

## Limitations

- NYC evidence is based on a dated snapshot, not a live feed.
- O*NET–ESCO direct skill mapping coverage is limited to eight reviewed equivalences.
- Missing observation in the processed postings dataset does not establish that employers do not require a skill.
- The model predicts observation status, not workforce scarcity, proficiency, or causal skill demand.
- SHAP explains predictions of the saved model; it does not validate those predictions or establish causation.
- The semantic-comparison threshold sweep is exploratory, and the original nested-CV study uses a different evaluation design.
- The application is a research prototype and has not undergone comprehensive security, user, or production-reliability testing.

## Attribution

Data sources: [O*NET Resource Center](https://www.onetcenter.org/), [ESCO](https://esco.ec.europa.eu/), and the historical NYC job-posting snapshot used for this study. Add the applicable official license/attribution text and dataset provenance before public release.

## Research status

The defined five-layer academic implementation and its additional representation-comparison experiment are complete within the validated mapping and dataset scope. Public-repository packaging and final paper documentation should preserve the distinctions above.
