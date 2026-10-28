#!/usr/bin/env python3
"""SHAP explanation for an existing TF-IDF + binary LogisticRegression pipeline.
No fitting, resampling, or rewriting baseline model/data. Explanations are in log-odds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]
TEXT_COLUMNS = ['occupation_label','skill_label','skill_type','relation_type']

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='data/processed/market_observation_model_grouped.joblib')
    p.add_argument('--dataset', default='data/processed/model_dataset_occupation_aware.csv')
    p.add_argument('--output-dir', default='data/processed/shap_v7')
    p.add_argument('--background-size', type=int, default=150)
    p.add_argument('--explain-size', type=int, default=100)
    p.add_argument('--top-features', type=int, default=25)
    args = p.parse_args()
    if args.background_size < 1 or args.explain_size < 1:
        p.error('Sample sizes must be positive')
    model_path, dataset_path = Path(args.model), Path(args.dataset)
    for f in (model_path,dataset_path):
        if not f.exists():
            raise FileNotFoundError(f'Missing {f}. See README for alternate paths.')
    try:
        import shap
    except ImportError as e:
        raise SystemExit('Missing SHAP. Run: python -m pip install shap') from e
    df = pd.read_csv(dataset_path)
    missing = [c for c in TEXT_COLUMNS if c not in df]
    if missing:
        raise ValueError(f'Dataset lacks model input columns: {missing}')
    # Reproduce original training scripts: .astype(str) turns missing values into 'nan'.
    if len(df) < 2:
        raise ValueError('Need at least two dataset rows')
    model = joblib.load(model_path)  # Only load models you trust
    if not hasattr(model,'named_steps') or set(['tfidf','classifier']) - set(model.named_steps):
        raise ValueError('Expected fitted pipeline steps named tfidf and classifier')
    vec, clf = model.named_steps['tfidf'], model.named_steps['classifier']
    if not hasattr(clf,'coef_') or clf.coef_.shape[0] != 1:
        raise ValueError('Expected binary linear Logistic Regression classifier')
    if list(clf.classes_) != [0,1]:
        raise ValueError(f'Unexpected classes {clf.classes_}; require [0,1]')
    text = (
    df[TEXT_COLUMNS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    )
    # Deterministic samples; explanations are for saved model, not OOF model evaluations.
    bg_idx = np.random.default_rng(42).choice(len(df), min(args.background_size,len(df)), replace=False)
    ex_idx = np.random.default_rng(43).choice(len(df), min(args.explain_size,len(df)), replace=False)
    X_bg = vec.transform(text.iloc[bg_idx]); X_ex = vec.transform(text.iloc[ex_idx])
    features = vec.get_feature_names_out()
    if X_bg.shape[1] != len(features) or clf.coef_.shape[1] != len(features):
        raise ValueError('Feature dimensions do not match saved model')
    # Interventional linear SHAP with sampled background. Results are log-odds (linear score).
    explainer = shap.LinearExplainer(clf, X_bg)
    sv = explainer(X_ex)
    values = np.asarray(sv.values)
    if values.ndim != 2:
        raise ValueError(f'Expected 2-D binary SHAP values, got {values.shape}')
    base = np.broadcast_to(np.asarray(sv.base_values).reshape(-1), (len(ex_idx),))
    decision = clf.decision_function(X_ex)
    additive_error = np.max(np.abs(base + values.sum(axis=1) - decision))
    if not np.isfinite(additive_error) or additive_error > 1e-5:
        raise AssertionError(f'SHAP additivity check failed: {additive_error}')
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    score = np.mean(np.abs(values), axis=0)
    glob_order = np.argsort(-score)[:args.top_features]
    glob = pd.DataFrame({'feature':features[glob_order], 'mean_abs_shap_log_odds':score[glob_order],
                         'model_coefficient':clf.coef_[0,glob_order]})
    glob.to_csv(out/'global_feature_importance.csv',index=False)
    info_cols = [c for c in ['occupation_uri','occupation_label','skill_uri','skill_label','market_observed'] if c in df]
    records=[]; local=[]
    pred_proba = clf.predict_proba(X_ex)[:,1]
    Xcsr = X_ex.tocsr()
    for j,original_idx in enumerate(ex_idx):
        base_row={c:df.iloc[original_idx][c] for c in info_cols}
        records.append({'source_row_index':int(original_idx),**base_row,
                        'observed_probability':float(pred_proba[j]),
                        'base_log_odds':float(base[j]), 'predicted_log_odds':float(decision[j]),
                        'sum_shap_log_odds':float(values[j].sum())})
        # Explain contributions for tokens PRESENT in this text. Baseline-absence contributions
        # still count in global SHAP and additive sum, but are not shown in local token table.
        present = Xcsr[j].indices
        order = present[np.argsort(-np.abs(values[j,present]))[:args.top_features]]
        for k in order:
            local.append({'source_row_index':int(original_idx),'feature':features[k],
                          'tfidf_value':float(Xcsr[j,k]),'shap_log_odds':float(values[j,k]),
                          'direction':'towards observed' if values[j,k]>0 else 'towards not observed'})
    pd.DataFrame(records).to_csv(out/'local_predictions.csv',index=False)
    pd.DataFrame(local,columns=['source_row_index','feature','tfidf_value','shap_log_odds','direction']).to_csv(out/'local_feature_contributions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    g=glob.iloc[::-1];ax.barh(g['feature'],g['mean_abs_shap_log_odds']);ax.set_xlabel('Mean |SHAP| (log-odds)')
    ax.set_title('TF-IDF model: global SHAP feature importance');fig.tight_layout()
    fig.savefig(out/'global_shap_importance.png',dpi=160);plt.close(fig)
    report={'model_file':str(model_path),'dataset_file':str(dataset_path),
            'samples_explained':len(ex_idx),'background_samples':len(bg_idx),
            'feature_count':len(features),'max_additivity_error':float(additive_error),
            'explanation_units':'log-odds of market_observed=1',
            'interpretation':'Explains predictions by saved model, not causal skill gaps; sampled dataset explanations are not out-of-fold metrics.',
            'no_training_performed':True, 'input_files_unchanged':True}
    (out/'shap_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Outputs:',out)

if __name__=='__main__':run()
