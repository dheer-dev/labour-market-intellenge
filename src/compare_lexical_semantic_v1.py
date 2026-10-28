"""Four-way lexical versus semantic model comparison using shared occupation-grouped CV.

Run from project root: python src/compare_lexical_semantic_v1.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

COLUMNS = ["occupation_uri", "occupation_label", "skill_label", "skill_type", "relation_type", "market_observed"]


def metrics(y, pred, prob):
    return dict(f1=float(f1_score(y,pred,zero_division=0)), precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)), roc_auc=float(roc_auc_score(y,prob)) if len(set(y))==2 else None,
                average_precision=float(average_precision_score(y,prob)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed/model_dataset_occupation_aware.csv')
    p.add_argument('--output-dir',default='data/processed/semantic_comparison_v1')
    p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--folds',type=int,default=5)
    a=p.parse_args()
    df=pd.read_csv(a.data).replace({np.nan:''})
    missing=[x for x in COLUMNS if x not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    df['market_observed']=pd.to_numeric(df.market_observed,errors='raise').astype(int)
    if not set(df.market_observed.unique()).issubset({0,1}): raise ValueError('Labels must be binary.')
    groups=df.occupation_uri.astype(str).to_numpy(); y=df.market_observed.to_numpy()
    occupation=df.occupation_label.astype(str)
    skill=(df.skill_label.astype(str)+' '+df.skill_type.astype(str)+' '+df.relation_type.astype(str))
    both=(occupation+' '+skill)
    text={'lexical_full':both.tolist(),'lexical_skill_only':skill.tolist(), 'semantic_full':both.tolist(), 'semantic_skill_only':skill.tolist()}
    print('Loading sentence embedding model:',a.model,flush=True)
    from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer(a.model)
    embeds={}
    for name in ('semantic_full','semantic_skill_only'):
        print('Encoding:',name,flush=True)
        embeds[name]=encoder.encode(text[name],batch_size=32,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    splits=list(StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
    output=Path(a.output_dir); output.mkdir(parents=True,exist_ok=True)
    folds=[]; preds=[]
    for idx,(train,test) in enumerate(splits,1):
        if set(groups[train])&set(groups[test]): raise AssertionError('Occupation leakage in fold')
        for name in text:
            clf=LogisticRegression(max_iter=1500,random_state=42)
            if name.startswith('lexical_'):
                pipeline=make_pipeline(TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_df=0.95),clf)
                pipeline.fit([text[name][i] for i in train],y[train])
                proba=pipeline.predict_proba([text[name][i] for i in test])[:,1]
            else:
                clf.fit(embeds[name][train],y[train]); proba=clf.predict_proba(embeds[name][test])[:,1]
            label=(proba>=0.5).astype(int)
            folds.append({'model':name,'fold':idx,'train_rows':len(train),'test_rows':len(test),**metrics(y[test],label,proba)})
            preds.extend({'model':name,'fold':idx,'row_index':int(j),'occupation_uri':groups[j], 'actual':int(y[j]),'probability':float(prob),'predicted':int(pr)} for j,prob,pr in zip(test,proba,label))
        print(f'Completed fold {idx}/{a.folds}',flush=True)
    fold_df=pd.DataFrame(folds); pred_df=pd.DataFrame(preds)
    fold_df.to_csv(output/'fold_metrics.csv',index=False)
    pred_df.to_csv(output/'oof_predictions.csv',index=False)
    summaries=[]
    for name,g in pred_df.groupby('model',sort=False):
        s=metrics(g.actual.to_numpy(),g.predicted.to_numpy(),g.probability.to_numpy()); summaries.append({'model':name,'rows':len(g),**s})
    pd.DataFrame(summaries).to_csv(output/'comparison_summary.csv',index=False)
    manifest={'dataset':a.data,'rows':len(df),'occupations':len(set(groups)),'positive':int(sum(y)),'folds':a.folds,
              'semantic_model':a.model,'embedding_normalized':True,'threshold':0.5,
              'comparison':'Same stratified occupation-grouped outer folds; fixed hyperparameters; preprocessing TF-IDF fit inside training folds; embedding model frozen.',
              'caveat':'This is 5-fold grouped CV, not nested hyperparameter-tuned CV. Do not directly equate scores with prior nested-CV metrics.'}
    (output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False)); print('Saved results to',output)

if __name__=='__main__': main()
