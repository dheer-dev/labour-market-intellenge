"""Read-only O*NET→ESCO candidate retrieval; never approves mappings.
Run from repo root: python src/propose_onet_esco_matches_v2.py
Optional semantic model: python src/propose_onet_esco_matches_v2.py --semantic-model intfloat/multilingual-e5-small
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = DATA / 'onet_esco_v2_review'

def clean(x):
    if pd.isna(x): return ''
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(x).casefold())).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--semantic-model', default=None, help='Optional sentence-transformers model; downloads weights on first run')
    ap.add_argument('--top-k', type=int, default=10)
    args=ap.parse_args()
    if args.top_k < 1: ap.error('--top-k must be >= 1')
    onet=pd.read_csv(DATA/'onet_target_occupation_skills.csv',dtype={'element_id':str})
    esco=pd.read_csv(DATA/'esco_skills.csv')
    old=pd.read_csv(DATA/'onet_esco_skill_mapping_initial_review.csv',dtype={'onet_element_id':str})
    needed_onet={'element_id','element_name','onet_skill_category'}
    needed_esco={'skill_uri','skill_label','skill_aliases','skill_description'}
    needed_old={'onet_element_id','review_decision','reviewed_esco_skill_uri'}
    for df,need,label in [(onet,needed_onet,'O*NET'),(esco,needed_esco,'ESCO'),(old,needed_old,'existing review')]:
        if need-set(df.columns): raise ValueError(f'{label} missing {sorted(need-set(df.columns))}')
    onet=onet[['element_id','element_name','onet_skill_category']].drop_duplicates(subset=['element_id']).fillna('')
    esco=esco.dropna(subset=['skill_uri','skill_label']).drop_duplicates('skill_uri').reset_index(drop=True)
    old=old.drop_duplicates('onet_element_id')
    if onet['element_id'].nunique()!=len(onet): raise ValueError('duplicate O*NET IDs')
    esco_label=esco.skill_label.fillna('').map(clean).tolist()
    esco_alias=esco.skill_aliases.fillna('').map(clean).tolist()
    esco_desc=esco.skill_description.fillna('').map(clean).tolist()
    onet_label=onet.element_name.fillna('').map(clean).tolist()
    # Score name/aliases separately from descriptions to reduce vague description matches.
    esco_search=[f'{l} {a}' for l,a in zip(esco_label,esco_alias)]
    matrices=[]
    for mode,weight in [('word',0.55),('char',0.45)]:
        vect=TfidfVectorizer(ngram_range=(1,2) if mode=='word' else (3,5), analyzer='word' if mode=='word' else 'char_wb',sublinear_tf=True)
        combined=vect.fit_transform(esco_search+onet_label)
        matrices.append((weight, cosine_similarity(combined[len(esco):],combined[:len(esco)])))
    lexical=sum(w*m for w,m in matrices)
    # Exact label/alias match promotes retrieval only; no automatic equivalence.
    aliases=[set([l]+[clean(a) for a in str(raw).split('\n') if clean(a)]) for l,raw in zip(esco_label,esco.skill_aliases.fillna(''))]
    for i,l in enumerate(onet_label):
        for j,a in enumerate(aliases):
            if l and l in a: lexical[i,j]=max(lexical[i,j],1.0)
    semantic=None
    if args.semantic_model:
        try: from sentence_transformers import SentenceTransformer
        except ImportError as e: raise SystemExit('Install optional model dependency: pip install sentence-transformers') from e
        m=SentenceTransformer(args.semantic_model)
        # E5 models require query:/passage: prefixes; otherwise use generic input.
        ise5='e5' in args.semantic_model.lower()
        q=[('query: ' if ise5 else '')+x for x in onet.element_name.astype(str)]
        docs=[('passage: ' if ise5 else '')+str(l)+'; aliases: '+str(a)+'; description: '+str(d)[:250] for l,a,d in zip(esco.skill_label.fillna(''),esco.skill_aliases.fillna(''),esco.skill_description.fillna(''))]
        qemb=m.encode(q,normalize_embeddings=True,show_progress_bar=True,batch_size=64)
        demb=m.encode(docs,normalize_embeddings=True,show_progress_bar=True,batch_size=64)
        semantic=np.asarray(qemb)@np.asarray(demb).T
        rank_score=0.3*lexical+0.7*np.clip(semantic,0,1)
    else:
        rank_score=lexical
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,row in onet.reset_index(drop=True).iterrows():
        inds=np.argsort(-rank_score[i])[:args.top_k]
        old_row=old.loc[old.onet_element_id==row.element_id]
        old_status=old_row.iloc[0].review_decision if len(old_row) else 'NOT_PREVIOUSLY_REVIEWED'
        existing_uri=old_row.iloc[0].reviewed_esco_skill_uri if len(old_row) else ''
        for rank,j in enumerate(inds,1):
            r=esco.iloc[int(j)]
            records.append({'onet_element_id':row.element_id,'onet_skill_label':row.element_name,'onet_skill_category':row.onet_skill_category,'previous_review_decision':old_status,'previous_approved_esco_uri':existing_uri if pd.notna(existing_uri) else '', 'candidate_rank':rank,'esco_skill_uri':r.skill_uri,'esco_skill_label':r.skill_label,'esco_skill_type':r.get('skill_type',''),'esco_skill_aliases':r.get('skill_aliases',''),'esco_skill_description':r.get('skill_description',''),'lexical_score':round(float(lexical[i,j]),4),'semantic_score':round(float(semantic[i,j]),4) if semantic is not None else '', 'retrieval_score':round(float(rank_score[i,j]),4),'proposed_review_decision':'PENDING_HUMAN_REVIEW','review_notes':''})
    result=pd.DataFrame(records)
    result.to_csv(OUT/'candidate_shortlist.csv',index=False)
    summary=pd.DataFrame([{'onet_skills':len(onet),'esco_skills_searched':len(esco),'candidate_rows':len(result),'existing_approved':int((old.review_decision=='APPROVED').sum()),'semantic_model':args.semantic_model or 'none (TF-IDF retrieval only)','mapping_automatically_approved':0}])
    summary.to_csv(OUT/'run_summary.csv',index=False)
    print('\nO*NET–ESCO candidate retrieval complete (NO existing mapping files modified).')
    print(summary.to_string(index=False))
    print('\nSaved:',OUT/'candidate_shortlist.csv')
    print('\nTop 3 per O*NET skill:')
    print(result[result.candidate_rank<=3][['onet_skill_label','candidate_rank','esco_skill_label','lexical_score','semantic_score']].to_string(index=False,max_rows=120))

if __name__=='__main__': main()
