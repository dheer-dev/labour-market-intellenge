#!/usr/bin/env python3
"""Export human-approved O*NET-ESCO equivalences and safe occupation-skill joins.
Never promotes AI proposals, semantic scores, or pending decisions to approvals.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

DEFAULT_REVIEW='data/processed/onet_esco_v4_review/definition_based_skill_review_v4.csv'
DEFAULT_ATTR='data/processed/onet_esco_v2_integration/onet_esco_occupation_skill_attributes_v2.csv'
DEFAULT_OUT='data/processed/onet_esco_v5_skill_crosswalk'


def load(p):
    p=Path(p)
    if not p.is_file(): raise SystemExit(f'Missing input file: {p}. Supply the correct path via CLI.')
    return pd.read_csv(p,dtype=str,keep_default_na=False,encoding='utf-8-sig').fillna('')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',default=DEFAULT_REVIEW)
    ap.add_argument('--occupation-attributes',default=DEFAULT_ATTR)
    ap.add_argument('--out',default=DEFAULT_OUT)
    args=ap.parse_args()
    review=load(args.review); attr=load(args.occupation_attributes)
    rcols=['onet_element_id','onet_skill_label','official_onet_description','suggested_esco_uri','suggested_esco_label','official_esco_description','esco_skill_type','human_review_status','human_decision','human_notes']
    acols=['element_id','target_occupation','onet_soc_code','esco_occupation_uri','onet_importance','onet_level']
    for name,d,cols in [('review',review,rcols),('occupation attributes',attr,acols)]:
        missing=set(cols)-set(d.columns)
        if missing: raise SystemExit(f'{name} missing columns: {sorted(missing)}')
    if review.onet_element_id.duplicated().any(): raise SystemExit('Duplicate O*NET concepts in review file; one concept per row required.')
    if not review.onet_element_id.is_unique: raise SystemExit('Review IDs must be unique.')
    review['human_review_status']=review.human_review_status.str.strip().str.upper()
    review['human_decision']=review.human_decision.str.strip().str.upper()
    allowed_status={'PENDING','REVIEWED'}
    allowed_decision={'','APPROVE','REJECT','PARTIAL','UNRESOLVED'}
    bad=review[~review.human_review_status.isin(allowed_status)|~review.human_decision.isin(allowed_decision)]
    if not bad.empty: raise SystemExit('Invalid human_review_status or human_decision. Allowed status PENDING/REVIEWED; decisions APPROVE/REJECT/PARTIAL/UNRESOLVED (or blank).')
    invalid=review[(review.human_review_status=='PENDING') & (review.human_decision!='')]
    if not invalid.empty: raise SystemExit('Decisions set while status PENDING. Set human_review_status=REVIEWED to commit each decision.')
    invalid=review[(review.human_review_status=='REVIEWED') & (review.human_decision=='')]
    if not invalid.empty: raise SystemExit('REVIEWED requires explicit human_decision.')
    approvals=review[(review.human_review_status=='REVIEWED') & (review.human_decision=='APPROVE')].copy()
    incomplete=approvals[(approvals.suggested_esco_uri.str.strip()=='') | (approvals.official_onet_description.str.strip()=='') | (approvals.official_esco_description.str.strip()=='') | (approvals.human_notes.str.strip()=='')]
    if not incomplete.empty: raise SystemExit('APPROVE requires URI, both definitions, and human_notes documenting the decision. IDs: '+', '.join(incomplete.onet_element_id.tolist()))
    wrongtype=approvals[~approvals.esco_skill_type.str.strip().str.lower().isin({'skill/competence','skill'})]
    if not wrongtype.empty: raise SystemExit('Knowledge-type ESCO terms cannot be approved as equivalent skills without an explicit different workflow: '+', '.join(wrongtype.onet_element_id.tolist()))
    if approvals.suggested_esco_uri.duplicated().any(): raise SystemExit('Multiple O*NET concepts map to one ESCO URI: requires separate ambiguity review.')
    if attr.duplicated(['target_occupation','element_id']).any(): raise SystemExit('Duplicate occupation-element pairs in attributes.')
    unknown=sorted(set(attr.element_id)-set(review.onet_element_id))
    if unknown: raise SystemExit(f'O*NET attribute concepts absent from review: {unknown}')
    cross=approvals[['onet_element_id','onet_skill_label','suggested_esco_uri','suggested_esco_label','official_onet_description','official_esco_description','esco_skill_type','human_notes']].rename(columns={'suggested_esco_uri':'approved_esco_skill_uri','suggested_esco_label':'approved_esco_skill_label','human_notes':'human_approval_notes'})
    cross.insert(2,'mapping_decision','HUMAN_APPROVED_EQUIVALENCE')
    joined=attr.merge(cross,left_on='element_id',right_on='onet_element_id',how='left',validate='many_to_one',indicator=True)
    joined['skill_mapping_status']=joined['_merge'].map({'both':'HUMAN_APPROVED_EQUIVALENCE','left_only':'UNMAPPED'}).astype(str)
    joined=joined.drop(columns=['_merge'])
    pending=review[review.human_review_status=='PENDING'].copy()
    nonapproved=review[review.human_decision.isin(['REJECT','PARTIAL','UNRESOLVED'])].copy()
    cov=joined.groupby('target_occupation',dropna=False).agg(total_skills=('element_id','size'),approved_skills=('skill_mapping_status',lambda x:int((x=='HUMAN_APPROVED_EQUIVALENCE').sum()))).reset_index()
    cov['unmapped_skills']=cov.total_skills-cov.approved_skills
    summary={'reviewed_concepts':int(len(review)),'human_approved_concepts':int(len(approvals)),'pending_concepts':int(len(pending)),'reviewed_nonapproved_concepts':int(len(nonapproved)),'occupation_rows_input':int(len(attr)),'occupation_rows_output':int(len(joined)),'rows_with_approved_esco_skill':int(joined.approved_esco_skill_uri.fillna('').ne('').sum()),'unique_occupations':int(attr.target_occupation.nunique()),'baseline_modified':False,'note':'Approval is a human decision, not semantic or AI inference. Unapproved concepts remain explicitly unmapped.'}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cross.to_csv(out/'approved_skill_crosswalk_v5.csv',index=False)
    joined.to_csv(out/'occupation_skill_join_v5.csv',index=False)
    cov.to_csv(out/'occupation_mapping_coverage_v5.csv',index=False)
    pending.to_csv(out/'pending_review_v5.csv',index=False)
    nonapproved.to_csv(out/'nonapproved_review_v5.csv',index=False)
    (out/'skill_crosswalk_validation_v5.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Outputs saved to:',out)

if __name__=='__main__': main()
