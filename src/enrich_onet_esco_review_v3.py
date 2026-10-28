#!/usr/bin/env python3
"""Read-only evidence enrichment. Does not approve mappings or mutate baselines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

DEFAULT_ONET_URL='https://www.onetcenter.org/dl_files/database/db_31_0_csv/Content%20Model%20Reference.csv'

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def unique_lookup(df, key, fields):
    if df[key].duplicated().any():
        raise ValueError(f'Duplicate identifiers in {key}')
    return df[[key]+fields]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',default='data/processed/onet_esco_v2_review/candidate_shortlist.csv')
    p.add_argument('--preliminary-review',default='data/processed/onet_esco_preliminary_review/preliminary_concept_review.csv')
    p.add_argument('--esco-skills',default='data/raw/skills_en.csv')
    p.add_argument('--onet-definitions',default='data/raw/content_model_reference.csv',help='Official O*NET Content Model Reference CSV or TSV; missing file is reported, not filled in')
    p.add_argument('--occupation-attributes',default='data/processed/onet_esco_v2_integration/onet_esco_occupation_skill_attributes_v2.csv')
    p.add_argument('--out',default='data/processed/onet_esco_v3_review')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    candidates=read_csv(a.candidates)
    reviews=read_csv(a.preliminary_review)
    esco=read_csv(a.esco_skills)
    required={'onet_element_id','esco_skill_uri','candidate_rank'}
    if not required.issubset(candidates):raise ValueError(f'Missing candidate columns: {required-set(candidates)}')
    if 'conceptUri' not in esco:raise ValueError('ESCO conceptUri column missing')
    assert not candidates.duplicated(['onet_element_id','esco_skill_uri']).any(), 'Duplicate candidate relationships'
    assert len(reviews)==reviews.onet_element_id.nunique(),'Review requires one row per O*NET concept'
    esco = esco.rename(columns={'conceptUri':'esco_skill_uri','preferredLabel':'official_esco_label','description':'official_esco_description','altLabels':'official_esco_aliases','skillType':'official_esco_skill_type'})
    esco_fields=[c for c in ['official_esco_label','official_esco_description','official_esco_aliases','official_esco_skill_type'] if c in esco.columns]
    esco=esco[['esco_skill_uri']+esco_fields].drop_duplicates()
    esco=unique_lookup(esco,'esco_skill_uri',esco_fields)
    defs_path=Path(a.onet_definitions)
    has_defs=defs_path.exists()
    if has_defs:
        sep='\t' if defs_path.suffix.lower()=='.txt' else ','
        defs=pd.read_csv(defs_path,sep=sep,dtype=str,keep_default_na=False,encoding='utf-8-sig')
        expected={'Element ID','Element Name','Description'}
        if not expected.issubset(defs):raise ValueError(f'O*NET definitions missing fields {expected-set(defs)}')
        defs=defs.rename(columns={'Element ID':'onet_element_id','Element Name':'official_onet_label','Description':'official_onet_description'})
        defs=unique_lookup(defs,'onet_element_id',['official_onet_label','official_onet_description'])
    else:
        defs=pd.DataFrame(columns=['onet_element_id','official_onet_label','official_onet_description'])
    enriched=candidates.merge(esco,on='esco_skill_uri',how='left',validate='many_to_one')
    enriched=enriched.merge(defs,on='onet_element_id',how='left',validate='many_to_one').fillna('')
    # Preliminary AI recommendations are not independent human approval.
    reviews=reviews[[c for c in ['onet_element_id','suggested_esco_uri','ai_preliminary_decision','ai_preliminary_reason','human_validation_decision','human_review_notes'] if c in reviews.columns]].rename(columns={'suggested_esco_uri':'preliminary_suggested_esco_uri'})
    enriched=enriched.merge(reviews,on='onet_element_id',how='left',validate='many_to_one').fillna('')
    enriched['is_preliminary_suggestion']=(enriched.esco_skill_uri==enriched.preliminary_suggested_esco_uri)
    enriched['official_onet_definition_available']=enriched.official_onet_description.str.strip().ne('')
    enriched['official_esco_definition_available']=enriched.official_esco_description.str.strip().ne('')
    enriched['review_decision_v3']='PENDING_HUMAN_REVIEW'
    enriched['review_notes_v3']=''
    enriched.to_csv(out/'enriched_candidates_v3.csv',index=False)
    # One row per O*NET skill with human-review fields intentionally unfilled.
    idx=enriched.sort_values(['onet_element_id','candidate_rank'],key=lambda s:pd.to_numeric(s,errors='coerce') if s.name=='candidate_rank' else s).groupby('onet_element_id',sort=True).head(1)
    summary_cols=['onet_element_id','onet_skill_label','official_onet_description','official_onet_definition_available','preliminary_suggested_esco_uri','ai_preliminary_decision','ai_preliminary_reason']
    one=idx[summary_cols].copy();one['review_decision_v3']='PENDING_HUMAN_REVIEW';one['review_notes_v3']=''
    one.to_csv(out/'concept_review_v3.csv',index=False)
    occupation_rows=None
    if Path(a.occupation_attributes).exists():
        attrs=read_csv(a.occupation_attributes)
        occupation_rows=len(attrs)
        attrs.merge(one[['onet_element_id','official_onet_description']],left_on='element_id',right_on='onet_element_id',how='left',validate='many_to_one').drop(columns='onet_element_id').to_csv(out/'occupation_skill_attributes_enriched_v3.csv',index=False)
    stats={'onet_skills':int(one.onet_element_id.nunique()),'candidate_rows':int(len(enriched)),'esco_descriptions_present':int(enriched.official_esco_definition_available.sum()),'candidate_rows_with_official_onet_definitions':int(enriched.official_onet_definition_available.sum()),'onet_definition_file_found':has_defs,'occupation_rows_preserved':occupation_rows,'auto_approvals':0,'original_crosswalk_modified':False,'baseline_gap_catalogue_modified':False,'official_onet_definitions_source':str(defs_path) if has_defs else 'MISSING — fetch official Content Model Reference'}
    (out/'enrichment_validation_v3.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
    print(f'Outputs: {out.resolve()}')

if __name__=='__main__':main()
