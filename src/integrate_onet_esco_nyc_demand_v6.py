#!/usr/bin/env python3
"""Read-only approved O*NET -> ESCO -> NYC demand overlay; no model retraining."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onet-join', default='data/processed/onet_esco_v5_skill_crosswalk/occupation_skill_join_v5.csv')
    p.add_argument('--demand-matrix', default='data/processed/occupation_skill_demand_matrix.csv')
    p.add_argument('--output-dir', default='data/processed/onet_esco_v6_demand')
    a = p.parse_args()
    df = pd.read_csv(a.onet_join, dtype={'element_id':'string', 'onet_soc_code':'string'}, keep_default_na=True)
    demand = pd.read_csv(a.demand_matrix)
    required_onet = {'target_occupation','esco_occupation_uri','onet_soc_code','element_id','onet_importance','onet_level','approved_esco_skill_uri','skill_mapping_status'}
    required_dem = {'occupation_uri','skill_uri','skill_posting_count','postings_with_skills','skill_prevalence','skill_prevalence_percent'}
    for name, actual, required in [('O*NET join',df,required_onet),('demand matrix',demand,required_dem)]:
        missing = required-set(actual.columns)
        if missing: raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if df.duplicated(['target_occupation','element_id']).any():
        raise ValueError('O*NET input contains duplicate target_occupation + element_id rows')
    if demand.duplicated(['occupation_uri','skill_uri']).any():
        raise ValueError('Demand matrix has duplicate occupation_uri + skill_uri keys; refusing many-to-many join')
    if df['esco_occupation_uri'].isna().any() or df['element_id'].isna().any():
        raise ValueError('Input occupation/skill keys must be non-null')
    linked = df['approved_esco_skill_uri'].notna() & df['approved_esco_skill_uri'].astype(str).str.strip().ne('')
    if linked.sum() and not df.loc[linked,'skill_mapping_status'].astype(str).str.contains('HUMAN_APPROVED', regex=False).all():
        raise ValueError('Some ESCO URIs are present without explicit human-approved mapping status')
    # Avoid accidental column name collisions and preserve original row order.
    dcols = ['occupation_uri','skill_uri','skill_label','skill_posting_count','postings_with_skills','skill_prevalence','skill_prevalence_percent']
    dsub = demand[dcols].rename(columns={'skill_label':'nyc_esco_skill_label',
        'skill_posting_count':'nyc_skill_posting_count','postings_with_skills':'nyc_postings_with_skills',
        'skill_prevalence':'nyc_skill_prevalence','skill_prevalence_percent':'nyc_skill_prevalence_percent'})
    joined = df.merge(dsub, how='left', left_on=['esco_occupation_uri','approved_esco_skill_uri'],
        right_on=['occupation_uri','skill_uri'], validate='many_to_one',sort=False)
    joined['nyc_evidence_status'] = 'NO_APPROVED_EQUIVALENCE'
    joined.loc[linked,'nyc_evidence_status'] = 'APPROVED_EQUIVALENCE_NO_OBSERVED_MATCH'
    observed = linked & joined['nyc_skill_posting_count'].notna()
    joined.loc[observed,'nyc_evidence_status'] = 'OBSERVED_IN_NYC_DEMAND_MATRIX'
    if len(joined)!=len(df):raise AssertionError('Output row count changed')
    coverage = joined.groupby('target_occupation',sort=True).agg(
        onet_skill_rows=('element_id','size'),
        approved_esco_links=('approved_esco_skill_uri',lambda s:int(s.notna().sum())),
        nyc_observed_skill_links=('nyc_evidence_status',lambda s:int((s=='OBSERVED_IN_NYC_DEMAND_MATRIX').sum())),
        approved_links_without_nyc_match=('nyc_evidence_status',lambda s:int((s=='APPROVED_EQUIVALENCE_NO_OBSERVED_MATCH').sum()))
    ).reset_index()
    coverage['nyc_observation_rate_among_approved'] = (coverage['nyc_observed_skill_links']/coverage['approved_esco_links'].replace(0,float('nan'))).round(4)
    if int(coverage['nyc_observed_skill_links'].sum()) != int(observed.sum()): raise AssertionError('Coverage counts inconsistent')
    out = Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out/'occupation_skill_nyc_evidence_v6.csv',index=False)
    coverage.to_csv(out/'occupation_demand_coverage_v6.csv',index=False)
    report = {
      'input_onet_occupation_skill_rows':int(len(df)),
      'output_rows':int(len(joined)),
      'unique_target_occupations':int(df['target_occupation'].nunique()),
      'unique_onet_skills':int(df['element_id'].nunique()),
      'approved_occupation_skill_links':int(linked.sum()),
      'approved_links_observed_in_nyc_matrix':int(observed.sum()),
      'approved_links_with_no_nyc_record':int((linked & ~observed).sum()),
      'onet_rows_without_approved_equivalence':int((~linked).sum()),
      'nyc_demand_matrix_rows':int(len(demand)),
      'matching_rule':'Exact ESCO occupation URI AND exact approved ESCO skill URI',
      'interpretation':'Unmatched means not present in this processed NYC dataset, NOT a missing labour-market skill or worker skill gap.',
      'outputs_written_only_to':str(out),
      'baseline_modified':False
    }
    (out/'demand_integration_validation_v6.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print('\nBy occupation:\n'+coverage.to_string(index=False))

if __name__ == '__main__': main()
