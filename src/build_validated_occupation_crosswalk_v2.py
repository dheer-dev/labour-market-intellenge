"""Create a NEW versioned five-occupation O*NET↔ESCO crosswalk.

No existing mappings, skill crosswalk, or skill-gap outputs are modified.
Runs from repository root; pandas required.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

EXPECTED = {
    'civil engineer': '17-2051.00',
    'construction manager': '11-9021.00',
    'environmental engineer': '17-2081.00',
    'lawyer': '23-1011.00',
    'project manager': '13-1082.00',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reviewed', type=Path, default=Path('data/processed/onet_esco_occupation_mapping_reviewed.csv'))
    p.add_argument('--project-manager-review', type=Path, default=Path('data/processed/onet_esco_next_review/project_manager_official_review.csv'))
    p.add_argument('--out-dir', type=Path, default=Path('data/processed/onet_esco_v2_occupation'))
    args = p.parse_args()
    existing = pd.read_csv(args.reviewed, dtype=str).fillna('')
    proposals = pd.read_csv(args.project_manager_review, dtype=str).fillna('')
    required = ['target_occupation','esco_occupation_uri','esco_occupation_label','onet_soc_code','onet_occupation_title']
    for label, df in [('reviewed',existing),('project-manager-review',proposals)]:
        absent = set(required)-set(df.columns)
        if absent: raise ValueError(f'{label} missing columns: {sorted(absent)}')
    ok = existing[existing.review_status.str.upper().str.strip() == 'APPROVED'].copy()
    if len(ok) != 4: raise ValueError(f'Expected 4 existing approved occupation rows, found {len(ok)}')
    pm = proposals[(proposals.target_occupation.str.strip().str.lower()=='project manager') & (proposals.onet_soc_code.str.strip()=='13-1082.00')]
    if len(pm) != 1: raise ValueError('Expected exactly one Project Management Specialists proposal, found '+str(len(pm)))
    if pm.iloc[0].esco_occupation_uri.strip() == '': raise ValueError('Project Manager ESCO URI is missing')
    pm_row = pm.iloc[0]
    fields = required + ['review_status','review_reason','mapping_source','evidence_status']
    rows=[]
    for _,r in ok.iterrows():
        rows.append({**{k:r[k] for k in required},'review_status':'APPROVED','review_reason':r.get('review_reason',''), 'mapping_source':'original reviewed crosswalk','evidence_status':'original human review'})
    rows.append({**{k:pm_row[k] for k in required},'review_status':'APPROVED','review_reason':'General Project Manager occupational match accepted for integration; retain specialized alternatives as unapproved.', 'mapping_source':'Project Manager 2026 review proposal; user approved inclusion', 'evidence_status':'user accepted proposal; independent expert validation not claimed'})
    output=pd.DataFrame(rows,columns=fields)
    output['target_occupation']=output['target_occupation'].str.strip().str.lower()
    actual=dict(zip(output.target_occupation,output.onet_soc_code))
    if len(output)!=5 or len(set(output.target_occupation))!=5 or actual!=EXPECTED:
        raise ValueError('Incorrect set of five occupation mappings: '+repr(actual))
    if output.esco_occupation_uri.duplicated().any():raise ValueError('Duplicate ESCO URIs among five targets')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    dest=args.out_dir/'onet_esco_occupation_crosswalk_v2.csv'
    if dest.exists():raise FileExistsError(f'Refusing to overwrite existing versioned output: {dest}')
    output.to_csv(dest,index=False)
    summary={'approved_target_occupations':5,'existing_approved_occupations_preserved':4,'project_manager_onet_soc':'13-1082.00','skill_mappings_modified':0,'baseline_gap_catalogue_modified':False,'output':str(dest)}
    (args.out_dir/'crosswalk_v2_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output[['target_occupation','onet_soc_code','onet_occupation_title','review_status']].to_string(index=False))
    print('\nPASS: 5/5 occupations mapped. Original inputs unchanged.')
    print('Output:',dest)

if __name__=='__main__':main()
