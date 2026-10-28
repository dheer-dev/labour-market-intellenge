"""Read-only integration of the five approved O*NET–ESCO occupations with O*NET skills.
Run from project root: python src/build_onet_integrated_attributes_v2.py
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Repository root')
    args = parser.parse_args()
    root = Path(args.root)
    processed = root / 'data' / 'processed'
    crosswalk_path = processed / 'onet_esco_v2_occupation' / 'onet_esco_occupation_crosswalk_v2.csv'
    crosswalk = pd.read_csv(crosswalk_path, dtype=str).fillna('')
    required_cw = {'target_occupation', 'esco_occupation_uri', 'onet_soc_code', 'review_status'}
    assert required_cw <= set(crosswalk.columns), f'Missing crosswalk columns: {required_cw-set(crosswalk.columns)}'
    approved = crosswalk[crosswalk.review_status.eq('APPROVED')].copy()
    assert len(approved) == 5 and approved.target_occupation.nunique() == 5, 'Expected exactly five approved occupations'
    assert approved.onet_soc_code.nunique() == 5, 'Each target must have a distinct O*NET code'
    assert approved.esco_occupation_uri.nunique() == 5, 'Each target must have a distinct ESCO URI'

    rows=[]
    for fn,category in [('onet_essential_skills_clean.csv','Essential Skills'), ('onet_transferable_skills_clean.csv','Transferable Skills')]:
        df=pd.read_csv(processed / fn, dtype={'onet_soc_code':str,'element_id':str,'scale_id':str})
        cols={'onet_soc_code','element_id','element_name','scale_id','data_value'}
        assert cols <= set(df.columns), f'Missing in {fn}: {cols-set(df.columns)}'
        df=df[df.onet_soc_code.isin(set(approved.onet_soc_code)) & df.scale_id.isin(['IM','LV'])].copy()
        df['onet_skill_category']=category
        rows.append(df)
    observations=pd.concat(rows, ignore_index=True)
    observations['data_value']=pd.to_numeric(observations.data_value, errors='coerce')
    assert observations.data_value.notna().all(), 'Some O*NET values are invalid'
    keys=['onet_soc_code','element_id','onet_skill_category']
    dup=observations.duplicated(keys+['scale_id']).sum()
    assert dup == 0, f'Duplicate source observations at scale grain: {dup}'
    wide=observations.pivot(index=keys, columns='scale_id', values='data_value').reset_index()
    wide.columns.name=None
    wide=wide.rename(columns={'IM':'onet_importance','LV':'onet_level'})
    if 'onet_importance' not in wide: wide['onet_importance']=float('nan')
    if 'onet_level' not in wide: wide['onet_level']=float('nan')
    names=observations[keys+['element_name']].drop_duplicates()
    assert not names.duplicated(keys).any(), 'Conflicting skill labels for same O*NET concept'
    wide=wide.merge(names,on=keys,validate='one_to_one')
    integrated=approved.merge(wide,on='onet_soc_code',how='left',validate='one_to_many',indicator=True)
    assert (integrated._merge=='both').all(), 'At least one mapped occupation has no O*NET skill observations'
    integrated=integrated.drop(columns='_merge')
    assert not integrated.duplicated(['target_occupation','element_id','onet_skill_category']).any()
    assert set(integrated.target_occupation)==set(approved.target_occupation)

    output_dir=processed / 'onet_esco_v2_integration'
    output_dir.mkdir(parents=True, exist_ok=True)
    output=output_dir / 'onet_esco_occupation_skill_attributes_v2.csv'
    integrated.to_csv(output,index=False)
    summary=(integrated.groupby(['target_occupation','onet_soc_code'],as_index=False)
             .agg(skill_rows=('element_id','size'),importance_non_null=('onet_importance','count'),level_non_null=('onet_level','count')))
    summary.to_csv(output_dir/'occupation_coverage_v2.csv',index=False)
    report={'approved_occupations':len(approved),'integrated_rows':len(integrated),
            'unique_onet_concepts':int(integrated.element_id.nunique()),
            'baseline_modified':False,'skill_crosswalk_modified':False,
            'output':str(output)}
    (output_dir/'integration_validation_v2.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
