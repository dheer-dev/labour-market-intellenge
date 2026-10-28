"""Read-only O*NET-ESCO occupation proposal; never modifies existing crosswalks."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/onet_esco_next_review'))
    args = parser.parse_args()
    root = args.project_root
    processed = root / 'data' / 'processed'
    reviewed = pd.read_csv(processed / 'onet_esco_occupation_mapping_reviewed.csv', dtype={'onet_soc_code': str})
    candidates = pd.read_csv(processed / 'onet_esco_occupation_mapping_candidates.csv', dtype={'onet_soc_code': str})
    skills = pd.read_csv(processed / 'onet_esco_skill_crosswalk.csv')
    target = 'project manager'
    current = reviewed[reviewed.target_occupation.str.casefold().eq(target)].copy()
    pm = candidates[candidates.target_occupation.str.casefold().eq(target)].copy()
    prioritized = ['13-1082.00', '15-1299.09', '11-9041.00', '11-9199.00']
    pm = pm[pm.onet_soc_code.isin(prioritized)].copy()
    pm['review_priority'] = pm.onet_soc_code.map({c:i+1 for i,c in enumerate(prioritized)})
    pm['manual_review_decision'] = 'PENDING'
    pm['official_definition_checked'] = False
    pm['review_notes'] = ''
    pm = pm.sort_values('review_priority')
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pm.to_csv(output / 'project_manager_review_candidates.csv', index=False)
    summary = pd.DataFrame([
        ('reviewed_occupation_rows', len(reviewed)),
        ('approved_occupation_rows', int(reviewed.review_status.eq('APPROVED').sum())),
        ('approved_target_occupations', reviewed.loc[reviewed.review_status.eq('APPROVED'),'target_occupation'].nunique()),
        ('project_manager_existing_review_rows', len(current)),
        ('project_manager_13_1082_present_in_candidates', int((pm.onet_soc_code=='13-1082.00').any())),
        ('onet_skill_elements', len(skills)),
        ('approved_skill_mappings', int(skills.review_decision.eq('APPROVED').sum())),
    ], columns=['metric','value'])
    summary.to_csv(output / 'mapping_audit_summary.csv',index=False)
    print(summary.to_string(index=False))
    print('\nReview file saved:', output / 'project_manager_review_candidates.csv')
    print('No original source or mapping files were changed.')


if __name__ == '__main__':
    main()
