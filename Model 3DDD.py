"""
TRIPLE DIFFERENCE-IN-DIFFERENCES EVENT STUDY ANALYSIS
=====================================================

MODEL SPECIFICATION:
- Triple-DiD with event studies around 8 regulatory milestones
- Treatment groups: EU x Carbon-intensive firms
- Outcomes: Net_Sentiment and Neutral_Disclosure
- Event windows: t-3 to t+3 years around each event
- Fixed effects: Firm and Year
- Standard errors: Clustered by firm

EVENTS ANALYZED:
EU Regulations:
1. 2014: EU Non-Financial Reporting Directive
2. 2018: EU Action Plan on Sustainable Finance
3. 2019: EU Green Deal
4. 2020: EU Taxonomy Regulation

Global Events:
5. 2015: Paris Agreement
6. 2016: TCFD Recommendations
7. 2021: COP26 Glasgow
8. 2021: SEC Climate Disclosure Proposal

EXPECTED OUTPUTS:
- Event study regression results for all 8 events
- Coefficient tables with significance stars
- Event study plots (5 figures)
- Pre-trends diagnostic assessment
- Summary statistics tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

df = pd.read_excel('Companies_Final_all_new_vars.xlsx')

df['EU'] = (df['Region'] == 'Europe').astype(int)
df['Carbon'] = df['Carbon Intensive'].astype(int)

events = {
    'EU_NFRD_2014': {'year': 2014, 'type': 'EU', 'name': 'EU NFRD'},
    'EU_Action_2018': {'year': 2018, 'type': 'EU', 'name': 'EU Action Plan'},
    'EU_Green_2019': {'year': 2019, 'type': 'EU', 'name': 'EU Green Deal'},
    'EU_Taxonomy_2020': {'year': 2020, 'type': 'EU', 'name': 'EU Taxonomy'},
    'Paris_2015': {'year': 2015, 'type': 'Global', 'name': 'Paris Agreement'},
    'TCFD_2016': {'year': 2016, 'type': 'Global', 'name': 'TCFD'},
    'COP26_2021': {'year': 2021, 'type': 'Global', 'name': 'COP26'},
    'SEC_2021': {'year': 2021, 'type': 'Global', 'name': 'SEC Proposal'}
}

outcomes = ['Net_Sentiment', 'Neutral_Disclosure']

results_all = {}

for event_name, event_info in events.items():
    event_year = event_info['year']
    df[f'EventTime_{event_name}'] = df['Year'] - event_year
    df[f'InWindow_{event_name}'] = ((df[f'EventTime_{event_name}'] >= -3) & 
                                     (df[f'EventTime_{event_name}'] <= 3)).astype(int)
    
    event_data = df[df[f'InWindow_{event_name}'] == 1].copy()
    
    for t in range(-3, 4):
        if t != -1:
            event_data[f'Post_{t}'] = (event_data[f'EventTime_{event_name}'] == t).astype(int)
            event_data[f'EU_Post_{t}'] = event_data['EU'] * event_data[f'Post_{t}']
            event_data[f'Carbon_Post_{t}'] = event_data['Carbon'] * event_data[f'Post_{t}']
            event_data[f'DDD_{t}'] = event_data['EU'] * event_data['Carbon'] * event_data[f'Post_{t}']
    
    results_all[event_name] = {}
    
    for outcome in outcomes:
        time_dummies = ' + '.join([f'Post_{t}' for t in range(-3, 4) if t != -1])
        eu_interact = ' + '.join([f'EU_Post_{t}' for t in range(-3, 4) if t != -1])
        carbon_interact = ' + '.join([f'Carbon_Post_{t}' for t in range(-3, 4) if t != -1])
        ddd_interact = ' + '.join([f'DDD_{t}' for t in range(-3, 4) if t != -1])
        
        formula = f'{outcome} ~ {time_dummies} + EU + Carbon + EU:Carbon + {eu_interact} + {carbon_interact} + {ddd_interact} + C(Company_id) + C(Year)'
        
        model = ols(formula, data=event_data).fit(cov_type='cluster', cov_kwds={'groups': event_data['Company_id']})
        
        results_all[event_name][outcome] = {
            'model': model,
            'event_year': event_year,
            'event_type': event_info['type'],
            'event_name': event_info['name'],
            'n_obs': len(event_data),
            'r_squared': model.rsquared,
            'coefficients': {},
            'significant_effects': []
        }
        
        for t in range(-3, 4):
            if t != -1:
                param_name = f'DDD_{t}'
                if param_name in model.params:
                    coef = model.params[param_name]
                    se = model.bse[param_name]
                    pval = model.pvalues[param_name]
                    ci_lower = model.conf_int().loc[param_name, 0]
                    ci_upper = model.conf_int().loc[param_name, 1]
                    
                    results_all[event_name][outcome]['coefficients'][t] = {
                        'coef': coef, 'se': se, 'pval': pval,
                        'ci_lower': ci_lower, 'ci_upper': ci_upper
                    }
                    
                    if pval < 0.10:
                        results_all[event_name][outcome]['significant_effects'].append({
                            'period': t, 'coef': coef, 'pval': pval
                        })

print("\n" + "="*80)
print("EVENT STUDY REGRESSION RESULTS SUMMARY")
print("="*80)

for event_name, event_info in events.items():
    print(f"\n{event_info['name']} ({event_info['year']}) - {event_info['type']}")
    print("-" * 60)
    for outcome in outcomes:
        sig_count = len(results_all[event_name][outcome]['significant_effects'])
        print(f"  {outcome}: {sig_count} significant effects (p<0.10)")

sig_table = []
for event_name, event_info in events.items():
    for outcome in outcomes:
        for effect in results_all[event_name][outcome]['significant_effects']:
            sig_table.append({
                'Event': event_info['name'],
                'Year': event_info['year'],
                'Type': event_info['type'],
                'Outcome': outcome,
                'Period': f"t={effect['period']}",
                'Coefficient': f"{effect['coef']:.4f}",
                'P-value': f"{effect['pval']:.4f}"
            })

sig_df = pd.DataFrame(sig_table)
print("\n" + "="*80)
print("SIGNIFICANT TRIPLE-DiD EFFECTS (p < 0.10)")
print("="*80)
print(sig_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = {'EU': '#2E86AB', 'Global': '#A23B72'}

for idx, outcome in enumerate(outcomes):
    ax = axes[idx]
    for event_name, event_info in events.items():
        periods = sorted([k for k in results_all[event_name][outcome]['coefficients'].keys()])
        coeffs = [results_all[event_name][outcome]['coefficients'][t]['coef'] for t in periods]
        ci_lower = [results_all[event_name][outcome]['coefficients'][t]['ci_lower'] for t in periods]
        ci_upper = [results_all[event_name][outcome]['coefficients'][t]['ci_upper'] for t in periods]
        
        ax.plot(periods, coeffs, marker='o', label=f"{event_info['name']}", 
                color=colors[event_info['type']], alpha=0.7, linewidth=2)
        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.2, color=colors[event_info['type']])
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Event')
    ax.set_xlabel('Event Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Triple-DiD Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'Panel {"AB"[idx]}: {outcome}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('event_study_both_outcomes.png', dpi=300, bbox_inches='tight')
print("\nSaved: event_study_both_outcomes.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, outcome in enumerate(outcomes):
    ax = axes[idx]
    for event_name, event_info in events.items():
        sig_effects = results_all[event_name][outcome]['significant_effects']
        if sig_effects:
            periods = [e['period'] for e in sig_effects]
            coeffs = [e['coef'] for e in sig_effects]
            ax.scatter(periods, coeffs, s=100, marker='*', 
                      color=colors[event_info['type']], label=event_info['name'], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Event Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'{outcome} - Significant Effects Only', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('significant_effects_only.png', dpi=300, bbox_inches='tight')
print("Saved: significant_effects_only.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Total events analyzed: {len(events)}")
print(f"Outcomes analyzed: {', '.join(outcomes)}")
print(f"Total regressions run: {len(events) * len(outcomes)}")
print(f"Total significant effects (p<0.10): {len(sig_df)}")
