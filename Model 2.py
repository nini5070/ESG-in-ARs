"""
PANEL REGRESSION WITH INDUSTRY-YEAR INTERACTIONS
================================================

MODEL SPECIFICATION:
- Dependent variable: Net_Sentiment (Opportunity - Risk)
- Fixed effects: Entity (firm-level) fixed effects
- Industry-year interactions: All industries (except reference) interacted with all years (except reference)
- Reference category: Industrials in 2011
- Standard errors: Clustered by firm (Company_id)
- Estimator: PanelOLS with entity fixed effects

MODEL STRUCTURE:
Net_Sentiment_it = alpha_i + sum(beta_t * Year_t) + sum(gamma_it * Industry_i × Year_t) + epsilon_it

Where:
- alpha_i = firm fixed effects
- beta_t = year fixed effects
- gamma_it = industry-specific year deviations from reference industry (Industrials)
- epsilon_it = error term clustered by firm

INDUSTRIES:
Reference: Industrials
Included: Communication Services, Consumer Discretionary, Consumer Staples, Energy,
          Financials, Health Care, Information Technology, Materials, Utilities

TIME PERIOD:
Reference year: 2011
Included years: 2012-2022

EXPECTED OUTPUTS:
- Full regression table with all coefficients
- Year fixed effects (deviations from 2011)
- Industry-year interaction effects (deviations from Industrials in each year)
- Significant interaction effects summary (p < 0.10)
- Industry-level descriptive statistics
- Model diagnostics (R-squared, F-statistics)
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('Companies_Final_Table.xlsx')

print("="*80)
print("DATA OVERVIEW")
print("="*80)
print(f"Total observations: {len(df)}")
print(f"Companies: {df['Company_id'].nunique()}")
print(f"Years: {df['Year'].min()} to {df['Year'].max()}")
print(f"\nIndustries:\n{df['Industry'].value_counts().sort_index()}")

industries = sorted(df['Industry'].unique())
print(f"\nREFERENCE INDUSTRY: Industrials")
print(f"OTHER INDUSTRIES: {[i for i in industries if i != 'Industrials']}")

for ind in industries:
    if ind != 'Industrials':
        clean_name = ind.replace(' ', '_').replace('-', '_')
        df[f'Ind_{clean_name}'] = (df['Industry'] == ind).astype(int)

years = sorted(df['Year'].unique())
print(f"\nREFERENCE YEAR: 2011")
print(f"OTHER YEARS: {[y for y in years if y != 2011]}")

for year in years:
    if year != 2011:
        df[f'Year_{year}'] = (df['Year'] == year).astype(int)

print(f"\n{'='*80}")
print("CREATING INDUSTRY-YEAR INTERACTIONS")
print("="*80)

interaction_terms = []
for year in years:
    if year != 2011:
        for ind in industries:
            if ind != 'Industrials':
                clean_name = ind.replace(' ', '_').replace('-', '_')
                interaction_name = f'Year_{year}_x_Ind_{clean_name}'
                df[interaction_name] = df[f'Year_{year}'] * df[f'Ind_{clean_name}']
                interaction_terms.append(interaction_name)

print(f"Created {len(interaction_terms)} interaction terms")

year_dummies = [f'Year_{y}' for y in years if y != 2011]
exog_vars = year_dummies + interaction_terms

print(f"\nMODEL VARIABLES:")
print(f"Year dummies: {len(year_dummies)}")
print(f"Interaction terms: {len(interaction_terms)}")
print(f"Total predictors: {len(exog_vars)}")

df['Net_Sentiment'] = df['Opportunity'] - df['Risk']

df = df.set_index(['Company_id', 'Year'])

exog = df[exog_vars]
endog = df['Net_Sentiment']

print(f"\n{'='*80}")
print("ESTIMATING MODEL")
print("="*80)

model = PanelOLS(endog, exog, entity_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True)

print(results.summary)

print("\n" + "="*80)
print("YEAR FIXED EFFECTS (vs 2011)")
print("="*80)

year_effects = []
for year in years:
    if year != 2011:
        param = f'Year_{year}'
        if param in results.params:
            year_effects.append({
                'Year': year,
                'Coefficient': f"{results.params[param]:.4f}",
                'Std Error': f"{results.std_errors[param]:.4f}",
                'P-value': f"{results.pvalues[param]:.4f}",
                'Sig': '***' if results.pvalues[param] < 0.01 else '**' if results.pvalues[param] < 0.05 else '*' if results.pvalues[param] < 0.10 else ''
            })

year_df = pd.DataFrame(year_effects)
print(year_df.to_string(index=False))

print("\n" + "="*80)
print("SIGNIFICANT INDUSTRY-YEAR INTERACTIONS (p < 0.10)")
print("="*80)

sig_interactions = []
for param in results.params.index:
    if 'Year_' in param and 'Ind_' in param and results.pvalues[param] < 0.10:
        parts = param.split('_x_')
        year_part = parts[0].replace('Year_', '')
        industry_part = parts[1].replace('Ind_', '').replace('_', ' ')
        
        sig_interactions.append({
            'Year': year_part,
            'Industry': industry_part,
            'Coefficient': f"{results.params[param]:.4f}",
            'Std Error': f"{results.std_errors[param]:.4f}",
            'P-value': f"{results.pvalues[param]:.4f}",
            'Sig': '***' if results.pvalues[param] < 0.01 else '**' if results.pvalues[param] < 0.05 else '*'
        })

sig_df = pd.DataFrame(sig_interactions)
if not sig_df.empty:
    sig_df = sig_df.sort_values(['Year', 'P-value'])
    print(sig_df.to_string(index=False))
    print(f"\nTotal significant interactions: {len(sig_df)}")
else:
    print("No significant interactions at p < 0.10")

print("\n" + "="*80)
print("INDUSTRY-YEAR EFFECTS BY YEAR")
print("="*80)

df_reset = df.reset_index()

for year in [2014, 2017, 2018, 2019, 2020, 2021]:
    print(f"\n{year} vs 2011:")
    year_sig = []
    for param in results.params.index:
        if f'Year_{year}_x_Ind_' in param and results.pvalues[param] < 0.10:
            industry = param.replace(f'Year_{year}_x_Ind_', '').replace('_', ' ')
            coef = results.params[param]
            se = results.std_errors[param]
            pval = results.pvalues[param]
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
            year_sig.append(f"  {industry:30s}: {coef:7.4f} (SE={se:.4f}, p={pval:.4f}) {sig}")
    
    if year_sig:
        for line in year_sig:
            print(line)
    else:
        print("  No significant effects")

print("\n" + "="*80)
print("INDUSTRY DESCRIPTIVE STATISTICS")
print("="*80)

industry_stats = df_reset.groupby('Industry').agg({
    'Net_Sentiment': ['count', 'mean', 'std', 'min', 'max'],
    'Company_id': 'nunique'
}).round(4)

industry_stats.columns = ['N_Obs', 'Mean', 'Std_Dev', 'Min', 'Max', 'N_Firms']
industry_stats = industry_stats.sort_values('Mean', ascending=False)
print(industry_stats.to_string())

print("\n" + "="*80)
print("MODEL DIAGNOSTICS")
print("="*80)
print(f"R-squared (within): {results.rsquared_within:.4f}")
print(f"R-squared (overall): {results.rsquared_overall:.4f}")
print(f"R-squared (between): {results.rsquared_between:.4f}")
print(f"Number of observations: {results.nobs:,}")
print(f"Number of entities: {results.entity_info['total']}")
print(f"Degrees of freedom: {results.df_resid:,}")
print(f"F-statistic: {results.f_statistic.stat:.4f}")
print(f"F p-value: {results.f_statistic.pval:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Total parameters estimated: {len(results.params)}")
print(f"Significant effects (p<0.10): {sum(results.pvalues < 0.10)}")
print(f"Highly significant effects (p<0.01): {sum(results.pvalues < 0.01)}")
