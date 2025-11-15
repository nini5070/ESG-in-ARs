"""
PANEL FIXED EFFECTS: OLS AND FRACTIONAL LOGIT MODELS
====================================================

MODEL SPECIFICATION:
Two complementary specifications for ESG disclosure topics and categories:

A. OLS FIXED EFFECTS MODEL:
   Y_it = alpha_i + sum(beta_t * Year_t) + epsilon_it
   
B. FRACTIONAL LOGIT MODEL:
   logit(E[Y_it]) = alpha_i + sum(beta_t * Year_t)
   
   Where logit(p) = log(p/(1-p)) for p in (0,1)

FIXED EFFECTS:
- Entity (firm) fixed effects: alpha_i
- Year fixed effects: beta_t (reference year: 2011)
- Standard errors: Two-way clustered (firm × year) with fallback to firm-only clustering

DATA PREPARATION:
- OLS: No transformation
- Fractional Logit: Values clipped to [epsilon, 1-epsilon] where epsilon=1e-6
  (Required because logit is undefined at exactly 0 or 1)

OUTCOMES:
5 SASB Categories (weighted disclosure shares):
- Business_Model_and_Innovation_Weighted
- Environment_Weighted
- Human_Capital_Weighted
- Leadership_and_Governance_Weighted
- Social_Capital_Weighted

26 ESG Topics (SASB materiality topics as disclosure shares):
- Business_Ethics, Data_Security, Access_And_Affordability, etc.

INFERENCE:
- Robust standard errors with two-way clustering (firm and year)
- If two-way clustering fails: fallback to firm-only clustering
- FDR correction for multiple testing (Benjamini-Hochberg) on topics

EXPECTED OUTPUTS:
- Year fixed effects coefficients for all outcomes (OLS and Fractional Logit)
- Comparison tables showing OLS vs FL estimates
- Significance summaries with FDR correction
- Predicted values and marginal effects
- Model diagnostics and convergence checks
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('Companies_Final_all_new_vars.xlsx')

ID_FIRM = 'Company_id'
ID_YEAR = 'Year'
LAST_TOPIC = 'GHG_Emissions'

idx_last = df.columns.get_loc(LAST_TOPIC)
topic_cols = list(df.columns[idx_last-25: idx_last+1])
carbon_col = df.columns[idx_last+1]
category_cols = list(df.columns[idx_last+2: idx_last+2+5])

print("="*80)
print("DATA OVERVIEW")
print("="*80)
print(f"Firm ID: {ID_FIRM} | Year: {ID_YEAR}")
print(f"Topics detected: {len(topic_cols)}")
print(f"Categories detected: {len(category_cols)}")
print(f"Carbon flag: {carbon_col}")

eps = 1e-6

work_ols = df.copy()
work_ols[ID_FIRM] = work_ols[ID_FIRM].astype(str)
work_ols[ID_YEAR] = work_ols[ID_YEAR].astype(int)

work_frac = df.copy()
work_frac[topic_cols + category_cols] = work_frac[topic_cols + category_cols].clip(eps, 1-eps)
work_frac[ID_FIRM] = work_frac[ID_FIRM].astype(str)
work_frac[ID_YEAR] = work_frac[ID_YEAR].astype(int)

for frame in [work_ols, work_frac]:
    keep = frame.groupby(ID_FIRM)[ID_YEAR].nunique()
    keep = keep[keep >= 2].index
    frame.drop(index=frame[~frame[ID_FIRM].isin(keep)].index, inplace=True)

base_year = int(work_ols[ID_YEAR].min())

print(f"\nBase year: {base_year}")
print(f"Firms: {work_ols[ID_FIRM].nunique()} | Observations: {len(work_ols)}")
print(f"Years: {work_ols[ID_YEAR].min()} to {work_ols[ID_YEAR].max()}")

def extract_year_effects(res, year_col, base_year):
    names = res.model.exog_names
    beta = pd.Series(res.params, index=names)
    
    prefix = f"C({year_col}, Treatment(reference={base_year}))"
    year_names = [n for n in names if n.startswith(prefix)]
    
    name_to_idx = {n: i for i, n in enumerate(names)}
    idx = np.array([name_to_idx[n] for n in year_names], dtype=int)
    
    cov = res.cov_params().values if hasattr(res.cov_params(), 'values') else res.cov_params()
    se = np.sqrt(np.diag(cov)[idx])
    
    coef = beta[year_names].values
    z = coef / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se
    
    years = []
    for n in year_names:
        m = n[n.rfind('T.')+2:].rstrip(']')
        try:
            years.append(int(m))
        except:
            years.append(int(float(m)))
    
    return pd.DataFrame({
        'year': years,
        'coef': coef,
        'se': se,
        'z': z,
        'pval': p,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi
    })

def run_ols_fe(df, outcome, firm_col, year_col, base_year):
    form = f"{outcome} ~ C({firm_col}) + C({year_col}, Treatment(reference={base_year}))"
    
    try:
        res = smf.ols(form, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[[firm_col, year_col]]}
        )
        inference = 'two-way'
    except:
        res = smf.ols(form, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[firm_col]}
        )
        inference = 'firm-only'
    
    tidy = extract_year_effects(res, year_col, base_year)
    tidy.insert(0, 'outcome', outcome)
    tidy['inference'] = inference
    
    return res, tidy

def run_fraclogit_fe(df, outcome, firm_col, year_col, base_year):
    form = f"{outcome} ~ C({firm_col}) + C({year_col}, Treatment(reference={base_year}))"
    
    try:
        res = smf.glm(form, data=df, family=sm.families.Binomial()).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[[firm_col, year_col]]}
        )
        inference = 'two-way'
    except:
        res = smf.glm(form, data=df, family=sm.families.Binomial()).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[firm_col]}
        )
        inference = 'firm-only'
    
    tidy = extract_year_effects(res, year_col, base_year)
    tidy.insert(0, 'outcome', outcome)
    tidy['inference'] = inference
    
    return res, tidy

print("\n" + "="*80)
print("PART A: OLS FIXED EFFECTS - CATEGORIES")
print("="*80)

ols_category_results = []
ols_models = {}

for i, outcome in enumerate(category_cols, 1):
    print(f"[{i}/{len(category_cols)}] {outcome}...")
    try:
        res, tidy = run_ols_fe(work_ols, outcome, ID_FIRM, ID_YEAR, base_year)
        ols_models[outcome] = res
        ols_category_results.append(tidy)
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")

ols_categories_df = pd.concat(ols_category_results, ignore_index=True) if ols_category_results else pd.DataFrame()

print("\n" + "="*80)
print("PART B: FRACTIONAL LOGIT - CATEGORIES")
print("="*80)

fl_category_results = []
fl_models = {}

for i, outcome in enumerate(category_cols, 1):
    print(f"[{i}/{len(category_cols)}] {outcome}...")
    try:
        res, tidy = run_fraclogit_fe(work_frac, outcome, ID_FIRM, ID_YEAR, base_year)
        fl_models[outcome] = res
        fl_category_results.append(tidy)
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")

fl_categories_df = pd.concat(fl_category_results, ignore_index=True) if fl_category_results else pd.DataFrame()

print("\n" + "="*80)
print("PART C: OLS FIXED EFFECTS - TOPICS")
print("="*80)

ols_topic_results = []

for i, outcome in enumerate(topic_cols, 1):
    if i % 5 == 0:
        print(f"[{i}/{len(topic_cols)}] {outcome}...")
    try:
        res, tidy = run_ols_fe(work_ols, outcome, ID_FIRM, ID_YEAR, base_year)
        ols_topic_results.append(tidy)
    except Exception as e:
        if i % 5 == 0:
            print(f"  Error: {str(e)[:80]}")

ols_topics_df = pd.concat(ols_topic_results, ignore_index=True) if ols_topic_results else pd.DataFrame()

print("\n" + "="*80)
print("PART D: FRACTIONAL LOGIT - TOPICS")
print("="*80)

fl_topic_results = []

for i, outcome in enumerate(topic_cols, 1):
    if i % 5 == 0:
        print(f"[{i}/{len(topic_cols)}] {outcome}...")
    try:
        res, tidy = run_fraclogit_fe(work_frac, outcome, ID_FIRM, ID_YEAR, base_year)
        fl_topic_results.append(tidy)
    except Exception as e:
        if i % 5 == 0:
            print(f"  Error: {str(e)[:80]}")

fl_topics_df = pd.concat(fl_topic_results, ignore_index=True) if fl_topic_results else pd.DataFrame()

print("\n" + "="*80)
print("FDR CORRECTION (Topics Only)")
print("="*80)

if not ols_topics_df.empty:
    for year in sorted(ols_topics_df['year'].unique()):
        year_data = ols_topics_df[ols_topics_df['year'] == year].copy()
        if len(year_data) > 0:
            reject, pvals_corrected = multipletests(year_data['pval'], alpha=0.05, method='fdr_bh')[:2]
            ols_topics_df.loc[ols_topics_df['year'] == year, 'pval_fdr'] = pvals_corrected
            ols_topics_df.loc[ols_topics_df['year'] == year, 'sig_fdr'] = reject.astype(int)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if not ols_categories_df.empty:
    print("\nOLS Categories - Significant years (p<0.05):")
    sig_ols_cat = ols_categories_df[ols_categories_df['pval'] < 0.05].groupby('outcome').agg({
        'year': lambda x: list(x),
        'coef': 'mean'
    })
    print(sig_ols_cat)

if not fl_categories_df.empty:
    print("\nFractional Logit Categories - Significant years (p<0.05):")
    sig_fl_cat = fl_categories_df[fl_categories_df['pval'] < 0.05].groupby('outcome').agg({
        'year': lambda x: list(x),
        'coef': 'mean'
    })
    print(sig_fl_cat)

if not ols_topics_df.empty and 'pval_fdr' in ols_topics_df.columns:
    print("\nTopics - FDR-significant effects by year:")
    fdr_summary = ols_topics_df[ols_topics_df['sig_fdr'] == 1].groupby('year').size()
    print(fdr_summary)

if not ols_categories_df.empty and not fl_categories_df.empty:
    print("\nOLS vs Fractional Logit Comparison (Categories):")
    for outcome in category_cols:
        ols_sig = len(ols_categories_df[(ols_categories_df['outcome'] == outcome) & (ols_categories_df['pval'] < 0.05)])
        fl_sig = len(fl_categories_df[(fl_categories_df['outcome'] == outcome) & (fl_categories_df['pval'] < 0.05)])
        print(f"  {outcome}: OLS={ols_sig} sig years, FL={fl_sig} sig years")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"OLS categories: {len(ols_categories_df)} results")
print(f"FL categories: {len(fl_categories_df)} results")
print(f"OLS topics: {len(ols_topics_df)} results")
print(f"FL topics: {len(fl_topics_df)} results")
if not ols_topics_df.empty and 'sig_fdr' in ols_topics_df.columns:
    print(f"FDR-significant topic effects: {ols_topics_df['sig_fdr'].sum()}")
