"""
EVENT STUDY ANALYSIS: EU REGULATIONS AND GLOBAL EVENTS ON ESG TOPICS
====================================================================

MODEL SPECIFICATION:
Two approaches for different event types:

A. EU REGULATORY EVENTS (Difference-in-Differences):
   Treatment: EU firms vs Non-EU firms
   Events: NFRD 2014, EU Action Plan 2018, EU Taxonomy 2020, CSRD Proposal 2021
   Model: Y_it = alpha_i + gamma_t + sum(beta_k * EU_i × Post_k,t) + epsilon_it

B. GLOBAL EVENTS (Triple Difference-in-Differences):
   Treatment: EU firms × Carbon-intensive firms
   Events: Paris/SDGs 2015, TCFD 2017, COVID-19 2020, COP26 2021
   Model: Y_it = alpha_i + gamma_t + sum(beta_k * EU_i × Carbon_i × Post_k,t) + lower-order interactions + epsilon_it

FIXED EFFECTS:
- Entity (firm) fixed effects
- Year fixed effects
- Standard errors: Clustered by firm

EVENT WINDOWS:
- Long window: 5 years before/after event (leads=5, lags=5)
- Short window: 3 years before/after event (leads=3, lags=3)

TIMING VARIANTS:
- Y: Event year as treatment year
- Y+1: One year after event (implementation lag)
- Donut: Excludes event year itself (k=0)

OUTCOMES:
5 Categories (weighted):
- Business_Model_and_Innovation_Weighted
- Environment_Weighted
- Human_Capital_Weighted
- Leadership_and_Governance_Weighted
- Social_Capital_Weighted

26 Topics (SASB materiality topics as disclosure shares)

EXPECTED OUTPUTS:
- Event study coefficients for all events, outcomes, and timing variants
- Pre-trend tests (F-test on pre-event coefficients)
- Significance counts by event/outcome
- FDR-corrected p-values for topics (Benjamini-Hochberg)
- Summary tables with k=0, k=1 effects and average post-event effects
- Comparison of long vs short windows
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('Companies_Final_all_new_vars.xlsx')

ID_FIRM = 'Company_id'
ID_YEAR = 'Year'

df['EU_Firm'] = df['Region'].astype(str).str.contains(r'\b(EU|Europe)\b', case=False).astype(int)

if 'Carbon Intensive' in df.columns:
    ci_raw = pd.to_numeric(df['Carbon Intensive'], errors='coerce').fillna(0)
    df['Carbon_Intensive'] = (ci_raw > 0.5).astype(int)
else:
    df['Carbon_Intensive'] = 0

category_cols = [
    'Business_Model_and_Innovation_Weighted',
    'Environment_Weighted',
    'Human_Capital_Weighted',
    'Leadership_and_Governance_Weighted',
    'Social_Capital_Weighted'
]

topic_cols = [
    'Business_Ethics', 'Data_Security', 'Access_And_Affordability', 'Business_Model_Resilience',
    'Competitive_Behavior', 'Critical_Incident_Risk_Management', 'Customer_Welfare', 'Director_Removal',
    'Employee_Engagement_Inclusion_And_Diversity', 'Employee_Health_And_Safety',
    'Human_Rights_And_Community_Relations', 'Labor_Practices', 'Management_Of_Legal_And_Regulatory_Framework',
    'Physical_Impacts_Of_Climate_Change', 'Product_Quality_And_Safety', 'Product_Design_And_Lifecycle_Management',
    'Selling_Practices_And_Product_Labeling', 'Supply_Chain_Management', 'Systemic_Risk_Management',
    'Waste_And_Hazardous_Materials_Management', 'Water_And_Wastewater_Management',
    'Air_Quality', 'Customer_Privacy', 'Ecological_Impacts', 'Energy_Management', 'GHG_Emissions'
]

print("="*80)
print("DATA OVERVIEW")
print("="*80)
print(f"Total observations: {len(df)}")
print(f"Firms: {df[ID_FIRM].nunique()}")
print(f"Years: {df[ID_YEAR].min()} to {df[ID_YEAR].max()}")
print(f"EU firms: {df['EU_Firm'].sum()} ({df['EU_Firm'].mean()*100:.1f}%)")
print(f"Carbon-intensive: {df['Carbon_Intensive'].sum()} ({df['Carbon_Intensive'].mean()*100:.1f}%)")

def run_event_did_eu(df, y_col, event_year, leads=5, lags=5, donut=False):
    d = df[[ID_FIRM, ID_YEAR, 'EU_Firm', y_col]].dropna().copy()
    d['et'] = d[ID_YEAR] - int(event_year)
    if donut:
        d = d[d['et'] != 0]
    
    def kname(k): 
        return f"EU_k_m{abs(k)}" if k < 0 else f"EU_k_p{k}"
    
    et_min, et_max = int(d['et'].min()), int(d['et'].max())
    k_low = max(-leads, et_min)
    k_high = min(lags, et_max)
    ks = [k for k in range(k_low, k_high+1) if k != -1 and ((d['EU_Firm'] == 1) & (d['et'] == k)).any()]
    
    if not ks:
        raise ValueError(f"No supported event-times for {y_col} at event {event_year}")
    
    terms = [f"C({ID_FIRM})", f"C({ID_YEAR})"]
    for k in ks:
        col = kname(k)
        d[col] = ((d['EU_Firm'] == 1) & (d['et'] == k)).astype(int)
        if d[col].sum() > 0:
            terms.append(col)
    
    formula = f"{y_col} ~ " + " + ".join(terms)
    res = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d[ID_FIRM]})
    
    rows = []
    for k in ks:
        nm = kname(k)
        if nm not in res.params.index:
            continue
        b, se = res.params[nm], res.bse[nm]
        t = b/se if np.isfinite(b) and se > 0 else np.nan
        p = 2*(1-norm.cdf(abs(t))) if np.isfinite(t) else np.nan
        rows.append([y_col, event_year, k, b, se, t, p, len(d)])
    
    out = pd.DataFrame(rows, columns=['outcome', 'event_year', 'event_time', 'coef_ols', 'se_ols', 't_ols', 'pval_ols', 'n_obs'])
    
    pre_ks = [k for k in ks if k < 0]
    if len(pre_ks) > 1:
        pre_params = [kname(k) for k in pre_ks if kname(k) in res.params.index]
        if len(pre_params) > 1:
            fstat = res.f_test(np.eye(len(pre_params), len(res.params))[:, [list(res.params.index).index(p) for p in pre_params]])
            out['pretrend_p'] = fstat.pvalue
            out['pretrend_df'] = f"{len(pre_params)},{int(res.df_resid)}"
    
    return res, out

def run_event_ddd_global(df, y_col, event_year, leads=5, lags=5, donut=False):
    d = df[[ID_FIRM, ID_YEAR, 'EU_Firm', 'Carbon_Intensive', y_col]].dropna().copy()
    d['et'] = d[ID_YEAR] - int(event_year)
    if donut:
        d = d[d['et'] != 0]
    
    def kname(prefix, k):
        return f"{prefix}_m{abs(k)}" if k < 0 else f"{prefix}_p{k}"
    
    et_min, et_max = int(d['et'].min()), int(d['et'].max())
    k_low, k_high = max(-leads, et_min), min(lags, et_max)
    ks = [k for k in range(k_low, k_high+1) if k != -1]
    
    terms = [f"C({ID_FIRM})", f"C({ID_YEAR})"]
    kept_ks = []
    
    for k in ks:
        m_triple = ((d['EU_Firm']==1) & (d['Carbon_Intensive']==1) & (d['et']==k)).astype(int)
        m_eu = ((d['EU_Firm']==1) & (d['et']==k)).astype(int)
        m_hc = ((d['Carbon_Intensive']==1) & (d['et']==k)).astype(int)
        
        if m_triple.sum() == 0:
            continue
        
        col_triple = kname("DDD", k)
        col_eu = kname("EU", k)
        col_hc = kname("HC", k)
        
        d[col_triple] = m_triple
        d[col_eu] = m_eu
        d[col_hc] = m_hc
        
        terms.extend([col_triple, col_eu, col_hc])
        kept_ks.append(k)
    
    if not kept_ks:
        raise ValueError(f"No DDD groups for {y_col} at event {event_year}")
    
    formula = f"{y_col} ~ " + " + ".join(terms)
    res = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d[ID_FIRM]})
    
    rows = []
    for k in kept_ks:
        nm = kname("DDD", k)
        if nm not in res.params.index:
            continue
        b, se = res.params[nm], res.bse[nm]
        t = b/se if np.isfinite(b) and se > 0 else np.nan
        p = 2*(1-norm.cdf(abs(t))) if np.isfinite(t) else np.nan
        rows.append([y_col, event_year, k, b, se, t, p, len(d)])
    
    out = pd.DataFrame(rows, columns=['outcome', 'event_year', 'event_time', 'coef_ddd', 'se_ddd', 't_ddd', 'pval_ddd', 'n_obs'])
    
    pre_ks = [k for k in kept_ks if k < 0]
    if len(pre_ks) > 1:
        pre_params = [kname("DDD", k) for k in pre_ks if kname("DDD", k) in res.params.index]
        if len(pre_params) > 1:
            fstat = res.f_test(np.eye(len(pre_params), len(res.params))[:, [list(res.params.index).index(p) for p in pre_params]])
            out['pretrend_p'] = fstat.pvalue
            out['pretrend_df'] = f"{len(pre_params)},{int(res.df_resid)}"
    
    return res, out

EU_EVENTS = [
    ('NFRD_2014', 2014),
    ('EU_ActionPlan_2018', 2018),
    ('EU_Taxonomy_2020', 2020),
    ('CSRD_Proposal_2021', 2021),
]

GLOBAL_EVENTS = [
    ('Paris_SDGS_2015', 2015),
    ('TCFD_2017', 2017),
    ('COVID19_2020', 2020),
    ('COP26_2021', 2021),
]

LEADS_SHORT, LAGS_SHORT = 3, 3

print("\n" + "="*80)
print("RUNNING EU EVENT STUDIES (DiD)")
print("="*80)

eu_results_topics = []
for label, yr in EU_EVENTS:
    print(f"\nProcessing {label} ({yr})...")
    for y in topic_cols:
        for timing, ev_year, donut in [("Y", yr, False), ("Y+1", yr+1, False), ("donut", yr, True)]:
            try:
                _, tab = run_event_did_eu(df, y, ev_year, LEADS_SHORT, LAGS_SHORT, donut)
                tab.insert(0, 'event_id', label)
                tab.insert(2, 'timing', timing)
                eu_results_topics.append(tab)
            except Exception as e:
                print(f"  Warning: {y} @ {timing}: {str(e)[:50]}")

eu_topics_df = pd.concat(eu_results_topics, ignore_index=True) if eu_results_topics else pd.DataFrame()

print("\n" + "="*80)
print("RUNNING GLOBAL EVENT STUDIES (DDD)")
print("="*80)

global_results_topics = []
for label, yr in GLOBAL_EVENTS:
    print(f"\nProcessing {label} ({yr})...")
    for y in topic_cols:
        for timing, ev_year, donut in [("Y", yr, False), ("Y+1", yr+1, False), ("donut", yr, True)]:
            try:
                _, tab = run_event_ddd_global(df, y, ev_year, LEADS_SHORT, LAGS_SHORT, donut)
                tab.insert(0, 'event_id', label)
                tab.insert(2, 'timing', timing)
                global_results_topics.append(tab)
            except Exception as e:
                print(f"  Warning: {y} @ {timing}: {str(e)[:50]}")

global_topics_df = pd.concat(global_results_topics, ignore_index=True) if global_results_topics else pd.DataFrame()

print("\n" + "="*80)
print("FDR CORRECTION (Benjamini-Hochberg)")
print("="*80)

if not eu_topics_df.empty:
    fdr_records = []
    for (eid, tming, k), g in eu_topics_df.groupby(['event_id', 'timing', 'event_time']):
        mask = g['pval_ols'].notna() & np.isfinite(g['pval_ols'])
        if mask.sum() > 0:
            rej, p_adj = fdrcorrection(g.loc[mask, 'pval_ols'].values, alpha=0.05, method='indep')
            out = g.loc[mask, ['outcome', 'event_id', 'timing', 'event_time', 'coef_ols', 'pval_ols']].copy()
            out['pval_fdr'] = p_adj
            out['sig_fdr_5%'] = rej.astype(int)
            fdr_records.append(out)
    
    eu_fdr_df = pd.concat(fdr_records, ignore_index=True) if fdr_records else pd.DataFrame()
    print(f"EU events FDR-corrected results: {len(eu_fdr_df)} rows")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if not eu_topics_df.empty:
    eu_sig = eu_topics_df[eu_topics_df['pval_ols'] < 0.05].groupby(['event_id', 'timing']).size()
    print("\nEU Events - Significant effects (p<0.05) by event and timing:")
    print(eu_sig)

if not global_topics_df.empty:
    global_sig = global_topics_df[global_topics_df['pval_ddd'] < 0.05].groupby(['event_id', 'timing']).size()
    print("\nGlobal Events - Significant DDD effects (p<0.05) by event and timing:")
    print(global_sig)

if not eu_topics_df.empty:
    eu_k0 = eu_topics_df[eu_topics_df['event_time'] == 0].groupby('event_id').agg({
        'coef_ols': 'mean',
        'pval_ols': lambda x: (x < 0.05).sum()
    })
    print("\nEU Events - Average k=0 coefficient and count of significant k=0:")
    print(eu_k0)

if not global_topics_df.empty:
    global_k0 = global_topics_df[global_topics_df['event_time'] == 0].groupby('event_id').agg({
        'coef_ddd': 'mean',
        'pval_ddd': lambda x: (x < 0.05).sum()
    })
    print("\nGlobal Events - Average k=0 DDD coefficient and count of significant k=0:")
    print(global_k0)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"EU event studies completed: {len(eu_topics_df)} results")
print(f"Global event studies completed: {len(global_topics_df)} results")
if not eu_topics_df.empty:
    print(f"EU events with FDR correction: {eu_fdr_df['sig_fdr_5%'].sum() if not eu_fdr_df.empty else 0} FDR-significant effects")
