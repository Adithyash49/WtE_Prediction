# === wte_dashboard.py ===

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from CoolProp.CoolProp import PropsSI

st.set_page_config(page_title='WtE Steam Predictor', layout='wide')
st.title('🔥 WtE Plant Steam Output Predictor')

# ── Load both models ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    ml_model     = joblib.load('xgb_steam_model.pkl')          # pure XGBoost
    hybrid_model = joblib.load('hybrid_correction_model.pkl')  # correction XGBoost
    return ml_model, hybrid_model

ml_model, hybrid_model = load_models()

# ── Physics function ──────────────────────────────────
def physics_predict_steam(waste_flow, NCV, excess_air, moi):
    Q_in    = waste_flow * 1000 * NCV / 3600
    eta     = 0.85 - 0.06 * (excess_air - 0.5) - 0.12 * (moi - 0.30)
    eta     = max(0.65, min(eta, 0.90))
    h_steam = PropsSI('H', 'T', 400 + 273.15, 'P', 40e5, 'Water') / 1000
    h_feed  = PropsSI('H', 'T', 105 + 273.15, 'P', 45e5, 'Water') / 1000
    delta_h = h_steam - h_feed
    return Q_in * eta * 3600 / delta_h

# ── Sidebar inputs — ranges from actual dataset ────────────────────────────────
st.sidebar.header('📅 Date')
month = st.sidebar.slider('Month', 1, 12, 7)

st.sidebar.header('♻️ Waste Properties')
waste_flow = st.sidebar.slider('Waste Flow (t/h)',      8.7,  21.3, 14.9, 0.1)
moisture   = st.sidebar.slider('Moisture (fraction)',   0.18,  0.52,  0.35, 0.01)
ash        = st.sidebar.slider('Ash (fraction)',        0.05,  0.20,  0.12, 0.01)
GCV        = st.sidebar.slider('GCV (MJ/kg)',           7.0,  18.5, 12.0, 0.1)

st.sidebar.header('⚙️ Operating Conditions')
excess_air = st.sidebar.slider('Excess Air Ratio',      0.30,  0.85,  0.55, 0.01)
T_comb     = st.sidebar.slider('T Combustion (°C)',   780,   908,   836,   1)
O2         = st.sidebar.slider('Flue Gas O2 (%)',       4.1,   8.8,   6.5, 0.1)
CO         = st.sidebar.slider('CO Emissions (mg/Nm³)', 11.5, 44.2, 28.5, 0.1)

# ── Feature engineering  ───────────────────────────────
NCV         = GCV - 2.442 * moisture              
combustible = 1-moisture - ash                     
Q_input     = waste_flow * 1000 * NCV / 3600    
combustion_quality = (                              
    (T_comb - 780) / (950 - 780)
    - (CO   - 5)   / (80 - 5)
    - abs(O2 - 7)  / 3
)

# ── Build feature array  ─────────────────
FEATURE_COLS = [
    'Waste Flow (t/h)', 'Moisture (fraction)', 'Ash (fraction)',
    'GCV (MJ/kg)', 'Excess Air Ratio', 'T Combustion (°C)',
    'Flue Gas O2 (%)', 'CO Emissions (mg/Nm³)',
    'month', 'NCV', 'Combustible', 'Q_input', 'Combustion_quality'
]

feature_values = {
    'Waste Flow (t/h)':       waste_flow,
    'Moisture (fraction)':    moisture,
    'Ash (fraction)':         ash,
    'GCV (MJ/kg)':            GCV,
    'Excess Air Ratio':       excess_air,
    'T Combustion (°C)':      T_comb,
    'Flue Gas O2 (%)':        O2,
    'CO Emissions (mg/Nm³)':  CO,
    'month':                  month,
    'NCV':                    NCV,
    'Combustible':            combustible,
    'Q_input':                Q_input,
    'Combustion_quality':     combustion_quality,
}

features = np.array([[feature_values[c] for c in FEATURE_COLS]])

# ── Three predictions —──────────────────────────

# 1. Physics only 
pred_physics = physics_predict_steam(waste_flow, NCV, excess_air, moisture)

# 2. Pure ML — xgb_steam_model.pkl predicts steam directly 
pred_ml = float(ml_model.predict(features)[0])

# 3. HYBRID — physics + correction 
#    hybrid_correction_model predicts the residual (actual - physics)
#    hybrid = physics + residual_correction
correction  = float(hybrid_model.predict(features)[0])
pred_hybrid = pred_physics + correction

# ── Display — hybrid is the primary metric ────────────────────────────────────
st.subheader('🎯 Predictions')

col1, col2, col3 = st.columns(3)
col1.metric(
    '⚛️ Physics Model',
    f'{pred_physics:.1f} t/h',
)
col2.metric(
    '🤖 Pure ML (XGBoost)',
    f'{pred_ml:.1f} t/h',
)
col3.metric(
    '🏆 Hybrid (Physics + ML)',
    f'{pred_hybrid:.1f} t/h',
)


# ── Calculated values row ──────────────────────────────────────────────────────
st.subheader('📐 Calculated Features')
c1, c2, c3, c4 = st.columns(4)
c1.metric('NCV',                f'{NCV:.2f} MJ/kg')
c2.metric('Thermal Input',      f'{Q_input:.1f} MW')
c3.metric('Combustible',        f'{combustible:.3f}')
c4.metric('Combustion Quality', f'{combustion_quality:.3f}')

# ── Status banner (based on hybrid) ───────────────────────────────────────────
st.divider()
if pred_hybrid > 55:
    st.success(f'✅ HIGH output: {pred_hybrid:.1f} t/h — Above average (avg: 47.4 t/h)')
elif pred_hybrid > 40:
    st.info(f'ℹ️ NORMAL output: {pred_hybrid:.1f} t/h — Near plant average (47.4 t/h)')
elif pred_hybrid > 25:
    st.warning(f'⚠️ BELOW AVERAGE: {pred_hybrid:.1f} t/h — Review waste quality')
else:
    st.error(f'🔴 LOW output: {pred_hybrid:.1f} t/h — Check waste and combustion conditions')

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DYNAMIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader('📋 Operator Recommendations')

recs = []

# Moisture
if moisture > 0.45:
    recs.append(('🔴 CRITICAL', 'MOISTURE',
        f'Moisture very high ({moisture:.0%}). '
        f'Evaporation is consuming {2.442 * moisture:.1f} MJ/kg that could make steam. '
        'Actions: (1) Blend in drier commercial/industrial waste batches, '
        '(2) Pre-dry waste on bunker floor for 2–3 days before feeding, '
        '(3) Increase grate speed to reduce residence time.'))
elif moisture > 0.40:
    recs.append(('🟠 HIGH', 'MOISTURE',
        f'Moisture elevated ({moisture:.0%}, dataset avg = 35%). '
        f'Estimated steam loss vs average: ~{(moisture - 0.35) * 10:.1f} t/h. '
        'Mix in drier waste batches and monitor bunker levels.'))
elif moisture < 0.22:
    recs.append(('🟡 NOTE', 'MOISTURE',
        f'Moisture unusually low ({moisture:.0%}) — likely dry industrial or packaging-heavy waste. '
        'Combustion temperature will run hotter. Watch grate for overheating above 900 °C.'))

# GCV
if GCV < 8.5:
    recs.append(('🔴 CRITICAL', 'GCV',
        f'GCV very low ({GCV:.1f} MJ/kg). NCV = {NCV:.1f} MJ/kg — near autothermal limit. '
        'Actions: (1) Inject auxiliary natural gas, '
        '(2) Remove low-energy fractions (wet food waste, soil), '
        '(3) Source higher-calorific waste batches.'))
elif GCV < 10:
    recs.append(('🟠 HIGH', 'GCV',
        f'GCV below normal ({GCV:.1f} MJ/kg, dataset avg = 12.0). '
        f'NCV = {NCV:.1f} MJ/kg. Blend with higher-GCV waste to recover output.'))
elif GCV > 16:
    recs.append(('🟡 NOTE', 'GCV',
        f'GCV high ({GCV:.1f} MJ/kg) — likely high plastics content. '
        'Risk of grate hotspots. Verify grate cooling water flow rate.'))

# Excess air
if excess_air > 0.70:
    recs.append(('🟠 HIGH', 'EXCESS AIR',
        f'Excess air high (λ = {excess_air:.2f}, dataset avg = 0.55). '
        f'Extra stack heat loss ≈ {(excess_air - 0.55) * 15:.1f}% of thermal input. '
        'Actions: (1) Reduce primary under-grate air by 10–15%, '
        '(2) Inspect furnace seals for air in-leakage, '
        '(3) Check O₂ sensor calibration.'))
elif excess_air < 0.38:
    recs.append(('🔴 CRITICAL', 'EXCESS AIR',
        f'Excess air dangerously low (λ = {excess_air:.2f}). '
        'Incomplete combustion — CO will spike. '
        'Increase primary and secondary air dampers immediately.'))

# Combustion temperature
if T_comb < 820:
    recs.append(('🔴 CRITICAL', 'TEMPERATURE',
        f'Temperature critically low ({T_comb} °C). '
        'EU Directive 2010/75/EU requires ≥850 °C for ≥2 seconds — regulatory breach risk. '
        'Actions: (1) Increase primary air, (2) inject auxiliary fuel, '
        '(3) reduce waste feed rate to concentrate heat.'))
elif T_comb < 850:
    recs.append(('🟠 HIGH', 'TEMPERATURE',
        f'Temperature below 850 °C regulatory threshold ({T_comb} °C, avg = 836 °C). '
        f'CO = {CO:.1f} mg/Nm³ — monitor CEMS closely. '
        'Reduce high-moisture waste input or increase under-grate air.'))

# CO
if CO > 40:
    recs.append(('🔴 CRITICAL', 'CO EMISSIONS',
        f'CO = {CO:.1f} mg/Nm³ — approaching EU IED daily average limit of 50 mg/Nm³. '
        'Actions: (1) Raise combustion temperature above 850 °C, '
        '(2) Increase secondary air above grate, '
        '(3) Reduce waste feed rate temporarily to stabilise flame.'))
elif CO > 35:
    recs.append(('🟠 HIGH', 'CO EMISSIONS',
        f'CO elevated ({CO:.1f} mg/Nm³, dataset avg = 28.5 mg/Nm³). '
        'Check combustion temperature and secondary air supply.'))

# O2
if O2 < 5.2:
    recs.append(('🟠 HIGH', 'FLUE GAS O₂',
        f'O₂ low ({O2:.1f}%, avg = 6.5%). Suggests air shortage. '
        'CO is likely elevated. Increase secondary air above the grate.'))
elif O2 > 8.0:
    recs.append(('🟡 NOTE', 'FLUE GAS O₂',
        f'O₂ high ({O2:.1f}%) — consistent with excess air above 0.75. '
        'Check for false air in-leakage before the O₂ probe.'))

# NCV
if NCV < 7:
    recs.append(('🔴 CRITICAL', 'NCV',
        f'NCV = {NCV:.1f} MJ/kg — below minimum for stable combustion. '
        'Autothermal operation not sustainable. Auxiliary fuel REQUIRED.'))

# Combustion quality
if combustion_quality < -0.7:
    recs.append(('🟠 HIGH', 'COMBUSTION QUALITY',
        f'Combustion quality score poor ({combustion_quality:.3f}, avg = −0.257). '
        'Multiple signals unfavourable simultaneously. '
        'Prioritise: (1) raise temperature, (2) check CO trend, (3) verify O₂.'))

if not recs:
    st.success('✅ All parameters within normal operating range. No action required.')
else:
    for severity, param, msg in recs:
        st.markdown(f'**{severity} — {param}**')
        st.write(msg)
        st.write('')

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VARIABLE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader('📊 Variable Insights')

tab1, tab2, tab3 = st.tabs(['Waste Properties', 'Operating Conditions', 'Calculated Values'])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### 💧 Moisture')
        st.write(f'**Current:** {moisture:.0%} | **Dataset range:** 18–52% | **Average:** 35%')
        st.progress(float((moisture - 0.18) / (0.52 - 0.18)))
        st.write(
            'Moisture is the biggest driver of steam loss. '
            f'Every extra 1% moisture costs ~{2.442 * 0.01 * waste_flow * 1000 / 3600:.2f} MW '
            'in latent heat — energy spent evaporating water instead of making steam.'
        )
        st.markdown('#### 🪨 Ash')
        st.write(f'**Current:** {ash:.0%} | **Dataset range:** 5–20% | **Average:** 12%')
        st.progress(float((ash - 0.05) / (0.20 - 0.05)))
        st.write(
            'Ash is inert — zero energy contribution. '
            'High ash means less organic matter per tonne of waste and increases '
            'bottom ash removal costs and grate wear rate.'
        )
    with c2:
        st.markdown('#### ⚡ GCV')
        st.write(f'**Current:** {GCV:.1f} MJ/kg | **Dataset range:** 7–18.5 | **Average:** 12.0')
        st.progress(float((GCV - 7.0) / (18.5 - 7.0)))
        st.write(
            f'Total chemical energy in the waste (bomb calorimeter). '
            f'After deducting moisture loss: NCV = {NCV:.1f} MJ/kg. '
            'Reference: plastics ~30, dry paper ~16, typical MSW ~10–14, wet garden waste ~5–8 MJ/kg.'
        )
        st.markdown('#### 🚚 Waste Flow')
        st.write(f'**Current:** {waste_flow:.1f} t/h | **Dataset range:** 8.7–21.3 | **Average:** 14.9')
        st.progress(float((waste_flow - 8.7) / (21.3 - 8.7)))
        st.write(
            f'More waste = more thermal input (currently {Q_input:.1f} MW). '
            'Exceeding grate design capacity causes poor burnout — '
            'unburnt carbon in bottom ash and rising CO.'
        )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### 🌬️ Excess Air Ratio (λ)')
        st.write(f'**Current:** {excess_air:.2f} | **Dataset range:** 0.30–0.85 | **Average:** 0.55')
        st.progress(float((excess_air - 0.30) / (0.85 - 0.30)))
        st.write(
            f'λ = 1.0 is stoichiometric; λ = {excess_air:.2f} means {excess_air*100:.0f}% more air '
            'than minimum. Too low → CO spike. Too high → stack losses rise. '
            'MSW needs λ = 0.50–0.65 because waste is inhomogeneous.'
        )
        st.markdown('#### 🌡️ Combustion Temperature')
        st.write(f'**Current:** {T_comb} °C | **Dataset range:** 780–908 °C | **Average:** 836 °C')
        st.progress(float((T_comb - 780) / (908 - 780)))
        st.write(
            'EU Directive 2010/75/EU requires ≥850 °C for ≥2 seconds — destroys dioxins/furans. '
            f'At {T_comb} °C: '
            f'{"dioxin destruction high (>99.99%)" if T_comb >= 850 else "⚠️ BELOW regulatory threshold"}.'
        )
    with c2:
        st.markdown('#### 🧪 Flue Gas O₂')
        st.write(f'**Current:** {O2:.1f}% | **Dataset range:** 4.1–8.8% | **Average:** 6.5%')
        st.progress(float((O2 - 4.1) / (8.8 - 4.1)))
        st.write(
            f'Real-time proxy for excess air. At λ = {excess_air:.2f}, '
            f'theory predicts O₂ ≈ {2 + 8*excess_air:.1f}%. You are reading {O2:.1f}%. '
            'Large gap → false air in-leakage or sensor drift.'
        )
        st.markdown('#### 💨 CO Emissions')
        st.write(f'**Current:** {CO:.1f} mg/Nm³ | **Dataset range:** 11.5–44.2 | **Average:** 28.5')
        st.progress(float((CO - 11.5) / (44.2 - 11.5)))
        st.write(
            'Primary indicator of incomplete combustion, read from CEMS. '
            'EU IED daily average limit = 50 mg/Nm³; half-hourly = 100 mg/Nm³. '
            'This is also a training feature — the model learned its relationship with steam output.'
        )

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### 🔥 NCV')
        st.write(f'**Current:** {NCV:.2f} MJ/kg | **Dataset range:** 5.8–18.0 | **Average:** 11.1')
        st.write(
            f'NCV = GCV − 2.442 × moisture\n\n'
            f'= {GCV:.1f} − 2.442 × {moisture:.3f} = **{NCV:.2f} MJ/kg**\n\n'
            'The usable energy after paying the cost of evaporating moisture. '
            'This is the most important single number for steam output prediction.'
        )
        st.markdown('#### ♻️ Combustible Fraction')
        st.write(f'**Current:** {combustible:.3f} | **Dataset range:** 0.03–0.43 | **Average:** 0.230')
        st.write(
            f'Combustible = moisture − ash = {moisture:.3f} − {ash:.3f} = **{combustible:.3f}**\n\n'
            'Captures the net balance between excess water and excess inert mineral. '
            'Higher = more organic matter available to burn.'
        )
        st.markdown('#### ⚡ Q_input (Thermal Input)')
        st.write(f'**Current:** {Q_input:.1f} MW | **Dataset range:** 17–95 MW | **Average:** 46 MW')
        st.write(
            f'Q_input = waste_flow × 1000 × NCV / 3600\n\n'
            f'= {waste_flow:.1f} × 1000 × {NCV:.2f} / 3600 = **{Q_input:.1f} MW**\n\n'
            'Total thermal power entering the boiler. XGBoost ranks this as the #1 feature by importance.'
        )
    with c2:
        st.markdown('#### ⭐ Combustion Quality Score')
        st.write(f'**Current:** {combustion_quality:.3f} | **Dataset range:** −1.23 to +0.43 | **Average:** −0.257')
        st.progress(float(np.clip((combustion_quality + 1.23) / (0.43 + 1.23), 0.0, 1.0)))
        st.markdown(
            '**Composite score — not a sensor reading.** Three CEMS signals combined:\n\n'
            f'| Term | Formula | Value |\n'
            f'|---|---|---|\n'
            f'| Temperature ↑ | (T−780)/170 | +{(T_comb-780)/(950-780):.3f} |\n'
            f'| CO penalty ↓ | −(CO−5)/75 | −{(CO-5)/(80-5):.3f} |\n'
            f'| O₂ deviation ↓ | −\|O₂−7\|/3 | −{abs(O2-7)/3:.3f} |\n\n'
            f'**Score = {combustion_quality:.3f}**\n\n'
            'Dataset average is −0.257 because typical temperature (836 °C) is below mid-range '
            'and CO/O₂ penalties are always present. A score near 0 or positive means '
            'all three signals are simultaneously good.'
        )
        st.markdown('#### 📅 Month')
        import calendar
        st.write(f'**Current:** {calendar.month_name[month]}')
        st.write(
            'Captures seasonal waste composition. Summer: drier waste, higher GCV → more steam. '
            'Winter: more food/garden waste → higher moisture, lower GCV → less steam. '
            'The model learned this pattern from a full year of training data.'
        )

st.divider()

# ── Debug expander ─────────────────────────────────────────────────────────────
with st.expander('🔍 Feature vector + raw model outputs'):
    st.write(f'**Physics:** {pred_physics:.4f} t/h')
    st.write(f'**Pure ML (raw):** {pred_ml:.4f} t/h')
    st.write(f'**Correction (residual):** {correction:.4f} t/h')
    st.write(f'**Hybrid = {pred_physics:.4f} + {correction:.4f} = {pred_hybrid:.4f} t/h**')
    st.dataframe(
        pd.DataFrame(features, columns=FEATURE_COLS).T
          .rename(columns={0: 'value'})
          .style.format('{:.4f}')
    )
