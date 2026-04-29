import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title='Telco Churn Analysis',
    page_icon='📡',
    layout='wide'
)

# ── Load model and scaler ─────────────────────────────────
@st.cache_resource
def load_models():
    with open('app/model/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('app/model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/Telco-Customer-Churn.csv')

model, scaler = load_models()
raw_df = load_data()

# ── Helper: preprocess ────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0).astype(float)

    cols_to_fix = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies','MultipleLines']
    for col in cols_to_fix:
        df[col] = df[col].replace({
            'No internet service': 'No',
            'No phone service':    'No'
        })

    binary_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                   'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                   'StreamingTV','StreamingMovies','PaperlessBilling']
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, columns=['Contract','InternetService','PaymentMethod'],
                        drop_first=True, dtype=int)

    expected_cols = model.get_booster().feature_names
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])
    
    df = df[expected_cols]


    return df

# ── Preprocess full dataset once ──────────────────────────
@st.cache_data
def get_processed():
    return preprocess(raw_df)

processed_df = get_processed()
all_probs    = model.predict_proba(processed_df)[:, 1]

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title('📡 Telco Churn Analysis')
st.sidebar.markdown('Navigate through the sections below.')
page = st.sidebar.radio('Go to', [
    '🏠 Overview',
    '🔮 Churn Prediction',
    '👥 Segment Analysis',
    '🔍 SHAP Explainability'
])

# ─────────────────────────────────────────────────────────
# PAGE 1 — Overview
# ─────────────────────────────────────────────────────────
if page == '🏠 Overview':
    st.title('Telco Customer Churn Analysis')
    st.markdown("""
    This dashboard combines CUSTOMER SEGMENATION and CHURN PREDICTION
    to deliver actionable retention strategies for a telecom company.
    """)

    st.markdown('---')

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total customers',  '7,043')
    col2.metric('Overall churn rate', '26.5%')
    col3.metric('Model AUC',        '0.833')
    col4.metric('Segments found',   '4')

    st.markdown('---')

    # Segment summary table
    st.subheader('Customer segment summary')
    summary = pd.DataFrame({
        'Segment':           ['High-value at-risk', 'New customers',
                              'High-value loyal',   'Long-term stable'],
        'Customers':         [1943, 2038, 1957, 1105],
        'Churn rate':        ['55%', '23%', '14%', '5%'],
        'Avg monthly ($)':   [84.71, 37.11, 91.72, 32.93],
        'Avg tenure (mo)':   [15, 12, 59, 54],
        'Priority':          ['🔴 URGENT', '🟡 NURTURE',
                              '🔵 MAINTAIN', '🟢 REWARD'],
        'Key action':        ['Contract upgrade offer',
                              '90-day onboarding program',
                              'Loyalty reward program',
                              'Referral + upsell program']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('---')

    # Quick churn overview chart
    st.subheader('Churn distribution')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        churn_counts = raw_df['Churn'].value_counts()
        ax.pie(churn_counts, labels=['No churn', 'Churned'],
               autopct='%1.1f%%',
               colors=['#5DCAA5', '#D85A30'], startangle=90)
        ax.set_title('Overall churn distribution')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        contract_churn = raw_df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        contract_churn.columns = ['Contract', 'Churn rate (%)']
        bars = ax.bar(contract_churn['Contract'],
                      contract_churn['Churn rate (%)'],
                      color=['#D85A30', '#EF9F27', '#5DCAA5'],
                      edgecolor='none')
        for bar, val in zip(bars, contract_churn['Churn rate (%)']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10)
        ax.set_title('Churn rate by contract type')
        ax.set_ylabel('Churn rate (%)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────
# PAGE 2 — Churn Prediction
# ─────────────────────────────────────────────────────────
elif page == '🔮 Churn Prediction':
    st.title('Churn prediction')
    st.markdown('Upload a CSV of customers to get churn probability scores.')

    uploaded_file = st.file_uploader('Upload customer CSV', type=['csv'])

    if uploaded_file:
        upload_df = pd.read_csv(uploaded_file)
        st.subheader('Uploaded data preview')
        st.dataframe(upload_df.head(), use_container_width=True)

        if st.button('Run prediction', type='primary'):
            with st.spinner('Running predictions...'):
                try:
                    processed = preprocess(upload_df)
                    probs     = model.predict_proba(processed)[:, 1]
                    preds     = (probs >= 0.5).astype(int)

                    results = upload_df.copy()
                    results['Churn_Probability'] = probs.round(3)
                    results['Churn_Predicted']   = preds
                    results['Risk_Level']        = pd.cut(
                        probs,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['🟢 Low', '🟡 Medium', '🔴 High']
                    )

                    st.markdown('---')
                    st.subheader('Results')

                    col1, col2, col3 = st.columns(3)
                    col1.metric('Total customers',    len(results))
                    col2.metric('Predicted churners', int(preds.sum()))
                    col3.metric('Avg churn probability',
                                f'{probs.mean()*100:.1f}%')

                    st.dataframe(
                        results[['Churn_Probability','Churn_Predicted','Risk_Level']],
                        use_container_width=True
                    )

                    csv = results.to_csv(index=False)
                    st.download_button('⬇️ Download predictions as CSV',
                                       csv, 'churn_predictions.csv', 'text/csv')

                except Exception as e:
                    st.error(f'Preprocessing error: {e}')
    else:
        st.info('Upload a CSV file with the same columns as the Telco dataset to get started.')

# ─────────────────────────────────────────────────────────
# PAGE 3 — Segment Analysis
# ─────────────────────────────────────────────────────────
elif page == '👥 Segment Analysis':
    st.title('Customer segment analysis')

    segment_data = pd.DataFrame({
        'Segment':        ['High-value at-risk', 'New customers',
                           'High-value loyal',   'Long-term stable'],
        'Customers':      [1943, 2038, 1957, 1105],
        'Churn rate (%)': [55,   23,   14,   5],
        'Avg monthly ($)':[84.71,37.11,91.72,32.93],
        'Avg tenure':     [15,   12,   59,   54]
    })
    colors = ['#D85A30', '#EF9F27', '#378ADD', '#5DCAA5']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Churn rate by segment')
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(segment_data['Segment'],
                      segment_data['Churn rate (%)'],
                      color=colors, edgecolor='none')
        for bar, val in zip(bars, segment_data['Churn rate (%)']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val}%', ha='center', fontsize=10)
        ax.set_ylabel('Churn rate (%)')
        ax.set_ylim(0, 70)
        plt.xticks(rotation=15, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader('Avg monthly charges')
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(segment_data['Segment'],
                      segment_data['Avg monthly ($)'],
                      color=colors, edgecolor='none')
        for bar, val in zip(bars, segment_data['Avg monthly ($)']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'${val}', ha='center', fontsize=10)
        ax.set_ylabel('Avg monthly charges ($)')
        plt.xticks(rotation=15, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('Business recommendations per segment')

    recs = {
        'High-value at-risk': {
            'priority': '🔴 URGENT',
            'actions': [
                'Offer 20-30% discount on 1 or 2-year contract upgrade',
                'Provide free 3-month trial of OnlineSecurity + TechSupport',
                'Incentivize switch from electronic check to auto-pay',
                'Flag for immediate customer service outreach (churn prob ≥ 0.5)'
            ]
        },
        'New customers': {
            'priority': '🟡 NURTURE',
            'actions': [
                'Launch structured 90-day onboarding program',
                'Offer discounted service bundle upgrade at month 3',
                'Provide $5/month discount for switching to auto-pay',
                'Offer free month loyalty bonus for early 1-year commitment'
            ]
        },
        'High-value loyal': {
            'priority': '🔵 MAINTAIN',
            'actions': [
                'Launch exclusive loyalty reward program for tenure > 36 months',
                'Run annual satisfaction surveys proactively',
                'Alert when churn probability crosses 0.3 for VIP outreach',
                'Offer premium add-on upsells (higher speed, device protection)'
            ]
        },
        'Long-term stable': {
            'priority': '🟢 REWARD',
            'actions': [
                'Acknowledge tenure milestones with thank-you offers',
                'Gently introduce fiber optic or premium bundles',
                'Launch referral program with bill credits',
                'Avoid over-investing — churn risk is only 5%'
            ]
        }
    }

    for segment, info in recs.items():
        with st.expander(f"{info['priority']} — {segment}"):
            for action in info['actions']:
                st.markdown(f'- {action}')

# ─────────────────────────────────────────────────────────
# PAGE 4 — SHAP Explainability
# ─────────────────────────────────────────────────────────
elif page == '🔍 SHAP Explainability':
    st.title('SHAP explainability')
    st.markdown('Understand **why** the model predicts a specific customer will churn.')

    # Customer selector
    customer_idx = st.slider('Select customer index',
                              0, len(processed_df)-1, 0)

    # Customer info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Churn probability',
                f'{all_probs[customer_idx]*100:.1f}%')
    col2.metric('Risk level',
                '🔴 High' if all_probs[customer_idx] >= 0.5 else '🟢 Low')
    col3.metric('Actual churn',
                raw_df.iloc[customer_idx]['Churn'])
    col4.metric('Monthly charges',
                f"${raw_df.iloc[customer_idx]['MonthlyCharges']}")

    st.subheader('Customer profile')
    st.dataframe(raw_df.iloc[[customer_idx]], use_container_width=True)

    if st.button('Explain this prediction', type='primary'):
        with st.spinner('Calculating SHAP values...'):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_df)

            shap_exp = shap.Explanation(
                values=shap_values[customer_idx],
                base_values=explainer.expected_value,
                data=processed_df.iloc[customer_idx].values,
                feature_names=processed_df.columns.tolist()
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(shap_exp, show=False, max_display=12)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown('---')
            st.subheader('How to read this chart')
            st.markdown("""
            - **Red bars (+)** → push the prediction **toward churn**
            - **Blue bars (-)** → push the prediction **away from churn**
            - **Longer bar** → stronger influence on the prediction
            - **E[f(x)]** → average prediction across all customers (starting point)
            - **f(x)** → final prediction score for this specific customer
            """)