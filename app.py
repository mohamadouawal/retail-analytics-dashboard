# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from logic.apriori_rules import generate_recommendations
from logic.anomaly_model import detect_anomalies
from utils.translations import UI_TEXT

# 1. Page Configuration
st.set_page_config(page_title="Retail Analytics", page_icon="🛒", layout="wide")

# Apply custom typography
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# 2. State Management for Language
lang = st.sidebar.radio("Language / Dil", ["EN", "TR"])
t = UI_TEXT[lang]

st.title(t["title"])
st.markdown(t["description"])

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# 3. File Uploader
uploaded_file = st.file_uploader(t["upload_label"], type=["csv", "xlsx"], help=t["upload_help"])

if uploaded_file is not None:
    # Handle both CSV and Excel uploads with encoding fallbacks
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    else:
        df = pd.read_excel(uploaded_file)
        
    # Extract dataset schema automatically
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if not num_cols: num_cols = all_cols
    if not cat_cols: cat_cols = all_cols

    # Sidebar Mapping Interface
    st.sidebar.header(t["map_header"])
    tx_col = st.sidebar.selectbox(t["map_tx"], all_cols, index=0)
    item_col = st.sidebar.selectbox(t["map_item"], all_cols, index=min(1, len(all_cols)-1))
    qty_col = st.sidebar.selectbox(t["map_qty"], num_cols, index=0)
    price_col = st.sidebar.selectbox(t["map_price"], num_cols, index=min(1, len(num_cols)-1))

    # Force numeric conversion for slider & metrics calculation
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[qty_col, price_col])

    if df.empty:
        st.warning("The selected metric columns do not contain valid numeric data. Please map different columns. / Seçilen metrik sütunları geçerli sayısal veri içermiyor. Lütfen farklı sütunlar eşleştirin.")
        st.stop()

    # Extract slider boundaries safely
    qty_min, qty_max = float(df[qty_col].min()), float(df[qty_col].max())
    price_min, price_max = float(df[price_col].min()), float(df[price_col].max())

    if qty_min == qty_max:
        qty_max = qty_min + 1.0
    if price_min == price_max:
        price_max = price_min + 1.0

    # Real-Time Interactive Filters
    st.sidebar.header(t["filter_header"])
    min_qty = st.sidebar.slider(t["filter_min_qty"].format(qty=qty_col), qty_min, qty_max, qty_min)
    max_price = st.sidebar.slider(t["filter_max_price"].format(price=price_col), price_min, price_max, price_max)

    # 4. Filter the dataset BEFORE passing it to the algorithms
    filtered_df = df[(df[qty_col] >= min_qty) & (df[price_col] <= max_price)].copy()

    if filtered_df.empty:
        st.warning("No data matches your current filter criteria. Please adjust the sliders. / Mevcut filtre kriterlerine uyan veri bulunamadı. Lütfen kaydırıcıları ayarlayın.")
        st.stop()

    # 5. Real-Time Model Execution
    with st.spinner("Analyzing Data in Real-Time..."):
        recommendations = generate_recommendations(filtered_df, tx_col, item_col, min_support=0.02, min_confidence=0.5)
        
        anomaly_results = detect_anomalies(filtered_df, feature_cols=[qty_col, price_col], contamination=0.01)
        filtered_df['Status'] = anomaly_results['Anomaly_Score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        anomalies = filtered_df[filtered_df['Status'] == 'Anomaly']

    # 6. Dashboard & Tab Rendering
    tab1, tab2, tab3 = st.tabs([t["tab_dashboard"], t["tab_recommendations"], t["tab_anomalies"]])

    with tab1:
        # Unique recommended products count
        if not recommendations.empty:
            unique_recs = len(set(item for sublist in recommendations['consequents'] for item in sublist))
        else:
            unique_recs = 0

        # Render 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["metric_products"], filtered_df[item_col].nunique())
        m2.metric(t["metric_recs"], unique_recs)
        m3.metric(t["metric_anomalies"], len(anomalies))
        m4.metric(t["metric_tx"], filtered_df[tx_col].nunique())

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            top_products = filtered_df.groupby(item_col)[qty_col].sum().nlargest(10).reset_index()
            fig_bar = px.bar(
                top_products, x=qty_col, y=item_col, 
                orientation='h', 
                title=t["chart_top_products"].format(item=item_col, qty=qty_col),
                color=qty_col, color_continuous_scale='Blues'
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

        with col2:
            status_counts = filtered_df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_pie = px.pie(
                status_counts, values='Count', names='Status', 
                title=t["chart_anomaly_dist"], 
                color='Status', color_discrete_map={'Normal':'#2ecc71', 'Anomaly':'#e74c3c'}
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

        fig_scatter = px.scatter(
            filtered_df, x=qty_col, y=price_col, color='Status',
            title=t["chart_scatter"].format(qty=qty_col, price=price_col), 
            hover_data=all_cols,
            color_discrete_map={'Normal':'#3498db', 'Anomaly':'#e74c3c'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart")

    with tab2:
        if not recommendations.empty:
            recommendations['antecedents'] = recommendations['antecedents'].apply(lambda x: ', '.join(map(str, x)))
            recommendations['consequents'] = recommendations['consequents'].apply(lambda x: ', '.join(map(str, x)))

            display_df = recommendations[['antecedents', 'consequents', 'confidence', 'lift']].head(50)
            display_df.columns = [t["col_antecedents"], t["col_consequents"], t["col_confidence"], t["col_lift"]]

            st.dataframe(display_df, use_container_width=True)
            st.download_button(label=t["btn_download_csv"], data=convert_df(display_df), file_name='recommendations.csv', mime='text/csv')
        else:
            st.info(t["msg_no_rules"])

    with tab3:
        st.write(f"Flagged {len(anomalies)} isolated transactions for review.")
        st.dataframe(anomalies, use_container_width=True)

        if not anomalies.empty:
            csv_anomalies = convert_df(anomalies)
            st.download_button(label=t["btn_download_csv"], data=csv_anomalies, file_name='anomalies.csv', mime='text/csv')