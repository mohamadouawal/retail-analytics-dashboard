# logic/anomaly_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, feature_cols, contamination=0.01):
    """Uses Isolation Forest on dynamically selected numerical columns."""
    clean_df = df.dropna(subset=feature_cols).copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    
    clean_df['Anomaly_Score'] = model.fit_predict(clean_df[feature_cols])
    return clean_df