# logic/apriori_rules.py
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
    """Generates rules using Apriori locally, and FP-Growth on the cloud."""
    
    if df.empty:
        return pd.DataFrame()

    # Pre-Pruning to save memory in both algorithms
    total_tx = df[tx_col].nunique()
    min_item_occurrences = total_tx * min_support
    
    item_counts = df[item_col].value_counts()
    viable_items = item_counts[item_counts >= min_item_occurrences].index
    filtered_df = df[df[item_col].isin(viable_items)]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Group items into a boolean matrix
    basket = (filtered_df.groupby([tx_col, item_col])[item_col]
              .count().unstack().fillna(0))
    basket_sets = basket > 0
    
    # ENVIRONMENT TOGGLE
    # Checks if we explicitly set the 'STREAMLIT_CLOUD' variable to 'true'
    is_cloud = os.environ.get("STREAMLIT_CLOUD") == "true"
    
    if is_cloud:
        # Use FP-Growth for the Cloud (Memory Efficient)
        frequent_itemsets = fpgrowth(
            basket_sets, 
            min_support=min_support, 
            use_colnames=True
        )
    else:
        # Use Apriori for Local/Academic Execution
        frequent_itemsets = apriori(
            basket_sets, 
            min_support=min_support, 
            use_colnames=True, 
            low_memory=True
        )
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values(by='lift', ascending=False)

