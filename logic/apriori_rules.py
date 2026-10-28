# logic/apriori_rules.py
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
    """Generates rules using TransactionEncoder for minimal RAM usage."""
    
    if df.empty:
        return pd.DataFrame()

    # 1. Pre-Pruning to save memory
    total_tx = df[tx_col].nunique()
    min_item_occurrences = total_tx * min_support
    
    item_counts = df[item_col].value_counts()
    viable_items = item_counts[item_counts >= min_item_occurrences].index
    filtered_df = df[df[item_col].isin(viable_items)]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # 2. ULTIMATE MEMORY OPTIMIZATION: Use TransactionEncoder instead of unstack()
    # This prevents Pandas from creating a massive float64 pivot table in RAM
    transactions = filtered_df.groupby(tx_col)[item_col].apply(list).values.tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=False)
    basket_sets = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 3. ENVIRONMENT TOGGLE
    # Ensure this is set in your Streamlit Cloud Secrets!
    is_cloud = os.environ.get("STREAMLIT_CLOUD") == "true"
    
    try:
        if is_cloud:
            # Cloud Execution (FP-Growth)
            frequent_itemsets = fpgrowth(
                basket_sets, 
                min_support=min_support, 
                use_colnames=True
            )
        else:
            # Local Execution (Apriori)
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
        
    except MemoryError:
        # Final fail-safe to prevent hard app crashes if the dataset is simply too large
        return pd.DataFrame()