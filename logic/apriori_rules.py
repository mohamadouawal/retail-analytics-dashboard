# logic/apriori_rules.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
    """Generates rules using dynamically provided transaction and item columns."""
    # Count occurrences of items per transaction
    basket = (df.groupby([tx_col, item_col])[item_col]
              .count().unstack().reset_index().fillna(0)
              .set_index(tx_col))
    
    # Map to explicitly boolean True/False to prevent memory overloads in mlxtend
    basket_sets = basket.map(lambda x: bool(x >= 1))
    
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values(by='lift', ascending=False)