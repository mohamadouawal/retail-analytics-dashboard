# # logic/apriori_rules.py
# import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules

# def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
#     """Generates rules using dynamically provided transaction and item columns."""
#     # Count occurrences of items per transaction
#     basket = (df.groupby([tx_col, item_col])[item_col]
#               .count().unstack().reset_index().fillna(0)
#               .set_index(tx_col))
    
#     # Map to explicitly boolean True/False to prevent memory overloads in mlxtend
#     basket_sets = basket.map(lambda x: bool(x >= 1))
    
#     frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
#     if frequent_itemsets.empty:
#         return pd.DataFrame()
        
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
#     return rules.sort_values(by='lift', ascending=False)

# logic/apriori_rules.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
    """Generates rules using dynamically provided transaction and item columns."""
    
    if df.empty:
        return pd.DataFrame()

    # MEMORY OPTIMIZATION 1: Prune mathematically unviable items.
    # If an item doesn't meet the minimum support on its own, it cannot be part 
    # of a frequent itemset. Dropping these before unstacking drastically shrinks matrix width.
    total_tx = df[tx_col].nunique()
    min_item_occurrences = total_tx * min_support
    
    item_counts = df[item_col].value_counts()
    viable_items = item_counts[item_counts >= min_item_occurrences].index
    filtered_df = df[df[item_col].isin(viable_items)]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Count occurrences of items per transaction
    basket = (filtered_df.groupby([tx_col, item_col])[item_col]
              .count().unstack().fillna(0))
    
    # MEMORY OPTIMIZATION 2: Native Vectorized Booleans
    # Replaces the slow, heavy lambda map with native Pandas boolean logic.
    # This specifically fixes the mlxtend deprecation warnings in your logs.
    basket_sets = basket > 0
    
    # MEMORY OPTIMIZATION 3: low_memory flag
    # Instructs mlxtend to use an iterator for combination searches rather than holding 
    # all candidates in RAM simultaneously.
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

# logic/apriori_rules.py
# import pandas as pd
# from mlxtend.frequent_patterns import fpgrowth, association_rules

# def generate_recommendations(df, tx_col, item_col, min_support=0.02, min_confidence=0.5):
#     """
#     Generates association rules using memory-efficient FP-Growth.
#     """
#     try:
#         if df.empty or len(df[item_col].dropna()) == 0:
#             return pd.DataFrame()

#         # 1. Cap items to the top 500 most frequent to prevent 1 GB RAM explosion
#         top_items = df[item_col].value_counts().nlargest(500).index
#         filtered = df[df[item_col].isin(top_items)].copy()

#         # 2. Group items into basket matrix
#         basket = (filtered.groupby([tx_col, item_col])[item_col]
#                   .count().unstack().fillna(0))

#         # 3. Convert strictly to bool type (requires much less memory than int/float)
#         basket_sets = basket.astype(bool)

#         if basket_sets.empty or basket_sets.shape[1] < 2:
#             return pd.DataFrame()

#         # 4. FP-Growth algorithm (memory-efficient & fast)
#         frequent_itemsets = fpgrowth(
#             basket_sets, 
#             min_support=min_support, 
#             use_colnames=True, 
#             max_len=3
#         )

#         if frequent_itemsets.empty:
#             return pd.DataFrame()

#         # 5. Generate association rules
#         rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
#         return rules.sort_values(by='lift', ascending=False)
        
#     except Exception:
#         # Prevent server termination on unexpected data shapes
#         return pd.DataFrame()