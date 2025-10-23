# %%
from search import Search
from dbManagement import Predicate
from settings import DISPLAY_COLS
search = Search()

# %% Test pre-filter strategy
print("=== Pre-filter Strategy ===")
k = 10
predicates = [
    Predicate(key="token_count", value=52, operator="<"), 
    Predicate(key="revdate", value="2025-02-01", operator=">="),
    ]
print(f"We are searching for {k} results with predicates: {predicates}")
search_results = search.pre_search("machine learning", predicates, k)
print(f"Do we have enough records to satisfy the {k} results? {search_results.is_k}")
search_results.to_df(show_cols=DISPLAY_COLS)
# %%
