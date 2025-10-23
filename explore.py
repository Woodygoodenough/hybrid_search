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
from dbManagement import DbManagement
db = DbManagement()
predicates = [
    Predicate(key="token_count", value=52, operator="<"), 
    Predicate(key="revdate", value="2025-02-01", operator=">="),
    Predicate(key="item_id", value=[12, 23, 46, 34567], operator="IN"),
]
records = db.predicates_search(predicates)
# %%
predicates = [
    Predicate(key="token_count", value=4000, operator=">"), 
    Predicate(key="item_id", value=[12, 23, 46, 34567], operator="IN"),
]
records = db.predicates_search(predicates)
records.to_df(show_cols=DISPLAY_COLS)

# %%
from search import Search
from dbManagement import Predicate
from settings import DISPLAY_COLS
search = Search()
predicates = [
    Predicate(key="token_count", value=52, operator="<"), 
    Predicate(key="revdate", value="2025-02-01", operator=">="),
]
search_results_pre = search.pre_search("machine learning", predicates, 10)
search_results_post = search.post_search("machine learning", predicates, 10)
search_results_pre.to_df(show_cols=DISPLAY_COLS)
# %%
search_results_post.to_df(show_cols=DISPLAY_COLS)
# %%
predicates = [
    Predicate(key="token_count", value=53, operator="="),  # Exact match
]
search_results_pre = search.pre_search("machine learning", predicates, 10)
search_results_post = search.post_search("machine learning", predicates, 10)
search_results_pre.to_df(show_cols=DISPLAY_COLS)
# %%
search_results_post.to_df(show_cols=DISPLAY_COLS)

# %%
