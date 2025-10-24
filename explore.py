# %%
from search import Search
from dbManagement import Predicate
from settings import DISPLAY_COLS
# %%
search = Search()
predicates = [Predicate(key="token_count", value=100, operator="<")]
results = search.hybrid_search("artificial intelligence", predicates, 3)
# %%
results.to_df(show_cols=DISPLAY_COLS)
# %%
