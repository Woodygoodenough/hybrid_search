# %%
from shared_dataclasses import Predicate
from search import Search

query = "artificial intelligence"
predicates = [Predicate(key="token_count", value=100, operator="<")]
k = 3


search = Search()
# %%
with search as search:
    results = search.search(query, predicates, k, method="post_search")
results


# %%
results.to_df()
# %%
