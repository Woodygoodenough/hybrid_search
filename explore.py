# %%
from timer import Timer, SearchMethod
from search import Search
from shared_dataclasses import Predicate

timer = Timer()
search = Search(timer)

query_embedding = search.embedder.encode_query("test query")
predicates = [Predicate(key="revdate", value="2025-01-15", operator=">=")]
k = 10

with timer.method_context(SearchMethod.BASE_PRE_SEARCH):
    for _ in range(10):
        with timer.run():
            results = search.base_pre_search(query_embedding, predicates, k)


# %%
timer.runs
# %%
timer._current_run
# %%
timer.method
# %%
