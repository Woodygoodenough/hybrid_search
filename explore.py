# %%
from timer import Timer
from search import Search
from shared_dataclasses import Predicate

timer = Timer()
search = Search(timer)

query_embedding = search.embedder.encode_query("test query")
predicates = [Predicate(key="item_id", value=[1, 2, 3], operator="IN")]
k = 10

results = search.base_pre_search(query_embedding, predicates, k)
# %%
timer.raw_times

# %%
timer.runs
# %%
timer.save_run()
# %%
timer.runs
# %%
