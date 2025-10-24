import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from dataclasses import dataclass
from typing import Optional, List, Tuple
from vector_index import Embedder, FaissIVFIndex
from settings import FAISS_PATH, NPROBE, NLIST, N
from dbManagement import DbManagement, DBRecord, DBRecords, Predicate
import pandas as pd
import time
from vector_index import AnnSearchResults
#decorator to time the function
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper

@dataclass
class HSearchResult:
    record: DBRecord
    similarity: float

@dataclass
class HSearchResults:
    results: List[HSearchResult]
    is_k: bool

    def to_df(self, show_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = DBRecords([result.record for result in self.results]).to_df()
        if show_cols is not None:
            df = df.loc[:, show_cols]
        df["similarity"] = [result.similarity for result in self.results] # always show similarity
        return df.sort_values(by="similarity", ascending=False)


class Search:
    def __init__(self):
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = DbManagement()


    def search(self, query: str, predicates: List[Predicate], k: int, method: str = "pre_search") -> HSearchResults:
        if method == "pre_search":
            return self.pre_search(query, predicates, k)
        elif method == "post_search":
            return self.post_search(query, predicates, k)
        elif method == "hybrid_search":
            pass
        else:
            raise ValueError(f"Invalid method: {method}")
    @time_function
    def pre_search(self, query: str, predicates: List[Predicate], k: int) -> HSearchResults:
        """
        Pre-filter strategy: Apply predicates first, then do ANN search.
        Compare ANN results with min(k, predicate_results).
        If len not same, increase nprobe and retry.
        """
        db_records = self.db.predicates_search(predicates)
        predicate_count = len(db_records)

        if predicate_count == 0:
            return HSearchResults(results=[], is_k=False)
        item_ids = [record.item_id for record in db_records.records]
        
        # Use min(k, predicate_count) as search target
        search_k = min(k, predicate_count)
        nprobe = NPROBE
        query_vec = self.embedder.encode_query(query)
        while nprobe <= NLIST:
            ann_results = self.index.search(
                query_vec,
                search_k,
                nprobe=nprobe,
                item_ids=item_ids,
            )
            valid_item_ids = [int(item_id) for item_id in (ann_results.item_ids if ann_results.item_ids.ndim > 0 else [ann_results.item_ids]) if item_id != -1]
            if len(valid_item_ids) == search_k:
                break
            elif len(valid_item_ids) > search_k:
                raise ValueError(f"Search results returned more than {search_k} results, check implementation")
            else:
                nprobe = nprobe * 2
        
        top_results = self._intersect(ann_results, db_records)
        return HSearchResults(results=top_results, is_k=len(top_results) == k)
    
    @time_function
    def post_search(self, query: str, predicates: List[Predicate], k: int) -> HSearchResults:
        """
        Post-filter strategy: Do ANN search first, then apply predicates to filter results.
        If not enough results after filtering, increase nprobe and retry.
        """
        query_vec = self.embedder.encode_query(query)
        nprobe = NPROBE
        
        search_k = min(k * 3, N)  # Search for 3x more results, cap at dataset size
        
        while True:
            ann_results = self.index.search(
                query_vec,
                search_k,
                nprobe=nprobe,
                item_ids=None,
            )
            
            # Get valid item IDs from ANN results
            valid_item_ids = [int(item_id) for item_id in (ann_results.item_ids if ann_results.item_ids.ndim > 0 else [ann_results.item_ids]) if item_id != -1]
            
            # Create a copy of predicates to avoid mutating the input
            combined_predicates = predicates.copy()
            item_id_predicate = Predicate(key="item_id", value=valid_item_ids, operator="IN")
            combined_predicates.append(item_id_predicate)
            
            # Apply predicates to filter the ANN results
            db_records = self.db.predicates_search(combined_predicates)
            if len(db_records) >= k:
                break
            else:
                if nprobe >= NLIST and search_k >= N:
                    break
                nprobe = min(nprobe * 2, NLIST)
                search_k = min(search_k * 3, N)  # Cap at dataset size
            

        all_results = self._intersect(ann_results, db_records)
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        top_results = all_results if len(all_results) <k else all_results[:k]
        return HSearchResults(results=top_results, is_k=len(top_results) == k)

    @time_function
    def hybrid_search(self, query: str, predicates: List[Predicate], k: int) -> HSearchResults:
        db_records = self.db.predicates_search(predicates)
        predicate_count = len(db_records)

        if predicate_count == 0:
            return HSearchResults(results=[], is_k=False)
        item_ids = [record.item_id for record in db_records.records]
        
        query_vec = self.embedder.encode_query(query)

        if predicate_count < k:
            print(f"We do not have enough records to satisfy the {k} results, at most we can return {predicate_count} results. But we still do the search to get similarities")
            # in this case, ann is going to be pointless, we do brute force search by specifying nprobe to NLIST
            ann_results = self.index.search(
                query_vec,
                predicate_count,
                nprobe=NLIST,
                item_ids=item_ids,
            )
            # notice here we do not need to filter out invalid item_ids: we are searching the entire dataset, and len(item_ids) == predicate_count
            results_count = len(ann_results)
        else:
            nprobe = self._find_optimal_nprobe(predicate_count, k, 0)
            while nprobe <= NLIST:
                print(f"Searching with # nprobe: {nprobe}")
                ann_results = self.index.search(
                    query_vec,
                    k,
                    nprobe=nprobe,
                    item_ids=item_ids,
                )
                valid_item_ids = [int(item_id) for item_id in (ann_results.item_ids if ann_results.item_ids.ndim > 0 else [ann_results.item_ids]) if item_id != -1]
                results_count = len(valid_item_ids)
                if results_count == k:
                    break
                elif results_count < k:
                    nprobe = self._find_optimal_nprobe(predicate_count, k - results_count, nprobe)
                    print(f"Search results returned {results_count} results, we will increase nprobe to {nprobe} to search for the remaining {k - results_count} results")

                else:
                    raise ValueError(f"Search results returned more than {k} results, check implementation")
        return HSearchResults(results=self._intersect(ann_results, db_records), is_k=results_count == k)
            
    @staticmethod
    def _intersect(ann_results: AnnSearchResults, db_records: DBRecords) -> List[HSearchResult]:
        db_records_dict = db_records.to_dict()
        ann_results_dict = ann_results.to_valid_dict()
        return [HSearchResult(record=db_records_dict[item_id], similarity=similarity) for item_id, similarity in ann_results_dict.items() if item_id in db_records_dict]
        
    @staticmethod
    def _find_optimal_nprobe(predicate_count: int, k_remaining: int, old_nprobe: int = 0) -> int:
        """
        Find the optimal nprobe for the query and predicates.
        """
        # each cluster is expected to have predicate_count / NLIST survivors
        # the expected number of clusters to search is k / (predicate_count / NLIST)
        expected_nprobe = int(k_remaining / (predicate_count / NLIST)) + old_nprobe
        # Use 3x multiplier for safety to ensure we get enough results
        return max(3, min(3 * expected_nprobe, NLIST))