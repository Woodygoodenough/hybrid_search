import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from dataclasses import dataclass
from typing import Optional, List, Tuple
from vector_index import Embedder, FaissIVFIndex
from settings import FAISS_PATH, ITEM_COLS_DB, NPROBE, NLIST
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

    def to_df(self) -> pd.DataFrame:
        db_records = []
        similarities = []
        for result in self.results:
            db_records.append(result.record)
            similarities.append(result.similarity)
        df = DBRecords(records=db_records).to_df()
        df["similarity"] = similarities
        return df.sort_values(by="similarity", ascending=False)


class Search:
    def __init__(self):
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = DbManagement()

    def simple_search(
        self, query: str, k: int, item_ids: Optional[List[int]] = None
    ) -> Tuple[List[DBRecord], List[float]]:
        query_vec = self.embedder.encode_query(query)
        similarities, indices = self.index.search(
            query_vec,
            k,
            item_ids=item_ids,
        )
        similarities, indices = similarities.squeeze(), indices.squeeze()
        # notice search always returns 2d arrays, so we need to squeeze them to 1d arrays for single query search
        # sqlite expects int, not numpy int64
        # explore further optimizations if needed
        records = [self.db.get_from_item_id(int(index)) for index in indices]
        return records, similarities

    @time_function
    def pre_search(self, query: str, predicates: List[Predicate], k: int) -> HSearchResults:
        """
        Pre-filter strategy: Apply predicates first, then do ANN search.
        Compare ANN results with min(k, predicate_results).
        If len not same, increase nprobe and retry.
        """
        db_records = self.db.predicates_search(predicates)
        predicate_count = len(db_records.records)
        item_ids = [record.item_id for record in db_records.records]
        
        # Use min(k, predicate_count) as search target
        search_k = min(k, predicate_count)
        # debug part
        if search_k != k:
            print(f"We do not have enough records to satisfy the {k} results, at most we can return {search_k} results")
        else:
            print(f"We have enough records to satisfy the {k} results, we will return {search_k} results")
        query_vec = self.embedder.encode_query(query)
        nprobe = NPROBE
        while nprobe < NLIST:
            # debug part
            print(f"Searching with # nprobe: {nprobe}")
            ann_results = self.index.search(
                query_vec,
                search_k,
                nprobe=nprobe,
                item_ids=item_ids,
            )
            valid_item_ids = [item_id for item_id in ann_results.item_ids if item_id != -1]
            if len(valid_item_ids) == search_k:
                print(f"Search results returned exactly {search_k} results, we will return them")
                break
            elif len(valid_item_ids) > search_k:
                raise ValueError(f"Search results returned more than {search_k} results, check implementation")
            else:
                print(f"Search results returned less than {search_k} results, we will increase nprobe to {nprobe * 2}")
                nprobe = nprobe * 2
        
        top_results = self._intersect(ann_results, db_records)
        return HSearchResults(results=top_results, is_k=len(top_results) == k)

    @staticmethod
    def _intersect(ann_results: AnnSearchResults, db_records: DBRecords) -> List[HSearchResult]:
        db_records_dict = db_records.to_dict()
        ann_results_dict = ann_results.to_valid_dict()
        return [HSearchResult(record=db_records_dict[item_id], similarity=similarity) for item_id, similarity in ann_results_dict.items() if item_id in db_records_dict]
        


def main():
    print("simple test")
    search = Search()
    results = search.simple_search("machine learning algorithms", k=1)
    print(results)


if __name__ == "__main__":
    main()
