import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from typing import Optional, List, Tuple
from vector_index import Embedder, FaissIVFIndex
from settings import FAISS_PATH
from dbManagement import DbManagement, ArticleRecord
import pandas as pd
from vector_index import FilterIds


class Search:
    def __init__(self):
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = DbManagement()

    def simple_search(
        self, query: str, k: int, item_ids: Optional[List[int]] = None
    ) -> Tuple[List[ArticleRecord], List[float]]:
        query_vec = self.embedder.encode_query(query)
        similarities, indices = self.index.search(
            query_vec,
            k,
            filter_ids=FilterIds.from_list(item_ids) if item_ids is not None else None,
        )
        similarities, indices = similarities.squeeze(), indices.squeeze()
        # notice search always returns 2d arrays, so we need to squeeze them to 1d arrays for single query search
        # sqlite expects int, not numpy int64
        # explore further optimizations if needed
        records = [self.db.get_from_item_id(int(index)) for index in indices]
        return records, similarities

    @staticmethod
    def search_results_to_df(
        records: List[ArticleRecord], similarities: List[float]
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            records,
            columns=["item_id", "title", "url", "revdate", "token_count", "entity"],
        )
        df["similarity"] = similarities
        return df


def main():
    print("simple test")
    search = Search()
    results = search.simple_search("machine learning algorithms", k=1)
    print(results)


if __name__ == "__main__":
    main()
