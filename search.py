import numpy as np
from typing import Optional, List, Tuple
from vector_index import Embedder, FaissIVFIndex
from settings import FAISS_PATH
class Search:
    def __init__(self):
        self.embedder = Embedder()
        self.index = FaissIVFIndex.load(FAISS_PATH)

    def simple_search(self, query: str, k: int) -> List[Tuple[int, float, float]]:
        query_vec = self.embedder.encode_query(query)
        distances, indices = self.index.search(query_vec, k)
        return [(indices[i], distances[i]) for i in range(len(indices))]

        

def main():
    search = Search()
    results = search.simple_search("machine learning algorithms", k=1)
    print(results)

if __name__ == "__main__":
    main()