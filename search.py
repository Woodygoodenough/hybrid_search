import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from dataclasses import dataclass
from typing import Optional, List, Literal
import numpy as np
from vector_index import Embedder, FaissIVFIndex, AnnSearchResults
from settings import FAISS_PATH, NPROBE, NLIST, N
from dbManagement import DbManagement, DBRecord, DBRecords
from shared_dataclasses import Predicate
from histo2d import Histo2D
import pandas as pd
from settings import N_PER_CLUSTER
from timer import Timer, Section


@dataclass
class HSearchResult:
    record: DBRecord
    similarity: float


@dataclass
class HSearchResults:
    results: List[HSearchResult]
    is_k: bool

    def __post_init__(self):
        self.results.sort(key=lambda x: x.similarity, reverse=True)

    def to_df(self, show_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = DBRecords([result.record for result in self.results]).to_df()
        if show_cols is not None:
            df = df.loc[:, show_cols]
        df["similarity"] = [
            result.similarity for result in self.results
        ]  # always show similarity
        return df.sort_values(by="similarity", ascending=False)


class Search:
    def __init__(self, timer: Timer = None):
        self.timer = timer or Timer()
        self.time_section = self.timer.section
        self.time_method = self.timer.method_context
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = DbManagement()
        self.histo = Histo2D.from_records(self.db.predicates_search([]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def close(self):
        """Close search resources."""
        self.db.close()

    def search(
        self,
        query: str,
        predicates: List[Predicate],
        k: int,
        method: Literal["pre_search", "post_search"] = "post_search",
    ) -> HSearchResults:
        embedding_query = self.embedder.encode_query(query)
        est_survivors = self.histo.estimate_survivors(predicates)
        if method == "pre_search":
            return self.base_pre_search(embedding_query, predicates, k)
        elif method == "post_search":
            return self.base_pos_search(embedding_query, predicates, k, est_survivors)
        else:
            raise ValueError(f"Invalid method: {method}")

    @staticmethod
    def _which_adap(
        est_survivors: int,
        k: int,
    ) -> Literal["adap_pre_search", "adap_pos_search"]:
        # based on how many we want to find and how many survivors we expect, we decide which adap method to use
        pass

    def base_pre_search(
        self, embedding_query: np.ndarray, predicates: List[Predicate], k: int
    ) -> HSearchResults:
        with self.time_section(Section.TOTAL):
            with self.time_section(Section.DB_SEARCH):
                pre_db_records = self.db.predicates_search(predicates)
                pre_predicate_count = len(pre_db_records)
                if pre_predicate_count == 0:
                    return HSearchResults(results=[], is_k=False)
            # this automatically goes to residual time
            pre_item_ids = [record.item_id for record in pre_db_records.records]
            nprobe = 64
            ann_results = AnnSearchResults.empty()
            search_k = min(pre_predicate_count, k)
            while len(ann_results) < search_k:
                with self.time_section(Section.FAISS_SEARCH):
                    ann_results = self.index.search(
                        embedding_query,
                        search_k,
                        nprobe=nprobe,
                        item_ids=pre_item_ids,
                    )
                with self.time_section(Section.UD_PARAMS):
                    nprobe = min(NLIST, nprobe * 3)
            with self.time_section(Section.INTERSECT):
                pre_top_results = self._intersect(ann_results, pre_db_records)
            with self.time_section(Section.FINALIZE):
                return HSearchResults(
                    results=pre_top_results[:search_k],
                    is_k=len(pre_top_results) == search_k,
                )

    def base_pos_search(
        self,
        embedding_query: np.ndarray,
        predicates: List[Predicate],
        k: int,
    ) -> HSearchResults:

        with self.time_section(Section.TOTAL):
            search_k = 100
            nprobe = 64
            top_results: List[HSearchResult] = []

            # for pos search, we do not know how many results we will get
            # we keep trying, untill we get k, or we reach the full dataset
            while len(top_results) < k:
                # Get next search parameters with aggressive growth
                # If we've reached the full dataset, search everything
                if search_k >= N:
                    search_k = N
                    nprobe = NLIST

                with self.time_section(Section.FAISS_SEARCH):
                    ann_results = self.index.search(
                        embedding_query, search_k, nprobe=nprobe, item_ids=None
                    )

                item_ids_predicate = Predicate(
                    key="item_id",
                    value=ann_results.item_ids.tolist(),
                    operator="IN",
                )
                predicates_copy = predicates.copy()
                predicates_copy.append(item_ids_predicate)
                with self.time_section(Section.DB_SEARCH):
                    post_db_records = self.db.predicates_search(predicates_copy)
                with self.time_section(Section.INTERSECT):
                    top_results = self._intersect(ann_results, post_db_records)

                # Break if we've reached the full dataset
                if search_k >= N:
                    break
                with self.time_section(Section.UD_PARAMS):
                    search_k = min(search_k * 3, N)
                    nprobe = min(NLIST, nprobe * 3)

            # HSearchResults always sorts by similarity
            with self.time_section(Section.FINALIZE):
                return HSearchResults(
                    # [:k] will limit the results to k items
                    # or all items if we have less than k (native python behavior)
                    results=top_results[:k],
                    is_k=len(top_results) == k,
                )

    def adap_pre_search(
        self, embedding_query: np.ndarray, predicates: List[Predicate], k: int
    ) -> HSearchResults:
        with self.time_section(Section.TOTAL):
            with self.time_section(Section.DB_SEARCH):
                pre_db_records = self.db.predicates_search(predicates)
            pre_predicate_count = len(pre_db_records)
            if pre_predicate_count == 0:
                return HSearchResults(results=[], is_k=False)

            pre_item_ids = [record.item_id for record in pre_db_records.records]
            nprobe = 0
            ann_results = AnnSearchResults.empty()
            # here we will know how many results we can get, so we can set search_k accordingly
            search_k = min(pre_predicate_count, k)
            while len(ann_results) < search_k:
                with self.time_section(Section.UD_PARAMS):
                    nprobe = self._opt_pre_nprobe(
                        pre_predicate_count, search_k - len(ann_results), nprobe
                    )
                with self.time_section(Section.FAISS_SEARCH):
                    ann_results = self.index.search(
                        embedding_query,
                        search_k,
                        nprobe=nprobe,
                        item_ids=pre_item_ids,
                    )
            with self.time_section(Section.INTERSECT):
                pre_top_results = self._intersect(ann_results, pre_db_records)
            with self.time_section(Section.FINALIZE):
                return HSearchResults(
                    # [:k] will limit the results to k items
                    # or all items if we have less than k (native python behavior)
                    # here we should always get correct number of pre_top_results because
                    # we set search_k to min(pre_predicate_count, k)
                    # but we simply keep the same finalizing code for simplicity
                    results=pre_top_results[:k],
                    is_k=len(pre_top_results) == k,
                )

    def adap_pos_search(
        self,
        embedding_query: np.ndarray,
        predicates: List[Predicate],
        k: int,
        est_survivors: int,
    ) -> HSearchResults:
        with self.time_section(Section.TOTAL):
            search_k = 0
            nprobe = 0
            top_results: List[HSearchResult] = []
            while len(top_results) < k:
                # Get next search parameters with aggressive growth
                with self.time_section(Section.UD_PARAMS):
                    search_k, nprobe = self._opt_pos_search_k_and_nprobe(
                        est_survivors, k - len(top_results), search_k
                    )
                # If we've reached the full dataset, search everything
                if search_k >= N:
                    search_k = N
                    nprobe = NLIST

                with self.time_section(Section.FAISS_SEARCH):
                    ann_results = self.index.search(
                        embedding_query, search_k, nprobe=nprobe, item_ids=None
                    )
                item_ids_predicate = Predicate(
                    key="item_id",
                    value=ann_results.item_ids.tolist(),
                    operator="IN",
                )
                predicates_copy = predicates.copy()
                predicates_copy.append(item_ids_predicate)

                with self.time_section(Section.DB_SEARCH):
                    post_db_records = self.db.predicates_search(predicates_copy)

                with self.time_section(Section.INTERSECT):
                    top_results = self._intersect(ann_results, post_db_records)

                # Break if we've reached the full dataset
                if search_k >= N:
                    break

            # HSearchResults always sorts by similarity
            with self.time_section(Section.FINALIZE):
                return HSearchResults(
                    results=top_results[:k],
                    is_k=len(top_results) == k,
                )

    def _opt_pre_nprobe(
        self, predicate_count: int, k_remaining: int, old_nprobe: int = 0
    ) -> int:
        """
        Find the optimal nprobe for the query and predicates.
        """
        # each cluster is expected to have predicate_count / NLIST survivors
        # the expected number of clusters to search is k / (predicate_count / NLIST)
        expected_nprobe = k_remaining * NLIST // predicate_count + old_nprobe
        # Use 3x multiplier for safety to ensure we get enough results
        return max(3, min(3 * expected_nprobe, NLIST))

    def _opt_pos_search_k_and_nprobe(
        self,
        est_survivors: int,
        k_remaining: int,
        old_search_k: int = 0,
        old_nprobe: int = 0,
    ):
        """
        Find the optimal search_k and nprobe for the query and est_survivors.
        Aggressive exponential growth to reach N quickly.
        """
        if est_survivors == 0:
            # If no survivors expected, search the entire dataset
            search_k = N
            nprobe = NLIST
        else:
            if old_search_k == 0:
                # Start with a reasonable initial search based on selectivity
                if est_survivors >= N * 0.1:  # Low selectivity
                    base_search_k = N // 10  # Start with 10% of dataset
                else:  # High selectivity
                    base_search_k = max(
                        k_remaining * 10, int(N / est_survivors * k_remaining * 5)
                    )
            else:
                # Exponential growth: 3x the previous size to converge quickly
                base_search_k = old_search_k * 3

            search_k = min(N, base_search_k)
            nprobe = min(NLIST, max(1, search_k // N_PER_CLUSTER * 4 + old_nprobe + 1))
        return search_k, nprobe

    @staticmethod
    def _intersect(
        ann_results: AnnSearchResults, db_records: DBRecords
    ) -> List[HSearchResult]:
        # Optimized version: avoid unnecessary dict conversions
        # Build set directly from records (O(n))
        db_item_ids = {record.item_id for record in db_records.records}
        # Build dict only once for record lookup (O(n))
        db_records_dict = {record.item_id: record for record in db_records.records}

        # Work directly with numpy arrays instead of converting to Python lists/dicts
        # Use numpy's isin for faster membership testing (O(m))
        if ann_results.is_empty():
            return []

        # Convert to numpy arrays if needed (handle 0-dim case)
        if ann_results.item_ids.ndim == 0:
            ann_item_ids = np.array([ann_results.item_ids])
            ann_distances = np.array([ann_results.distances])
        else:
            ann_item_ids = ann_results.item_ids
            ann_distances = ann_results.distances

        # Use numpy's isin for fast membership testing
        # np.isin can work with sets, but for better performance with large sets,
        # we can convert to array. For small sets, set lookup might be faster.
        # Using list conversion as numpy's isin is optimized for array-like inputs
        db_item_ids_array = np.array(list(db_item_ids), dtype=ann_item_ids.dtype)
        mask = np.isin(ann_item_ids, db_item_ids_array)
        matching_indices = np.where(mask)[0]

        # Build results list efficiently
        return [
            HSearchResult(
                record=db_records_dict[int(ann_item_ids[i])],
                similarity=float(ann_distances[i]),
            )
            for i in matching_indices
        ]
