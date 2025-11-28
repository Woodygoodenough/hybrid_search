import faiss  # do not remove this import, there is an apple silicon issue with faiss's importing order
from dataclasses import dataclass
from typing import Optional, List, Literal
import numpy as np
from vector_index import Embedder, FaissIVFIndex, AnnSearchResults
from settings import FAISS_PATH, NLIST, N
from dbManagement import DbManagement, DBRecord, DBRecords
from shared_dataclasses import Predicate
from histo2d import Histo2D
import pandas as pd
from timer import Timer, Section, SearchMethod


# ============================================================================
# HARDCODED LOGISTIC REGRESSION MODEL FOR PRE/POS SELECTION
# Auto-generated from model_evaluation.py
# This eliminates pkl loading overhead for faster inference
# ============================================================================

# Scaler parameters (StandardScaler) - will be populated by model_evaluation.py
_SCALER_MEAN = np.array([2687.2571942446, 81211.6438848921, 0.5414109592])
_SCALER_SCALE = np.array([3341.8898775446, 35212.3631437106, 0.2347490876])

# Logistic Regression parameters - will be populated by model_evaluation.py
_LR_COEF = np.array([-1.3276164995, 2.2549849548, 2.2549849548])
_LR_INTERCEPT = 9.9547678062


def _predict_pos_faster_hardcoded(k: int, num_survivors: int, total_docs: int = 150000) -> bool:
    """
    Predict whether POS search will be faster than PRE search using hardcoded model.
    This is much faster than loading a pkl file.

    Args:
        k: Number of results requested
        num_survivors: Estimated number of survivors after predicates
        total_docs: Total number of documents (default: 150000)

    Returns:
        True if POS is predicted to be faster, False if PRE is predicted to be faster
    """
    # If model parameters are not set, use simple heuristic
    if _LR_COEF is None or _LR_INTERCEPT is None:
        # Simple fallback: use POS if selectivity is low
        return num_survivors < k * 100

    # Calculate selectivity
    selectivity = num_survivors / total_docs

    # Create feature vector
    features = np.array([k, num_survivors, selectivity])

    # Scale features
    features_scaled = (features - _SCALER_MEAN) / _SCALER_SCALE

    # Calculate logistic regression prediction
    logit = np.dot(features_scaled, _LR_COEF) + _LR_INTERCEPT

    # Apply sigmoid and get prediction
    probability_pos = 1 / (1 + np.exp(-logit))

    # Predict POS if probability > 0.5
    return probability_pos > 0.5


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
    def __init__(self, timer: Timer = None, db: DbManagement = None):
        self.timer = timer or Timer()
        self.time_section = self.timer.section
        self.time_method = self.timer.method_context
        self.index = FaissIVFIndex.load(FAISS_PATH)
        self.embedder = Embedder()
        self.db = db or DbManagement()
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
        method: SearchMethod,
    ) -> HSearchResults:
        embedding_query = self.embedder.encode_query(query)
        est_survivors = self.histo.estimate_survivors(predicates)
        if method == SearchMethod.BASE_PRE_SEARCH:
            return self.base_pre_search(embedding_query, predicates, k)
        elif method == SearchMethod.BASE_POS_SEARCH:
            return self.base_pos_search(embedding_query, predicates, k)
        elif method == SearchMethod.ADAP_PRE_SEARCH:
            return self.adap_pre_search(embedding_query, predicates, k)
        elif method == SearchMethod.ADAP_POS_SEARCH:
            return self.adap_pos_search(embedding_query, predicates, k, est_survivors)
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
            # we do not need histo at this point, but we want to time it because it
            # is part of the adap search
            # delete THIS when deploying
            with self.time_section(Section.HISTO_FILTER):
                _ = self.histo.estimate_survivors(predicates)
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
            # we already have est_survivors, but we still want to time est
            # delete THIS when deploying
            with self.time_section(Section.HISTO_FILTER):
                _ = self.histo.estimate_survivors(predicates)
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

    def lr_based_adap_search(
        self,
        embedding_query: np.ndarray,
        predicates: List[Predicate],
        k: int,
    ) -> HSearchResults:
        """
        Logistic Regression-based adaptive search that automatically selects
        between PRE and POS search methods based on the hardcoded LR model.

        This uses hardcoded model parameters instead of loading a pkl file,
        eliminating file I/O overhead for maximum inference speed.

        Args:
            embedding_query: Query embedding vector
            predicates: List of predicates for filtering
            k: Number of results to return

        Returns:
            HSearchResults object with search results
        """
        # Estimate survivors using histogram
        est_survivors = self.histo.estimate_survivors(predicates)

        # Use hardcoded model to predict (no file loading overhead!)
        use_pos = _predict_pos_faster_hardcoded(k, est_survivors)

        # Execute the predicted best method
        if use_pos:
            # POS is predicted to be faster
            return self.adap_pos_search(embedding_query, predicates, k, est_survivors)
        else:
            # PRE is predicted to be faster
            return self.adap_pre_search(embedding_query, predicates, k)

    def _opt_pre_nprobe(
        self, predicate_count: int, k_remaining: int, old_nprobe: int = 0
    ) -> int:
        """
        Find the optimal nprobe for the query and predicates.
        """
        # each cluster is expected to have predicate_count / NLIST survivors
        # the expected number of clusters to search is k / (predicate_count / NLIST)
        # we use a 1.5x multiplier to be safe
        expected_nprobe = (
            int(1.5 * k_remaining * NLIST // predicate_count)
            # refer to the comment in _opt_pos_search_k_and_nprobe for the 3 multiplier
            + 3 * old_nprobe
        )

        return max(1, min(expected_nprobe, NLIST))

    def _opt_pos_search_k_and_nprobe(
        self,
        est_survivors: int,
        k_remaining: int,
        old_search_k: int = 0,
        old_nprobe: int = 0,
    ):
        if est_survivors == 0:
            # If no survivors expected, search the entire dataset
            search_k = N
            nprobe = NLIST
        else:
            # possibility of any one entry to be a survivor is est_survivors / N
            # we expect to find k survivors, so we need to search for k * N / est_survivors entries
            # we use a 1.5x multiplier to be safe
            expected_search_k = (
                int(1.5 * k_remaining * N / max(est_survivors, 1))
                # note the 3 below is not "optimal", we should not add this multiple if we stick with statistical expectation
                # however, let's consider an edge case. assume there is actually 0 survivors, and we only request 3 results,
                # search_k will only grow linearly, which takes a lot of iterations to reach N before algo terminates
                # so we applied this multiplier to handle edge cases, this should add minimum overhead,
                # since in most cases only the first iteration is required
                + 3 * old_search_k
            )
            search_k = min(N, expected_search_k)
            # per cluster, we expect to have est_survivors / NLIST survivors
            # we expect to find search_k results, so we need to search for search_k * NLIST / est_survivors clusters
            # note we already applied 1.5x multiplier to search_k, so for now nprobe should already be large enough
            expected_nprobe = search_k * NLIST // max(est_survivors, 1) + old_nprobe
            nprobe = min(NLIST, max(1, expected_nprobe))
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
