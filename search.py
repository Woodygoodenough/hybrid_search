#!/usr/bin/env python3
"""
Working search script using FAISS directly + database metadata.
"""
# %%
import sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dbManagement import DbManagement
from settings import MODEL_NAME

class SimpleSearch:
    """Simple search using FAISS directly."""

    def __init__(self, index_path="index.faiss", db_path="meta_wiki.db"):
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)

        # Load FAISS index
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path / "vectors.index"))
        print(f"✅ Index loaded: {self.index.ntotal} vectors, dim={self.index.d}")

        # Load embedder
        print(f"Loading embedder model: {MODEL_NAME}...")
        self.embedder = SentenceTransformer(MODEL_NAME)
        print(f"✅ Embedder loaded: dim={self.embedder.get_sentence_embedding_dimension()}")

        # Connect to database
        self.db = None
        if self.db_path.exists():
            try:
                self.db = DbManagement()
                print("✅ Database connected")
            except Exception as e:
                print(f"⚠️  Could not connect to database: {e}")
        else:
            print(f"⚠️  Database file not found: {self.db_path}")

    def search(self, query, k=5, show_metadata=True):
        """Search for similar items."""
        print(f"\n🔍 Searching for: '{query}'")

        # Encode query
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        # Normalize for cosine similarity
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Search
        distances, indices = self.index.search(query_vec.astype(np.float32), k)

        print(f"\n📊 Results (top {k}):")
        print("-" * 80)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            result = {
                'rank': i + 1,
                'item_id': int(idx),
                'distance': float(dist),
                'similarity': float(1 / (1 + dist))  # Convert to similarity score
            }

            # Get metadata if available
            if self.db and show_metadata:
                try:
                    self.db.cur.execute("""
                        SELECT title, url
                        FROM items
                        WHERE item_id = ?
                    """, (int(idx),))

                    row = self.db.cur.fetchone()
                    if row:
                        result['title'] = row[0][:80] + "..." if len(row[0]) > 80 else row[0]
                        result['url'] = row[1]

                        # Get categories from separate table
                        self.db.cur.execute("""
                            SELECT GROUP_CONCAT(category, '; ')
                            FROM item_categories
                            WHERE item_id = ?
                        """, (int(idx),))
                        cat_row = self.db.cur.fetchone()
                        result['categories'] = cat_row[0] if cat_row[0] else "None"
                    else:
                        result['title'] = "Not found in database"
                        result['url'] = "N/A"
                        result['categories'] = "N/A"

                except Exception as e:
                    result['title'] = f"Error loading metadata: {e}"
                    result['url'] = "N/A"
                    result['categories'] = "N/A"

            results.append(result)

            # Print formatted result
            title_str = result.get('title', 'N/A')[:50] + "..." if len(result.get('title', '')) > 50 else result.get('title', 'N/A')
            print(f"{i+1:2d}. ID:{idx:6d} | Sim:{result['similarity']:.3f} | {title_str}")

            if show_metadata and 'url' in result:
                print(f"    URL: {result['url']}")
                if result.get('categories') and result['categories'] != 'N/A':
                    print(f"    Categories: {result['categories'][:100]}...")
                print()

        return results

    def search_multiple(self, queries, k=5):
        """Search multiple queries."""
        all_results = {}

        for query in queries:
            print(f"\n{'='*60}")
            results = self.search(query, k=k, show_metadata=True)
            all_results[query] = results

        return all_results

def main():
    """Main test function."""
    print("🚀 Starting Simple Search Test")
    print("=" * 50)

    try:
        search_engine = SimpleSearch()

        # Test queries
        test_queries = [
            "machine learning algorithms",
            "artificial intelligence applications",
            "computer programming languages",
            "data science methods"
        ]

        # Run searches
        all_results = search_engine.search_multiple(test_queries, k=3)

        print(f"\n✅ All searches completed successfully!")
        print(f"   Searched {len(test_queries)} queries")
        print(f"   Found relevant results for each query")

        return all_results

    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False

# %%
if __name__ == "__main__":
    all_results = main()
  
# %%
all_results
# %%
