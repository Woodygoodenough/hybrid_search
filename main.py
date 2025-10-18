#!/usr/bin/env python3
"""
Main entry point for the Hybrid Search System.
Run this to start the search interface.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from search import SimpleSearch

def main():
    """Main search interface."""
    print("🚀 Hybrid Search System")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session.")
    print()

    try:
        # Initialize search engine
        search_engine = SimpleSearch()

        while True:
            try:
                # Get user query
                query = input("\n🔍 Enter your search query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not query:
                    print("Please enter a search query.")
                    continue

                # Perform search
                results = search_engine.search(query, k=5, show_metadata=True)

                print(f"\n✅ Found {len(results)} results for '{query}'")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize search system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)