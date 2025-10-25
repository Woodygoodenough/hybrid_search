#!/bin/bash
# Complete workflow setup script for Hybrid Search System

set -e  # Exit on any error

echo "🏗️  Hybrid Search System - Complete Workflow Setup"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "settings.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Step 1: Data fetching
echo ""
echo "📥 Step 1: Fetching Wikipedia data..."
if [ ! -f "wikipedia_sample_150k.csv" ]; then
    python datafetch.py
    echo "✅ Data fetched successfully!"
else
    echo "⚠️  Data file already exists. Skipping data fetch."
fi

# Step 2: Add IDs
echo ""
echo "🔢 Step 2: Adding sequential IDs..."
if [ ! -f "wikipedia_sample_150k_with_ids.csv" ]; then
    python add_item_id_to_csv.py wikipedia_sample_150k.csv
    echo "✅ IDs added successfully!"
else
    echo "⚠️  ID file already exists. Skipping ID addition."
fi

# Step 3: Create database
echo ""
echo "🗄️  Step 3: Creating metadata database..."
if [ ! -f "meta_wiki.db" ]; then
    python dbManagement.py
    echo "✅ Database created successfully!"
else
    echo "⚠️  Database already exists. Skipping database creation."
fi

# Step 4: Create vector index
echo ""
echo "🔍 Step 4: Creating FAISS vector index..."
if [ ! -d "index.faiss.new" ]; then
    python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv
    echo "✅ Vector index created successfully!"
else
    echo "⚠️  Vector index already exists. Skipping index creation."
fi

echo ""
echo "🎉 Setup complete! You can now:"
echo ""
echo "1. Start interactive search:"
echo "   python main.py"
echo ""
echo "2. Run tests:"
echo "   python test_hybrid_strategies.py"
echo ""
echo "3. Explore the system:"
echo "   python explore.py"
echo ""
echo "📊 System ready for hybrid search!"
