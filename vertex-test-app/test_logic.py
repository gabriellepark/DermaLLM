# test_logic.py
import time
import pandas as pd
import vertexai

# 1. Initialize Environment
vertexai.init(project="skin-care-recommender", location="us-central1")

# 2. Import RAG Agent
try:
    from rag_agent.agent import SkincareAgent
    print("Initializing Agent for Testing...")
    agent = SkincareAgent()
    vectorstore = agent.vectorstore  # Direct access for the coverage test
    print("Agent Ready!\n" + "="*50 + "\n")
except Exception as e:
    print(f"Error initializing: {e}")
    print("Make sure you are running this from the 'vertex-skincare-app' folder")
    print("and that 'faiss_index' exists.")
    exit()



def test_efficiency():
    """Test response speed across different query types"""

    test_queries = [
        "I need a moisturizer",
        "Moisturizer under $30",
        "Vegan cleanser",
        "Vegan retinol for anti-aging under $50",
    ]

    results = []

    print("Testing Query Speed...\n")
    print(f"{'Query':<50} {'Time (s)':<10} {'Products':<10}")

    for query in test_queries:
        start = time.time()
        agent.chat(query, show_details=False)
        products = agent.get_last_products()
        elapsed = time.time() - start

        results.append({'query': query, 'time': elapsed, 'num_products': len(products)})
        print(f"{query:<50} {elapsed:<10.2f} {len(products):<10}")

    return pd.DataFrame(results)

def test_accuracy():
    """Test if filters are working correctly"""
    print("\n" + "="*20 + "\nTesting Filter Accuracy\n" + "="*20)

    # Test 1: Budget
    print("TEST 1: Budget Filter ($20)")
    agent.chat("Show me products under $20", show_details=False)
    products = agent.get_last_products()
    over = [p for p in products if p.get('price', 0) > 20]
    if over: print(f"❌ FAILED: {len(over)} products over budget.")
    else: print(f"✅ PASSED: All under $20")

    # Test 2: Vegan
    print("\nTEST 2: Vegan Filter")
    agent.chat("Show me vegan products", show_details=False)
    products = agent.get_last_products()
    not_vegan = [p for p in products if not p.get('vegan')]
    if not_vegan: print(f"❌ FAILED: {len(not_vegan)} non-vegan products.")
    else: print(f"✅ PASSED: All vegan")

    # Test 3: Intent Extraction
    print("\nTEST 3: Intent Extraction Logic")
    query = "I need retinol for anti-aging under $50"
    intent = agent.intent_extractor.analyze(query)
    print(f"Query: {query}")
    print(f"Extracted: {intent}")
    
    if intent['budget'] == 50 and 'retinol' in intent['ingredients']:
        print("✅ PASSED: Correctly identified budget and ingredient.")
    else:
        print("❌ FAILED: Extraction mismatch.")

def test_feature_coverage():
    """Test how many products have each feature"""
    print("\n" + "="*20 + "\nTesting Feature Coverage\n" + "="*20)
    
    # We use the agent's internal vectorstore
    all_docs = vectorstore.similarity_search("skincare", k=100)
    
    features = {'vegan': 0, 'cruelty_free': 0, 'fragrance_free': 0}
    
    for doc in all_docs:
        for key in features:
            if doc.metadata.get(key): features[key] += 1
            
    for f, count in features.items():
        print(f"{f:<20} {count:<10} {(count/len(all_docs))*100:.1f}%")

# --- Run All Tests ---
if __name__ == "__main__":
    test_efficiency()
    test_accuracy()
    test_feature_coverage()