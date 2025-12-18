import pandas as pd
import numpy as np
import warnings
import os
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

print("Importing libraries...")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("✓ All imports successful!")

# ============================================================================
# LOAD PRE-BUILT VECTOR STORE
# ============================================================================

print("\nLoading pre-built FAISS vector store...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)

print("✓ Vector store loaded!")

# ============================================================================
# LOAD YOUR DATA WITH SENTIMENT
# ============================================================================

print("\nLoading data files...")

medical_df = pd.read_csv('medical_info.csv')
print(f"✓ Loaded {len(medical_df):,} medical Q&As")

ingredient_df = pd.read_csv('ingredient_list_final.csv')
print(f"✓ Loaded {len(ingredient_df):,} ingredients")

# Load YOUR sentiment analysis from CSV
print("Loading YOUR sentiment analysis from CSV...")
try:
    sentiment_df = pd.read_csv('sentiment_analysis.csv')
    print(f"✓ Loaded YOUR sentiment for {len(sentiment_df):,} products!")
    print(f"   Columns: {list(sentiment_df.columns)}")
    has_sentiment = True
except FileNotFoundError:
    print("⚠️  sentiment_analysis.csv not found!")
    print("   Run CONVERT_PICKLE_COLAB.py in Colab to create it")
    print("   Continuing without sentiment...")
    sentiment_df = None
    has_sentiment = False
except Exception as e:
    print(f"⚠️  Error loading sentiment CSV: {e}")
    print("   Continuing without sentiment...")
    sentiment_df = None
    has_sentiment = False

# Load products
products = pd.read_csv('product_info.csv', low_memory=False)
skincare = products[products['primary_category'] == 'Skincare'].copy()

# Merge with YOUR sentiment analysis
if has_sentiment and sentiment_df is not None:
    print("Merging sentiment data...")
    
    # Make sure product_id types match
    skincare['product_id'] = skincare['product_id'].astype(str)
    sentiment_df['product_id'] = sentiment_df['product_id'].astype(str)
    
    # Merge
    skincare = skincare.merge(
        sentiment_df[['product_id', 'avg_rating', 'total_reviews', 
                     'predicted_sentiment', 'positive_rating_pct']],
        on='product_id', 
        how='left',
        suffixes=('', '_sentiment')
    )
    
    # Count how many products got sentiment
    with_sentiment = skincare['predicted_sentiment'].notna().sum()
    print(f"✓ Merged sentiment for {with_sentiment:,} products!")
    
else:
    # No sentiment data
    skincare['avg_rating'] = 0
    skincare['total_reviews'] = 0
    skincare['predicted_sentiment'] = 'unknown'
    skincare['positive_rating_pct'] = 0
    print("⚠️  Products have no sentiment data")

print(f"✓ Loaded {len(skincare):,} total products")

# ============================================================================
# LOAD LLM - Using a working free model
# ============================================================================

print("\nConnecting to HuggingFace Inference API...")

hf_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

if not hf_token:
    print("❌ ERROR: HUGGINGFACEHUB_API_TOKEN not found!")
    raise ValueError("Missing API token")

# Use Mistral with the correct configuration
from langchain_huggingface import ChatHuggingFace

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.3,
        max_new_tokens=1500,
        huggingfacehub_api_token=hf_token,
    )
)

print("✓ Connected to Inference API!")


# ============================================================================
# SKINCARE AGENT - Updated to work with ChatHuggingFace
# ============================================================================

class SkincareAgent:
    """Skincare consultant agent"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        self.conversation_history = []
        self.last_results = []

        # Updated prompt for chat format
        from langchain_core.prompts import ChatPromptTemplate
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a professional skincare consultant with expertise in dermatology, 
            cosmetic chemistry, and product recommendations. Provide helpful, accurate, and 
            personalized responses. When listing recommendations:
            - Give 3-5 specific products with brief explanations
            - Complete all numbered points
            - Be thorough but concise"""),
            ("human", """Based on this information:

{context}

Answer this question: {question}

Provide specific product recommendations when relevant, explain ingredients, and consider any 
budget mentioned. Highlight customer sentiment when available.""")
        ])

    def chat(self, query: str, show_details: bool = False) -> str:
        """Main chat function"""
        docs = self.retriever.invoke(query)

        self.last_results = [
            {
                'type': doc.metadata.get('type'),
                'name': doc.metadata.get('name', ''),
                'brand': doc.metadata.get('brand', ''),
                'price': doc.metadata.get('price', 0),
                'sentiment': doc.metadata.get('sentiment', 'neutral'),
                'rating': doc.metadata.get('rating', 0),
                'positive_pct': doc.metadata.get('positive_pct', 0),
            }
            for doc in docs[:10]
        ]

        context = "\n\n".join([doc.page_content for doc in docs[:8]])

        # Use invoke instead of the old chain syntax
        messages = self.prompt_template.format_messages(
            context=context,
            question=query
        )
        
        response = self.llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)

        return text.strip()

    def get_last_results(self):
        return self.last_results

    def clear_history(self):
        self.conversation_history = []
        self.last_results = []


# Create agent
print("\nCreating agent...")
agent = SkincareAgent(vectorstore, llm)
print("✓ Agent ready!")

print("\n" + "=" * 70)
print("SKINCARE AGENT READY!")
print("=" * 70)
if has_sentiment:
    print("✅ WITH YOUR SENTIMENT ANALYSIS!")
    print(f"✅ {with_sentiment:,} products have ratings & sentiment")
else:
    print("⚠️ Running without sentiment")
print("✅ Inference API (fast & free)")
print("=" * 70)