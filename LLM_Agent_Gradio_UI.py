
import os

import warnings
import os
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.generation').setLevel(logging.ERROR)

print("Warnings suppressed!")

# %%time
# !pip install -q langchain-community langchain-core chromadb
# !pip install -q sentence-transformers transformers torch accelerate bitsandbytes
# !pip install -q kaggle pandas numpy
# !pip install -q faiss-cpu
print("All packages installed!")

import pandas as pd
import numpy as np
import os, json, warnings, re, ast, pickle
from typing import Optional, List, Dict, Any
warnings.filterwarnings('ignore')

# LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

print("All imports successful!")

# %%time
# Set up Kaggle credentials
# !mkdir -p ~/.kaggle
# with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
#     json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
# !chmod 600 ~/.kaggle/kaggle.json

# Download dataset
# !kaggle datasets download -d nadyinky/sephora-products-and-skincare-reviews -q
# !unzip -q sephora-products-and-skincare-reviews.zip

print("Kaggle data downloaded!")

# from google.colab import drive
# drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/DermaLLM/'

print(f"Google Drive mounted!")
print(f"Using path: {DRIVE_PATH}")
print("\nMake sure this folder contains:")
print("   1. medical_info.csv")
print("   2. reviews_prod_lvl_12_17_2025.pkl")
print("   3. ingredient_list_final.csv")

# %%time
print("Loading medical information...")
medical_df = pd.read_csv(DRIVE_PATH + 'medical_info.csv')
print(f"Loaded {len(medical_df):,} medical Q&A pairs")
print(f"   Categories: {', '.join(medical_df['category'].unique())}")
print(f"   Conditions: {len(medical_df['condition'].unique())} unique")

print("\nLoading ingredient database...")
ingredient_df = pd.read_csv(DRIVE_PATH + 'ingredient_list_final.csv')
ingredient_df = ingredient_df[ingredient_df['name'].notna()]
print(f"Loaded {len(ingredient_df):,} ingredients")
print(f"   Examples: {', '.join(ingredient_df['name'].head(5).tolist())}")

print("\nLoading sentiment analysis (Updated 12/17/2025)...")
with open(DRIVE_PATH + 'reviews_prod_lvl_12_17_2025.pkl', 'rb') as f:
    sentiment_df = pickle.load(f)

print(f"Loaded sentiment for {len(sentiment_df):,} products")
print(f"\nSentiment Distribution:")
sentiment_counts = sentiment_df['predicted_sentiment'].value_counts()
for sent, count in sentiment_counts.items():
    pct = (count / len(sentiment_df)) * 100
    print(f"   {sent.capitalize()}: {count:,} ({pct:.1f}%)")

print(f"\nQuality Metrics:")
print(f"   Avg sentiment score: {sentiment_df['predicted_sentiment_score'].mean():.4f}")
print(f"   Avg review quality: {sentiment_df['avg_review_quality'].mean():.4f}")

print("\nAll additional data loaded successfully!")

# %%time
print("Loading product data...")
products = pd.read_csv('product_info.csv', low_memory=False)
skincare = products[products['primary_category'] == 'Skincare'].copy()
print(f"Loaded {len(skincare):,} skincare products")

# Clean text fields
text_cols = ['product_name', 'brand_name', 'description', 'ingredients', 'highlights', 'how_to_use']
for col in text_cols:
    if col in skincare.columns:
        skincare[col] = skincare[col].fillna('').astype(str)

# DON'T load reviews from CSV - use sentiment pickle instead!
print("\nMerging sentiment analysis (includes ratings)...")
before_merge = len(skincare)

skincare = skincare.merge(
    sentiment_df[[
        'product_id',
        'total_reviews',
        'avg_rating',
        'predicted_sentiment',
        'predicted_sentiment_score',
        'avg_review_quality',
        'dominant_rating_sentiment',
        'positive_rating_pct',
        'neutral_rating_pct',
        'negative_rating_pct',
        'review_sample'
    ]],
    on='product_id',
    how='left'
)

# Use avg_rating from sentiment file as the primary rating
if 'rating' in skincare.columns:
    # If there's a rating column from product_info, fill missing with sentiment avg_rating
    skincare['rating'] = skincare['rating'].fillna(skincare['avg_rating'])
else:
    # Otherwise just use avg_rating
    skincare['rating'] = skincare['avg_rating']

# Fill missing values
skincare['rating'] = skincare['rating'].fillna(0)
skincare['total_reviews'] = skincare['total_reviews'].fillna(0)

sentiment_count = skincare['predicted_sentiment'].notna().sum()
print(f"Sentiment merged for {sentiment_count:,} products ({sentiment_count/before_merge*100:.1f}%)")

# Show stats
with_reviews = (skincare['total_reviews'] > 0).sum()
print(f"Products with reviews: {with_reviews:,}")
print(f"Products without reviews: {len(skincare) - with_reviews:,}")

print(f"\nFinal dataset: {len(skincare):,} products ready for feature extraction")

# %%time

# PARSE HIGHLIGHTS
def parse_highlights(x):
    if pd.isna(x):
        return []
    try:
        if x.strip().startswith("["):
            return [h.lower().strip() for h in ast.literal_eval(x)]
    except:
        pass
    return [h.lower().strip() for h in re.split(",|;", x)]

skincare['highlight_list'] = skincare['highlights'].apply(parse_highlights)
print("Parsed highlights")

# HIGHLIGHT MAPPING
highlight_mapping = {
    "vegan": "vegan",
    "cruelty-free": "cruelty_free",
    "without parabens": "paraben_free",
    "without sulfates sls & sles": "sulfate_free",
    "without silicones": "silicone_free",
    "without mineral oil": "mineral_free",
    "fragrance free": "fragrance_free",
    "fresh scent": "has_fragrance",
    "floral scent": "has_fragrance",
    "woody & earthy scent": "has_fragrance",
    "warm &spicy scent": "has_fragrance",
    "unisex/ genderless scent": "has_fragrance",
    "oil free": "oil_free",
    "clean at sephora": "clean_at_sephora",
    "non-comedogenic": "non_comedogenic",
    "good for: dullness/uneven texture": "dullness",
    "good for: dryness": "dryness",
    "good for: anti-aging": "anti_aging",
    "good for: acne/blemishes": "acne",
    "good for: dark spots": "dark_spots",
    "good for: redness": "redness",
    "plumping": "plumping",
    "hyaluronic acid": "hyaluronic_acid",
    "niacinamide": "niacinamide",
    "retinol": "retinol",
    "salicylic acid": "salicylic_acid",
    "aha/glycolic acid": "aha_glycolic_acid",
    "spf": "spf",
    "reef safe spf": "spf",
    "alcohol free": "alcohol_free",
    "hypoallergenic": "hypoallergenic"
}

# Apply mapping
for highlight, col_name in highlight_mapping.items():
    skincare[col_name] = skincare['highlight_list'].apply(
        lambda hl: int(any(highlight.lower() in h.lower() for h in hl)))

print("Extracted concerns, actives, and preferences")

# ALLURE AWARDS
def check_allure_best_of_beauty(highlights):
    return int(any("allure" in h.lower() and "best of beauty" in h.lower() for h in highlights))

skincare['allure_best_of_beauty'] = skincare['highlight_list'].apply(check_allure_best_of_beauty)
print("Extracted awards")

# SKIN TYPES
SKIN_TYPE_MAPPING = {
    "best for oily skin": ["oily_skin"],
    "best for dry skin": ["dry_skin"],
    "best for normal skin": ["normal_skin"],
    "best for combination skin": ["combination_skin"],
    "best for oily, combo, normal skin": ["oily_skin", "combination_skin", "normal_skin"],
    "best for dry, combo, normal skin": ["dry_skin", "combination_skin", "normal_skin"]
}
SKIN_TYPE_COLUMNS = ["oily_skin", "dry_skin", "normal_skin", "combination_skin"]

def apply_skin_types(highlights):
    flags = {col: 0 for col in SKIN_TYPE_COLUMNS}
    for h in highlights:
        h_lower = h.lower()
        for pattern, cols in SKIN_TYPE_MAPPING.items():
            if pattern in h_lower:
                for col in cols:
                    flags[col] = 1
    return pd.Series(flags)

skin_type_df = skincare['highlight_list'].apply(apply_skin_types)
skincare = pd.concat([skincare, skin_type_df], axis=1)
print("Extracted skin types")

# SENSITIVE SKIN
skincare['sensitive_skin'] = (
    (skincare['hypoallergenic'] == 1) | (skincare['fragrance_free'] == 1)).astype(int)

print("\nFeature extraction complete!")
print(f"   Extracted 50+ features for {len(skincare):,} products")

# %%time
def build_product_text_with_sentiment(row):
    """Build rich product description with all features + sentiment"""
    parts = []

    # Helper function to safely get value
    def safe_get(key, default=0):
        try:
            val = row[key]
            return val if pd.notna(val) else default
        except:
            return default

    # 1. Name + brand
    parts.append(f"{safe_get('product_name', '')} by {safe_get('brand_name', '')}.")

    # 2. Skin concerns
    concerns = []
    for c in ['dullness', 'dryness', 'anti_aging', 'acne', 'dark_spots', 'redness', 'plumping']:
        if safe_get(c) == 1:
            concerns.append(c.replace("_", " "))
    if concerns:
        parts.append(f"Targets {', '.join(concerns)}.")

    # 3. Active ingredients
    actives = []
    for a in ['hyaluronic_acid', 'niacinamide', 'retinol', 'salicylic_acid', 'aha_glycolic_acid', 'spf']:
        if safe_get(a) == 1:
            actives.append(a.replace("_", " "))
    if actives:
        parts.append(f"Contains {', '.join(actives)}.")

    # 4. Preferences
    prefs = []
    for p in ['vegan', 'cruelty_free', 'paraben_free', 'sulfate_free', 'silicone_free',
              'mineral_free', 'alcohol_free', 'hypoallergenic', 'clean_at_sephora']:
        if safe_get(p) == 1:
            prefs.append(p.replace("_", " "))
    if prefs:
        parts.append(f"{', '.join(prefs).title()}.")

    # 5. Fragrance
    if safe_get('fragrance_free') == 1:
        parts.append("Fragrance free.")
    elif safe_get('has_fragrance') == 1:
        parts.append("Contains fragrance.")

    # 6. Formulation
    formulation = []
    for f in ['oil_free', 'non_comedogenic']:
        if safe_get(f) == 1:
            formulation.append(f.replace("_", " "))
    if formulation:
        parts.append(f"{', '.join(formulation).title()}.")

    # 7. Awards
    if safe_get('allure_best_of_beauty') == 1:
        parts.append("Allure Best of Beauty award winner.")

    # 8. SENTIMENT
    sentiment = safe_get('predicted_sentiment', None)
    if sentiment is not None:
        parts.append(f"Customer sentiment: {sentiment}.")

        pos_pct = safe_get('positive_rating_pct', None)
        if pos_pct is not None:
            parts.append(f"{pos_pct:.0f}% positive reviews.")

        quality = safe_get('avg_review_quality', None)
        if quality is not None:
            if quality > 0.75:
                parts.append("Highly detailed customer reviews.")
            elif quality > 0.6:
                parts.append("Good review quality.")

    # 9. Price and rating
    price = safe_get('price_usd', None)
    if price is not None:
        parts.append(f"Price ${price:.2f}.")

    rating = safe_get('rating', None)
    if rating is not None:
        parts.append(f"Rating {rating:.1f}/5.0.")

    # 10. Skin types
    skin_types = []
    for st in ['dry_skin', 'oily_skin', 'combination_skin', 'normal_skin']:
        if safe_get(st) == 1:
            skin_types.append(st.replace("_", " "))
    if skin_types:
        parts.append(f"Best for {', '.join(skin_types)}.")

    # 11. Sensitive skin
    if safe_get('sensitive_skin') == 1:
        parts.append("Suitable for sensitive skin.")

    # 12. Size
    size = safe_get('size', None)
    if size is not None:
        parts.append(f"Size {size}.")

    return " ".join(parts)

print("Building enhanced product descriptions...")
skincare['product_text'] = skincare.apply(build_product_text_with_sentiment, axis=1)

print(f"\nEnhanced {len(skincare):,} products with rich descriptions!")
print(f"\nSample product_text (first 400 chars):\n")
print(skincare['product_text'].iloc[0][:400] + "...")

# %%time
print("Creating medical knowledge documents...")

medical_documents = []

for _, row in medical_df.iterrows():
    text = f"MEDICAL KNOWLEDGE\n\n"
    text += f"CONDITION: {row['condition'].title()}\n"
    text += f"CATEGORY: {row['category']}\n\n"
    text += f"QUESTION: {row['instruction']}\n\n"
    text += f"ANSWER: {row['output']}"

    doc = Document(
        page_content=text,
        metadata={
            'type': 'medical',
            'condition': row['condition'],
            'category': row['category']
        }
    )
    medical_documents.append(doc)

print(f"Created {len(medical_documents):,} medical documents")

# %%time
print("Creating ingredient knowledge documents...")

ingredient_documents = []

for _, row in ingredient_df.iterrows():
    # Parse good_for list
    try:
        good_for = eval(row['who_is_it_good_for']) if pd.notna(row['who_is_it_good_for']) else []
    except:
        good_for = []

    text = f"INGREDIENT INFORMATION\n\n"
    text += f"INGREDIENT: {row['name']}\n\n"

    if pd.notna(row['what_is_it']):
        text += f"WHAT IS IT: {row['what_is_it']}\n\n"

    if pd.notna(row['what_does_it_do']):
        text += f"WHAT IT DOES: {row['what_does_it_do']}\n\n"

    if good_for:
        text += f"GOOD FOR: {', '.join(good_for)}\n"

    doc = Document(
        page_content=text,
        metadata={
            'type': 'ingredient',
            'ingredient_name': row['name'],
            'good_for': good_for
        }
    )
    ingredient_documents.append(doc)

print(f"Created {len(ingredient_documents):,} ingredient documents")
print(f"\nSample (first 300 chars):\n{ingredient_documents[0].page_content[:300]}...")

# %%time
def create_enhanced_document(row):
    """Create document with ALL features + sentiment"""

    # Helper function to safely get value
    def safe_get(key, default=''):
        try:
            val = row[key]
            return val if pd.notna(val) else default
        except:
            return default

    sections = []

    sections.append(f"PRODUCT: {safe_get('product_name')}")
    sections.append(f"BRAND: {safe_get('brand_name')}")
    sections.append(f"CATEGORY: {safe_get('secondary_category')}")

    price = safe_get('price_usd', 0)
    sections.append(f"PRICE: ${price:.2f}")

    product_text = safe_get('product_text', None)
    if product_text:
        sections.append(f"\nPRODUCT DETAILS: {product_text}")

    description = safe_get('description', None)
    if description:
        sections.append(f"\nDESCRIPTION: {description}")

    ingredients = safe_get('ingredients', None)
    if ingredients:
        sections.append(f"\nINGREDIENTS: {ingredients}")

    # UPDATED: Use avg_rating and total_reviews from sentiment pickle
    rating = safe_get('avg_rating', 0)
    total_reviews = safe_get('total_reviews', 0)

    if rating > 0 and total_reviews > 0:
        sections.append(f"\nCUSTOMER RATING: {rating:.1f}/5.0 ({int(total_reviews)} reviews)")

        sentiment = safe_get('predicted_sentiment', None)
        if sentiment:
            sections.append(f"SENTIMENT: {sentiment.upper()}")

            pos_pct = safe_get('positive_rating_pct', None)
            if pos_pct is not None:
                sections.append(f"POSITIVE REVIEWS: {pos_pct:.0f}%")

        # UPDATED: Use review_sample (correct column name from pickle)
        review_sample = safe_get('review_sample', None)
        if review_sample:
            sections.append(f"SAMPLE REVIEWS: {review_sample}")

    text = "\n".join(sections)

    # Enhanced metadata with safe access
    concerns = []
    for c in ['dullness', 'dryness', 'anti_aging', 'acne', 'dark_spots', 'redness', 'plumping']:
        if safe_get(c, 0) == 1:
            concerns.append(c.replace('_', ' '))

    actives = []
    for a in ['hyaluronic_acid', 'niacinamide', 'retinol', 'salicylic_acid', 'aha_glycolic_acid', 'spf']:
        if safe_get(a, 0) == 1:
            actives.append(a.replace('_', ' '))

    skin_types = []
    for st in ['dry_skin', 'oily_skin', 'combination_skin', 'normal_skin', 'sensitive_skin']:
        if safe_get(st, 0) == 1:
            skin_types.append(st.replace('_', ' '))

    metadata = {
        'type': 'product',
        'name': str(safe_get('product_name', '')),
        'brand': str(safe_get('brand_name', '')),
        'price': float(price),
        'category': str(safe_get('secondary_category', '')),
        'rating': float(rating),
        'total_reviews': int(total_reviews),
        'concerns': concerns,
        'actives': actives,
        'skin_types': skin_types,
        'vegan': safe_get('vegan', 0) == 1,
        'cruelty_free': safe_get('cruelty_free', 0) == 1,
        'fragrance_free': safe_get('fragrance_free', 0) == 1,
        'clean_beauty': safe_get('clean_at_sephora', 0) == 1,
        'sentiment': str(safe_get('predicted_sentiment', 'unknown')),
        'sentiment_score': float(safe_get('predicted_sentiment_score', 0)),
        'review_quality': float(safe_get('avg_review_quality', 0)),
        'positive_pct': float(safe_get('positive_rating_pct', 0))
    }

    return Document(page_content=text, metadata=metadata)

print("Creating product documents...")
product_documents = [create_enhanced_document(row) for _, row in skincare.iterrows()]

print(f"\nCreated {len(product_documents):,} product documents")
print(f"\nSample metadata:")
import pprint
pprint.pprint(product_documents[0].metadata, width=100, compact=True)

# %%time
print("Combining all documents...")
all_documents = product_documents + medical_documents + ingredient_documents

print(f"\nTotal documents: {len(all_documents):,}")
print(f"   - Products: {len(product_documents):,}")
print(f"   - Medical Q&A: {len(medical_documents):,}")
print(f"   - Ingredients: {len(ingredient_documents):,}")

print("\nLoading BGE-Large embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("\nCreating FAISS vector store (10-15 minutes)...")
vectorstore = FAISS.from_documents(documents=all_documents, embedding=embeddings)
vectorstore.save_local("faiss_index")

print("\nVector store ready!")
print("   Knowledge base: Products + Medical + Ingredients + Sentiment")

# !pip install -U bitsandbytes -q

# %%time
import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.generation').setLevel(logging.ERROR)

print("Loading Mistral-7B...\n")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.15,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)
print("Mistral-7B loaded!")

class EnhancedIntentExtractor:
    """Understands product, medical, and ingredient queries"""

    def __init__(self):
        self.concerns = [
            'acne', 'pimples', 'breakouts', 'blemishes',
            'wrinkles', 'fine lines', 'aging', 'anti-aging', 'anti aging',
            'dark spots', 'hyperpigmentation',
            'dryness', 'dry skin', 'flaky',
            'oily', 'shine', 'greasy',
            'redness', 'sensitive', 'irritation',
            'dullness', 'brightening', 'plumping', 'pores',
            'psoriasis', 'eczema', 'dermatitis', 'rosacea', 'vitiligo', 'melanoma', 'hives'
        ]

        self.ingredients = [
            'retinol', 'vitamin c', 'vitamin-c', 'niacinamide', 'hyaluronic acid', 'hyaluronic',
            'salicylic acid', 'salicylic', 'glycolic acid', 'glycolic', 'aha', 'bha', 'spf', 'sunscreen'
        ]

        self.preferences = [
            'vegan', 'cruelty free', 'cruelty-free', 'paraben free', 'paraben-free',
            'sulfate free', 'sulfate-free', 'fragrance free', 'unscented',
            'hypoallergenic', 'clean beauty', 'clean'
        ]

        self.medical_patterns = [
            'what is', 'what are', 'symptoms of', 'cause of', 'causes of',
            'treatment for', 'how to treat', 'medication for', 'what causes', 'etiology'
        ]

        self.categories = [
            'cleanser', 'toner', 'serum', 'moisturizer', 'cream', 'oil', 'mask', 'exfoliant', 'sunscreen'
        ]

    def is_medical_query(self, text: str) -> bool:
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.medical_patterns)

    def extract_preferences(self, text: str) -> Dict[str, bool]:
        text = text.lower()
        prefs = {}
        if 'vegan' in text: prefs['vegan'] = True
        if 'cruelty' in text: prefs['cruelty_free'] = True
        if 'fragrance free' in text or 'unscented' in text: prefs['fragrance_free'] = True
        if 'clean' in text: prefs['clean_beauty'] = True
        return prefs

    def extract_ingredients(self, text: str) -> List[str]:
        text = text.lower()
        return [ing for ing in self.ingredients if ing in text]

    def extract_budget(self, text: str) -> Optional[float]:
        text = text.lower()
        patterns = [
            r'under\s*\$?(\d+)', r'below\s*\$?(\d+)', r'less than\s*\$?(\d+)',
            r'up to\s*\$?(\d+)', r'around\s*\$?(\d+)', r'max\s*\$?(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def extract_concerns(self, text: str) -> List[str]:
        text = text.lower()
        return [concern for concern in self.concerns if concern in text]

    def extract_category(self, text: str) -> Optional[str]:
        text = text.lower()
        for category in self.categories:
            if category in text:
                return category
        return None

    def analyze(self, text: str) -> Dict[str, Any]:
        return {
            'budget': self.extract_budget(text),
            'concerns': self.extract_concerns(text),
            'category': self.extract_category(text),
            'preferences': self.extract_preferences(text),
            'ingredients': self.extract_ingredients(text),
            'is_medical': self.is_medical_query(text),
            'original_text': text
        }

intent_extractor = EnhancedIntentExtractor()
print("Intent extractor ready!")

import sys
from io import StringIO

class SkincareAgent:
    """ULTIMATE Agent with Products + Medical + Ingredients + Sentiment"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.intent_extractor = EnhancedIntentExtractor()
        self.conversation_history = []

        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        self.prompt_template = """[INST] You are a friendly skincare expert with medical and ingredient knowledge.

CONTEXT:
{context}

USER: {question}

Provide helpful, accurate advice. If medical info or ingredient details are provided, explain clearly. Recommend 3-4 products when relevant. Be warm and conversational. [/INST]"""

        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def _search_with_filters(self, query: str, budget: Optional[float] = None,
                            concerns: List[str] = None, category: Optional[str] = None,
                            preferences: Dict[str, bool] = None, ingredients: List[str] = None,
                            is_medical: bool = False) -> List[Document]:

        enhanced_query = query
        if concerns:
            enhanced_query += " " + " ".join(concerns)
        if category:
            enhanced_query += " " + category
        if ingredients:
            enhanced_query += " " + " ".join(ingredients)

        has_filters = bool(budget or preferences)
        k_value = 12 if has_filters else 5

        docs = self.vectorstore.similarity_search(enhanced_query, k=k_value)

        if is_medical:
            medical_docs = [d for d in docs if d.metadata.get('type') == 'medical']
            product_docs = [d for d in docs if d.metadata.get('type') == 'product']
            return medical_docs[:3] + product_docs[:2]

        if not has_filters:
            return [d for d in docs if d.metadata.get('type') != 'medical'][:5]

        filtered = []
        for doc in docs:
            if doc.metadata.get('type') == 'medical':
                continue

            if budget and doc.metadata.get('price', 0) > budget:
                continue

            if preferences:
                if preferences.get('vegan') and not doc.metadata.get('vegan'):
                    continue
                if preferences.get('cruelty_free') and not doc.metadata.get('cruelty_free'):
                    continue
                if preferences.get('fragrance_free') and not doc.metadata.get('fragrance_free'):
                    continue
                if preferences.get('clean_beauty') and not doc.metadata.get('clean_beauty'):
                    continue

            filtered.append(doc)
            if len(filtered) >= 5:
                break

        return filtered[:5]

    def chat(self, user_message: str, show_details: bool = True) -> Optional[str]:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()

        try:
            intent = self.intent_extractor.analyze(user_message)

            products = self._search_with_filters(
                user_message,
                budget=intent['budget'],
                concerns=intent['concerns'],
                category=intent['category'],
                preferences=intent.get('preferences', {}),
                ingredients=intent.get('ingredients', []),
                is_medical=intent.get('is_medical', False)
            )

            response = self.rag_chain.invoke(user_message)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        clean_response = response
        if "[/INST]" in clean_response:
            clean_response = clean_response.split("[/INST]")[-1]
        if "[INST]" in clean_response:
            clean_response = clean_response.split("[INST]")[0]
        clean_response = clean_response.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"').replace("\\", "")
        clean_response = clean_response.replace("**", "").replace("* ", "").strip()

        if clean_response and not clean_response[-1] in '.!?':
            sentences = clean_response.split('.')
            if len(sentences) > 1:
                clean_response = '.'.join(sentences[:-1]) + '.'

        self.conversation_history.append({
            'user': user_message,
            'agent': clean_response,
            'intent': intent,
            'results': [p.metadata for p in products]
        })

        if show_details:
            print(f"You: {user_message}")

            if intent['budget'] or intent['concerns'] or intent.get('preferences') or intent.get('ingredients'):
                info = []
                if intent.get('is_medical'):
                    info.append("Medical Query")
                if intent['budget']:
                    info.append(f"Budget: ${intent['budget']:.0f}")
                if intent['concerns']:
                    info.append(', '.join(intent['concerns']).title())
                if intent.get('ingredients'):
                    info.append(f"Ingredients: {', '.join(intent['ingredients'])}")
                if intent.get('preferences'):
                    prefs = [k.replace('_', ' ').title() for k, v in intent['preferences'].items() if v]
                    if prefs:
                        info.append(f"{', '.join(prefs)}")
                print(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ {' | '.join(info)}")

            print("\nAnalyzing...\n")
            print("Skincare Expert:")
            print()

            for line in clean_response.split('\n'):
                if line.strip():
                    print(line.strip())
            print()

            product_results = [p for p in products if p.metadata.get('type') == 'product']
            if product_results:
                print("Recommended Products")

                for i, p in enumerate(product_results, 1):
                    rating = p.metadata['rating']

                    # FIXED: Show stars or "No reviews yet"
                    if rating >= 3:
                        stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â" * min(5, max(1, int(rating)))
                    elif rating > 0:
                        stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â"
                    else:
                        stars = "No reviews yet"

                    sentiment = p.metadata.get('sentiment', 'unknown')
                    sentiment_emoji = {"positive": "ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€¹Ã…â€œÃƒâ€¦Ã‚Â ", "neutral": "ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€¹Ã…â€œÃƒâ€šÃ‚Â", "negative": "ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€¹Ã…â€œÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢"}.get(sentiment, "")

                    print(f"\n{i}. {p.metadata['brand']} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {p.metadata['name']}")
                    sentiment_display = f" ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¢ {sentiment_emoji} {sentiment}" if sentiment_emoji else ""
                    print(f"   ${p.metadata['price']:.2f} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¢ {stars}{sentiment_display}")

                    if p.metadata.get('positive_pct', 0) > 0:
                        print(f"   {p.metadata['positive_pct']:.0f}% positive reviews")

            return None
        else:
            return clean_response

    def get_history(self) -> List[Dict]:
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []

    def get_last_results(self) -> List[Dict]:
        if self.conversation_history:
            return self.conversation_history[-1]['results']
        return []

agent = SkincareAgent(vectorstore, llm)

print("ULTIMATE AGENT READY!")


######################################
# App for demo
#######################################

import gradio as gr
def chat_with_filters(message, budget, concerns, vegan, cruelty_free, fragrance_free, history):
    """
    Chat function with filter integration
    """
    # Build enhanced query with filters
    enhanced_message = message
    
    if concerns:
        enhanced_message += f" I'm concerned about {', '.join(concerns)}."
    
    if budget < 200:
        enhanced_message += f" My budget is ${budget}."
    
    preference_list = []
    if vegan:
        preference_list.append("vegan")
    if cruelty_free:
        preference_list.append("cruelty-free")
    if fragrance_free:
        preference_list.append("fragrance-free")
    
    if preference_list:
        enhanced_message += f" I prefer {' and '.join(preference_list)} products."
    
    # Get response from agent
    response = agent.chat(enhanced_message, show_details=False)
    
    # Get products
    products = agent.get_last_products()
    
    print("Enhanced message:", enhanced_message)
    print("Agent response:", response)
    print("Products:", products)
    
    # Format response with products
    if products:
        response += "\n\n **Recommended Products:**\n\n"
        for i, product in enumerate(products[:5], 1):
            rating = product.get('rating', 0)
            if rating >= 4.5:
                stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â"
            elif rating >= 4.0:
                stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â"
            elif rating >= 3.0:
                stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â"
            elif rating > 0:
                stars = "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚Â­Ãƒâ€šÃ‚Â"
            else:
                stars = "New"
            
            response += f"\n**{i}. {product['brand']} - {product['name']}**\n"
            response += f"    ${product['price']:.2f} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¢ {stars}\n"
            
            badges = []
            if product.get('vegan'):
                badges.append("Vegan")
            if product.get('cruelty_free'):
                badges.append("Cruelty-Free")
            if product.get('fragrance_free'):
                badges.append("Fragrance-Free")
            if product.get('clean_beauty'):
                badges.append("Clean Beauty")
            if badges:
                response += f"   {' ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¢ '.join(badges)}\n"
    
    return response

# Create interface with custom layout
with gr.Blocks(theme=gr.themes.Soft(), title="DermaLLM") as demo:
    gr.Markdown("""
    #  DermaLLM - Your AI Skincare Advisor
    Get personalized skincare recommendations from our AI expert!
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
            value=[],  # initialize as empty list
            height=500,
            label="Chat with your skincare expert",
            show_label=True,
            avatar_images=(None, "ÃƒÂ°Ã…Â¸Ã‚Â§Ã¢â‚¬ËœÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã…Â¡Ã¢â‚¬Â¢ÃƒÂ¯Ã‚Â¸Ã‚Â")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything about skincare...",
                    show_label=False,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### Filters (Optional)")
            gr.Markdown("*Or just chat naturally!*")
            
            budget = gr.Slider(
                minimum=10,
                maximum=200,
                value=100,
                step=5,
                label="Max Budget ($)"
            )
            
            concerns = gr.CheckboxGroup(
                choices=["Acne", "Anti-Aging", "Dark Spots", "Redness", "Dryness", "Sensitivity", "Texture"],
                label="Skin Concerns"
            )
            
            gr.Markdown("### Product Preferences")
            vegan = gr.Checkbox(label="Vegan", value=False)
            cruelty_free = gr.Checkbox(label="Cruelty-Free", value=False)
            fragrance_free = gr.Checkbox(label="Fragrance-Free", value=False)
            
            clear = gr.Button("Clear Chat", variant="secondary")
    
    gr.Markdown("### Try asking:")
    with gr.Row():
        gr.Examples(
            examples=[
                "I need a moisturizer for dry sensitive skin",
                "What's the best affordable acne treatment?",
                "Recommend an anti-aging serum",
                "I want a gentle cleanser for oily skin"
            ],
            inputs=msg,
            label=None
        )
    
    # Chat functionality
    def respond(message, budget, concerns, vegan, cruelty_free, fragrance_free, chat_history):
      if chat_history is None:
        chat_history = []

      if not message:
        return chat_history, ""
    
      bot_response = chat_with_filters(
        message, budget, concerns, vegan, cruelty_free, fragrance_free, chat_history
      )
    
      # Append as dicts
      chat_history.append({"role": "user", "content": message})
      chat_history.append({"role": "assistant", "content": bot_response})
    
      return chat_history, ""
    
    # Event handlers
    submit.click(
        respond,
        inputs=[msg, budget, concerns, vegan, cruelty_free, fragrance_free, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        respond,
        inputs=[msg, budget, concerns, vegan, cruelty_free, fragrance_free, chatbot],
        outputs=[chatbot, msg]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch with public URL (valid for 72 hours)
demo.launch(share=True, debug=True)
