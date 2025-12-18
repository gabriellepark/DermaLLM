import re
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import ChatVertexAI  # <--- NEW CLASS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- 1. Intent Extractor ---
class EnhancedIntentExtractor:
    def __init__(self):
        self.concerns = ['acne', 'wrinkles', 'aging', 'dark spots', 'dryness', 'oily', 'redness', 'sensitive', 'dullness']
        self.ingredients = ['retinol', 'vitamin c', 'niacinamide', 'hyaluronic', 'salicylic', 'glycolic', 'spf']
        self.categories = ['cleanser', 'toner', 'serum', 'moisturizer', 'cream', 'mask', 'sunscreen']
        self.preferences = ['vegan', 'cruelty_free', 'fragrance_free', 'clean_beauty']

    def analyze(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        return {
            'budget': self._extract_budget(text),
            'concerns': [c for c in self.concerns if c in text],
            'category': next((c for c in self.categories if c in text), None),
            'ingredients': [i for i in self.ingredients if i in text],
            'preferences': {p: True for p in self.preferences if p.replace('_', ' ') in text or p.replace('_', '-') in text}
        }

    def _extract_budget(self, text):
        match = re.search(r'under\s*\$?(\d+)', text)
        return float(match.group(1)) if match else None

# --- 2. Main Agent Class ---
class SkincareAgent:
    def __init__(self):
        # Load Embeddings
        # (We silence the warning by knowing it works, even if deprecated)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5", # <--- Change to "large"
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load Vector Store
        self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)

        # Switching to 2.5 because it works better w/ Gemini and is updated version



        self.llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=2048, # <--- 4x the previous limit
        )
        
        self.intent_extractor = EnhancedIntentExtractor()
        self.conversation_history = [] 

        # Template
        self.prompt_template = """You are a helpful skincare expert. 
        Based on these products, recommend the best ones for the user.
        
        PRODUCTS:
        {context}

        USER REQUEST: {question}

        - Recommend 3 products.
        - Mention the price and brand.
        - Explain why they fit the user's needs.
        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def _search_with_filters(self, query: str, intent: Dict) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=20)
        filtered = []
        budget = intent.get('budget')
        prefs = intent.get('preferences', {})

        for doc in docs:
            meta = doc.metadata
            if budget and meta.get('price', 0) > budget: continue
            if prefs.get('vegan') and not meta.get('vegan'): continue
            if prefs.get('cruelty_free') and not meta.get('cruelty_free'): continue
            if prefs.get('fragrance_free') and not meta.get('fragrance_free'): continue
            if prefs.get('clean_beauty') and not meta.get('clean_beauty'): continue
            filtered.append(doc)
            if len(filtered) >= 4: break 
            
        return filtered if filtered else docs[:4]

    def get_response(self, user_text: str):
        return self.chat(user_text, show_details=False)

    def chat(self, user_text: str, show_details: bool = False):
        intent = self.intent_extractor.analyze(user_text)
        products = self._search_with_filters(user_text, intent)
        
        # Store for testing logic
        product_metadata = [p.metadata for p in products]
        self.conversation_history.append({'products': product_metadata})

        context_str = "\n\n".join([p.page_content for p in products])
        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke(user_text)
        return response

    def get_last_products(self):
        if self.conversation_history:
            return self.conversation_history[-1]['products']
        return []