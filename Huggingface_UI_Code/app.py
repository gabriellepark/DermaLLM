import gradio as gr
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import traceback
import warnings

warnings.filterwarnings('ignore')

print("Starting Skincare Agent...")

# Import agent
from agent_code import agent

print("✓ Agent loaded!")

# ============================================================================
# CHAT FUNCTION - Returns plain text (ChatInterface handles formatting)
# ============================================================================

def chat_fn(message, history):
    """Chat function - returns just the response text"""
    try:
        print(f"\n[USER] {message}")
        start = time.time()
        
        # Call agent with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.chat, message, False)
            try:
                response = future.result(timeout=60)
                elapsed = time.time() - start
                print(f"[AGENT] ✅ {elapsed:.1f}s")
                return response  # Just return the string
                
            except TimeoutError:
                return "⏱️ Timeout (>60s). Please try a simpler question."
                
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        return f"❌ Error: {str(e)[:200]}"

# ============================================================================
# GRADIO UI - Simple ChatInterface (NO type parameter)
# ============================================================================

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🫧SkinWise",
    description="""
### Your Personal Skincare Expert Powered by Medical Knowledge, Ingredient Science & Customer Reviews. 
#### Note: Always read product labels for allergy awareness and consult medical professionals for chronic skin issues.

Ask me anything about skincare products, ingredients, or skin conditions!
    """,
    examples=[
        "What is retinol and what does it do?",
        "Show me a good moisturizer under $50",
        "Recommend vegan products for dry skin",
        "What helps with acne and redness?",
        "Products with vitamin C for dullness",
        "What causes eczema?"
    ],
    cache_examples=False,
)

# Launch
if __name__ == "__main__":
    print("🚀 Launching Gradio...")
    demo.queue(max_size=10)
    demo.launch()
    print("\n✅ App ready!")