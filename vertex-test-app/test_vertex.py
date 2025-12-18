import vertexai
from vertexai.generative_models import GenerativeModel

# Test file
# Initializing with project ID
vertexai.init(project="skin-care-recommender", location="us-central1")
try:
    print("Attempting to talk to Gemini...")
    model = GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Hello, are you working?")
    print("SUCCESS! Response:", response.text)
except Exception as e:
    print("\nFAILED. Here is the real error:\n", e)