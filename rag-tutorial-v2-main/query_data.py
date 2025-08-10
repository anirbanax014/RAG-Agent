import argparse
import os
import sys
from langchain_chroma import Chroma  # ✅ Updated import
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from get_embedding_function import get_embedding_function

# ==== Gemini Setup ====
GEMINI_API_KEY = "AIzaSyDrkN31a7bE7WDyF6ELAWMoWLXhxrstamE"  # 🔑 Replace with your real API key
if not GEMINI_API_KEY:
    print("❌ Gemini API key is missing. Set it in query_data.py.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    # Ensure DB exists
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print(f"❌ Chroma DB not found in '{CHROMA_PATH}'. Run populate_database.py first.")
        sys.exit(1)

    # Load embeddings and DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        print("⚠️ No results found.")
        return

    # Build context from retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Detect available Gemini model
    model_name = None
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        if "models/gemini-1.5-pro" in available_models:
            model_name = "gemini-1.5-pro"
        elif "models/gemini-1.5-flash" in available_models:
            model_name = "gemini-1.5-flash"
        else:
            model_name = available_models[0]  # fallback
    except Exception as e:
        print(f"⚠️ Could not list models: {e}, defaulting to gemini-1.5-pro")
        model_name = "gemini-1.5-pro"

    # Query Gemini
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        response_text = response.text
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        sys.exit(1)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text

if __name__ == "__main__":
    main()
