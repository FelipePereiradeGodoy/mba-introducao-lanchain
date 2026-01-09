import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

for key in ("OPENAI_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION_NAME"):
    if not os.getenv(key):
        raise RuntimeError(f"Environment variable {key} is not set")
    
query = "Tell me amore about the gpt-5 thinking evaluation and performance results comparing to gpt-4"

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=3)

for count, (doc, score) in enumerate(results, start=1):
    print("="*30)
    print(f"Resultado {count} (score: {score: .2f}):")
    print("="*30)

    print("\nTexto:\n")
    print(doc.page_content.strip())

    print("\nMetadados:\n")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")