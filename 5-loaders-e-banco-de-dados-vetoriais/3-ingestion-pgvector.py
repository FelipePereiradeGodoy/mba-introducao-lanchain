import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for key in ("OPENAI_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION_NAME"):
    if not os.getenv(key):
        raise RuntimeError(f"Environment variable {key} is not set")
    
current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=False
).split_documents(docs)

if not splits:
    raise SystemExit(0)

enriched = [
    Document(
        page_content=doc.page_content,
        metadata={key: value for key, value in doc.metadata.items() if value not in ("", None)}
    )
    for doc in splits
]

ids = [f"doc-{count}" for count in range(len(enriched))]

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION_NAME"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

store.add_documents(documents=enriched, ids=ids)

