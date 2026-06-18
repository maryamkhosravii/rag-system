from pathlib import Path
from litellm import completion
from pydantic import BaseModel, Field
from pypdf import PdfReader
from tqdm import tqdm
from tenacity import wait_exponential, retry
from multiprocessing import Pool
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv



load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent/"preprocessed_db")
collection_name = "docs"

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent/"knowledge-base"

EMBEDDING_MODEL = "text-embedding-3-large"

AVERAGE_CHUNK_SIZE = 100

wait = wait_exponential (multiplier=1, min=10, max=240)

WORKERS = 4

openai = OpenAI()





class Result (BaseModel):
    page_content: str
    metadata: dict



class Chunk (BaseModel):
    headline: str = Field (description="A brief heading for this chunk")
    summary: str = Field (description="A short summary of the chunk")
    original_text: str = Field (description="Original chunk text")

    def as_result (self, document):
        metadata = {
            "source": document.get("source"),
            "type": document.get("type")
        }

        page_content = (
            self.headline + "\n\n" +
            self.summary + "\n\n" +
            self.original_text
        )

        return Result (
            page_content=page_content,
            metadata=metadata
        )

    


class Chunks (BaseModel):
    chunks: list[Chunk]





def fetch_documents():
    """Load all PDF documents from knowledge-base"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():

        if not folder.is_dir():
            continue

        doc_type = folder.name

        for file in folder.rglob("*.pdf"):
            reader = PdfReader (file)
            text = "\n".join (page.extract_text() or "" for page in reader.pages)

            documents.append (
                {
                    "type": doc_type,
                    "source": file.as_posix(),
                    "text": text
                }
            )

    print (f"Loaded {len(documents)} documents")

    return documents






def make_prompt (document):

    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1

    return f"""
You are an expert financial document analyst.

Your task is to split a banking or financial regulation document into overlapping chunks for a Retrieval-Augmented Generation (RAG) knowledge base.

document type: {document["type"]}
document source: {document["source"]}

The document contains banking, risk management, liquidity, stress testing, or financial stability information.

A financial assistant chatbot will use these chunks to answer questions about regulations, model risk, liquidity risk, stress testing, and financial governance.

Guidelines:
- Divide the document into approximately {how_many} chunks.
- Ensure the entire document is covered.
- Do not omit any important information.
- Use semantic boundaries whenever possibile.
- Create overlapping chunks (approximately 20-25% overlap).
- Each chunk should represent a coherent topic or concept.
- Preserve important definitions, requirements, procedures, and regulatory guidance.

for each chunk provide:
1. headline (a short descriptive title)
2. summary (2-4 sentences summarizing the key information)
3. original text (the exact text belonging to that chunk)

The collection of chunks should completely represent the document with appropriate overlap.

Documents:
{document["text"]}

Return the chunks in a structured format.
"""





def make_messages (document):
    return [{"role": "user", "content": make_prompt(document)},]





@retry(wait=wait)
def process_document (document):
    messages = make_messages (document)
    response = completion (model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]





def create_chunks (documents):
    """ Create chunks using a number of workers in parallel. If you get a rate limit error, set the WORKERS to 1."""

    chunks = []
    with Pool (processes=WORKERS) as pool:
        for result in tqdm (pool.imap_unordered (process_document, documents), total=len(documents)):
            chunks.extend (result)

    return chunks





def create_embeddings (chunks):
    chroma = PersistentClient (path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
         chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create (model=EMBEDDING_MODEL, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection (collection_name)
   
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add (ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    print (f"Vectorstore created with {collection.count()} documents")
    





if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks (documents)
    create_embeddings (chunks)
    print ("Integration Completed")