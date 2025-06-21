import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os

import asyncio
load_dotenv()
HF_API = os.getenv("HUGGING_FACE_API")
MISTRAL_API = os.getenv("MISTRAL_API")

#Loading the docs from langgraph documentation
loader_multiple_pages = WebBaseLoader(
    ["https://python.langchain.com/docs/introduction/?_gl=1*18piwnm*_ga*MTkzNjkxMTcwNy4xNzQ1NDg4MTQ3*_ga_47WX3HKKY2*czE3NDY1MzA1MjAkbzIkZzAkdDE3NDY1MzA1MjAkajAkbDAkaDA.", 
    "https://langchain-ai.github.io/langgraph/",
    "https://langchain-ai.github.io/langgraph/how-tos/state-reducers/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/#prompt-chaining"
    ]
)
langgraph_docs = []
async def load_langgraph_docs():
    async for doc in loader_multiple_pages.alazy_load():
        langgraph_docs.append(doc)
asyncio.run(load_langgraph_docs())

#docs = [WebBaseLoader(url).load() for url in loader_multiple_pages]

# Load the Attention is all you need paper from pdf
file_path = r"C:\Users\lenovo\Downloads\attentionisallyouneed.pdf"
loader = PyPDFLoader(file_path)
attention_paper_pages = []
async def load_attention_paper_pages():
    async for doc in loader.alazy_load():
        attention_paper_pages.append(doc)
asyncio.run(load_attention_paper_pages())




#Splitting the documents into chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
attention_splits = splitter.split_documents(attention_paper_pages)
langgraph_splits = splitter.split_documents(langgraph_docs)


# Add source label
for doc in langgraph_splits:
    doc.metadata = {"source": "langgraph"}

for doc in attention_splits:
    doc.metadata = {"source": "attention"}



# defining embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
DB_DIR = "./chroma_docs_rag"
if not os.path.exists(DB_DIR):
    # Combine and add
    all_chunks = langgraph_splits + attention_splits
    vector_store = Chroma(
        collection_name="Documentations",
        embedding_function=embeddings,
        persist_directory = DB_DIR
        )
    vector_store.add_documents(all_chunks)
    print("Built VECTOR DB and saved!!!")
else:
    vector_store = Chroma(
        collection_name="Documentations",
        embedding_function=embeddings,
        persist_directory = DB_DIR
        )
    print("Vector DB loaded from disk !! 🚀")

query = " what is attention score"

docs_and_scores = vector_store.similarity_search_with_score(query, k=2)
relevant_doc = [doc for doc, score in docs_and_scores]  
# lower score = more similar becuase we are calculatin cosine distance, the less the distance the more close the vectors are !!!!

print(relevant_doc)
