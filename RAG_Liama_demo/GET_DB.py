import glob
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

bed_server = OllamaEmbeddings(base_url="http://localhost:11434" , model="nomic-embed-text")
oll_server = Ollama(base_url="http://localhost:11434",model="llama3:8b")
db_path = "./DB"

pdf_file_path = sorted(glob.glob(os.path.join('./data/', "*.pdf")))
for file_path in pdf_file_path:
    print(file_path)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=128)
    all_splits = text_splitter.split_documents(docs)

    vectorstire_to_db = Chroma.from_documents(
        documents= all_splits,
        embedding= bed_server,
        persist_directory= db_path
    )
