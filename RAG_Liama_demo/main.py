from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

db_path = './DB'
bed_server = OllamaEmbeddings(base_url="http://localhost:11434" , model="nomic-embed-text")
oll_server = Ollama(base_url="http://localhost:11434",model="llama3:8b")

vectorstire_from_db = Chroma(
    persist_directory= db_path,
    embedding_function= bed_server
)

#question = "小米su7轿跑的配置如何"
#question = "麻雀算法对推荐算法有什么作用"
while(True):
    try:
        question = input()
        docs = vectorstire_from_db.similarity_search(question )
        qachain = RetrievalQA.from_chain_type(oll_server , retriever=vectorstire_from_db.as_retriever())
        ans = qachain.invoke({"query":"请使用中文回答：" + question})
        print(ans['result'])
    except KeyboardInterrupt:
        break