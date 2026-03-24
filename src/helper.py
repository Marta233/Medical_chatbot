# extract text from pdf files
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    return documents
def filter_to_minimal_document(documents: List[Document]) -> List[Document]:
    minimal_documents:List[Document] = []
    for doc in documents:
        minimal_documents.append(Document(
            page_content=doc.page_content, 
            metadata={'source': doc.metadata.get('source')}))
    return minimal_documents
# Chunk the text into smaller pieces
def text_spliter(minimal_documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_chunk = text_splitter.split_documents(minimal_documents)
    return split_chunk
# Convert chunks into vector Operation
from langchain.embeddings import HuggingFaceEmbeddings
def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings