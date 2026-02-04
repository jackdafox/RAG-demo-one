import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from typing import List

from langchain_core.documents import Document

load_dotenv()


def load_documents(docs_path="docs") -> List[Document]:
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} doesn't exist")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found at {docs_path}")


    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0) -> list[Document]:
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chromadb"):
    print("Creating embeddings and storing in ChromaDB...")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("-- Finish --")
    
    print(f"Vector store created and saved to {persist_directory}")
    
    return vectorstore


def main():
    print("main function")
    documents = load_documents(docs_path="docs")
    chunked_documents = split_documents(documents, chunk_size=800)
    
    vector_store = create_vector_store(chunked_documents)


if __name__ == "__main__":
    main()
