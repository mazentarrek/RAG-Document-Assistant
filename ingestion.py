import asyncio
import os
import ssl
from pathlib import Path
from typing import Any,Dict, List

import certifi
from dotenv import load_dotenv

# (RecursiveCharacterTextSplitter) A langchain helper class to help us split the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# (Chroma) A vector store to store everything locally if you do not want to use a cloud-based vector store
from langchain_chroma import Chroma
# (Document) A Class abstraction for handling text data
from langchain_core.documents import Document
# (OpenAIEmbeddings) Our Embedding model from OpenAI
from langchain_openai import OpenAIEmbeddings
# (PineconeVectorStore) A Cloud based vector store
from langchain_pinecone import PineconeVectorStore
# Tavily Crawl is what gets the documentations
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

#Configure SSL Context to use certifi certificates
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10, api_key=os.environ.get("OPENAI_API_KEY")
)

vectorStore = PineconeVectorStore(index_name="rag-document-assistant", embedding = embeddings)
tavily_extract = TavilyExtract()
tavily_crawl = TavilyCrawl()

async def main():
    """Main async function to orchestrate the entire ingestion process"""
    print("DOCUMENTATION INGESTION PIPELINE")
    print()
    print("TavilyCrawl: Startig to crawl documentation at https://en.wikipedia.org/wiki/Hossam_Hassan")
    res = tavily_crawl.invoke({
        "url": "https://en.wikipedia.org/wiki/Hossam_Hassan",
        "max_depth":1, 
        "extract_depth":"advanced"
    })

    all_docs=[Document(page_content=result['raw_content'], metadata={"source":result['url']}) for result in res['results']]
    print(all_docs)



if __name__ == "__main__":
    asyncio.run(main())