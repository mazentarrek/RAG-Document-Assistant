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

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Async function to process documents and index tthem in batches"""
    print("Vector Storage Phase \n")
    # Create batches
    batches=[
        documents[i : i+batch_size] for i in range (0, len(documents), batch_size)
    ]

    print(f"Documents Split into {len(batches)} batches \n")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            vectorStore.add_documents(batch)
            print(f"Successfully added batch number: {batch_num}/{len(batches)}")
        except Exception as e:
            print(f"Failed to add batch {batch_num}/{len(batches)}: {type(e).__name__}: {e}")
            return False
        return True
    
    #Process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        print("All batches processed successfully")
    else:
        print(f"Processed {successful/len(batches)} batches successfully")

async def main():
    """Main async function to orchestrate the entire ingestion process"""
    print("DOCUMENTATION INGESTION PIPELINE \n")
    print("TavilyCrawl: Startig to crawl documentation at https://en.wikipedia.org/wiki/Hossam_Hassan")
    res = tavily_crawl.invoke({
        "url": "https://en.wikipedia.org/wiki/Hossam_Hassan",
        "max_depth":1, 
        "extract_depth":"advanced"
    })

    all_docs=[Document(page_content=result['raw_content'], metadata={"source":result['url']}) for result in res['results']]
    print(f"{all_docs} \n")

    # Split Documents into CHunks
    print("Splitting Documents into Chunks \n")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs=text_splitter.split_documents(all_docs)

    #Index documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    print("Pipeline Completed")






if __name__ == "__main__":
    asyncio.run(main())