import os
from typing import Any,Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
# A way to initialize a chat client to make the LLM request
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
# To create a tool that will retieve a certain query from vector store
from langchain.tools import tool
# To retrieve simila vectors from vector store for context to LLM
from langchain_pinecone import PineconeVectorStore
# To embed the query of the user
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10, api_key=os.environ.get("OPENAI_API_KEY")
)

# Initialize Vector Store
vectorStore = PineconeVectorStore(index_name="rag-document-assistant", embedding = embeddings)

# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve the most relevant documents from the vector store for a user query."""

    # retrieve top 4 most similar documents
    retrieved_docs = vectorStore.as_retriever().invoke(query, k=4)

    # Serialize documents to be one big string to give to the model
    serialized="\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')} \n\n Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
        Run the RAG retrieval pipeline to answeer a query using the retrieved documents

        Args:
            query: The user's question

        Returns:
            Dictionary containing:
                - answer: the generated answer
                - context: List or retrieved documents
    """
    # Create the agent with the retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation"
        "You have access to a tool that retrieves relevant documentation"
        "Use the tool to find relevant onformation before answering questions"
        "Always cite the resources you use in your answer"
        "If you cannot find the answer in the retrieved documentation say so"   
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    response = agent.invoke({"query": query})

    answer = response["messages"][-1].content

    return answer


if __name__ == "__main__":
    result = run_llm("What year did hossam hassan stop playing football")
    print(result)


