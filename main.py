# /// script
# dependencies = [
#   "langchain-community",
#   "langchain-openai",
#   "langchain",
#   "supabase",
#   "python-dotenv",
#   "langchain-core",
# ]
# ///



import os
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(".env")

def notion_to_supabase():
    """Load documents from a Notion database and process them."""
    

    NOTION_TOKEN = os.environ["NOTION_TOKEN"]
    DATABASE_ID = os.environ["DATABASE_ID"]

    loader = NotionDBLoader(
        integration_token=NOTION_TOKEN,
        database_id=DATABASE_ID,
        request_timeout_sec=30  # Optional, defaults to 10
    )
    docs = loader.load()
    return docs

def create_supabase_client(docs: list):
    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = OpenAIEmbeddings()

    vector_store = SupabaseVectorStore.from_documents(
        docs, # list of Document objects from Notion
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500
    )
    return vector_store


def create_RAG_chain(retriever):
    base_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-3.5-turbo"), temperature=1)

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | base_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    """Main function to execute the Notion to Supabase data transfer."""

    import argparse
    parser = argparse.ArgumentParser(description="Transfer Notion database to Supabase Vector Store")
    parser.add_argument("--query", type=str, help="Query to test the vector store", required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use for RAG")

    args = parser.parse_args()
    docs = notion_to_supabase()
    vector_store = create_supabase_client(docs)
    retriever = vector_store.as_retriever()
    retriever.get_relevant_documents(query=args.query)
    rag_chain = create_RAG_chain(retriever=retriever)

    answer = rag_chain.invoke({"question": args.query})
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
