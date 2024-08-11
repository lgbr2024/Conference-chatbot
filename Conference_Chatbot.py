import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]


# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Define ModifiedPineconeVectorStore class
class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Tuple[Document, float]]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        return [
            (
                Document(
                    page_content=result["metadata"].get(self._text_key, ""),
                    metadata={k: v for k, v in result["metadata"].items() if k != self._text_key}
                ),
                result["score"],
            )
            for result in results["matches"]
        ]

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20,
        lambda_mult: float = 0.5, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Document]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        if not results['matches']:
            return []
        
        embeddings = [match['values'] for match in results['matches']]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=min(k, len(results['matches'])),
            lambda_mult=lambda_mult
        )
        
        return [
            Document(
                page_content=results['matches'][i]['metadata'].get(self._text_key, ""),
                metadata={
                    'source': results['matches'][i]['metadata'].get('source', '').split('data\\')[-1] if 'source' in results['matches'][i]['metadata'] else 'Unknown'
                }
            )
            for i in mmr_selected
        ]

# Define maximal_marginal_relevance function
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[List[float]],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    # ... (paste the entire function definition here)

# Streamlit app
def main():
    st.title("Conference Q&A System")

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)

    # Select GPT model
    option = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"))
    llm = ChatOpenAI(model=option)

    # Set up Pinecone vector store
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )

    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.75}
    )

    # Set up prompt template and chain
    template = """
    You are a Korean assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    You should answer in KOREAN and please give rich sentences to make the answer much better.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            content = doc.page_content if doc.page_content else "No content available"
            formatted.append(f"Source: {source}\nContent Snippet: {content[:200]}...")
        return "\n\n" + "\n\n".join(formatted)

    format = itemgetter("docs") | RunnableLambda(format_docs)
    answer = prompt | llm | StrOutputParser()

    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )

    # User input
    question = st.text_input("Enter your question about the conference:")

    if st.button("Ask"):
        if question:
            response = chain.invoke(question)
            answer = response['answer']
            source_documents = response['docs'][:10]  # Get up to 10 documents

            st.write("Answer:", answer)
            st.write("\nReference Documents:")
            for i, doc in enumerate(source_documents, 1):
                st.write(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"   Content: {doc.page_content[:100]}...")  # Display part of the content
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
