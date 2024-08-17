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
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    similarity_scores = cosine_similarity([query_embedding], embedding_list)[0]
    selected_indices = []
    candidate_indices = list(range(len(embedding_list)))
    for _ in range(k):
        if not candidate_indices:
            break
        
        mmr_scores = [
            lambda_mult * similarity_scores[i] - (1 - lambda_mult) * max(
                [cosine_similarity([embedding_list[i]], [embedding_list[s]])[0][0] for s in selected_indices] or [0]
            )
            for i in candidate_indices
        ]
        max_index = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(max_index)
        candidate_indices.remove(max_index)
    return selected_indices
def main():
    st.title("Conference Q&A System")
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "conference"
    index = pc.Index(index_name)
    # Select GPT model
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"
    
    st.session_state.gpt_model = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"), index=("gpt-4o", "gpt-4o-mini").index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)
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
    <prompt>
    Question: {question} 
    Context: {context} 
    Answer:

    <context>
    <role>
      Strategic consultant for LG Group, tasked with uncovering new trends and insights based on conference trends.
    </role>
    <audience>
      - LG Group individual business executives
      - LG Group representative
    </audience>
    <knowledge_base>
      - Conference file saved in vector database
    </knowledge_base>
    <goal>
      Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights.
    </goal>
    </context>
    <task>
    <description>
      Prepare a report of about 4 A4 pages for each [question], covering changes, issues, and response strategies
    </description>
    <format>
 
    Answer structure:
    [Overall Conference]:
    - Explain the overall context of the conference related to the question
    - Introduce the main points or topics
    
    [Contents]:
    - Analyze the key content discussed at the conference and reference
    - Present relevant data or case studies
    - Show 2~3 data, file sources for each key contents
    
    [Conclusion]:
    - Summarize new trends based on the conference content
    - Present derived insights
    - Suggest future strategic directions
    </format>
 

      
    
    <style>
      Business writing with clear and concise sentences targeted at executives
    </style>
    <constraints>
 	- Use the following retrieved context to answer the question. 
    	- If you don't know the answer, just say that you don't know. 
    	- Please answer in Korean and provide rich sentences to enhance the quality of the answer.
        - report of about 4 A4 pages for each [question]
        - [Overall Conference] (about 35% of the total answer) /  [Contents] (about 40% of the total answer) / [Conclusion] (about 25% of the total answer)
    </constraints>
    </task>
    <team>
    <member>
      <name>John</name>
      <role>15-year consultant skilled in hypothesis-based thinking</role>
      <expertise>Special ability in business planning and creating outlines</expertise>
    </member>
    <member>
      <name>EJ</name>
      <role>20-year electronics industry research expert</role>
      <expertise>Special ability in finding new business cases and fact-based findings</expertise>
    </member>
    <member>
      <name>JD</name>
      <role>20-year business problem-solving expert</role>
      <expertise>
        - Advancing growth methods for electronics manufacturing companies
        - Future of customer changes and electronics business
        - Future AI development directions
        - Problem-solving and decision-making regarding the future of manufacturing
      </expertise>
    </member>
    <member>
      <name>DS</name>
      <role>25-year consultant leader, Ph.D. in Business Administration</role>
      <expertise>Special ability to refine content for delivery to LG affiliate CEOs and LG Group representatives</expertise>
    </member>
    <member>
      <name>YM</name>
      <role>30-year Ph.D. in Economics and Business Administration</role>
      <expertise>Overall leader overseeing the general quality of content</expertise>
    </member>
   </team>
   </prompt>
    """
    prompt = ChatPromptTemplate.from_template(template)
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)
    format = itemgetter("docs") | RunnableLambda(format_docs)
    answer = prompt | llm | StrOutputParser()
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # User input
    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            response = chain.invoke(question)
            answer = response['answer']
            source_documents = response['docs'][:5]  # Get up to 5 documents
            st.markdown(answer)
            
            with st.expander("Reference Documents"):
                for i, doc in enumerate(source_documents, 1):
                    st.write(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
    
    # Add Plex.tv link
            st.markdown("---")
            st.markdown("[Watch related conference videos (Plex.tv)](https://app.plex.tv)")
        
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
if __name__ == "__main__":
    main()
