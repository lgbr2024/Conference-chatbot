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
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import xml.etree.ElementTree as ET

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

# XML System Prompt
SYSTEM_PROMPT_XML = """
<system_prompt>
  <role>You are a strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends.</role>
  
  <audience>
    <item>LG Group individual business executives</item>
    <item>LG Group representative</item>
  </audience>
  
  <knowledge_base>Conference file saved in a vector database</knowledge_base>
  
  <goal>Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights.</goal>
  
  <research_principles>
    <principle>Insightful Analysis and Insight Generation</principle>
    <principle>Long-term Perspective and Proactive Response</principle>
    <principle>Sensitivity and Adaptability to Change</principle>
    <principle>Value Creation and Inducing Practical Change</principle>
    <principle>Importance of Networking and Collaboration</principle>
    <principle>Proactive Researcher Role</principle>
    <principle>Practical and Specific Approach</principle>
  </research_principles>
  
  <task>
    <description>Describe about 15,000+ words covering industrial changes, issues, and response strategies related to the conference. Explicitly reflect and incorporate the research principles throughout your analysis and recommendations.</description>
    
    <format>
      <section>
        <name>Conference Overview</name>
        <word_count>about 4,000 words</word_count>
        <content>
          <item>Explain the overall context of the conference related to the question</item>
          <item>Introduce the main points or topics</item>
        </content>
      </section>
      
      <section>
        <name>Contents</name>
        <word_count>about 7,000 words</word_count>
        <content>
          <item>Analyze the key content discussed at the conference and reference.</item>
          <item>For each key session or topic:
            <subitem>Provide a detailed description of approximately 5 sentences.</subitem>
            <subitem>Include specific examples, data points, or case studies mentioned in the session.</subitem>
            <subitem>Show 2~3 file sources for each key content</subitem>
          </item>
        </content>
      </section>
      
      <section>
        <name>Conclusion</name>
        <word_count>about 4,000 words</word_count>
        <content>
          <item>Summarize new trends based on the conference content</item>
          <item>Present derived insights, emphasizing the 'Value Creation and Inducing Practical Change' principle</item>
          <item>Suggest future strategic directions, incorporating the 'Proactive Researcher Role' principle</item>
          <item>Propose concrete next steps that reflect the 'Practical and Specific Approach'</item>
          <item>Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3~4 sentences)</item>
        </content>
      </section>
    </format>
  </task>
  
  <constraints>
    <item>Use the provided context to answer the question</item>
    <item>If you don't know the answer, admit it honestly</item>
    <item>Answer in Korean and provide rich sentences to enhance the quality of the answer</item>
    <item>Adhere to the length constraints for each section</item>
    <item>Suggest appropriate data visualizations (e.g., charts, graphs) where relevant</item>
    <item>Explicitly mention and apply the research principles throughout the response</item>
  </constraints>
</system_prompt>
"""

# Parse XML to extract text content
def parse_xml_to_text(xml_string):
    root = ET.fromstring(xml_string)
    text_content = ET.tostring(root, encoding='unicode', method='text')
    return text_content.strip()

SYSTEM_PROMPT = parse_xml_to_text(SYSTEM_PROMPT_XML)

class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 8, filter: Dict[str, Any] = None, namespace: str = None
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
        self, embedding: List[float], k: int = 8, fetch_k: int = 30,
        lambda_mult: float = 0.7, filter: Dict[str, Any] = None, namespace: str = None
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

def generate_response_in_parts(llm, question, context):
    sections = ["[Conference Overview]", "[Contents]", "[Conclusion]"]
    full_response = ""
    
    for i, section in enumerate(sections):
        section_prompt = f"""
        Response so far: {full_response}
        
        Please write the next section: {section}
        """
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\nContext: {context}\n\n{section_prompt}")
        ]
        response = llm.invoke(messages)
        section_content = response.content
        
        full_response += section_content + "\n\n"
        progress = (i + 1) / len(sections)
        
        yield progress, full_response.strip()

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
    llm = ChatOpenAI(model=st.session_state.gpt_model, max_tokens=4096, temperature=0.56)
    
    # Set up Pinecone vector store
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

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
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            response_placeholder = st.empty()
            
            try:
                # Step 1: Query Processing
                status_placeholder.text("Processing query...")
                progress_bar.progress(0.1)
                time.sleep(0.5)
                
                # Step 2: Searching Database
                status_placeholder.text("Searching database...")
                progress_bar.progress(0.2)
                context = format_docs(retriever.invoke(question))
                time.sleep(0.5)
                
                # Step 3: Generating Answer
                status_placeholder.text("Generating answer...")
                for progress, partial_response in generate_response_in_parts(llm, question, context):
                    progress_bar.progress(0.2 + progress * 0.8)  # Start from 20% to 100%
                    response_placeholder.markdown(partial_response)
                    time.sleep(0.1)  # Short pause for better visualization
                
                # Step 4: Finalizing Response
                status_placeholder.text("Response complete")
                progress_bar.progress(1.0)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": partial_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clear status displays
                status_placeholder.empty()
                progress_bar.empty()
            
            # Display sources
            with st.expander("Sources"):
                for doc in retriever.invoke(question):
                    st.write(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    main()
