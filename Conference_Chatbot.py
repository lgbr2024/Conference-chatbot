import os
import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import ModifiedPineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from typing import List
from langchain_core.documents import Document
from operator import itemgetter

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
    def strategic_consultant(question: str, context: str) -> str:
        '''
        Analyze conference trends and provide strategic insights.
        
        Args:
            question: The user's question about the conference.
            context: Retrieved information related to the question.
        
        Returns:
            A structured answer following the format below.
        '''
        # Role definition
        role = "Strategic consultant for a large corporation"
        
        # Answer structure
        structure = {
            "introduction": {
                "weight": 0.5,
                "content": [
                    "Overall context of the conference",
                    "Main points or topics"
                ]
            },
            "main_body": {
                "weight": 0.3,
                "content": [
                    "Analysis of key conference discussions",
                    "Relevant data or case studies"
                ]
            },
            "conclusion": {
                "weight": 0.2,
                "content": [
                    "Summary of new trends",
                    "Derived insights",
                    "Suggested future strategic directions"
                ]
            }
        }
        
        # Generate answer
        answer = f"As a {role}, here's my analysis based on the conference information:\n\n"
        
        for section, details in structure.items():
            answer += f"{section.capitalize()}:\n"
            for point in details['content']:
                answer += f"- {point}\n"
            answer += "\n"
        
        return answer

    # User's question: {question}
    # Context: {context}
    
    # Execute the function and provide the analysis:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)
    
    format = itemgetter("docs") | RunnableParallel(format_docs)
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
    if question := st.chat_input("컨퍼런스에 대해 질문해주세요:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            response = chain.invoke(question)
            answer = response['answer']
            source_documents = response['docs'][:5]  # Get up to 5 documents
            st.markdown(answer)
            
            with st.expander("참조 문서"):
                for i, doc in enumerate(source_documents, 1):
                    st.write(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
    
        # Add Plex.tv link
        st.markdown("---")
        st.markdown("[관련 컨퍼런스 영상 보기 (Plex.tv)](https://app.plex.tv)")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
