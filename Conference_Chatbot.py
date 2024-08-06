import os
from typing import List
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
import pinecone

# API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone"]["api_key"]

# Streamlit UI 설정
st.header("Chat with the Conference 2022-2024 💬 📚")
option = st.selectbox("GPT 모델을 선택해주세요.", ("gpt-4", "gpt-3.5-turbo"))
llm = ChatOpenAI(model=option)

# Pinecone 초기화
pinecone.init(api_key=st.secrets["pinecone"]["api_key"], environment=st.secrets["pinecone"]["environment"])

# Pinecone 인덱스 설정
index_name = "gtc2024"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(index_name, embeddings)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.75}
)

# 프롬프트 템플릿 설정
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
    """Convert Documents to a single string."""
    formatted = [
        f"Article Title: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

# RunnableLambda for formatting docs
format_docs_lambda = RunnableLambda(lambda docs: format_docs(docs))

# Answer generation chain
answer = prompt | llm | StrOutputParser()

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format_docs_lambda)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

# 채팅 인터페이스 설정
if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Conference에서 공개된 내용에 대해 질문해보세요!"}
    ]

if prompt_message := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt_message})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": prompt_message})
            answer = response['answer']
            source_documents = response['docs']
            st.markdown(answer)

            with st.expander("참고 문서 확인"):
                for i, doc in enumerate(source_documents[:3], 1):
                    st.markdown(f"{i}. {doc.metadata['source']}", help=doc.page_content)
            message = {"role": "assistant", "content": response['answer']}
            st.session_state.messages.append(message)
