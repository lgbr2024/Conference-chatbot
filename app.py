!pip install python-dotenv langchain-pinecone langchain-openai

# 셀 1: 필요한 라이브러리 임포트 및 환경 설정
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

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

## 셀 2: ModifiedPineconeVectorStore 클래스 정의
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

# 셀 3: maximal_marginal_relevance 함수 정의
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[List[float]],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    if not embedding_list:
        return []
    
    # 이미 숫자 리스트이므로 변환 불필요
    embedding_list = np.array(embedding_list, dtype=np.float32)
    
    if embedding_list.size == 0:
        return []
    
    similarity_to_query = cosine_similarity(query_embedding.reshape(1, -1), embedding_list)[0]
    
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        for i, emb in enumerate(embedding_list):
            if i in idxs:
                continue
            
            similarity_to_query = cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
            similarities_to_selected = cosine_similarity(emb.reshape(1, -1), selected)[0]
            mmr_score = lambda_mult * similarity_to_query - (1 - lambda_mult) * np.max(similarities_to_selected)
            
            if mmr_score > best_score:
                best_score = mmr_score
                idx_to_add = i
        
        if idx_to_add == -1:
            break
        
        idxs.append(idx_to_add)
        selected = np.vstack((selected, embedding_list[idx_to_add]))
    
    return idxs

# 셀 4: Pinecone 초기화 및 벡터 스토어 설정
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "conference"
index = pc.Index(index_name)

# GPT 모델 선택
option = input("GPT 모델을 선택해주세요 (gpt-4o 또는 gpt-4o-mini): ")
llm = ChatOpenAI(model=option)

# Pinecone 벡터 스토어 설정
vectorstore = ModifiedPineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    text_key="source"
)

# retriever 설정
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.75}
)

# 셀 6: 프롬프트 템플릿, 체인 설정 및 ask_question 함수 정의

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

def ask_question(question):
    response = chain.invoke(question)
    answer = response['answer']
    source_documents = response['docs'][:10]  # 최대 10개까지만 가져옵니다
    
    print("답변:", answer)
    print("\n참고 문서:")
    for i, doc in enumerate(source_documents, 1):
        print(f"{i}. 출처: {doc.metadata.get('source', 'Unknown')}")
        print(f"   내용: {doc.page_content[:100]}...")  # 내용의 일부만 출력
    
    return answer, source_documents

# 메인 루프
print("conference에 대해 질문해보세요. 종료하려면 'quit'을 입력하세요.")

while True:
    question = input("\n질문을 입력하세요: ")
    if question.lower() == 'quit':
        print("프로그램을 종료합니다.")
        break
    ask_question(question)

IAA 컨퍼런스 내용 중 BMW의 내용은?


