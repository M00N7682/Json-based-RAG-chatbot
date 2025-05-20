# main.py
import os, json, argparse
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 환경변수 확인
google_api_key = os.environ.get("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is missing")

# 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 가맹본부 데이터를 기반으로 질문에 정확하게 답변하는 AI입니다. 아래 문서를 참고하여 구체적으로 답변해주세요. 모를 경우 '모르겠습니다'라고 답변하세요.

[문서 정보]
{context}

[질문]
{question}

[답변]
"""
)

# 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
args = parser.parse_args()

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

# Knowledge base 로딩 및 청크 분할
documents = []
kb_path = Path("data/train")

for file in kb_path.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    entries = kb_data if isinstance(kb_data, list) else [kb_data]

    for entry in entries:
        ql = entry.get("QL", {})
        context = ql.get("EXTRACTED_SUMMARY_TEXT") or ql.get("extracted_summary_text")
        if context:
            chunks = text_splitter.split_text(context)
            for chunk in chunks:
                documents.append(Document(page_content=chunk))

# 벡터 DB 임베딩
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
vectordb = FAISS.from_documents(documents, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM 구성
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0,
    google_api_key=google_api_key
)

# QA 체인 구성
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# 테스트 질문 로딩 및 추론 수행
with open(args.data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

qas = test_data if isinstance(test_data, list) else test_data.get("QAs", [])

results = []
for item in qas:
    question = item.get("question")
    original_text = item.get("original_text", "")

    if not question:
        results.append({
            "original_text": original_text,
            "question": "",
            "answer": "오류 발생: 질문 항목이 존재하지 않습니다."
        })
        continue

    try:
        result = qa_chain({"query": question})
        results.append({
            "original_text": original_text,
            "question": question,
            "answer": result["result"]
        })
    except Exception as e:
        results.append({
            "original_text": original_text,
            "question": question,
            "answer": "오류 발생: " + str(e)
        })

# 결과 저장
with open("/app/test_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)