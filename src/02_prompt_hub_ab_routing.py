"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Đổi thành tên duy nhất của bạn để tránh xung đột trên Hub công cộng nếu cần
PROMPT_V1_NAME = "student-2a202600819-rag-prompt-v1"
PROMPT_V2_NAME = "student-2a202600819-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# SYSTEM_V1 — phong cách ngắn gọn, trả lời 2-4 câu
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời câu hỏi.\n"
    "Giữ câu trả lời thật ngắn gọn, súc tích và trực tiếp vào vấn đề (khoảng 2-4 câu).\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# SYSTEM_V2 — phong cách chuyên gia, expert tone, 3-5 câu, có cấu trúc rõ ràng
SYSTEM_V2 = (
    "Bạn là chuyên gia AI giàu kinh nghiệm. Đọc kỹ context dưới đây và trích xuất các facts liên quan.\n"
    "Viết câu trả lời rõ ràng, có cấu trúc mạch lạc, chặt chẽ (khoảng 3-5 câu).\n"
    "Trình bày chuyên nghiệp và giải thích logic nếu cần.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    # Push PROMPT_V1
    try:
        # client.push_prompt returns None or string URL
        client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn, 2-4 câu")
        print(f"✅ Đã push V1 thành công lên Hub: {PROMPT_V1_NAME}")
    except Exception as e:
        print(f"⚠️  V1 push lỗi: {e}")

    # Push PROMPT_V2
    try:
        client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc, chuyên gia, 3-5 câu")
        print(f"✅ Đã push V2 thành công lên Hub: {PROMPT_V2_NAME}")
    except Exception as e:
        print(f"⚠️  V2 push lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    # Pull PROMPT_V1_NAME
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}' (Lỗi: {e})")

    # Pull PROMPT_V2_NAME
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}' (Lỗi: {e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


import time

# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    """
    # Retrieve docs từ retriever
    docs = retriever.invoke(question)

    # Ghép page_content thành 1 string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Chạy chain và lấy answer
    chain = prompt | llm | StrOutputParser()
    
    max_retries = 5
    base_delay = 5
    for attempt in range(max_retries):
        try:
            answer = chain.invoke({"context": context, "question": question})
            return {"question": question, "answer": answer, "version": version}
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"\n  ⚠️  Lỗi xảy ra (thường là 429), thử lại sau {delay}s... (Lần thử {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
            else:
                raise e


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Tạo LangSmith Client với API key từ config
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # Pull cả 2 prompts từ Hub (dùng dict trả về)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và 2 LLMs cho A/B test
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm_v1      = get_llm(model="meta-llama/llama-3-8b-instruct")
    llm_v2      = get_llm(model="qwen/qwen-2.5-7b-instruct")

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        # Lấy version key từ request_id qua get_prompt_version()
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        model_name  = "llama-3-8b" if version_tag == "v1" else "qwen-2.5-7b"
        llm         = llm_v1 if version_tag == "v1" else llm_v2
        
        # Dùng chung prompt V1 để đánh giá khách quan hiệu năng của 2 model
        prompt      = prompts[PROMPT_V1_NAME]

        # Gọi ask_ab() với đúng arguments
        result = ask_ab(retriever, llm, prompt, question, f"{version_tag}-{model_name}")

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [model-{model_name}] {question[:55]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
