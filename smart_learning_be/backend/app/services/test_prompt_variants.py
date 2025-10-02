from .chain import QA_PROMPT
from .llm import get_llm
from .hybrid import hybrid_retrieve
from .reranker import rerank
from langchain.chains import LLMChain

# --- Load LLM & reranker ---
llm = get_llm()
reranker = rerank

# --- Prompt (subject = General) ---
prompt = QA_PROMPT.partial(subject="General")
llm_chain = LLMChain(llm=llm, prompt=prompt)

def answer_question(query: str):
    # 1. Hybrid retrieval
    candidates = hybrid_retrieve(query, k_dense=12, k_sparse=12, top_after_rrf=30)
    if not candidates:
        return f"(Không tìm thấy ngữ cảnh cho câu hỏi: {query})"

    # 2. Rerank top 3
    reranked_texts = reranker.rerank(query, [d.page_content for d in candidates], top_k=3)
    reranked_docs = [d for d in candidates if d.page_content in reranked_texts]

    # 3. Build context
    context = "\n\n".join([d.page_content for d in reranked_docs])

    # 4. LLM generate answer
    result = llm_chain.invoke({"query": query, "context": context})
    return result["text"]

# --- Test ---
tests = [
    {"q": "What is the Vietnam War?", "label": "Lịch sử"},
    {"q": "Giải phương trình x² - 5x + 6 = 0", "label": "Toán"},
    {"q": "What is recursion in programming?", "label": "Lập trình"},
    {"q": "When was Isaac Newton born?", "label": "Ngoài context"},
]

for t in tests:
    print(f"\n=== {t['label']} ===")
    print(answer_question(t["q"]))
