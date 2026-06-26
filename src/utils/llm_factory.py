"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class SafeChatOpenAI(ChatOpenAI):
    """
    Wrapper của ChatOpenAI để hỗ trợ tham số n > 1 bằng cách gửi nhiều yêu cầu tuần tự hoặc song song.
    Điều này giải quyết lỗi của Groq API khi không hỗ trợ tham số n > 1.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", None) or self.n or 1
        if n <= 1:
            orig_n = self.n
            self.n = 1
            try:
                return ChatOpenAI._generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            finally:
                self.n = orig_n
        
        generations = []
        res = None
        orig_n = self.n
        self.n = 1
        try:
            for _ in range(n):
                res = ChatOpenAI._generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
        finally:
            self.n = orig_n
        return ChatResult(generations=generations, llm_output=res.llm_output if res else None)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", None) or self.n or 1
        if n <= 1:
            orig_n = self.n
            self.n = 1
            try:
                return await ChatOpenAI._agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            finally:
                self.n = orig_n
        
        orig_n = self.n
        self.n = 1
        try:
            tasks = [
                ChatOpenAI._agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            self.n = orig_n
        generations = []
        for res in results:
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=results[0].llm_output if results else None)


def get_llm(provider: str = None, temperature: float = 0.0, model: str = None):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)
        model       : Tên model tùy chỉnh (chỉ áp dụng cho các provider tương thích)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    import os
    provider = (provider or config.PROVIDER).lower()

    # Kiểm tra API Key và tự động fallback về Ollama nếu thiếu API Key
    missing_key = False
    if provider == "openai" and not config.OPENAI_API_KEY:
        missing_key = True
    elif provider == "gemini" and not config.GOOGLE_API_KEY:
        missing_key = True
    elif provider == "anthropic" and not config.ANTHROPIC_API_KEY:
        missing_key = True
    elif provider == "openrouter" and not config.OPENROUTER_API_KEY:
        missing_key = True
    elif provider == "groq" and not os.getenv("GROQ_API_KEY"):
        missing_key = True

    if missing_key and provider != "ollama":
        print(f"⚠️  Không tìm thấy API Key cho provider '{provider}'. Tự động chuyển sang sử dụng Ollama (local LLM).")
        provider = "ollama"

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model or config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_model = model or config.GEMINI_MODEL
        if gemini_model:
            gemini_model_lower = gemini_model.lower()
            if "llama-3" in gemini_model_lower:
                gemini_model = "gemini-2.5-flash"
            elif "qwen" in gemini_model_lower:
                gemini_model = "gemini-1.5-flash-8b"
            elif not any(x in gemini_model_lower for x in ("gemini", "embedding")):
                gemini_model = config.GEMINI_MODEL

        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        local_model = model or config.OLLAMA_MODEL
        
        # Ánh xạ các model cloud lớn sang model local siêu nhẹ tương ứng
        if local_model:
            model_lower = local_model.lower()
            if "llama-3" in model_lower:
                local_model = config.OLLAMA_MODEL  # qwen2.5:0.5b
            elif "qwen-2.5-7b" in model_lower or "qwen/qwen-2.5-7b-instruct" in model_lower:
                # Dành cho A/B testing
                local_model = "smollm2:135m"
            elif "/" in model_lower:
                local_model = config.OLLAMA_MODEL

        return ChatOllama(
            model=local_model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=1000,
            max_retries=10,
        )

    elif provider == "groq":
        # Groq dùng OpenAI-compatible API hỗ trợ tham số n > 1
        groq_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if groq_model:
            groq_model_lower = groq_model.lower()
            if "llama-3" in groq_model_lower:
                groq_model = "llama-3.1-8b-instant"
            elif "qwen" in groq_model_lower:
                # Dùng llama-3.3-70b-versatile trên Groq cho A/B testing
                groq_model = "llama-3.3-70b-versatile"
            elif "/" in groq_model_lower:
                groq_model = "llama-3.1-8b-instant"

        return SafeChatOpenAI(
            model=groq_model,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_retries=10,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter, groq"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    # Kiểm tra API Key và tự động fallback sang local embeddings nếu thiếu key
    missing_key = False
    if provider == "openai" and not config.OPENAI_API_KEY:
        missing_key = True
    elif provider == "gemini" and not config.GOOGLE_API_KEY:
        missing_key = True
    elif provider == "anthropic" and not config.OPENAI_API_KEY:
        missing_key = True
    elif provider == "openrouter" and not config.OPENROUTER_API_KEY:
        missing_key = True

    if missing_key and provider != "ollama":
        print(f"⚠️  Không tìm thấy API Key cho embedding provider '{provider}'. Tự động dùng local HuggingFace embeddings.")
        provider = "local"

    if provider in ("openai", "openrouter", "local", "groq") or (provider == "ollama" and getattr(config, "OLLAMA_EMBEDDING_MODEL", "") == "local") or (provider == "gemini" and getattr(config, "GEMINI_EMBEDDING_MODEL", "") == "local"):
        # Sử dụng model embedding local nhẹ theo yêu cầu của user để tránh lỗi thanh toán/credit
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
