"""
=============================================================================
VECTOR STORE - Armazenamento Vetorial para RAG
=============================================================================

O que são Embeddings?
- São representações numéricas (vetores) de texto
- Textos com significado similar têm vetores próximos
- Permite busca por SIMILARIDADE SEMÂNTICA

Exemplo:
- "cachorro" e "cão" terão vetores muito próximos
- "cachorro" e "avião" terão vetores distantes

Vector Store (Banco de Vetores):
- Armazena os embeddings dos documentos
- Permite busca rápida por similaridade
- Exemplos: FAISS, Chroma, Pinecone, Weaviate

Neste exemplo usamos FAISS (gratuito e local)

Provedores de Embeddings suportados:
- OpenAI: text-embedding-ada-002, text-embedding-3-small/large
- Azure OpenAI: Mesmos modelos, hospedados na Azure
- Gemini: models/embedding-001, models/text-embedding-004
- Ollama: nomic-embed-text, mxbai-embed-large (local e gratuito)
- HuggingFace: sentence-transformers (local e gratuito, fallback)

Configuração via variáveis de ambiente (.env):
- EMBEDDING_PROVIDER: Provedor a usar (openai, azure, gemini, ollama, huggingface)
- EMBEDDING_MODEL: Modelo específico do provedor escolhido
- AZURE_OPENAI_EMBEDDING_ENDPOINT: URL específica para embeddings Azure (opcional)

Prioridade de configuração:
1. Parâmetro no código (provider=, model=)
2. Variável de ambiente (EMBEDDING_PROVIDER, EMBEDDING_MODEL)
3. Detecção automática (pelas API keys configuradas)

=============================================================================
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Provedores de embeddings suportados
SUPPORTED_PROVIDERS = ["openai", "azure", "gemini", "ollama", "huggingface"]


class VectorStoreManager:
    """
    Gerenciador de Vector Store para RAG.

    Esta classe encapsula a lógica de:
    - Criar embeddings dos documentos
    - Armazenar em um vector store
    - Buscar documentos relevantes

    Provedores suportados:
    - OpenAI (text-embedding-ada-002, text-embedding-3-small/large)
    - Azure OpenAI (mesmos modelos, hospedados na Azure)
    - Google Gemini (models/embedding-001, models/text-embedding-004)
    - Ollama (nomic-embed-text, mxbai-embed-large - local e gratuito)
    - HuggingFace (sentence-transformers - local e gratuito, fallback)

    Attributes:
        vector_store: O banco de vetores (FAISS)
        embeddings: O modelo de embeddings usado
        provider: O provedor de embeddings selecionado
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Inicializa o gerenciador.

        A configuração de embeddings segue a seguinte prioridade:
        1. Parâmetro direto (embeddings, provider, model)
        2. Variáveis de ambiente (EMBEDDING_PROVIDER, EMBEDDING_MODEL)
        3. Detecção automática (pelas API keys configuradas)

        Args:
            embeddings: Modelo de embeddings a usar (instância pronta).
                       Se None, detecta automaticamente pelo provider ou variáveis de ambiente.
            provider: Provedor de embeddings a usar.
                     Opções: "openai", "azure", "gemini", "ollama", "huggingface".
                     Se None, tenta EMBEDDING_PROVIDER do .env, depois detecta automaticamente.
            model: Nome/ID do modelo de embeddings.
                  Se None, tenta EMBEDDING_MODEL do .env, depois usa o padrão de cada provedor.

        Variáveis de ambiente suportadas:
            EMBEDDING_PROVIDER: Provedor de embeddings (ex: "ollama", "openai", "gemini")
            EMBEDDING_MODEL: Modelo de embeddings (ex: "nomic-embed-text", "text-embedding-3-small")

        Examples:
            >>> # Detecção automática (usa variáveis de ambiente)
            >>> manager = VectorStoreManager()

            >>> # Provedor explícito via código
            >>> manager = VectorStoreManager(provider="ollama", model="nomic-embed-text")

            >>> # Via .env (sem parâmetros no código):
            >>> # EMBEDDING_PROVIDER=gemini
            >>> # EMBEDDING_MODEL=models/text-embedding-004
            >>> manager = VectorStoreManager()

            >>> # Instância pronta
            >>> from langchain_openai import OpenAIEmbeddings
            >>> manager = VectorStoreManager(embeddings=OpenAIEmbeddings())
        """
        self.vector_store = None
        self.embeddings = embeddings
        self.provider = provider

        # Se não foi passado um modelo, tenta criar um
        if self.embeddings is None:
            self._initialize_embeddings(provider=provider, model=model)

    def _initialize_embeddings(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Inicializa o modelo de embeddings.

        Prioridade de configuração:
        1. Parâmetros diretos (provider, model) — passados no código
        2. Variáveis de ambiente (EMBEDDING_PROVIDER, EMBEDDING_MODEL)
        3. Detecção automática pelas API keys configuradas:
           a. OpenAI (OPENAI_API_KEY)
           b. Azure OpenAI (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)
           c. Google Gemini (GOOGLE_API_KEY)
           d. Ollama (sempre disponível localmente)
           e. HuggingFace / sentence-transformers (fallback local)

        Args:
            provider: Provedor explícito (opcional)
            model: Modelo específico (opcional)
        """
        # Prioridade: parâmetro > variável de ambiente
        resolved_provider = provider or os.getenv("EMBEDDING_PROVIDER")
        resolved_model = model or os.getenv("EMBEDDING_MODEL")

        # Log das variáveis de ambiente detectadas (útil para debug)
        env_provider = os.getenv("EMBEDDING_PROVIDER")
        env_model = os.getenv("EMBEDDING_MODEL")
        if env_provider or env_model:
            print(f"🔧 Variáveis de ambiente detectadas: "
                  f"EMBEDDING_PROVIDER={env_provider or '(não definido)'}, "
                  f"EMBEDDING_MODEL={env_model or '(não definido)'}")

        # Se um provider foi definido (via parâmetro ou env var), usa ele
        if resolved_provider:
            resolved_provider = resolved_provider.lower().strip()
            if resolved_provider not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Provedor '{resolved_provider}' não suportado. "
                    f"Opções: {', '.join(SUPPORTED_PROVIDERS)}"
                )
            initializer = {
                "openai": self._init_openai_embeddings,
                "azure": self._init_azure_embeddings,
                "gemini": self._init_gemini_embeddings,
                "ollama": self._init_ollama_embeddings,
                "huggingface": self._init_huggingface_embeddings,
            }
            initializer[resolved_provider](model=resolved_model)
            return

        # Detecção automática por variáveis de ambiente (API keys)
        # 1. OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                self._init_openai_embeddings(model=resolved_model)
                return
            except (ImportError, Exception) as e:
                print(f"⚠️ OpenAI Embeddings indisponível: {e}")

        # 2. Azure OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            try:
                self._init_azure_embeddings(model=resolved_model)
                return
            except (ImportError, Exception) as e:
                print(f"⚠️ Azure OpenAI Embeddings indisponível: {e}")

        # 3. Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self._init_gemini_embeddings(model=resolved_model)
                return
            except (ImportError, Exception) as e:
                print(f"⚠️ Gemini Embeddings indisponível: {e}")

        # 4. Ollama (local, sem API key)
        try:
            self._init_ollama_embeddings(model=resolved_model)
            return
        except (ImportError, Exception) as e:
            print(f"⚠️ Ollama Embeddings indisponível: {e}")

        # 5. HuggingFace (fallback local)
        try:
            self._init_huggingface_embeddings(model=model)
            return
        except (ImportError, Exception) as e:
            print(f"⚠️ HuggingFace Embeddings indisponível: {e}")

        raise ValueError(
            "Nenhum modelo de embeddings disponível.\n"
            "Opções:\n"
            "  - Configure no .env: EMBEDDING_PROVIDER=openai e EMBEDDING_MODEL=text-embedding-3-small\n"
            "  - Defina OPENAI_API_KEY para usar OpenAI\n"
            "  - Defina AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT para usar Azure\n"
            "  - Defina GOOGLE_API_KEY para usar Gemini\n"
            "  - Instale e inicie o Ollama (https://ollama.ai)\n"
            "  - Instale sentence-transformers para usar HuggingFace local"
        )

    # =========================================================================
    # INICIALIZADORES POR PROVEDOR
    # =========================================================================

    def _init_openai_embeddings(self, model: Optional[str] = None):
        """
        Inicializa embeddings da OpenAI.

        Modelos disponíveis:
        - text-embedding-3-small (mais barato, boa qualidade)
        - text-embedding-3-large (melhor qualidade)
        - text-embedding-ada-002 (modelo legado)

        Args:
            model: Nome do modelo. Padrão: text-embedding-3-small
        """
        from langchain_openai import OpenAIEmbeddings

        self.embeddings = OpenAIEmbeddings(
            model=model or "text-embedding-3-small"
        )
        self.provider = "openai"
        print(f"✅ Usando OpenAI Embeddings (modelo: {model or 'text-embedding-3-small'})")

    def _init_azure_embeddings(self, model: Optional[str] = None):
        """
        Inicializa embeddings do Azure OpenAI.

        Requer variáveis de ambiente:
        - AZURE_OPENAI_API_KEY: Chave de acesso
        - AZURE_OPENAI_ENDPOINT: URL do recurso (fallback)
        - AZURE_OPENAI_EMBEDDING_ENDPOINT: URL específica para embeddings (opcional, prioritário)
        - AZURE_OPENAI_API_VERSION: Versão da API (opcional)

        O parâmetro 'model' aqui corresponde ao nome do DEPLOYMENT no Azure.

        Args:
            model: Nome do deployment no Azure. Padrão: text-embedding-3-small
        """
        from langchain_openai import AzureOpenAIEmbeddings

        # Usa endpoint específico para embeddings, com fallback para o endpoint principal
        azure_embedding_endpoint = (
            os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=model or "text-embedding-3-small",
            azure_endpoint=azure_embedding_endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        self.provider = "azure"
        print(f"✅ Usando Azure OpenAI Embeddings (deployment: {model or 'text-embedding-3-small'}, endpoint: {azure_embedding_endpoint})")

    def _init_gemini_embeddings(self, model: Optional[str] = None):
        """
        Inicializa embeddings do Google Gemini.

        Modelos disponíveis:
        - models/embedding-001 (padrão)
        - models/text-embedding-004 (mais recente)

        Requer variável de ambiente:
        - GOOGLE_API_KEY: Chave da API do Google AI

        Args:
            model: Nome do modelo. Padrão: models/text-embedding-004
        """
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model or "models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.provider = "gemini"
        print(f"✅ Usando Gemini Embeddings (modelo: {model or 'models/text-embedding-004'})")

    def _init_ollama_embeddings(self, model: Optional[str] = None):
        """
        Inicializa embeddings do Ollama (local e gratuito).

        Modelos populares (baixe com `ollama pull <modelo>`):
        - nomic-embed-text (768 dims, bom equilíbrio)
        - mxbai-embed-large (1024 dims, melhor qualidade)
        - all-minilm (384 dims, leve e rápido)
        - snowflake-arctic-embed (1024 dims)

        Requer:
        - Ollama instalado e rodando (https://ollama.ai)
        - Modelo de embedding baixado: `ollama pull nomic-embed-text`

        Args:
            model: Nome do modelo. Padrão: nomic-embed-text
        """
        from langchain_ollama import OllamaEmbeddings

        embedding_model = model or "nomic-embed-text"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )
        self.provider = "ollama"
        print(f"✅ Usando Ollama Embeddings (modelo: {embedding_model}, url: {base_url})")

    def _init_huggingface_embeddings(self, model: Optional[str] = None):
        """
        Inicializa embeddings do HuggingFace (local e gratuito).

        Usa a biblioteca sentence-transformers.
        Modelos populares:
        - sentence-transformers/all-MiniLM-L6-v2 (384 dims, rápido)
        - sentence-transformers/all-mpnet-base-v2 (768 dims, melhor qualidade)
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilíngue)

        Args:
            model: Nome do modelo. Padrão: sentence-transformers/all-MiniLM-L6-v2
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedding_model = model or "sentence-transformers/all-MiniLM-L6-v2"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        self.provider = "huggingface"
        print(f"✅ Usando HuggingFace Embeddings (modelo: {embedding_model})")

    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Cria o vector store a partir de documentos.

        Este é o processo de INDEXAÇÃO:
        1. Valida e sanitiza cada documento
        2. Cada documento é convertido em embedding
        3. Os embeddings são armazenados no FAISS

        Args:
            documents: Lista de documentos a indexar

        Example:
            >>> manager = VectorStoreManager()
            >>> manager.create_from_documents(my_docs)
        """
        try:
            from langchain_community.vectorstores import FAISS

            # Sanitiza documentos: garante que page_content é sempre uma string válida
            sanitized = []
            skipped = 0
            for doc in documents:
                content = doc.page_content

                # Converte para string se não for
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""

                # Remove caracteres nulos que podem causar problemas
                content = content.replace("\x00", "")

                # Ignora documentos vazios
                if not content.strip():
                    skipped += 1
                    continue

                doc.page_content = content
                sanitized.append(doc)

            if skipped > 0:
                print(f"⚠️ {skipped} documento(s) vazios foram ignorados")

            if not sanitized:
                raise ValueError(
                    "❌ Nenhum documento com conteúdo válido para indexar.\n"
                    "Verifique se os arquivos contêm texto extraível."
                )

            print(f"📊 Indexando {len(sanitized)} documentos...")

            # Cria o vector store
            # Isso pode demorar dependendo da quantidade de documentos
            self.vector_store = FAISS.from_documents(
                documents=sanitized,
                embedding=self.embeddings
            )

            print(f"✅ Vector store criado com sucesso!")

        except ImportError:
            raise ImportError(
                "FAISS não está instalado. "
                "Execute: pip install faiss-cpu"
            )

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Busca documentos similares a uma query.

        Este é o processo de RETRIEVAL (recuperação):
        1. A query é convertida em embedding
        2. Buscamos os k embeddings mais próximos
        3. Retornamos os documentos correspondentes

        Args:
            query: Texto de busca
            k: Número de resultados a retornar

        Returns:
            Lista dos k documentos mais relevantes

        Example:
            >>> results = manager.similarity_search("O que é Python?")
            >>> for doc in results:
            ...     print(doc.page_content[:100])
        """
        if self.vector_store is None:
            raise ValueError("Vector store não foi inicializado. Use create_from_documents primeiro.")

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Busca documentos com score de similaridade.

        Útil para entender o quão relevante cada resultado é.
        Score menor = mais similar.

        Args:
            query: Texto de busca
            k: Número de resultados

        Returns:
            Lista de tuplas (Document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store não inicializado.")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def save(self, path: str) -> None:
        """
        Salva o vector store em disco.

        Útil para não precisar reindexar toda vez.

        Args:
            path: Caminho do diretório para salvar
        """
        if self.vector_store is None:
            raise ValueError("Vector store não inicializado.")

        self.vector_store.save_local(path)
        print(f"💾 Vector store salvo em: {path}")

    def load(self, path: str) -> None:
        """
        Carrega um vector store salvo.

        Args:
            path: Caminho do diretório
        """
        from langchain_community.vectorstores import FAISS

        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 Vector store carregado de: {path}")


def create_simple_knowledge_base(
    texts: List[str],
    metadatas: List[dict] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> VectorStoreManager:
    """
    Cria uma base de conhecimento simples a partir de textos.

    Função de conveniência para criar rapidamente uma KB.

    Args:
        texts: Lista de textos a indexar
        metadatas: Metadados opcionais para cada texto
        provider: Provedor de embeddings (opcional).
                 Opções: "openai", "azure", "gemini", "ollama", "huggingface".
                 Se None, detecta automaticamente.
        model: Modelo de embeddings específico (opcional).

    Returns:
        VectorStoreManager configurado

    Example:
        >>> # Detecção automática
        >>> kb = create_simple_knowledge_base(["texto1", "texto2"])

        >>> # Com provedor explícito
        >>> kb = create_simple_knowledge_base(["texto1"], provider="ollama")
        >>> kb = create_simple_knowledge_base(["texto1"], provider="gemini")
    """
    # Cria documentos a partir dos textos
    if metadatas is None:
        metadatas = [{"source": f"text_{i}"} for i in range(len(texts))]

    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(texts, metadatas)
    ]

    # Cria e retorna o manager
    manager = VectorStoreManager(provider=provider, model=model)
    manager.create_from_documents(documents)

    return manager


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Exemplo: criar uma base de conhecimento simples

    sample_texts = [
        """
        Python é uma linguagem de programação de alto nível, interpretada 
        e de propósito geral. Foi criada por Guido van Rossum e lançada 
        em 1991. Python enfatiza legibilidade de código.
        """,
        """
        LangChain é um framework para desenvolvimento de aplicações 
        alimentadas por modelos de linguagem. Permite criar agentes, 
        chains, e integrar com diversas fontes de dados.
        """,
        """
        RAG (Retrieval Augmented Generation) é uma técnica que combina 
        busca de informação com geração de texto. Permite que LLMs 
        respondam perguntas usando documentos específicos como contexto.
        """,
        """
        FAISS (Facebook AI Similarity Search) é uma biblioteca para 
        busca eficiente de vetores similares. É muito usada em 
        aplicações de RAG para armazenar embeddings.
        """
    ]

    print("🚀 Criando base de conhecimento de exemplo...")
    print(f"📋 Provedores suportados: {', '.join(SUPPORTED_PROVIDERS)}")
    print()

    try:
        # Detecção automática — usa o melhor provedor disponível
        kb = create_simple_knowledge_base(sample_texts)
        print(f"🎯 Provedor selecionado: {kb.provider}")

        # Teste de busca
        query = "O que é RAG?"
        print(f"\n🔍 Buscando: '{query}'")

        results = kb.similarity_search(query, k=2)

        for i, doc in enumerate(results, 1):
            print(f"\n📄 Resultado {i}:")
            print(doc.page_content.strip()[:200] + "...")

    except Exception as e:
        print(f"❌ Erro: {e}")
        print(
            "💡 Opções para resolver:\n"
            "  - Defina OPENAI_API_KEY para usar OpenAI\n"
            "  - Defina AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT para usar Azure\n"
            "  - Defina GOOGLE_API_KEY para usar Gemini\n"
            "  - Inicie o Ollama: ollama serve && ollama pull nomic-embed-text\n"
            "  - Instale sentence-transformers: pip install sentence-transformers"
        )

