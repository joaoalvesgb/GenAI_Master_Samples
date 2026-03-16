"""
=============================================================================
MCP AGENT - Agente com Model Context Protocol (Implementação Real)
=============================================================================

Este módulo implementa um agente que se conecta REALMENTE a servidores MCP
(Model Context Protocol) para usar ferramentas externas.

O QUE É MCP?
------------
MCP (Model Context Protocol) é um protocolo aberto desenvolvido pela Anthropic
que permite que agentes de IA se conectem a servidores externos que fornecem:
- Tools (ferramentas)
- Resources (recursos/dados)
- Prompts (templates de prompts)

REQUISITOS:
-----------
pip install mcp langchain-mcp-adapters

Além disso, para usar servidores MCP via npx, você precisa de Node.js instalado.

SERVIDORES MCP DISPONÍVEIS:
---------------------------
- fetch: Busca conteúdo de URLs (não requer API key)
- filesystem: Acesso a arquivos locais (não requer API key)
- memory: Memória persistente (não requer API key)
- brave-search: Busca na web (requer BRAVE_API_KEY)
- github: Acesso a repositórios (requer GITHUB_TOKEN)

=============================================================================
"""

import os
import asyncio
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory

# Verifica disponibilidade das bibliotecas MCP
MCP_AVAILABLE = False
MCP_IMPORT_ERROR = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_IMPORT_ERROR = str(e)


# =============================================================================
# CONFIGURAÇÃO DOS SERVIDORES MCP
# =============================================================================

MCP_SERVERS = {
    "fetch": {
        "name": "Fetch",
        "description": "Busca e extrai conteúdo de URLs da web",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "env_required": [],
        "example_queries": [
            "Busque o conteúdo de https://python.org",
            "Extraia informações de https://langchain.com"
        ]
    },
    "filesystem": {
        "name": "Filesystem",
        "description": "Lê e escreve arquivos no sistema local",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "env_required": [],
        "example_queries": [
            "Liste os arquivos do diretório atual",
            "Leia o conteúdo do arquivo README.md"
        ]
    },
    "memory": {
        "name": "Memory",
        "description": "Armazena e recupera informações na memória",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env_required": [],
        "example_queries": [
            "Lembre que meu nome é João",
            "O que você lembra sobre mim?"
        ]
    },
    "brave_search": {
        "name": "Brave Search",
        "description": "Busca na web usando Brave Search API",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_required": ["BRAVE_API_KEY"],
        "env_url": "https://brave.com/search/api/",
        "example_queries": [
            "Pesquise as últimas notícias sobre IA",
            "Busque informações sobre Python 3.12"
        ]
    },
    "github": {
        "name": "GitHub",
        "description": "Acesso a repositórios e dados do GitHub",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_required": ["GITHUB_TOKEN"],
        "env_url": "https://github.com/settings/tokens",
        "example_queries": [
            "Liste os repositórios do usuário langchain-ai",
            "Busque issues abertas no repositório langchain"
        ]
    },
    "sqlite": {
        "name": "SQLite",
        "description": "Consultas em banco de dados SQLite",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./data.db"],
        "env_required": [],
        "example_queries": [
            "Liste as tabelas do banco de dados",
            "Execute SELECT * FROM users LIMIT 10"
        ]
    },
    "time": {
        "name": "Time",
        "description": "Informações de data, hora e fuso horário",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-time"],
        "env_required": [],
        "example_queries": [
            "Que horas são agora?",
            "Qual a data atual em Tokyo?"
        ]
    }
}


class MCPAgent(BaseAgent):
    """
    Agente que se conecta a servidores MCP reais para usar ferramentas externas.

    Este agente:
    1. Inicia um servidor MCP como subprocesso
    2. Carrega as ferramentas disponíveis no servidor
    3. Usa essas ferramentas via LangGraph ReAct agent
    4. Encerra o servidor ao finalizar

    Example:
        >>> agent = MCPAgent(
        ...     provider="openai",
        ...     mcp_server_name="fetch"  # Servidor que busca URLs
        ... )
        >>> response = agent.process_message("Busque o conteúdo de https://python.org")
        >>> print(response)
    """

    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4"],
        "google": ["gemini-2.0-flash", "gemini-2.5-flash-preview", "gemini-3-flash-preview"]
    }

    def __init__(
        self,
        name: str = "MCP Agent",
        description: str = "Agente conectado a servidor MCP externo",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        # Configuração MCP
        mcp_server_name: str = "fetch",
        mcp_server_command: Optional[str] = None,
        mcp_server_args: Optional[List[str]] = None,
        mcp_env: Optional[Dict[str, str]] = None,
        # Memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "mcp_agent"
    ):
        """
        Inicializa o Agente MCP.

        Args:
            name: Nome do agente
            description: Descrição do agente
            provider: "openai" ou "google"
            model: Modelo do LLM
            temperature: Criatividade (0.0 a 2.0)
            max_tokens: Máximo de tokens na resposta
            top_p: Nucleus sampling
            presence_penalty: Penalidade de presença (OpenAI)
            frequency_penalty: Penalidade de frequência (OpenAI)
            top_k: Top-K sampling (Google)
            api_key: API key do provider
            system_prompt: Prompt de sistema customizado
            mcp_server_name: Nome do servidor MCP pré-configurado
            mcp_server_command: Comando customizado (sobrescreve mcp_server_name)
            mcp_server_args: Args customizados
            mcp_env: Variáveis de ambiente extras
            memory_type: Tipo de memória
            memory_max_messages: Máximo de mensagens
            memory_storage_path: Caminho para memória
            memory_session_id: ID da sessão
        """
        super().__init__(name, description)

        # Verifica se MCP está disponível
        if not MCP_AVAILABLE:
            raise ImportError(
                f"❌ Bibliotecas MCP não instaladas!\n"
                f"Erro: {MCP_IMPORT_ERROR}\n\n"
                f"Execute:\n"
                f"  pip install mcp langchain-mcp-adapters\n\n"
                f"Você também precisa de Node.js instalado para usar servidores npx."
            )

        self.provider = provider.lower()
        self.model = model

        # Validar provider
        if self.provider not in ["openai", "google"]:
            raise ValueError(f"❌ Provider '{provider}' não suportado!")

        # Modelo padrão
        if self.model is None:
            self.model = "gpt-4o-mini" if self.provider == "openai" else "gemini-2.0-flash"

        # API Key
        self.api_key = self._get_api_key(api_key)

        # Criar LLM
        self.llm = self._create_llm(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_k=top_k
        )

        # Configuração do servidor MCP
        self.mcp_server_name = mcp_server_name
        self.mcp_env = mcp_env or {}

        # Usa configuração customizada ou pré-configurada
        if mcp_server_command:
            self.mcp_command = mcp_server_command
            self.mcp_args = mcp_server_args or []
        else:
            self._configure_mcp_server(mcp_server_name)

        # System prompt
        self.system_prompt = system_prompt or self._get_mcp_system_prompt()

        # Memória
        self.memory_type = memory_type
        self.memory = self._setup_memory(
            memory_type=memory_type,
            max_messages=memory_max_messages,
            storage_path=memory_storage_path,
            session_id=memory_session_id
        )

        # Tools serão carregadas dinamicamente
        self.tools = []

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """Obtém a API key do LLM provider."""
        if api_key:
            return api_key

        env_var = "OPENAI_API_KEY" if self.provider == "openai" else "GOOGLE_API_KEY"
        key = os.getenv(env_var)

        if not key:
            raise ValueError(f"❌ {env_var} não configurada!")

        return key

    def _create_llm(
        self,
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        top_k: Optional[int]
    ):
        """Cria a instância do LLM."""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                api_key=self.api_key
            )
        else:
            kwargs = {
                "model": self.model,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
                "google_api_key": self.api_key
            }
            if top_k is not None:
                kwargs["top_k"] = top_k
            return ChatGoogleGenerativeAI(**kwargs)

    def _configure_mcp_server(self, server_name: str):
        """Configura o servidor MCP baseado no nome."""
        if server_name not in MCP_SERVERS:
            available = list(MCP_SERVERS.keys())
            raise ValueError(
                f"❌ Servidor MCP '{server_name}' não encontrado!\n"
                f"Servidores disponíveis: {available}"
            )

        config = MCP_SERVERS[server_name]
        self.mcp_command = config["command"]
        self.mcp_args = config["args"]

        # Verifica variáveis de ambiente necessárias
        for env_var in config.get("env_required", []):
            if env_var not in self.mcp_env and not os.getenv(env_var):
                env_url = config.get("env_url", "")
                raise ValueError(
                    f"❌ Servidor '{server_name}' requer {env_var}!\n"
                    f"Obtenha em: {env_url}\n"
                    f"Configure via: export {env_var}='sua_key'"
                )

    def _get_mcp_system_prompt(self) -> str:
        """Retorna o system prompt especializado para MCP."""
        server_config = MCP_SERVERS.get(self.mcp_server_name, {})
        server_name = server_config.get("name", self.mcp_server_name)
        server_desc = server_config.get("description", "Servidor MCP")
        examples = server_config.get("example_queries", [])

        examples_text = ""
        if examples:
            examples_text = "\n## 💡 Exemplos de uso:\n" + "\n".join(f"- {ex}" for ex in examples)

        return f"""Você é o {self.name}, um assistente conectado ao servidor MCP "{server_name}".

        ## 🔌 SERVIDOR MCP: {server_name}
        {server_desc}
        
        ## 📋 INSTRUÇÕES
        
        1. **Use as ferramentas MCP** disponíveis para responder perguntas
        2. As ferramentas são fornecidas pelo servidor MCP conectado
        3. Seja claro sobre o que você está fazendo
        4. Se houver erro, explique o problema
        5. Sempre responda em português brasileiro
        {examples_text}
        
        ## ⚠️ IMPORTANTE
        
        - As ferramentas vêm do servidor MCP externo
        - A disponibilidade depende do servidor estar funcionando
        - Alguns servidores podem ter rate limits
        """

    def _setup_memory(
        self,
        memory_type: str,
        max_messages: int,
        storage_path: str,
        session_id: str
    ):
        """Configura a memória."""
        if memory_type == "none":
            return None
        elif memory_type == "short_term":
            return ShortTermMemory(max_messages=max_messages)
        elif memory_type == "long_term":
            return LongTermMemory(storage_path=storage_path, session_id=session_id)
        elif memory_type == "combined":
            return CombinedMemory(
                max_short_term_messages=max_messages,
                storage_path=storage_path,
                session_id=session_id
            )
        return None

    def _extract_text_from_content(self, content) -> str:
        """Extrai texto do conteúdo da resposta."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        text_parts.append(text)
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)
        return str(content)

    async def _process_with_mcp(self, message: str) -> str:
        """
        Processa uma mensagem conectando ao servidor MCP.

        Este método:
        1. Inicia o servidor MCP como subprocesso
        2. Carrega as ferramentas disponíveis
        3. Cria um agente ReAct com essas ferramentas
        4. Processa a mensagem
        5. Retorna a resposta
        """
        # Prepara variáveis de ambiente
        env = {**os.environ, **self.mcp_env}

        # Parâmetros do servidor MCP
        server_params = StdioServerParameters(
            command=self.mcp_command,
            args=self.mcp_args,
            env=env
        )

        # Conecta ao servidor MCP
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Inicializa a sessão MCP
                await session.initialize()

                # Carrega as ferramentas do servidor MCP
                tools = await load_mcp_tools(session)
                self.tools = tools  # Salva para referência

                # Cria o agente ReAct com as ferramentas MCP
                agent = create_react_agent(
                    model=self.llm,
                    tools=tools
                )

                # Prepara as mensagens
                messages = [SystemMessage(content=self.system_prompt)]

                # Adiciona histórico
                if self.memory_type == "combined" and self.memory:
                    messages.extend(self.memory.get_short_term_messages())
                elif self.memory_type == "short_term" and self.memory:
                    messages.extend(self.memory.messages)
                else:
                    messages.extend(self.chat_history)

                # Adiciona a mensagem atual
                messages.append(HumanMessage(content=message))

                # Invoca o agente
                result = await agent.ainvoke({"messages": messages})

                # Extrai a resposta
                response_messages = result.get("messages", [])
                if response_messages:
                    last_message = response_messages[-1]
                    return self._extract_text_from_content(last_message.content)

                return "❌ Não foi possível processar a mensagem."

    def process_message(self, message: str) -> str:
        """
        Processa uma mensagem do usuário usando o servidor MCP.

        Args:
            message: Mensagem do usuário

        Returns:
            Resposta do agente
        """
        try:
            # Executa a função assíncrona
            response = asyncio.run(self._process_with_mcp(message))

            # Atualiza memória
            self._update_memory(message, response)

            return response

        except FileNotFoundError as e:
            return (
                f"❌ Servidor MCP não encontrado!\n\n"
                f"Verifique se Node.js está instalado:\n"
                f"  node --version\n\n"
                f"Se não estiver, instale em: https://nodejs.org/\n\n"
                f"Erro: {str(e)}"
            )
        except Exception as e:
            error_str = str(e)

            if "ENOENT" in error_str or "not found" in error_str.lower():
                return (
                    f"❌ Não foi possível iniciar o servidor MCP.\n\n"
                    f"Verifique:\n"
                    f"1. Node.js está instalado (node --version)\n"
                    f"2. npx está disponível (npx --version)\n"
                    f"3. Conexão com internet está ativa\n\n"
                    f"Erro: {error_str}"
                )

            return f"❌ Erro: {error_str}"

    def _update_memory(self, user_message: str, ai_response: str) -> None:
        """Atualiza a memória com a interação."""
        self.add_to_history(user_message, ai_response)

        if self.memory is None:
            return

        if self.memory_type == "short_term":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)
        elif self.memory_type == "combined":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)

    def list_tools(self) -> List[str]:
        """Retorna lista de tools carregadas do MCP."""
        if self.tools:
            return [tool.name for tool in self.tools]
        return [f"(conecte ao servidor '{self.mcp_server_name}' para ver as tools)"]

    def has_rag(self) -> bool:
        """Retorna False."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo."""
        return {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "mcp_server": self.mcp_server_name,
            "mcp_command": self.mcp_command,
            "specialization": "mcp"
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """Retorna informações da memória."""
        info = {"type": self.memory_type, "enabled": self.memory is not None}

        if self.memory_type == "short_term" and self.memory:
            info["messages_count"] = len(self.memory.messages)
            info["max_messages"] = self.memory.max_messages
        elif self.memory_type == "long_term" and self.memory:
            info["memories_count"] = len(self.memory.memories)
        elif self.memory_type == "combined" and self.memory:
            info["short_term_messages"] = len(self.memory.short_term.messages)
            info["long_term_memories"] = len(self.memory.long_term.memories)

        return info

    @classmethod
    def get_available_servers(cls) -> Dict[str, Dict]:
        """Retorna os servidores MCP disponíveis."""
        return MCP_SERVERS

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """Retorna modelos disponíveis para um provider."""
        return cls.AVAILABLE_MODELS.get(provider.lower(), [])


# =============================================================================
# CLASSE DEMO (PARA USO SEM DEPENDÊNCIAS MCP)
# =============================================================================

class MCPAgentDemo(BaseAgent):
    """
    Versão demonstrativa do MCPAgent que funciona sem as dependências MCP.

    Útil para:
    - Testar a interface sem instalar dependências
    - Demonstrar o conceito de MCP
    - Ambiente onde Node.js não está disponível
    """

    MCP_SERVERS = MCP_SERVERS  # Usa a mesma configuração

    def __init__(
        self,
        name: str = "MCP Agent (Demo)",
        description: str = "Demonstração do conceito MCP",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        mcp_server_name: str = "fetch",
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, description)

        self.provider = provider.lower()
        self.model = model or ("gpt-4o-mini" if provider == "openai" else "gemini-2.0-flash")
        self.mcp_server_name = mcp_server_name
        self.temperature = temperature

        # API Key
        env_var = "OPENAI_API_KEY" if self.provider == "openai" else "GOOGLE_API_KEY"
        self.api_key = os.getenv(env_var)

        if not self.api_key:
            raise ValueError(f"❌ {env_var} não configurada!")

        # Cria o LLM
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=temperature,
                api_key=self.api_key
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=temperature,
                google_api_key=self.api_key
            )

        # System prompt: usa o customizado se fornecido, senão o padrão demo
        self.system_prompt = system_prompt or self._get_demo_system_prompt()

    def _get_demo_system_prompt(self) -> str:
        """System prompt para o modo demo."""
        server_info = MCP_SERVERS.get(self.mcp_server_name, {})

        return f"""Você é um assistente demonstrando o conceito de MCP (Model Context Protocol).

        ## 🔌 O QUE É MCP?
        
        MCP (Model Context Protocol) é um protocolo aberto da Anthropic que permite
        agentes de IA se conectarem a servidores externos para usar ferramentas.
        
        ## 📡 SERVIDOR SIMULADO: {server_info.get('name', self.mcp_server_name)}
        **Descrição:** {server_info.get('description', 'Servidor MCP')}
        
        ## 🎯 MODO DEMONSTRAÇÃO
        
        Como esta é uma versão demo (sem conexão real ao MCP), você deve:
        1. Explicar o que o servidor MCP faria
        2. Simular o tipo de resposta esperada
        3. Mencionar que para funcionar de verdade, precisa instalar as dependências
        
        ## 📦 PARA USAR O MCP REAL:
        
        ```bash
        pip install mcp langchain-mcp-adapters
        ```
        
        Também é necessário ter Node.js instalado.
        
        ## 💡 SERVIDORES MCP DISPONÍVEIS
        
        - **fetch**: Busca conteúdo de URLs
        - **filesystem**: Acesso a arquivos locais
        - **memory**: Memória persistente
        - **brave_search**: Busca na web (requer API key)
        - **github**: Acesso a repositórios (requer token)
        - **time**: Informações de data/hora
        
        Responda sempre em português brasileiro.
        """

    def process_message(self, message: str) -> str:
        """Processa mensagem no modo demo."""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=message)
            ]
            messages.extend(self.chat_history)

            response = self.llm.invoke(messages)
            result = response.content

            self.add_to_history(message, result)
            return result

        except Exception as e:
            return f"❌ Erro: {str(e)}"

    def list_tools(self) -> List[str]:
        """Lista tools simuladas."""
        return [f"mcp_{self.mcp_server_name} (modo demo)"]

    def has_rag(self) -> bool:
        return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mcp_server": self.mcp_server_name,
            "mode": "demo"
        }


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def check_mcp_requirements() -> Dict[str, Any]:
    """
    Verifica se os requisitos para MCP estão instalados.

    Returns:
        Dicionário com status de cada requisito
    """
    import shutil

    status = {
        "mcp_library": MCP_AVAILABLE,
        "mcp_error": MCP_IMPORT_ERROR,
        "node_installed": shutil.which("node") is not None,
        "npx_installed": shutil.which("npx") is not None,
    }

    status["all_ready"] = (
        status["mcp_library"] and
        status["node_installed"] and
        status["npx_installed"]
    )

    return status


def get_mcp_server_info(server_name: str) -> Optional[Dict]:
    """Retorna informações de um servidor MCP específico."""
    return MCP_SERVERS.get(server_name)


def list_mcp_servers() -> List[str]:
    """Lista os servidores MCP disponíveis."""
    return list(MCP_SERVERS.keys())
