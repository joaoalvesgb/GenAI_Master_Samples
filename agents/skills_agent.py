"""
=============================================================================
SKILLS AGENT - Agente com Skills (Habilidades) via Azure OpenAI
=============================================================================

Este módulo implementa um agente que utiliza SKILLS em vez de apenas Tools,
demonstrando um padrão avançado de arquitetura de agentes de IA.

CONCEITO: Skills vs Tools
=========================

    ┌──────────────────────────────────────────────────────────────────┐
    │                        AGENTE (SkillsAgent)                     │
    │                                                                  │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │                    SKILLS (Alto Nível)                   │    │
    │  │                                                         │    │
    │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐│    │
    │  │  │ Research     │ │ Summarize    │ │ Content Creation ││    │
    │  │  │ Skill        │ │ Skill        │ │ Skill            ││    │
    │  │  │              │ │              │ │                  ││    │
    │  │  │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────────┐ ││    │
    │  │  │ │web_search│ │ │ │calculator│ │ │ │datetime_tool │ ││    │
    │  │  │ │wikipedia │ │ │ │  regex   │ │ │ │  templates   │ ││    │
    │  │  │ └──────────┘ │ │ └──────────┘ │ │ └──────────────┘ ││    │
    │  │  └──────────────┘ └──────────────┘ └──────────────────┘│    │
    │  │                     TOOLS (Baixo Nível)                 │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │                                                                  │
    │  LLM: Azure OpenAI (GPT-4o)                                    │
    │  Framework: LangGraph (ReAct Agent)                             │
    └──────────────────────────────────────────────────────────────────┘

O agente "pensa" em termos de SKILLS:
- "Preciso pesquisar? → Research Skill"
- "Preciso resumir? → Summarize Skill"
- "Preciso criar conteúdo? → Content Creation Skill"

Cada Skill internamente compõe múltiplas Tools para realizar a tarefa.

BOAS PRÁTICAS DEMONSTRADAS:
1. Separação de Responsabilidades: Skills isolam lógica complexa
2. Composição sobre Herança: Skills compõem tools, não herdam delas
3. Single Responsibility: Cada skill faz UMA coisa bem feita
4. Open/Closed Principle: Novas skills podem ser adicionadas sem modificar o agente
5. Dependency Inversion: O agente depende de abstrações (BaseSkill), não de implementações

Configuração necessária (variáveis de ambiente):
- AZURE_OPENAI_API_KEY: Chave de acesso ao recurso Azure OpenAI
- AZURE_OPENAI_ENDPOINT: URL do recurso (ex: https://meu-recurso.openai.azure.com/)
- AZURE_OPENAI_API_VERSION: Versão da API (padrão: 2024-08-01-preview)

=============================================================================
"""

import os
from typing import Optional, List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory

# Importa tools auxiliares (usadas diretamente pelo agente)
from tools import calculator_tool, get_current_datetime
from tools import rag_search_tool, set_vector_store

# ── Importa as SKILLS (o diferencial deste agente!) ──
from skills import (
    research_skill_tool,
    summarize_skill_tool,
    content_creation_skill_tool,
)


class SkillsAgent(BaseAgent):
    """
    Agente com Skills (Habilidades) via Azure OpenAI.

    Este agente demonstra o padrão de SKILLS, onde capacidades de alto nível
    são encapsuladas em módulos reutilizáveis que compõem múltiplas tools.

    CONCEITO DIDÁTICO: Pirâmide de Abstração
    ------------------------------------------
        Nível 3 (Alto):   AGENTE     → Orquestra tudo
        Nível 2 (Médio):  SKILLS     → Capacidades compostas
        Nível 1 (Baixo):  TOOLS      → Ações atômicas
        Nível 0 (Base):   APIs/Libs  → Serviços externos

    O agente opera no nível mais alto, delegando trabalho complexo
    para Skills, que por sua vez usam Tools para executar ações.

    Skills disponíveis:
    - 🔍 Research Skill: Pesquisa aprofundada (web + Wikipedia)
    - 📋 Summarize Skill: Análise e resumo inteligente de textos
    - ✉️ Content Creation Skill: Criação de e-mails, relatórios e posts

    Tools auxiliares:
    - 🧮 Calculator: Cálculos matemáticos
    - 🕐 DateTime: Data e hora atual

    Example:
        >>> agent = SkillsAgent()
        >>> # Usa a Research Skill automaticamente
        >>> response = agent.process_message("Pesquise sobre energia solar")
        >>> # Usa a Content Creation Skill
        >>> response = agent.process_message("Escreva um e-mail sobre a reunião de amanhã")
        >>> # Usa a Summarize Skill
        >>> response = agent.process_message("Resuma: A IA está transformando...")
    """

    # ─── Skills e Tools padrão do agente ───
    # Note a separação conceitual: Skills (capacidades compostas) + Tools (ações atômicas)
    DEFAULT_SKILLS = [
        research_skill_tool,         # Skill: Pesquisa aprofundada
        summarize_skill_tool,        # Skill: Resumo inteligente
        content_creation_skill_tool, # Skill: Criação de conteúdo
    ]

    AUXILIARY_TOOLS = [
        calculator_tool,       # Tool: Cálculos matemáticos
        get_current_datetime,  # Tool: Data/hora atual
    ]

    def __init__(
        self,
        name: str = "Skills Assistant",
        description: str = "Assistente com habilidades avançadas (Skills) via Azure OpenAI",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        vector_store_manager=None,
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "skills_agent"
    ):
        """
        Inicializa o Agente com Skills via Azure OpenAI.

        Args:
            name: Nome do agente (exibido no chat)
            description: Descrição do agente
            model: Nome do deployment no Azure (ex: gpt-4o, gpt-4)
            temperature: Criatividade (0.0 = preciso, 2.0 = criativo)
            max_tokens: Limite de tokens na resposta (None = sem limite)
            top_p: Nucleus sampling
            presence_penalty: Penaliza repetição de tópicos (-2.0 a 2.0)
            frequency_penalty: Penaliza repetição de palavras (-2.0 a 2.0)
            system_prompt: Prompt de sistema customizado
            api_key: API key do Azure OpenAI (ou usa env AZURE_OPENAI_API_KEY)
            azure_endpoint: Endpoint do Azure (ou usa env AZURE_OPENAI_ENDPOINT)
            api_version: Versão da API (ou usa env AZURE_OPENAI_API_VERSION)
            tools: Lista customizada de tools/skills (se None, usa padrão)
            vector_store_manager: Gerenciador de vector store para RAG
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            memory_max_messages: Máximo de mensagens no curto prazo
            memory_storage_path: Caminho para salvar memória de longo prazo
            memory_session_id: ID da sessão de memória

        Raises:
            ValueError: Se API key ou endpoint não forem encontrados

        Note:
            O agente combina Skills + Tools auxiliares automaticamente.
            Para adicionar novas Skills, use o método add_skill().
        """
        super().__init__(name, description)

        # ─── Validar credenciais Azure ───
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ API Key do Azure OpenAI não encontrada!\n"
                "Configure AZURE_OPENAI_API_KEY ou passe api_key.\n"
                "Obtenha no Azure Portal: https://portal.azure.com"
            )

        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError(
                "❌ Endpoint do Azure OpenAI não encontrado!\n"
                "Configure AZURE_OPENAI_ENDPOINT ou passe azure_endpoint.\n"
                "Formato: https://seu-recurso.openai.azure.com/"
            )

        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        )

        # ─── Inicializar o LLM (Azure OpenAI) ───
        self.llm = AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        # ─── Configurar Skills + Tools ───
        # CONCEITO: Combinamos Skills (alto nível) com Tools (baixo nível)
        # O agente pode usar ambos conforme a necessidade
        if tools is not None:
            self.tools = list(tools)
        else:
            self.tools = list(self.DEFAULT_SKILLS) + list(self.AUXILIARY_TOOLS)

        # ─── Configurar RAG se fornecido ───
        self.vector_store_manager = vector_store_manager
        self._setup_rag()

        # ─── Configurar Memória ───
        self.memory_type = memory_type
        self.memory = self._setup_memory(
            memory_type=memory_type,
            max_messages=memory_max_messages,
            storage_path=memory_storage_path,
            session_id=memory_session_id
        )

        # ─── System Prompt ───
        self.system_prompt = system_prompt or self._get_skills_system_prompt()

        # ─── Criar agente ReAct com LangGraph ───
        self._create_agent()

    # =====================================================================
    # CONFIGURAÇÃO INTERNA
    # =====================================================================

    def _setup_memory(
        self,
        memory_type: str,
        max_messages: int,
        storage_path: str,
        session_id: str
    ):
        """
        Configura o sistema de memória.

        Tipos disponíveis:
        - none: Sem memória persistente
        - short_term: Últimas N mensagens (janela deslizante)
        - long_term: Persiste em disco (fatos importantes)
        - combined: Curto + Longo prazo
        """
        if memory_type == "none":
            return None
        elif memory_type == "short_term":
            return ShortTermMemory(max_messages=max_messages)
        elif memory_type == "long_term":
            return LongTermMemory(
                storage_path=storage_path,
                session_id=session_id
            )
        elif memory_type == "combined":
            return CombinedMemory(
                max_short_term_messages=max_messages,
                storage_path=storage_path,
                session_id=session_id
            )
        return None

    def _setup_rag(self):
        """Configura o RAG (Retrieval Augmented Generation) se disponível."""
        if self.vector_store_manager is not None:
            set_vector_store(self.vector_store_manager)
            if rag_search_tool not in self.tools:
                self.tools.append(rag_search_tool)

    def _create_agent(self):
        """Cria o agente ReAct com LangGraph."""
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )

    def _get_skills_system_prompt(self) -> str:
        """
        Retorna o system prompt especializado para o Skills Agent.

        CONCEITO: System Prompt para Skills
        ------------------------------------
        O prompt é otimizado para que o LLM entenda a diferença entre
        Skills e Tools e saiba quando usar cada uma.

        Estrutura do prompt:
        1. Identidade e papel do agente
        2. Lista de SKILLS disponíveis (com quando usar)
        3. Lista de TOOLS auxiliares
        4. Instruções de uso
        5. Tom de comunicação
        """
        base_prompt = f"""Você é o {self.name}, {self.description}.

## 🧠 CONCEITO: SKILLS (Habilidades)

Você possui SKILLS (habilidades avançadas) que vão além de simples ferramentas.
Cada Skill encapsula uma capacidade completa com múltiplas etapas.

## 🎯 SUAS SKILLS

### 1. 🔍 Research Skill (research_skill)
**Quando usar:** Quando o usuário pedir pesquisa, investigação ou informações detalhadas.
- Combina busca web + Wikipedia automaticamente
- Gera relatório estruturado com fontes
- Suporta profundidades: rápida, normal, profunda
- **Exemplos:** "Pesquise sobre...", "Quero saber mais sobre...", "Investigue..."

### 2. 📋 Summarize Skill (summarize_skill)
**Quando usar:** Quando o usuário pedir resumo, análise ou síntese de texto.
- Analisa métricas do texto (palavras, complexidade)
- Extrai pontos-chave automaticamente
- Estilos: executivo, pontos_chave, analise_critica
- **Exemplos:** "Resuma...", "Quais os pontos principais...", "Analise este texto..."

### 3. ✉️ Content Creation Skill (content_creation_skill)
**Quando usar:** Quando o usuário pedir para criar/redigir conteúdo formatado.
- Cria e-mails profissionais com templates
- Gera relatórios estruturados
- Cria posts para LinkedIn e Twitter/X
- **Exemplos:** "Escreva um e-mail...", "Crie um relatório...", "Faça um post..."

## 🛠️ TOOLS AUXILIARES

Além das Skills, você tem ferramentas simples:
- **calculator**: Cálculos matemáticos
- **get_current_datetime**: Data e hora atual
"""

        # Adiciona instruções de RAG se habilitado
        if self.vector_store_manager is not None:
            base_prompt += """
- **knowledge_base_search**: Busca na base de conhecimento do usuário

IMPORTANTE SOBRE A BASE DE CONHECIMENTO:
- Use knowledge_base_search para perguntas sobre documentos carregados
- Cite a fonte quando usar informações da base
"""

        base_prompt += """
## 📋 INSTRUÇÕES

1. **Prefira Skills** para tarefas complexas (pesquisa, resumo, criação)
2. **Use Tools** para tarefas simples (cálculos, data/hora)
3. **Combine** Skills e Tools quando necessário
4. **Responda em português brasileiro**
5. **Seja didático** - explique o que está fazendo quando relevante
6. **Use formatação markdown** para melhor legibilidade

## 🗣️ TOM DE COMUNICAÇÃO

- Profissional mas acessível
- Didático quando explicar conceitos
- Use emojis com moderação para melhorar legibilidade
- Sempre em português brasileiro
"""
        return base_prompt

    # =====================================================================
    # PROCESSAMENTO DE MENSAGENS
    # =====================================================================

    def process_message(self, message: str) -> str:
        """
        Processa uma mensagem do usuário e retorna uma resposta.

        Fluxo de processamento:
        1. Monta o contexto (system prompt + memória + histórico)
        2. Adiciona a mensagem do usuário
        3. Invoca o agente ReAct (que decide qual Skill/Tool usar)
        4. Extrai e formata a resposta
        5. Atualiza a memória

        O agente ReAct (Reasoning + Acting) do LangGraph:
        - PENSA sobre qual Skill/Tool usar
        - AGE executando a Skill/Tool escolhida
        - OBSERVA o resultado
        - Repete se necessário
        - RESPONDE ao usuário

        Args:
            message: Mensagem/pergunta do usuário

        Returns:
            Resposta do agente (pode incluir resultados de Skills)
        """
        try:
            # ─── Monta contexto ───
            messages = [SystemMessage(content=self.system_prompt)]

            # Adiciona contexto da memória de longo prazo
            if self.memory_type in ["long_term", "combined"]:
                long_term_context = self._get_long_term_context()
                if long_term_context:
                    messages.append(SystemMessage(content=long_term_context))

            # Adiciona histórico de curto prazo
            if self.memory_type == "combined" and self.memory:
                messages.extend(self.memory.get_short_term_messages())
            elif self.memory_type == "short_term" and self.memory:
                messages.extend(self.memory.messages)
            else:
                messages.extend(self.chat_history)

            # Adiciona mensagem atual
            messages.append(HumanMessage(content=message))

            # ─── Invoca o agente ReAct ───
            result = self.agent.invoke({"messages": messages})

            # ─── Extrai resposta ───
            response_messages = result.get("messages", [])

            if response_messages:
                last_message = response_messages[-1]
                response = self._extract_text_from_content(last_message.content)
            else:
                response = "❌ Erro ao processar sua solicitação."

            # ─── Atualiza memória ───
            self._update_memory(message, response)

            return response

        except Exception as e:
            error_msg = str(e)
            if "AZURE_OPENAI" in error_msg:
                return (
                    "❌ Erro de configuração do Azure OpenAI.\n\n"
                    "Verifique:\n"
                    "1. `AZURE_OPENAI_API_KEY` está configurada\n"
                    "2. `AZURE_OPENAI_ENDPOINT` está correto\n"
                    "3. O deployment do modelo existe no seu recurso Azure\n\n"
                    "Acesse: https://portal.azure.com"
                )
            return f"❌ Erro: {error_msg}"

    def _extract_text_from_content(self, content) -> str:
        """
        Extrai texto do conteúdo da resposta.

        Normaliza o conteúdo que pode vir em diferentes formatos:
        - String: retorna diretamente
        - Lista de blocos: extrai texto de cada bloco
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)

        return str(content)

    # =====================================================================
    # GERENCIAMENTO DE MEMÓRIA
    # =====================================================================

    def _get_long_term_context(self) -> str:
        """Retorna o contexto da memória de longo prazo."""
        if self.memory is None:
            return ""

        if self.memory_type == "long_term":
            return self.memory.get_memories_as_text(limit=5)
        elif self.memory_type == "combined":
            return self.memory.long_term.get_memories_as_text(limit=5)

        return ""

    def _update_memory(self, user_message: str, ai_response: str) -> None:
        """Atualiza a memória com a nova interação."""
        self.add_to_history(user_message, ai_response)

        if self.memory is None:
            return

        if self.memory_type == "short_term":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)
        elif self.memory_type == "combined":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)

    def save_to_long_term(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 5
    ) -> None:
        """
        Salva uma informação na memória de longo prazo.

        Args:
            content: Conteúdo a salvar
            memory_type: Tipo (fact, preference, summary)
            importance: Importância de 1 a 10
        """
        if self.memory is None:
            return

        if self.memory_type == "long_term":
            self.memory.add_memory(content, memory_type, importance)
        elif self.memory_type == "combined":
            self.memory.add_to_long_term(content, memory_type, importance)

    def get_memory_info(self) -> Dict[str, Any]:
        """Retorna informações sobre a memória atual."""
        info = {
            "type": self.memory_type,
            "enabled": self.memory is not None
        }

        if self.memory_type == "short_term" and self.memory:
            info["messages_count"] = len(self.memory.messages)
            info["max_messages"] = self.memory.max_messages
        elif self.memory_type == "long_term" and self.memory:
            info["memories_count"] = len(self.memory.memories)
        elif self.memory_type == "combined" and self.memory:
            info["short_term_messages"] = len(self.memory.short_term.messages)
            info["long_term_memories"] = len(self.memory.long_term.memories)

        return info

    # =====================================================================
    # GERENCIAMENTO DE SKILLS E TOOLS
    # =====================================================================

    def add_skill(self, skill_tool: BaseTool) -> None:
        """
        Adiciona uma nova Skill ao agente.

        CONCEITO: Open/Closed Principle
        --------------------------------
        O agente está ABERTO para extensão (adicionar skills)
        mas FECHADO para modificação (não precisa alterar código existente).

        Args:
            skill_tool: A skill como LangChain BaseTool

        Example:
            >>> from skills import minha_nova_skill_tool
            >>> agent.add_skill(minha_nova_skill_tool)
        """
        self.tools.append(skill_tool)
        self._create_agent()  # Recria o agente com a nova skill

    def add_tool(self, tool: BaseTool) -> None:
        """Adiciona uma tool auxiliar ao agente."""
        self.tools.append(tool)
        self._create_agent()

    def list_skills(self) -> List[str]:
        """
        Lista as Skills disponíveis (diferenciando de Tools simples).

        Returns:
            Lista de nomes das skills
        """
        skill_names = [t.name for t in self.tools if "skill" in t.name.lower()]
        return skill_names

    def list_tools(self) -> List[str]:
        """
        Lista todas as tools e skills disponíveis.

        Returns:
            Lista de nomes de todas as tools (incluindo skills)
        """
        return [t.name for t in self.tools]

    def set_vector_store(self, manager):
        """
        Define o vector store manager para RAG.

        Args:
            manager: Instância de VectorStoreManager
        """
        self.vector_store_manager = manager
        self._setup_rag()
        if "knowledge_base_search" not in self.system_prompt:
            self.system_prompt = self._get_skills_system_prompt()
        self._create_agent()

    def has_rag(self) -> bool:
        """Retorna True se o RAG está habilitado."""
        return self.vector_store_manager is not None

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Retorna informações completas sobre o agente.

        Útil para debug e para exibir na interface.
        """
        return {
            "name": self.name,
            "description": self.description,
            "provider": "Azure OpenAI",
            "skills": self.list_skills(),
            "tools": [t for t in self.list_tools() if "skill" not in t.lower()],
            "memory": self.get_memory_info(),
            "has_rag": self.has_rag()
        }

