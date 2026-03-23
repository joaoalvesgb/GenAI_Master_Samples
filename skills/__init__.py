"""
=============================================================================
SKILLS - Módulo de Habilidades (Skills) para Agentes de IA
=============================================================================

O que é uma Skill?
------------------
Uma Skill (habilidade) é uma abstração de NÍVEL SUPERIOR às Tools.

Enquanto uma **Tool** realiza UMA única ação (ex: buscar na web, calcular),
uma **Skill** encapsula uma CAPACIDADE COMPLETA que pode:
- Combinar múltiplas tools internamente
- Ter sua própria lógica de raciocínio multi-etapa
- Possuir prompt engineering especializado
- Entregar um resultado mais elaborado e contextualizado

Analogia do Mundo Real:
-----------------------
    Tool  = Martelo, Chave de Fenda, Serra  (ferramentas individuais)
    Skill = Carpintaria, Encanamento, Pintura (habilidades que usam ferramentas)

    Um carpinteiro (agente) usa a SKILL de "construir móveis",
    que internamente usa múltiplas tools: serra, martelo, lixa, etc.

Diferença entre Tool e Skill:
-----------------------------
    ┌─────────────┬───────────────────────────┬──────────────────────────────────┐
    │             │ Tool                      │ Skill                            │
    ├─────────────┼───────────────────────────┼──────────────────────────────────┤
    │ Escopo      │ Ação única e atômica      │ Capacidade completa e composta   │
    │ Complexidade│ Simples (1 função)        │ Complexa (multi-etapa)           │
    │ Composição  │ Independente              │ Pode usar múltiplas Tools        │
    │ Contexto    │ Sem estado                │ Pode ter contexto/memória        │
    │ Exemplo     │ web_search("clima SP")    │ research("impacto climático SP") │
    └─────────────┴───────────────────────────┴──────────────────────────────────┘

Skills disponíveis:
- ResearchSkill: Pesquisa aprofundada (web + Wikipedia + síntese)
- SummarizeSkill: Análise e resumo estruturado de textos
- ContentCreationSkill: Criação de conteúdo formatado (e-mails, relatórios)

Exemplo de uso:
    from skills import research_skill_tool, summarize_skill_tool, content_creation_skill_tool

    # Cada skill é exportada como uma LangChain Tool
    # para ser usada diretamente por um agente ReAct
    agent = create_react_agent(model=llm, tools=[research_skill_tool, ...])

=============================================================================
"""

from .base_skill import BaseSkill
from .research_skill import research_skill_tool, ResearchSkillInput
from .summarize_skill import summarize_skill_tool, SummarizeSkillInput
from .content_skill import content_creation_skill_tool, ContentCreationInput

__all__ = [
    "BaseSkill",
    "research_skill_tool",
    "ResearchSkillInput",
    "summarize_skill_tool",
    "SummarizeSkillInput",
    "content_creation_skill_tool",
    "ContentCreationInput",
]

