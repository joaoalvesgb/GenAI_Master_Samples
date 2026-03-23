"""
=============================================================================
RESEARCH SKILL - Skill de Pesquisa Aprofundada
=============================================================================

Esta Skill demonstra o conceito de COMPOSIÇÃO DE TOOLS.

O que esta Skill faz:
---------------------
Realiza uma pesquisa aprofundada sobre um tópico, combinando:
1. Busca na Web (via DuckDuckGo/Tavily) → informações atuais
2. Busca na Wikipedia → conhecimento enciclopédico
3. Síntese → combina todas as fontes em um resultado coerente

Por que é uma Skill e não uma Tool?
------------------------------------
Uma Tool faria APENAS a busca (ex: web_search retorna links).
Esta Skill vai além:
- Faz MÚLTIPLAS buscas em fontes diferentes
- SINTETIZA os resultados em um relatório coerente
- FORMATA o resultado com seções e referências
- Tem LÓGICA de fallback (se uma fonte falha, usa outra)

Fluxo de Execução:
    ┌──────────┐     ┌────────────┐     ┌──────────────┐
    │ Input:   │────→│ Etapa 1:   │────→│ Etapa 2:     │
    │ Tópico   │     │ Web Search │     │ Wikipedia    │
    └──────────┘     └────────────┘     └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │ Etapa 3:     │
                                        │ Síntese      │
                                        └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │ Output:      │
                                        │ Relatório    │
                                        └──────────────┘

=============================================================================
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

from .base_skill import BaseSkill


# =============================================================================
# SCHEMA DE ENTRADA (Pydantic)
# =============================================================================

class ResearchSkillInput(BaseModel):
    """
    Schema de entrada para a Skill de Pesquisa.

    O Pydantic valida os dados automaticamente e as descrições
    dos Fields ajudam o LLM a entender como preencher os parâmetros.
    """
    topic: str = Field(
        description="O tópico a ser pesquisado. Exemplo: 'inteligência artificial', 'energia solar', 'história do Brasil'"
    )
    language: str = Field(
        default="pt",
        description="Idioma para busca na Wikipedia (pt, en, es). Padrão: pt"
    )
    depth: str = Field(
        default="normal",
        description="Profundidade da pesquisa: 'rapida' (só web), 'normal' (web + wiki), 'profunda' (web + wiki + detalhes)"
    )


# =============================================================================
# IMPLEMENTAÇÃO DA SKILL
# =============================================================================

class ResearchSkill(BaseSkill):
    """
    Skill de Pesquisa Aprofundada.

    Combina busca web + Wikipedia para entregar uma pesquisa completa.

    CONCEITO DIDÁTICO: Composição
    -----------------------------
    Esta skill COMPÕE múltiplas tools para criar uma capacidade
    mais poderosa. É como um chef (skill) que usa vários
    utensílios (tools) para criar um prato completo.
    """

    def __init__(self):
        super().__init__(
            name="research",
            description="Pesquisa aprofundada sobre um tópico combinando web e Wikipedia",
            required_tools=["web_search", "wikipedia_summary"],
            version="1.0.0"
        )

    def execute(self, topic: str, language: str = "pt", depth: str = "normal") -> str:
        """
        Executa a pesquisa aprofundada.

        Etapas:
        1. Busca na web por informações atuais
        2. Busca na Wikipedia por conhecimento consolidado
        3. Sintetiza tudo em um relatório estruturado

        Args:
            topic: Tópico a pesquisar
            language: Idioma da Wikipedia
            depth: Profundidade (rapida, normal, profunda)

        Returns:
            Relatório de pesquisa formatado em markdown
        """
        sections = []
        sources = []

        # ─── ETAPA 1: Busca na Web ───
        web_results = self._search_web(topic)
        if web_results:
            sections.append(web_results)
            sources.append("🌐 Busca Web")

        # ─── ETAPA 2: Busca na Wikipedia (se depth >= normal) ───
        if depth in ["normal", "profunda"]:
            wiki_results = self._search_wikipedia(topic, language)
            if wiki_results:
                sections.append(wiki_results)
                sources.append("📚 Wikipedia")

        # ─── ETAPA 3: Síntese ───
        report = self._build_report(topic, sections, sources, depth)

        return report

    def _search_web(self, topic: str) -> Optional[str]:
        """
        Etapa 1: Busca na web usando as tools existentes.

        Note:
            Importamos e usamos as tools INTERNAMENTE, não como
            dependência do agente. A skill encapsula essa complexidade.
        """
        try:
            from tools.web_search import search_with_duckduckgo

            results = search_with_duckduckgo(topic, num_results=5)
            if not results:
                return None

            text = "### 🌐 Informações da Web\n\n"
            for i, r in enumerate(results, 1):
                title = r.get("title", "Sem título")
                snippet = r.get("snippet", "")
                url = r.get("url", "")
                text += f"**{i}. {title}**\n"
                text += f"   {snippet}\n"
                if url:
                    text += f"   🔗 Fonte: {url}\n"
                text += "\n"

            return text

        except Exception as e:
            return f"⚠️ Busca web indisponível: {str(e)}\n"

    def _search_wikipedia(self, topic: str, language: str = "pt") -> Optional[str]:
        """
        Etapa 2: Busca na Wikipedia.

        Tenta obter o resumo do artigo sobre o tópico.
        Se não encontrar exatamente, busca artigos relacionados.
        """
        try:
            from tools.wikipedia import get_article_summary, search_articles

            # Tenta obter resumo direto
            summary_data = get_article_summary(topic, language=language)

            if "error" not in summary_data:
                text = "### 📚 Wikipedia\n\n"
                text += f"**{summary_data.get('title', topic)}**\n\n"
                text += f"{summary_data.get('extract', 'Sem resumo disponível.')}\n"
                if summary_data.get("url"):
                    text += f"\n🔗 Artigo completo: {summary_data['url']}\n"
                return text

            # Se não encontrou, busca artigos relacionados
            search_data = search_articles(topic, language=language, limit=3)
            if search_data.get("results"):
                text = "### 📚 Wikipedia - Artigos Relacionados\n\n"
                for article in search_data["results"]:
                    text += f"- **{article.get('title', '')}**: {article.get('description', 'Sem descrição')}\n"
                return text

            return None

        except Exception as e:
            return f"⚠️ Wikipedia indisponível: {str(e)}\n"

    def _build_report(
        self,
        topic: str,
        sections: list,
        sources: list,
        depth: str
    ) -> str:
        """
        Etapa 3: Constrói o relatório final de pesquisa.

        Monta um relatório estruturado com todas as informações
        coletadas nas etapas anteriores.
        """
        report = f"## 🔍 Relatório de Pesquisa: {topic}\n\n"
        report += f"**Profundidade:** {depth} | **Fontes consultadas:** {', '.join(sources)}\n\n"
        report += "---\n\n"

        if sections:
            for section in sections:
                report += section + "\n"
        else:
            report += "⚠️ Não foi possível encontrar informações sobre este tópico.\n"
            report += "Tente reformular sua busca ou usar termos mais específicos.\n"

        report += "\n---\n"
        report += f"*Pesquisa realizada pela Research Skill v{self.version}*\n"

        return report


# =============================================================================
# INSTÂNCIA DA SKILL E TOOL LANGCHAIN
# =============================================================================

# Cria a instância da skill
_research_skill = ResearchSkill()


@tool("research_skill", args_schema=ResearchSkillInput)
def research_skill_tool(topic: str, language: str = "pt", depth: str = "normal") -> str:
    """
    Pesquisa aprofundada sobre um tópico combinando múltiplas fontes.

    Esta SKILL (não é uma simples tool!) faz pesquisa em múltiplas fontes
    simultaneamente e sintetiza os resultados em um relatório estruturado.

    Use quando o usuário pedir:
    - "Pesquise sobre..." / "Faça uma pesquisa sobre..."
    - "Quero saber mais sobre..."
    - "Me dê informações detalhadas sobre..."
    - "Investigue..." / "Aprofunde..."

    Fontes consultadas:
    - Web (DuckDuckGo): Informações atualizadas da internet
    - Wikipedia: Conhecimento enciclopédico consolidado

    Profundidades:
    - rapida: Apenas busca web (mais rápido)
    - normal: Web + Wikipedia (recomendado)
    - profunda: Web + Wikipedia + detalhes extras

    Args:
        topic: O tópico a pesquisar
        language: Idioma da Wikipedia (pt, en, es)
        depth: Profundidade (rapida, normal, profunda)

    Returns:
        Relatório de pesquisa formatado com múltiplas fontes
    """
    return _research_skill.execute(
        topic=topic,
        language=language,
        depth=depth
    )


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Teste a skill diretamente
    result = research_skill_tool.invoke({
        "topic": "Inteligência Artificial",
        "language": "pt",
        "depth": "normal"
    })
    print(result)

