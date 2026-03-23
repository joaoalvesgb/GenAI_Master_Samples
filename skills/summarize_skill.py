"""
=============================================================================
SUMMARIZE SKILL - Skill de Análise e Resumo Inteligente
=============================================================================

Esta Skill demonstra o conceito de PROCESSAMENTO MULTI-ETAPA.

O que esta Skill faz:
---------------------
Analisa e resume textos de forma inteligente, aplicando diferentes
estratégias de resumo conforme o tipo de conteúdo:
- Resumo executivo (para relatórios e documentos longos)
- Pontos-chave (bullet points com os destaques)
- Análise crítica (com prós, contras e conclusão)

Por que é uma Skill e não uma Tool?
------------------------------------
Uma Tool simplesmente "cortaria" o texto (ex: pegar os primeiros N caracteres).
Esta Skill:
- ANALISA o conteúdo para identificar o tipo de texto
- EXTRAI as informações mais relevantes
- ESTRUTURA o resultado conforme o estilo solicitado
- ADICIONA metadados úteis (contagem de palavras, complexidade)

CONCEITO DIDÁTICO: Processamento em Pipeline
----------------------------------------------
    ┌──────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────┐
    │ Input:   │────→│ Pré-        │────→│ Análise e  │────→│ Output:  │
    │ Texto    │     │ processamento│     │ Resumo     │     │ Resumo   │
    └──────────┘     └─────────────┘     └────────────┘     └──────────┘

    Cada etapa do pipeline transforma os dados progressivamente,
    similar a uma linha de montagem em uma fábrica.

=============================================================================
"""

import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base_skill import BaseSkill


# =============================================================================
# SCHEMA DE ENTRADA (Pydantic)
# =============================================================================

class SummarizeSkillInput(BaseModel):
    """
    Schema de entrada para a Skill de Resumo.

    Define os parâmetros que o LLM deve preencher ao usar esta skill.
    """
    text: str = Field(
        description="O texto a ser analisado e resumido. Pode ser um parágrafo, artigo ou documento."
    )
    style: str = Field(
        default="pontos_chave",
        description=(
            "Estilo do resumo: "
            "'executivo' (resumo formal conciso), "
            "'pontos_chave' (lista com bullet points dos destaques), "
            "'analise_critica' (análise com prós, contras e conclusão)"
        )
    )
    max_points: int = Field(
        default=5,
        description="Número máximo de pontos ou parágrafos no resumo (1-10)"
    )


# =============================================================================
# IMPLEMENTAÇÃO DA SKILL
# =============================================================================

class SummarizeSkill(BaseSkill):
    """
    Skill de Análise e Resumo Inteligente.

    Processa textos e gera resumos estruturados em diferentes estilos.

    CONCEITO DIDÁTICO: Strategy Pattern
    -------------------------------------
    Esta skill usa o padrão Strategy: o método de resumo muda
    conforme o estilo selecionado, mas a interface permanece a mesma.

    Estilos disponíveis:
    - executivo: Resumo formal e conciso (1-2 parágrafos)
    - pontos_chave: Lista de bullet points com os destaques
    - analise_critica: Análise estruturada com prós, contras e conclusão
    """

    def __init__(self):
        super().__init__(
            name="summarize",
            description="Analisa e resume textos em diferentes estilos",
            required_tools=["calculator"],  # Usado para métricas
            version="1.0.0"
        )

    def execute(
        self,
        text: str,
        style: str = "pontos_chave",
        max_points: int = 5
    ) -> str:
        """
        Executa a análise e resumo do texto.

        Pipeline de processamento:
        1. Pré-processamento: Limpa e normaliza o texto
        2. Análise: Extrai métricas e identifica estrutura
        3. Resumo: Gera o resumo no estilo solicitado

        Args:
            text: Texto a ser resumido
            style: Estilo do resumo (executivo, pontos_chave, analise_critica)
            max_points: Número máximo de pontos no resumo

        Returns:
            Resumo formatado em markdown
        """
        # ─── ETAPA 1: Pré-processamento ───
        cleaned_text = self._preprocess(text)
        metrics = self._analyze_metrics(cleaned_text)

        # ─── ETAPA 2: Extração de sentenças-chave ───
        key_sentences = self._extract_key_sentences(cleaned_text, max_points)

        # ─── ETAPA 3: Gerar resumo no estilo escolhido ───
        if style == "executivo":
            summary = self._format_executive(cleaned_text, key_sentences, metrics)
        elif style == "analise_critica":
            summary = self._format_critical_analysis(cleaned_text, key_sentences, metrics)
        else:  # pontos_chave (padrão)
            summary = self._format_key_points(key_sentences, metrics)

        return summary

    def _preprocess(self, text: str) -> str:
        """
        Etapa 1: Pré-processamento do texto.

        Limpa e normaliza o texto para facilitar a análise.
        Remove espaços extras, normaliza quebras de linha, etc.
        """
        # Remove espaços duplicados
        text = re.sub(r'\s+', ' ', text)
        # Remove espaços no início e fim
        text = text.strip()
        return text

    def _analyze_metrics(self, text: str) -> dict:
        """
        Analisa métricas do texto.

        Usa conceitos básicos de NLP para extrair informações
        quantitativas sobre o texto.

        Returns:
            Dicionário com métricas:
            - word_count: Total de palavras
            - sentence_count: Total de sentenças
            - avg_sentence_length: Média de palavras por sentença
            - complexity: Nível estimado de complexidade
        """
        words = text.split()
        word_count = len(words)

        # Divide em sentenças (heurística simples)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        avg_sentence_length = word_count / max(sentence_count, 1)

        # Estimativa simples de complexidade baseada no tamanho médio das palavras
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        if avg_word_length > 6 and avg_sentence_length > 20:
            complexity = "Alta"
        elif avg_word_length > 5 or avg_sentence_length > 15:
            complexity = "Média"
        else:
            complexity = "Baixa"

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "complexity": complexity
        }

    def _extract_key_sentences(self, text: str, max_points: int) -> list:
        """
        Extrai as sentenças mais importantes do texto.

        CONCEITO: Extractive Summarization
        -----------------------------------
        Este é um exemplo simplificado de sumarização extrativa,
        onde selecionamos as sentenças originais mais relevantes.
        (Em produção, usaríamos TF-IDF, TextRank ou embeddings)

        Heurística usada:
        1. Sentenças mais longas tendem a ser mais informativas
        2. Primeira e última sentenças são frequentemente importantes
        3. Sentenças com palavras-chave indicativas são priorizadas
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return [text[:500]]  # Fallback: primeiros 500 caracteres

        # Pontua cada sentença
        scored = []
        keywords = [
            "importante", "principal", "fundamental", "essencial",
            "resultado", "conclusão", "portanto", "em resumo",
            "destaque", "significativo", "crucial", "impacto"
        ]

        for i, sentence in enumerate(sentences):
            score = 0
            # Primeira e última sentença ganham bônus
            if i == 0:
                score += 3
            if i == len(sentences) - 1:
                score += 2

            # Sentenças mais longas (com limite) ganham pontos
            words_in_sentence = len(sentence.split())
            if 10 <= words_in_sentence <= 40:
                score += 2
            elif words_in_sentence > 5:
                score += 1

            # Palavras-chave indicativas
            for kw in keywords:
                if kw.lower() in sentence.lower():
                    score += 2

            scored.append((score, sentence))

        # Ordena por pontuação e pega as top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:max_points]]

    def _format_executive(self, text: str, key_sentences: list, metrics: dict) -> str:
        """
        Formata como resumo executivo.

        Estilo formal e conciso, ideal para relatórios e decisões rápidas.
        """
        result = "## 📋 Resumo Executivo\n\n"
        result += f"*Texto original: {metrics['word_count']} palavras | "
        result += f"Complexidade: {metrics['complexity']}*\n\n"

        # Combina as sentenças-chave em parágrafos coesos
        if key_sentences:
            result += " ".join(key_sentences[:3])
            result += "\n\n"
            if len(key_sentences) > 3:
                result += " ".join(key_sentences[3:])
                result += "\n"
        else:
            result += text[:500] + "...\n"

        result += f"\n---\n*Resumo gerado pela Summarize Skill v{self.version}*\n"
        return result

    def _format_key_points(self, key_sentences: list, metrics: dict) -> str:
        """
        Formata como lista de pontos-chave.

        Estilo com bullet points, ideal para apresentações e revisão rápida.
        """
        result = "## 🎯 Pontos-Chave\n\n"
        result += f"*Texto original: {metrics['word_count']} palavras | "
        result += f"{metrics['sentence_count']} sentenças | "
        result += f"Complexidade: {metrics['complexity']}*\n\n"

        for i, sentence in enumerate(key_sentences, 1):
            result += f"**{i}.** {sentence}\n\n"

        if not key_sentences:
            result += "⚠️ Não foi possível extrair pontos-chave. O texto pode ser muito curto.\n"

        result += f"\n---\n*Análise gerada pela Summarize Skill v{self.version}*\n"
        return result

    def _format_critical_analysis(
        self,
        text: str,
        key_sentences: list,
        metrics: dict
    ) -> str:
        """
        Formata como análise crítica.

        Estilo analítico com estrutura: Resumo + Observações + Conclusão.
        """
        result = "## 🔎 Análise Crítica\n\n"
        result += f"*Texto original: {metrics['word_count']} palavras | "
        result += f"Complexidade: {metrics['complexity']}*\n\n"

        # Resumo
        result += "### 📝 Resumo\n"
        if key_sentences:
            result += " ".join(key_sentences[:2]) + "\n\n"
        else:
            result += text[:300] + "...\n\n"

        # Observações
        result += "### 📊 Observações\n"
        result += f"- O texto contém **{metrics['word_count']} palavras** "
        result += f"em **{metrics['sentence_count']} sentenças**\n"
        result += f"- Comprimento médio das sentenças: **{metrics['avg_sentence_length']} palavras**\n"
        result += f"- Nível de complexidade estimado: **{metrics['complexity']}**\n\n"

        # Pontos de destaque
        if len(key_sentences) > 2:
            result += "### 💡 Destaques Adicionais\n"
            for s in key_sentences[2:]:
                result += f"- {s}\n"
            result += "\n"

        result += f"\n---\n*Análise gerada pela Summarize Skill v{self.version}*\n"
        return result


# =============================================================================
# INSTÂNCIA DA SKILL E TOOL LANGCHAIN
# =============================================================================

_summarize_skill = SummarizeSkill()


@tool("summarize_skill", args_schema=SummarizeSkillInput)
def summarize_skill_tool(
    text: str,
    style: str = "pontos_chave",
    max_points: int = 5
) -> str:
    """
    Analisa e resume textos de forma inteligente em diferentes estilos.

    Esta SKILL analisa o texto e gera um resumo estruturado com métricas.

    Use quando o usuário pedir:
    - "Resuma este texto..." / "Faça um resumo de..."
    - "Quais os pontos principais de..."
    - "Me dê um resumo executivo de..."
    - "Analise este texto..." / "Faça uma análise crítica..."

    Estilos disponíveis:
    - executivo: Resumo formal e conciso (ideal para decisões)
    - pontos_chave: Lista de bullet points (ideal para revisão)
    - analise_critica: Análise com observações e destaques

    Args:
        text: O texto a ser resumido (pode ser longo)
        style: Estilo do resumo (executivo, pontos_chave, analise_critica)
        max_points: Número máximo de pontos no resumo (1-10)

    Returns:
        Resumo estruturado com métricas e formatação markdown
    """
    return _summarize_skill.execute(
        text=text,
        style=style,
        max_points=max_points
    )


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    texto_exemplo = """
    A inteligência artificial (IA) é um campo da ciência da computação que se dedica
    a criar sistemas capazes de realizar tarefas que normalmente requerem inteligência
    humana. Isso inclui aprendizado, raciocínio, percepção, compreensão de linguagem
    natural e tomada de decisões. A IA tem aplicações em diversas áreas, como saúde,
    finanças, educação, transporte e entretenimento. Os avanços recentes em aprendizado
    profundo e processamento de linguagem natural têm impulsionado significativamente
    o desenvolvimento de assistentes virtuais, sistemas de recomendação e veículos
    autônomos. No entanto, a IA também levanta questões éticas importantes sobre
    privacidade, viés algorítmico e o impacto no mercado de trabalho. É fundamental
    que o desenvolvimento de IA seja feito de forma responsável e transparente.
    """

    # Testa diferentes estilos
    for style in ["executivo", "pontos_chave", "analise_critica"]:
        print(f"\n{'='*60}")
        result = summarize_skill_tool.invoke({
            "text": texto_exemplo,
            "style": style,
            "max_points": 5
        })
        print(result)

