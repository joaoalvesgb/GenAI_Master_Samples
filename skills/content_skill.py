"""
=============================================================================
CONTENT CREATION SKILL - Skill de Criação de Conteúdo
=============================================================================

Esta Skill demonstra o conceito de TEMPLATES DINÂMICOS + CONTEXTO.

O que esta Skill faz:
---------------------
Gera conteúdo formatado profissionalmente usando templates pré-definidos:
- E-mails profissionais (formal, informal, follow-up)
- Relatórios estruturados (com seções e dados)
- Posts para redes sociais (LinkedIn, Twitter/X)

Por que é uma Skill e não uma Tool?
------------------------------------
Uma Tool geraria texto genérico sem estrutura.
Esta Skill:
- USA TEMPLATES especializados para cada tipo de conteúdo
- ADICIONA CONTEXTO temporal (data/hora atual)
- FORMATA conforme o canal/plataforma destino
- APLICA REGRAS de boas práticas para cada formato

CONCEITO DIDÁTICO: Template Method + Factory
----------------------------------------------
- Template Method: Cada tipo de conteúdo segue um template específico
- Factory: O tipo de conteúdo determina qual template é usado

    Usuário pede          Skill identifica          Skill aplica
    "escreva um           tipo = "email"            Template de
     e-mail formal"  ───→ tom = "formal"   ────→   e-mail formal
                          assunto = "..."           com estrutura

=============================================================================
"""

from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base_skill import BaseSkill


# =============================================================================
# SCHEMA DE ENTRADA (Pydantic)
# =============================================================================

class ContentCreationInput(BaseModel):
    """
    Schema de entrada para a Skill de Criação de Conteúdo.

    O LLM preenche estes campos baseado no pedido do usuário.
    """
    content_type: str = Field(
        description=(
            "Tipo de conteúdo a criar: "
            "'email' (e-mail profissional), "
            "'report' (relatório estruturado), "
            "'social_post' (post para redes sociais)"
        )
    )
    topic: str = Field(
        description="Assunto ou tema do conteúdo. Exemplo: 'reunião de projeto', 'resultados trimestrais'"
    )
    tone: str = Field(
        default="profissional",
        description="Tom do conteúdo: 'formal', 'profissional', 'casual', 'entusiasmado'"
    )
    key_points: str = Field(
        default="",
        description="Pontos-chave a incluir, separados por vírgula. Exemplo: 'prazo, orçamento, próximos passos'"
    )
    recipient: str = Field(
        default="",
        description="Destinatário (para e-mails) ou audiência-alvo. Exemplo: 'equipe de desenvolvimento', 'gerente'"
    )


# =============================================================================
# TEMPLATES DE CONTEÚDO
# =============================================================================

EMAIL_TEMPLATES = {
    "formal": {
        "greeting": "Prezado(a) {recipient}",
        "closing": "Atenciosamente",
        "structure": "saudação → contexto → corpo → ação → despedida"
    },
    "profissional": {
        "greeting": "Olá {recipient}",
        "closing": "Abraços",
        "structure": "saudação → contexto → corpo → ação → despedida"
    },
    "casual": {
        "greeting": "Oi {recipient}",
        "closing": "Valeu!",
        "structure": "saudação → corpo direto → despedida"
    }
}

SOCIAL_TEMPLATES = {
    "linkedin": {
        "max_chars": 3000,
        "format": "profissional com hashtags",
        "elements": ["hook", "contexto", "insight", "cta", "hashtags"]
    },
    "twitter": {
        "max_chars": 280,
        "format": "conciso e impactante",
        "elements": ["mensagem_curta", "hashtags"]
    }
}


# =============================================================================
# IMPLEMENTAÇÃO DA SKILL
# =============================================================================

class ContentCreationSkill(BaseSkill):
    """
    Skill de Criação de Conteúdo Profissional.

    Gera diferentes tipos de conteúdo usando templates especializados.

    CONCEITO DIDÁTICO: Factory Pattern
    -----------------------------------
    O método execute() atua como uma Factory: com base no content_type,
    delega para o método de criação apropriado.

    content_type="email"       → _create_email()
    content_type="report"      → _create_report()
    content_type="social_post" → _create_social_post()
    """

    def __init__(self):
        super().__init__(
            name="content_creation",
            description="Cria conteúdo profissional formatado (e-mails, relatórios, posts)",
            required_tools=["get_current_datetime"],
            version="1.0.0"
        )

    def execute(
        self,
        content_type: str,
        topic: str,
        tone: str = "profissional",
        key_points: str = "",
        recipient: str = ""
    ) -> str:
        """
        Executa a criação de conteúdo.

        Factory Method: direciona para o criador específico
        baseado no tipo de conteúdo solicitado.

        Args:
            content_type: Tipo de conteúdo (email, report, social_post)
            topic: Assunto/tema
            tone: Tom de comunicação
            key_points: Pontos-chave separados por vírgula
            recipient: Destinatário ou audiência

        Returns:
            Conteúdo formatado em markdown
        """
        # Processa pontos-chave
        points = [p.strip() for p in key_points.split(",") if p.strip()]

        # Obtém data/hora atual para contexto
        current_date = datetime.now().strftime("%d/%m/%Y")
        current_time = datetime.now().strftime("%H:%M")

        context = {
            "topic": topic,
            "tone": tone,
            "points": points,
            "recipient": recipient or "Destinatário",
            "date": current_date,
            "time": current_time
        }

        # Factory: direciona para o criador apropriado
        if content_type == "email":
            return self._create_email(context)
        elif content_type == "report":
            return self._create_report(context)
        elif content_type == "social_post":
            return self._create_social_post(context)
        else:
            return self._create_generic(context)

    def _create_email(self, context: dict) -> str:
        """
        Cria um e-mail profissional usando templates.

        Estrutura do e-mail:
        1. Saudação (baseada no tom)
        2. Abertura contextual
        3. Corpo com pontos-chave
        4. Call to action
        5. Despedida
        """
        tone = context["tone"]
        template = EMAIL_TEMPLATES.get(tone, EMAIL_TEMPLATES["profissional"])

        greeting = template["greeting"].format(recipient=context["recipient"])
        closing = template["closing"]

        result = "## ✉️ E-mail Gerado\n\n"
        result += f"*Tom: {tone} | Data: {context['date']}*\n\n"
        result += "---\n\n"

        # Saudação
        result += f"**{greeting},**\n\n"

        # Abertura
        if tone == "formal":
            result += f"Venho por meio deste comunicar sobre **{context['topic']}**.\n\n"
        elif tone == "casual":
            result += f"Tudo bem? Queria falar sobre **{context['topic']}**.\n\n"
        else:
            result += f"Espero que esteja bem. Gostaria de tratar sobre **{context['topic']}**.\n\n"

        # Corpo com pontos-chave
        if context["points"]:
            result += "Os principais pontos são:\n\n"
            for point in context["points"]:
                result += f"- **{point.capitalize()}**\n"
            result += "\n"

        # Call to action
        if tone == "formal":
            result += "Fico à disposição para quaisquer esclarecimentos adicionais.\n\n"
        else:
            result += "Me avise se tiver alguma dúvida ou se precisar de mais detalhes!\n\n"

        # Despedida
        result += f"**{closing},**\n"
        result += "*[Seu Nome]*\n"

        result += f"\n---\n*Gerado pela Content Creation Skill v{self.version}*\n"
        return result

    def _create_report(self, context: dict) -> str:
        """
        Cria um relatório estruturado.

        Estrutura do relatório:
        1. Cabeçalho com metadados
        2. Sumário executivo
        3. Detalhamento dos pontos
        4. Próximos passos
        """
        result = f"## 📊 Relatório: {context['topic']}\n\n"
        result += f"**Data:** {context['date']} | **Hora:** {context['time']}\n"
        result += f"**Destinatário:** {context['recipient']}\n\n"
        result += "---\n\n"

        # Sumário executivo
        result += "### 📋 Sumário Executivo\n\n"
        result += f"Este relatório aborda **{context['topic']}** e apresenta "
        result += f"os pontos principais para análise e tomada de decisão.\n\n"

        # Detalhamento
        if context["points"]:
            result += "### 📌 Pontos Principais\n\n"
            for i, point in enumerate(context["points"], 1):
                result += f"#### {i}. {point.capitalize()}\n"
                result += f"- **Status:** A ser definido\n"
                result += f"- **Observações:** Detalhar conforme necessidade\n\n"

        # Próximos passos
        result += "### ➡️ Próximos Passos\n\n"
        result += "1. Revisar os pontos apresentados\n"
        result += "2. Alinhar prioridades com a equipe\n"
        result += "3. Definir cronograma de ações\n\n"

        # Rodapé
        result += "---\n"
        result += f"*Relatório gerado em {context['date']} às {context['time']}*\n"
        result += f"*Content Creation Skill v{self.version}*\n"
        return result

    def _create_social_post(self, context: dict) -> str:
        """
        Cria posts para redes sociais.

        Gera versões para LinkedIn e Twitter/X com formatação adequada.
        """
        result = "## 📱 Posts para Redes Sociais\n\n"
        result += f"*Tema: {context['topic']} | Tom: {context['tone']}*\n\n"

        # LinkedIn
        result += "### 💼 LinkedIn\n\n"
        result += f"🔹 **{context['topic'].upper()}**\n\n"

        if context["points"]:
            result += "Gostaria de compartilhar algumas reflexões:\n\n"
            for point in context["points"]:
                result += f"✅ {point.capitalize()}\n"
            result += "\n"
        else:
            result += f"Hoje quero compartilhar minha visão sobre {context['topic']}.\n\n"

        result += "O que vocês acham? Compartilhem nos comentários! 👇\n\n"

        # Gera hashtags
        hashtags = self._generate_hashtags(context["topic"])
        result += f"{hashtags}\n\n"

        # Twitter/X
        result += "### 🐦 Twitter/X\n\n"
        tweet = f"💡 {context['topic']}"
        if context["points"]:
            tweet += f" - destaque: {context['points'][0]}"
        tweet += f"\n\n{hashtags}"

        if len(tweet) > 280:
            tweet = tweet[:277] + "..."

        result += f"{tweet}\n"
        result += f"\n*({len(tweet)} caracteres)*\n"

        result += f"\n---\n*Gerado pela Content Creation Skill v{self.version}*\n"
        return result

    def _create_generic(self, context: dict) -> str:
        """Cria conteúdo genérico quando o tipo não é reconhecido."""
        result = f"## 📝 Conteúdo: {context['topic']}\n\n"
        result += f"*Data: {context['date']} | Tom: {context['tone']}*\n\n"

        if context["points"]:
            result += "### Pontos Abordados\n\n"
            for point in context["points"]:
                result += f"- {point.capitalize()}\n"
            result += "\n"

        result += (
            f"ℹ️ Tipo de conteúdo não reconhecido. "
            f"Use: 'email', 'report' ou 'social_post'.\n"
        )

        result += f"\n---\n*Content Creation Skill v{self.version}*\n"
        return result

    def _generate_hashtags(self, topic: str) -> str:
        """
        Gera hashtags relevantes baseadas no tópico.

        Estratégia simples: transforma palavras-chave do tópico em hashtags
        e adiciona hashtags genéricas relevantes.
        """
        # Palavras do tópico como hashtags
        words = topic.split()
        hashtags = []

        for word in words:
            clean_word = word.strip(".,;:!?").capitalize()
            if len(clean_word) > 2:
                hashtags.append(f"#{clean_word}")

        # Adiciona hashtags genéricas
        generic_tags = ["#Inovação", "#Produtividade", "#Conhecimento"]
        hashtags.extend(generic_tags[:2])

        return " ".join(hashtags[:6])  # Máximo 6 hashtags


# =============================================================================
# INSTÂNCIA DA SKILL E TOOL LANGCHAIN
# =============================================================================

_content_skill = ContentCreationSkill()


@tool("content_creation_skill", args_schema=ContentCreationInput)
def content_creation_skill_tool(
    content_type: str,
    topic: str,
    tone: str = "profissional",
    key_points: str = "",
    recipient: str = ""
) -> str:
    """
    Cria conteúdo profissional formatado (e-mails, relatórios, posts).

    Esta SKILL gera conteúdo usando templates profissionais especializados.

    Use quando o usuário pedir:
    - "Escreva um e-mail sobre..." / "Crie um e-mail para..."
    - "Faça um relatório sobre..." / "Gere um relatório de..."
    - "Crie um post para LinkedIn sobre..." / "Escreva um tweet sobre..."
    - "Redija um comunicado sobre..."

    Tipos de conteúdo:
    - email: E-mail profissional com saudação, corpo e despedida
    - report: Relatório estruturado com seções e metadados
    - social_post: Posts para LinkedIn e Twitter/X

    Tons disponíveis:
    - formal: Linguagem formal e corporativa
    - profissional: Profissional mas acessível (padrão)
    - casual: Informal e descontraído
    - entusiasmado: Positivo e motivacional

    Args:
        content_type: Tipo (email, report, social_post)
        topic: Assunto/tema do conteúdo
        tone: Tom de comunicação
        key_points: Pontos-chave separados por vírgula
        recipient: Destinatário ou audiência-alvo

    Returns:
        Conteúdo formatado profissionalmente em markdown
    """
    return _content_skill.execute(
        content_type=content_type,
        topic=topic,
        tone=tone,
        key_points=key_points,
        recipient=recipient
    )


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Teste: Criar e-mail
    print("=" * 60)
    result = content_creation_skill_tool.invoke({
        "content_type": "email",
        "topic": "Reunião de alinhamento do projeto",
        "tone": "profissional",
        "key_points": "prazo final, orçamento atualizado, próximos marcos",
        "recipient": "Equipe de Desenvolvimento"
    })
    print(result)

    # Teste: Criar relatório
    print("\n" + "=" * 60)
    result = content_creation_skill_tool.invoke({
        "content_type": "report",
        "topic": "Resultados Q1 2025",
        "tone": "formal",
        "key_points": "receita, custos, crescimento, projeções",
        "recipient": "Diretoria"
    })
    print(result)

    # Teste: Criar post social
    print("\n" + "=" * 60)
    result = content_creation_skill_tool.invoke({
        "content_type": "social_post",
        "topic": "Inteligência Artificial no mercado de trabalho",
        "tone": "entusiasmado",
        "key_points": "automação, novas profissões, upskilling",
        "recipient": ""
    })
    print(result)

