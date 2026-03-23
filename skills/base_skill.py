"""
=============================================================================
BASE SKILL - Classe Base Abstrata para Todas as Skills
=============================================================================

Este módulo define a classe base que todas as Skills devem seguir.

O que é uma Skill?
------------------
Uma Skill é uma capacidade de alto nível que um agente pode executar.
Diferente de uma Tool (que faz uma ação atômica), uma Skill encapsula
um fluxo completo de trabalho com múltiplas etapas.

Padrão de Projeto: Template Method + Strategy
----------------------------------------------
- Template Method: Define a estrutura base (validate → execute → format)
- Strategy: Cada skill implementa sua própria estratégia de execução

Conceitos Importantes:
- ABC (Abstract Base Class): Classe que não pode ser instanciada diretamente
- @abstractmethod: Método que DEVE ser implementado pelas classes filhas
- Composição: Skills podem compor múltiplas Tools internamente
- Encapsulamento: A complexidade interna fica oculta do agente

Exemplo de como criar uma nova Skill:
    class MinhaSkill(BaseSkill):
        def __init__(self):
            super().__init__(
                name="minha_skill",
                description="Faz algo incrível",
                required_tools=["web_search", "calculator"]
            )

        def execute(self, **kwargs) -> str:
            # Lógica multi-etapa aqui
            return "Resultado"

=============================================================================
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseSkill(ABC):
    """
    Classe base abstrata para todas as Skills.

    Uma Skill encapsula uma capacidade completa do agente, podendo
    combinar múltiplas Tools, lógica de raciocínio e formatação
    de resultados em um único componente reutilizável.

    Attributes:
        name (str): Nome único da skill (ex: "research", "summarize")
        description (str): Descrição clara do que a skill faz
        required_tools (List[str]): Lista de tools que esta skill utiliza
        version (str): Versão da skill para controle de mudanças

    Ciclo de Vida de uma Skill:
        1. Inicialização: Configura dependências e parâmetros
        2. Validação: Verifica se os inputs são válidos
        3. Execução: Realiza o trabalho (multi-etapa)
        4. Formatação: Estrutura o resultado para o agente

    Example:
        >>> class PesquisaSkill(BaseSkill):
        ...     def execute(self, topic: str) -> str:
        ...         # 1. Busca na web
        ...         # 2. Busca na Wikipedia
        ...         # 3. Sintetiza resultados
        ...         return "Resultado da pesquisa"
    """

    def __init__(
        self,
        name: str,
        description: str,
        required_tools: Optional[List[str]] = None,
        version: str = "1.0.0"
    ):
        """
        Inicializa a Skill base.

        Args:
            name: Nome identificador da skill (usado como nome da tool)
            description: Descrição do que a skill faz (usada pelo LLM)
            required_tools: Lista de nomes das tools que esta skill compõe
            version: Versão da skill (semântica: major.minor.patch)

        Note:
            O campo required_tools é informativo - serve para documentar
            quais tools a skill utiliza internamente. A skill é responsável
            por importar e usar essas tools diretamente.
        """
        self.name = name
        self.description = description
        self.required_tools = required_tools or []
        self.version = version

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Executa a skill com os parâmetros fornecidos.

        Este método DEVE ser implementado por todas as Skills concretas.
        Aqui é onde a lógica multi-etapa acontece.

        Args:
            **kwargs: Parâmetros específicos de cada skill

        Returns:
            Resultado da execução como string formatada

        Raises:
            NotImplementedError: Se a classe filha não implementar
        """
        pass

    def validate_input(self, **kwargs) -> bool:
        """
        Valida os parâmetros de entrada antes da execução.

        Pode ser sobrescrito pelas classes filhas para adicionar
        validações específicas.

        Args:
            **kwargs: Parâmetros a validar

        Returns:
            True se os parâmetros são válidos
        """
        return True

    def format_result(self, raw_result: str, output_format: str = "markdown") -> str:
        """
        Formata o resultado da execução.

        Args:
            raw_result: Resultado bruto da execução
            output_format: Formato desejado (markdown, plain, json)

        Returns:
            Resultado formatado
        """
        if output_format == "plain":
            # Remove markdown básico
            result = raw_result.replace("**", "").replace("##", "").replace("###", "")
            return result
        # Por padrão, retorna como está (markdown)
        return raw_result

    def get_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre a skill.

        Útil para debug e para o agente entender as capacidades disponíveis.

        Returns:
            Dicionário com metadados da skill
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_tools": self.required_tools,
            "version": self.version
        }

    def __repr__(self) -> str:
        return f"Skill(name='{self.name}', version='{self.version}')"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

