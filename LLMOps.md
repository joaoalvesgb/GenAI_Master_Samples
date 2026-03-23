# LLMOps — Operações para Large Language Models

## 📌 O que é LLMOps?

**LLMOps** (Large Language Model Operations) é um conjunto de práticas, ferramentas e processos dedicados ao ciclo de vida completo de modelos de linguagem de grande escala — desde o desenvolvimento, fine-tuning e avaliação até o deploy, monitoramento e manutenção em produção.

Assim como o **MLOps** trouxe disciplina para Machine Learning tradicional, o **LLMOps** surgiu para endereçar os desafios únicos que LLMs introduzem: custos elevados de inferência, gerenciamento de prompts, alucinações, latência, segurança e governança de dados.

---

## 🎯 Por que LLMOps é importante?

| Desafio | Como LLMOps ajuda |
|---|---|
| **Custos de inferência** | Monitoramento de tokens, caching inteligente, roteamento de modelos (modelo menor para tarefas simples) |
| **Qualidade das respostas** | Avaliação sistemática com benchmarks, testes de regressão de prompts |
| **Alucinações** | Guardrails, RAG (Retrieval-Augmented Generation), validação de fatos |
| **Latência** | Otimização de modelos, quantização, batching, streaming |
| **Segurança e compliance** | Filtros de conteúdo, PII detection, audit logs |
| **Versionamento** | Controle de versão de prompts, modelos e datasets |
| **Observabilidade** | Tracing de chamadas, métricas de performance, dashboards |

---

## 🏗️ Componentes-chave de uma Pipeline LLMOps

### 1. Gerenciamento de Prompts (Prompt Management)
- Versionamento de prompts como código
- A/B testing de diferentes formulações
- Templates reutilizáveis com variáveis

### 2. Fine-Tuning e Treinamento
- Preparação e curadoria de datasets
- Fine-tuning supervisionado (SFT)
- RLHF (Reinforcement Learning from Human Feedback)
- LoRA / QLoRA para treinamento eficiente

### 3. Avaliação e Testes (Evaluation)
- Métricas automatizadas (BLEU, ROUGE, BERTScore)
- Avaliação LLM-as-a-Judge
- Testes de regressão em pipelines CI/CD
- Red teaming para segurança

### 4. Deploy e Serving
- Containerização (Docker/Kubernetes)
- Auto-scaling baseado em demanda
- Model routing (escolher modelo por complexidade da tarefa)
- Canary deployments e blue-green

### 5. Monitoramento e Observabilidade
- Tracing distribuído de chamadas LLM
- Monitoramento de custo por request
- Detecção de drift na qualidade das respostas
- Alertas para anomalias (latência, erros, toxicidade)

### 6. Guardrails e Segurança
- Filtros de entrada/saída
- Detecção de prompt injection
- Remoção de PII (dados pessoais)
- Rate limiting e abuse prevention

---

## 🔧 Como usar LLMOps no dia a dia

### Passo 1: Trate prompts como código
```
project/
├── prompts/
│   ├── v1/
│   │   └── summarize.txt
│   ├── v2/
│   │   └── summarize.txt
│   └── prompt_registry.yaml
```

Versione seus prompts no Git, use templates parametrizados e mantenha um registro de qual versão está em produção.

### Passo 2: Implemente avaliação contínua
```python
# Exemplo: avaliação automatizada em CI/CD
def evaluate_prompt(prompt_template, test_cases):
    results = []
    for case in test_cases:
        response = llm.generate(prompt_template.format(**case["input"]))
        score = compute_similarity(response, case["expected"])
        results.append({"input": case["input"], "score": score})
    
    avg_score = sum(r["score"] for r in results) / len(results)
    assert avg_score >= 0.85, f"Qualidade abaixo do threshold: {avg_score}"
    return results
```

### Passo 3: Monitore custos e performance
- Configure dashboards para acompanhar: tokens consumidos, custo por request, latência p50/p95/p99
- Defina budgets e alertas de custo

### Passo 4: Implemente caching
- Cache respostas para queries idênticas ou semanticamente similares
- Use embeddings para cache semântico

### Passo 5: Use guardrails em produção
```python
# Exemplo simplificado de guardrail
def safe_generate(prompt, model):
    # Verificar input
    if detect_prompt_injection(prompt):
        raise SecurityError("Prompt injection detectado")
    
    response = model.generate(prompt)
    
    # Verificar output
    if contains_pii(response):
        response = redact_pii(response)
    
    if is_toxic(response):
        return "Desculpe, não posso gerar essa resposta."
    
    return response
```

### Passo 6: Automatize o deploy
- Use CI/CD para testar e deployar mudanças de prompts e configurações
- Implemente canary releases para mudanças de modelo
- Mantenha rollback automatizado

---

## 🛠️ Ferramentas e Plataformas

| Categoria | Ferramentas |
|---|---|
| **Observabilidade** | [LangSmith](https://smith.langchain.com/), [Langfuse](https://langfuse.com/), [Arize Phoenix](https://phoenix.arize.com/) |
| **Orquestração** | [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/) |
| **Avaliação** | [Ragas](https://docs.ragas.io/), [DeepEval](https://docs.confident-ai.com/), [promptfoo](https://www.promptfoo.dev/) |
| **Guardrails** | [Guardrails AI](https://www.guardrailsai.com/), [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) |
| **Serving** | [vLLM](https://docs.vllm.ai/), [TGI](https://huggingface.co/docs/text-generation-inference), [Ollama](https://ollama.com/) |
| **Fine-Tuning** | [Hugging Face](https://huggingface.co/), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [Unsloth](https://unsloth.ai/) |
| **Plataformas Cloud** | [Azure AI Studio](https://ai.azure.com/), [AWS Bedrock](https://aws.amazon.com/bedrock/), [Google Vertex AI](https://cloud.google.com/vertex-ai) |
| **Prompt Management** | [PromptLayer](https://promptlayer.com/), [Humanloop](https://humanloop.com/) |

---

## 📊 Métricas essenciais para acompanhar

- **Latência** — Tempo total de resposta (p50, p95, p99)
- **Throughput** — Requests por segundo
- **Custo por request** — Tokens de entrada + saída × preço do modelo
- **Qualidade** — Scores de avaliação automatizada e feedback humano
- **Taxa de erro** — % de falhas, timeouts e respostas inválidas
- **Toxicidade/Segurança** — % de respostas flagradas por guardrails
- **Cache hit rate** — Eficiência do caching

---

## 📚 Referências e Leitura Recomendada

- [What is LLMOps? — Databricks](https://www.databricks.com/glossary/llmops)
- [LLMOps: MLOps for Large Language Models — Weights & Biases](https://wandb.ai/site/articles/llmops-mlops-for-large-language-models)
- [A Practical Guide to LLMOps — Google Cloud](https://cloud.google.com/discover/what-is-llmops)
- [LLMOps — Chip Huyen (Blog)](https://huyenchip.com/2023/06/07/generative-ai-strategy.html)
- [Building LLM Applications for Production — Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html)
- [Patterns for Building LLM-based Systems — Eugene Yan](https://eugeneyan.com/writing/llm-patterns/)
- [MLOps vs LLMOps — Neptune.ai](https://neptune.ai/blog/mlops-vs-llmops)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Langfuse — Open Source LLM Observability](https://langfuse.com/docs)
- [Ragas — Evaluation Framework for RAG](https://docs.ragas.io/en/latest/)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Prompt Engineering Guide — DAIR.AI](https://www.promptingguide.ai/)
- [Microsoft Responsible AI Toolbox](https://responsibleaitoolbox.ai/)
- [The Shift from Models to Compound AI Systems — Berkeley AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)

---

## 🔑 Takeaways

1. **LLMOps não é opcional** — sem ele, aplicações com LLM em produção acumulam dívida técnica rapidamente.
2. **Comece simples** — logging, versionamento de prompts e avaliação básica já trazem enorme valor.
3. **Automatize avaliação** — testes manuais não escalam; crie suites de avaliação automatizadas.
4. **Monitore custos** — LLMs podem se tornar caros rapidamente sem visibilidade.
5. **Segurança é prioridade** — guardrails devem estar desde o dia 1 em produção.

---

> *"LLMOps é a ponte entre um protótipo de IA impressionante e um produto de IA confiável em produção."*

