"""Avaliação das estratégias de RAG
Autor: Rodrigo Aguiar
Data: 30/07/2026
Objetivo: Avaliação comparativa das estratégias de RAG definidas em rag_strategies.py.
"""

# Monkey patch: corrige incompatibilidade do ragas 0.4.3 com
# langchain-community moderno (ChatVertexAI foi movido para
# langchain-google-vertexai)
import sys
import types

try:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _mod = types.ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _mod
except ImportError:
    pass

# -------------------------

from datasets import Dataset
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from config import EMBEDDING_MODEL, LLM_EVAL_MODEL

METRICAS_RAGAS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Configuração ajustada para execução local via Ollama:
# max_workers=1 evita chamadas concorrentes disputando o mesmo
# servidor local; timeout generoso dá tempo real ao modelo responder.
RUN_CONFIG_LOCAL = RunConfig(timeout=600, max_workers=1)


def montar_avaliador_ragas() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:  # type: ignore[valid-type]
    """Prepara o LLM juiz e o modelo de embeddings usados internamente
    pelo RAGAS para julgar cada métrica."""
    llm_avaliador = LangchainLLMWrapper(OllamaLLM(model=LLM_EVAL_MODEL))
    embeddings_avaliador = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    return llm_avaliador, embeddings_avaliador


def coletar_amostras(
    retriever: BaseRetriever,
    chain: Runnable,
    perguntas_gabarito: list[dict[str, str]],
) -> list[dict]:
    """Executa uma chain de RAG para cada pergunta do gabarito e
    monta as amostras no formato esperado pelo RAGAS.

    Args:
        retriever: retriever usado para recuperar o contexto de cada
            pergunta (necessário para calcular context_precision e
            context_recall).
        chain: chain de RAG já montada (ex.: rag_simples), usada
            para gerar a resposta final.
        perguntas_gabarito: lista de dicionários com "query" e
            "answer" (resposta de referência).

    Returns:
        Lista de dicionários no formato exigido pelo RAGAS:
        user_input, response, retrieved_contexts e reference.
    """
    amostras = []
    for item in perguntas_gabarito:
        pergunta = item["test_query"]
        resposta = chain.invoke(pergunta)
        documentos = retriever.invoke(pergunta)

        amostras.append(
            {
                "user_input": pergunta,
                "response": resposta,
                "retrieved_contexts": [doc.page_content for doc in documentos],
                "reference": item["answer"],
            }
        )
    return amostras


def avaliar_estrategia(
    nome_estrategia: str,
    retriever: BaseRetriever,
    chain: Runnable,
    perguntas_gabarito: list[dict[str, str]],
) -> None:
    """Gera as amostras de uma estratégia de RAG, roda as métricas
    do RAGAS e imprime o resultado."""
    amostras = coletar_amostras(retriever, chain, perguntas_gabarito)
    dataset = Dataset.from_list(amostras)

    llm_avaliador, embeddings_avaliador = montar_avaliador_ragas()
    resultado = evaluate(
        dataset=dataset,
        metrics=METRICAS_RAGAS,
        llm=llm_avaliador,
        embeddings=embeddings_avaliador,
    )

    print(f"\n[{nome_estrategia}]")
    print(resultado)