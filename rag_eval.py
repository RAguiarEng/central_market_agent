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
from langsmith import Client
from langchain_openai import ChatOpenAI

from config import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_JUDGE_MODEL

DATASET_NOME = "avaliacao-rag-mercado"
METRICAS_RAGAS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Configuração ajustada para execução local via Ollama:
# max_workers=1 evita chamadas concorrentes disputando o mesmo
# servidor local; timeout generoso dá tempo real ao modelo responder.
RUN_CONFIG_LOCAL = RunConfig(timeout=600, max_workers=1)


def montar_avaliador_ragas() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:  # type: ignore[valid-type]
    """Prepara o LLM juiz e o modelo de embeddings usados internamente
    pelo RAGAS para julgar cada métrica."""
    llm_avaliador = LangchainLLMWrapper(
        ChatOpenAI(
            model=OPENROUTER_JUDGE_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        )
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


def criar_ou_obter_dataset(perguntas_gabarito: list[dict[str, str]]) -> str:
    """Cria (ou reaproveita, se já existir) um dataset no LangSmith
    a partir do gabarito de perguntas e respostas.

    Args:
        perguntas_gabarito: lista de dicionários com "test_query" e
            "answer".

    Returns:
        Nome do dataset criado/reaproveitado no LangSmith.
    """
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NOME):
        print(f"Dataset '{DATASET_NOME}' já existe, reaproveitando.")
        return DATASET_NOME

    dataset = client.create_dataset(
        dataset_name=DATASET_NOME,
        description="Gabarito de perguntas e respostas sobre o Mercado, usado para comparar estratégias de RAG.",
    )

    examples = [
        {
            "inputs": {"question": item["test_query"]},
            "outputs": {"answer": item["answer"]},
        }
        for item in perguntas_gabarito
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Dataset '{DATASET_NOME}' criado com {len(examples)} exemplos.")
    return DATASET_NOME


def criar_target(chain: Runnable):
    """Cria uma função target compatível com client.evaluate(),
    que invoca a chain de RAG fornecida a partir dos inputs
    do dataset do LangSmith.
    """
    def target(inputs: dict) -> dict:
        resposta = chain.invoke(inputs["question"])
        return {"answer": resposta}
    return target