"""Estratégias de RAG: chains que combinam recuperação e geração.
Autor: Rodrigo Aguiar
Data: 30/07/2026
"""

# Bibliotecas externas
from itertools import chain as it_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable



def montar_chain_resposta(modelo: BaseLanguageModel) -> Runnable:
    """Chain de resposta: recebe {contexto, user_query} e gera a resposta."""
    prompt_resposta = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Responda usando exclusivamente o conteúdo fornecido. "
             "Seja breve na resposta.\n\nContexto:\n{contexto}"),
            ("human", "{user_query}"),
        ]
    )
    return prompt_resposta | modelo | StrOutputParser()


def montar_chain_reescrita(modelo: BaseLanguageModel) -> Runnable:
    """
    Chain de reescrita: recebe a pergunta bruta e devolve uma
    consulta otimizada para busca semântica no vector DB.
    """
    prompt_reescrita = PromptTemplate.from_template(
        """Gere uma consulta de pesquisa para o banco de dados de vetores
(Vector DB) a partir da pergunta do usuário, permitindo uma resposta
mais precisa por meio da busca semântica.
Retorne apenas a consulta revisada, sem aspas e sem explicações.

Pergunta do usuário: {user_query}
Consulta revisada:"""
    )
    return prompt_reescrita | modelo | StrOutputParser()


def montar_chain_multi_query(modelo: BaseLanguageModel) -> Runnable:
    """
    Chain de múltiplas queries: recebe a pergunta bruta e devolve uma lista 
    de reformulações para ampliar a cobertura da busca semântica no vector DB.
    """
    prompt_multi_query = PromptTemplate.from_template(
        """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar 
        três versões diferentes da pergunta do usuário para recuperar documentos 
        relevantes de um banco de dados vetorial. Ao gerar múltiplas perspectivas 
        sobre a pergunta do usuário, seu objetivo é ajudar a superar limitações da 
        busca por similaridade baseada em distância.
        Forneça apenas as perguntas alternativas, uma por linha, sem numeração.
        Pergunta original: {user_query}"""
    )
    return (
        prompt_multi_query 
        | modelo 
        | StrOutputParser()
        |RunnableLambda(_parse_multi_queries)
    )


def _parse_multi_queries(texto: str) -> list[str]:
    """Transforma a saída do LLM (uma pergunta por linha) em uma lista de strings, 
    descartando linhas vazias."""
    linhas = [linha.strip() for linha in texto.strip().split("\n")]
    return [linha for linha in linhas if linha]


def _formatar_documentos(docs: list[Document]) -> str:
    return '\n\n'.join(doc.page_content for doc in docs)


def _remover_documentos_duplicados(listas_de_docs: list[list[Document]]) -> list[Document]:
    """Achata as listas de documentos vindas de múltiplas buscas e remove duplicatas 
    com base no conteúdo do texto."""
    todos_docs = it_chain.from_iterable(listas_de_docs)
    vistos = set()
    unicos = []
    for doc in todos_docs:
        if doc.page_content not in vistos:
            vistos.add(doc.page_content)
            unicos.append(doc)
    return unicos


# --- Tipos de RAG

def montar_rag_simples(retriever: BaseRetriever, modelo: BaseLanguageModel) -> Runnable:
    """RAG direto: busca com a query original do usuário."""
    chain_resposta = montar_chain_resposta(modelo)
    chain = (
        {
            "contexto": retriever | _formatar_documentos,
            "user_query": RunnablePassthrough(),
        }
        | chain_resposta
    )

    return chain.with_config(
        run_name="rag_simples", 
        tags=["rag_simples", "sem_reescrita"]
        )


def montar_rag_com_reescrita(
    retriever: BaseRetriever, 
    modelo_reescrita: BaseLanguageModel, 
    modelo_resposta: BaseLanguageModel
    ) -> Runnable:
    """RAG com reescrita: primeiro reescreve a query, depois busca
    e responde usando a query original no prompt final."""
    chain_reescrita = montar_chain_reescrita(modelo_reescrita).with_config(
        run_name="etapa_reescrita_query",
        tags=["reescrita"],
    )
    chain_resposta = montar_chain_resposta(modelo_resposta).with_config(
        run_name="etapa_geracao_resposta",
        tags=["geracao_resposta"],
    )

    chain = (
        {
            "contexto": (
                {"user_query": RunnablePassthrough()}
                | chain_reescrita
                | retriever
                | _formatar_documentos
            ),
            "user_query": RunnablePassthrough(),
        }
        | chain_resposta
    )

    return chain.with_config(
        run_name="rag_com_reescrita",
        tags=["rag_com_reescrita", "reescrita"],
    )



def montar_rag_multi_query(
    retriever: BaseRetriever, 
    modelo_reescrita: BaseLanguageModel, 
    modelo_resposta: BaseLanguageModel
    ) -> Runnable:
    """RAG com múltiplas queries: gera variações da pergunta original,
    busca documentos para cada uma, unifica os resultados removendo 
    duplicatas e responde usando a pergunta original no prompt final."""
    chain_multi_query = montar_chain_multi_query(modelo_reescrita).with_config(
        run_name="etapa_geracao_multi_query",
        tags=["multi_query"]
    )
    chain_resposta = montar_chain_resposta(modelo_resposta).with_config(
        run_name="etapa_geracao_resposta",
        tags=["geracao_resposta"],
    )

    # Closure aninhada para evidenciar a ação do RunnableLambda dentro da chain.
    def _buscar_para_cada_query(queries: list[str]) -> list[Document]:
        listas_docs = retriever.batch(queries)
        return _remover_documentos_duplicados(listas_docs)
    
    chain = (
        {
            "contexto": (
                {"user_query": RunnablePassthrough()}
                | chain_multi_query 
                | RunnableLambda(_buscar_para_cada_query) 
                | _formatar_documentos 
            ),
            "user_query": RunnablePassthrough(),
        }
        | chain_resposta
    )

    return chain.with_config(
        run_name="rag_multi_query", 
        tags=["rag_multi_query", "multi_query"]
    )