""" RAG
Autor: Rodrigo Aguiar
Data: 30/07/2026
Objetivo: Comparar diferentes estratégias de RAG
"""

import time

# Bibliotecas externas
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Bibliotecas internas
from config import *
from indexing import carregar_ou_criar_vectorstore
from rag_strategies import montar_rag_simples, montar_rag_com_reescrita, montar_rag_multi_query
from rag_eval import avaliar_estrategia, criar_ou_obter_dataset, criar_target


# ----------- 

def cronometrar(nome_etapa: str, funcao, *args, **kwargs):
    """Executa uma função, medindo e imprimindo o tempo decorrido."""
    inicio = time.perf_counter()
    resultado = funcao(*args, **kwargs)
    tempo_decorrido = time.perf_counter() - inicio
    print(f"[{nome_etapa}] concluído em {tempo_decorrido:.2f}s")
    return resultado



def main() -> None:
    """Função principal: executa os RAGs escolhidos."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = cronometrar("Indexação", carregar_ou_criar_vectorstore, embeddings) #carregar_ou_criar_vectorstore(embeddings)
    retriever = vectorstore.as_retriever()

    modelo_resposta = OllamaLLM(model=LLM_MODEL)
    modelo_reescrita = OllamaLLM(model=QUERY_MODEL)
    
    user_query = 'Quais são os tipos de pagamento aceitos pelo Mercado?'
    print(f'Pergunta: {user_query}')

    rag_simples = cronometrar("Montagem de RAG simples", montar_rag_simples, retriever, modelo_resposta) # montar_rag_simples(retriever, modelo_resposta)
    #rag_com_reescrita = cronometrar("Montagem de RAG com reescrita", montar_rag_com_rescrita(retriever, modelo_reescrita, modelo_resposta) #montar_rag_com_reescrita(retriever, modelo_reescrita, modelo_resposta)
    rag_multi_query = cronometrar("Montagem de RAG com muilti-query", montar_rag_multi_query, retriever, modelo_reescrita, modelo_resposta) # montar_rag_multi_query(retriever, modelo_reescrita, modelo_resposta)

    target_simples = criar_target(rag_simples)
    #target_reescrita = criar_target(rag_com_reescrita)
    target_multi_query = criar_target(rag_multi_query)

    metadata_comum = {
        "pergunta_original": user_query,
        "modelo_reescrita": QUERY_MODEL,
        "modelo_resposta": LLM_MODEL,
    }

    resposta_simples = cronometrar(
        "Invoke RAG simples",
        rag_simples.invoke,
        user_query, 
        config={"metadata": {**metadata_comum, "abordagem": "sem_reescrita"}}
        )

    #resposta_reescrita = rag_com_reescrita.invoke(
    #    user_query, 
    #    config={"metadata": {**metadata_comum, "abordagem": "com_reescrita"}},
    #    )

    resposta_multi_query = cronometrar(
        "Invoke RAG multi-query",
        rag_multi_query.invoke,
        user_query,
        config={'metadata': {**metadata_comum, "abordagem": "multi_query"}}
    )

    print(f'\n[RAG simples]\n{resposta_simples}')
    #print(f'\n[RAG com reescrita]\n{resposta_reescrita}')
    print(f'\n[RAG multi query]\n{resposta_multi_query}')

    # --- Avaliação das estruturas
    if AVALIAR_RAG: # Ativação da avaliação quando necessário
        perguntas_gabarito = [
            {
                "test_query": "Onde localizo os carrinhos de compra do Mercado?",
                "answer": "Eles estão posicionados logo na entrada principal e nos subsolos de estacionamento."
            },
            {
                "test_query": "Em qual horário do dia surge cheiro de pão quente?",
                "answer": "Nos horários das 6h, 12h, 18h e 00h, que são os horários de destaque das fornadas, quando surge cheiro de pão quente."
            },
            {
                "test_query": "Quais são as formas de pagamento que o Mercado aceita?",
                "answer": "O Mercado aceita as principais bandeiras de cartão de crédito e débito (Visa, Mastercard, Elo, American Express e Hipercard), pix por QR Code, vales-alimentação (Alelo, Sodexo, Ticket Alimentação e VR Alimentação). Vale-Refeição é aceito exclusivamente para itens de consumo imediato (padaria, rotisseria e lanchonete), pagamento por aproximação (NFC) também é aceiro (Apple Pay, Google Pay e Samsung Pay). Não se aceita cheque. "
            }
        ]

        nome_dataset = criar_ou_obter_dataset(perguntas_gabarito)

        print('[Início da avaliação]')
        cronometrar("Avaliação de RAG simples", 
                    avaliar_estrategia,
                    nome_estrategia="rag_simples",
                    retriever=retriever,
                    chain=rag_simples,
                    perguntas_gabarito=perguntas_gabarito)
        #cronometrar("Avaliação de RAG com reescrita",avaliar_estrategia, "rag_com_reescrita", retriever, rag_com_reescrita, perguntas_gabarito)
        cronometrar("Avaliação de RAG com multi-query",
                    avaliar_estrategia,
                    nome_estrategia="rag_multi_query",
                    retriever=retriever,
                    chain=rag_multi_query,
                    perguntas_gabarito=perguntas_gabarito)

if __name__ == "__main__":
    main()