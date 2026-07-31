""" RAG
Autor: Rodrigo Aguiar
Data: 30/07/2026
Objetivo: Comparar diferentes estratégias de RAG
"""

# Bibliotecas externas
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Bibliotecas internas
from config import *
from indexing import carregar_ou_criar_vectorstore
from rag_strategies import montar_rag_simples, montar_rag_com_reescrita, montar_rag_multi_query
from rag_eval import avaliar_estrategia


# ----------- 

def main() -> None:
    """Função principal: executa os RAGs escolhidos."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = carregar_ou_criar_vectorstore(embeddings)
    retriever = vectorstore.as_retriever()

    modelo_resposta = OllamaLLM(model=LLM_MODEL)
    modelo_reescrita = OllamaLLM(model=QUERY_MODEL)
    
    user_query = 'Quais são os tipos de pagamento aceitos pelo Mercado?'
    print(f'Pergunta: {user_query}')

    rag_simples = montar_rag_simples(retriever, modelo_resposta)
    #rag_com_reescrita = montar_rag_com_reescrita(retriever, modelo_reescrita, modelo_resposta)
    rag_multi_query = montar_rag_multi_query(retriever, modelo_reescrita, modelo_resposta)

    metadata_comum = {
        "pergunta_original": user_query,
        "modelo_reescrita": QUERY_MODEL,
        "modelo_resposta": LLM_MODEL,
    }

    resposta_simples = rag_simples.invoke(
        user_query, 
        config={"metadata": {**metadata_comum, "abordagem": "sem_reescrita"}},
        )

    #resposta_reescrita = rag_com_reescrita.invoke(
    #    user_query, 
    #    config={"metadata": {**metadata_comum, "abordagem": "com_reescrita"}},
    #    )

    resposta_multi_query = rag_multi_query.invoke(
        user_query,
        config={'metadata': {**metadata_comum, "abordagem": "multi_query"}},
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

        avaliar_estrategia("rag_simples", retriever, rag_simples, perguntas_gabarito)
        #avaliar_estrategia("rag_com_reescrita", retriever, rag_com_reescrita, perguntas_gabarito)
        avaliar_estrategia("rag_multi_query", retriever, rag_multi_query, perguntas_gabarito)

if __name__ == "__main__":
    main()