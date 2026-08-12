"""Definição do ACPMessage
Autor: Rodrigo Aguiar
Data: 12/08/2026
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ACPMessage(BaseModel):
    """
    Protocolo de Comunicação entre Agentes (ACP - Agent Communication Protocol).
    Define a estrutura padronizada para mensagens trocadas entre agentes.
    """
    sender: str = Field(..., description="Identificador do agente remetente.")
    receiver: str = Field(..., description="Identificador do agente destinatário.")
    intent: str = Field(..., description="A intenção da mensagem (ex: 'query_retrieval', 'final_answer', 'clarification').")

    # Dicionário flexível que conterá os dados reais da mensagem:
    payload: Dict[str, Any] = Field(default_factory=dict, description="Conteúdo da mensagem, pode variar conforme a intenção.")

    # Útil para rastreamento:
    context_id: str = Field(..., description="ID da sessão ou contexto da conversa para rastreamento.")

    # Útil para auditoria e ordenação de eventos:
    timestamp: str = Field(..., description="Timestamp da mensagem no formato ISO 8601.")

    # Úteis para suportar a mitigação de Cascata de alucinações e para avaliação de Precisão e Recall,
    # permitindo que os agentes reportem a qualidade de suas informações:
    confidence: Optional[float] = Field(None, description="Nível de confiança da informação no payload (0.0 a 1.0).")
    sources: Optional[List[str]] = Field(None, description="Lista de fontes ou documentos utilizados para gerar o payload.")

    def to_dict(self) -> Dict[str, Any]:
        """Converte a mensagem ACP para um dicionário."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPMessage":
        """Cria uma instância de ACPMessage a partir de um dicionário."""
        return cls(**data)

    def __str__(self):
        return f"ACPMessage(sender={self.sender}, receiver={self.receiver}, intent={self.intent}, context_id={self.context_id})"
