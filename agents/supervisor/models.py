"""Definição das estruturas de dados do projeto
Autor: Rodrigo Aguiar
Data: 13/08/2026
"""

from pydantic.v1 import BaseModel, Field

class AgentSelection(BaseModel):
    """
    Representa a decisão do supervisor sobre qual agente deve processar a query.
    """
    next_agent: str = Field(..., description="O nome do agente especialista selecionado ou 'geral'.")
