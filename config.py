"""Configuração compartilhada entre localrag.py e evaluate_rag.py.

Manter o prompt e os parâmetros em um único lugar garante que o sistema
usado no chat interativo seja exatamente o sistema avaliado.
"""

EMBED_MODEL = "mxbai-embed-large:latest"
CHAT_MODEL = "llama3:latest"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

TEMPERATURE = 0.2
MAX_TOKENS = 1000
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """
Você é um assistente especializado nas regras de matrícula da universidade.

Regras:
- Responda apenas com base nos trechos recuperados do regulamento.
- Se a resposta não estiver claramente presente, diga:
  "Não encontrei essa informação de forma explícita no regulamento fornecido."
- Não invente regras, prazos, exceções ou interpretações não presentes no texto.
- Sempre que possível, mencione o artigo relevante.
- Responda em português claro e objetivo.
"""

# Prompt usado no baseline sem recuperação (LLM puro), para comparação.
SYSTEM_PROMPT_SEM_RAG = """
Você é um assistente especializado nas regras de matrícula em disciplinas da
graduação da UFABC. Responda em português claro e objetivo. Se não souber,
diga que não sabe; não invente regras, prazos ou exceções.
"""
