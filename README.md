# ChefBot - Chatbot Hibrido de Receitas (NLTK + LLM)

Chatbot tematico de culinaria desenvolvido como Trabalho Final de IA e ML.
Implementa arquitetura **hibrida**: base de conhecimento estruturada
consultada **antes** do LLM, com fallback automatico quando a base nao
tem resposta confiavel.

## Arquitetura

```
                     +------------------+
   Mensagem -------> |  NLP (NLTK)      |  tokenizacao, stopwords,
                     |  + Naive Bayes   |  lematizacao, intent class.
                     +--------+---------+
                              |
                  intencao + confianca
                              |
                     +--------v---------+
                     | Base estruturada |  receitas, FAQ (TF-IDF)
                     +--------+---------+
                              |
              base resolveu?  |  nao -> fallback
                              v
                     +--------+---------+
                     | LLM (Qwen2.5-7B) |  4-bit NF4 na GPU
                     +------------------+
```

## Componentes

| Arquivo | Funcao |
|---|---|
| [nlp.py](nlp.py) | Pre-processamento NLTK + classificador de intencao (Naive Bayes) |
| [kb.py](kb.py) | Base de conhecimento: receitas + FAQ culinaria (busca TF-IDF) |
| [receitas.py](receitas.py) | Dataset estruturado de receitas |
| [llm.py](llm.py) | Qwen2.5-3B-Instruct em fp16 (GPU) |
| [chatbot.py](chatbot.py) | Orquestrador: aplica o fluxo KB -> LLM |
| [app.py](app.py) | Servidor Flask com endpoints `/chat`, `/stats`, `/chat/reset` |

## Modelo LLM

**Qwen/Qwen2.5-3B-Instruct** carregado em **fp16** diretamente na GPU.
- VRAM: ~6 GB (cabe na RTX 4060 8GB)
- Sem gating no Hugging Face
- Multilingue forte (PT-BR nativo na geracao)
- Tempo de resposta ~1s na RTX 4060
- Carregamento lazy + pre-load em thread de fundo
- Sem quantizacao, stack minimo (so torch + transformers)

## Como rodar

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Acesse http://127.0.0.1:5000

> Primeira chamada que cair no LLM faz o download dos pesos (~5 GB).
> O `precarregar_async()` em `app.py` ja inicia esse download assim que o
> servidor sobe, em segundo plano.

## Endpoints

- `POST /chat` -> `{ "resposta": str, "fonte": "base|base-faq|llm", "tempo_ms": float }`
- `GET /stats` -> contadores de uso da base vs LLM, tempo medio, status do LLM
- `POST /chat/reset` -> limpa sessao

## Para a apresentacao (checkpoint)

O endpoint `/stats` ja fornece:
- Taxa de respostas resolvidas pela base
- Taxa de uso do LLM
- Tempo medio de resposta de cada caminho

Util para a secao "Resultados e Discussao" do trabalho.

## Integrantes

- Gabriel Krepker
- Gabriel Monteiro
- Gustavo Lopes
- Joao Victor da Costa
- Rafael Lima Henriques
