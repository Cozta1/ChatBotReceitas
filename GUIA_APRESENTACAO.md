# ChefBot — Guia de Apresentação Acadêmica

**Disciplina:** Inteligência Artificial e Machine Learning  
**Projeto:** ChatBot de Receitas com Arquitetura Híbrida KB + LLM  
**Tecnologias:** Python · Flask · NLTK · scikit-learn · Transformers (Qwen2.5)

---

## 1. Visão Geral do Projeto

O **ChefBot** é um chatbot conversacional especializado em culinária brasileira. Ele combina duas abordagens clássicas em IA conversacional:

- **Agente 1 — NLP Clássico (NLTK + Naive Bayes):** classifica a intenção do usuário com rapidez e baixo custo computacional.
- **Agente 2 — LLM (Qwen2.5-3B-Instruct):** entra como fallback para perguntas complexas ou abertas que a base de conhecimento não consegue responder.

Essa arquitetura é chamada de **híbrida** porque prioriza respostas rápidas da base de conhecimento e usa o LLM somente quando necessário, economizando recursos e reduzindo latência.

---

## 2. Arquitetura do Sistema

```
Mensagem do Usuário
        │
        ▼
┌───────────────────┐
│  Pré-processamento │  ← normalizar, tokenizar, lematizar (NLTK)
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    State Machine                         │
│  estado: inicio │ sondagem │ passo_a_passo               │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐        ┌───────────────────────────┐
│  Naive Bayes        │──conf──▶   Base de Conhecimento    │
│  (scikit-learn)     │        │  110 receitas + 24 FAQs   │
└────────┬───────────┘        └───────────────────────────┘
         │ sem confiança
         ▼
┌────────────────────┐
│  LLM Fallback       │  ← Qwen2.5-3B-Instruct (fp16)
│  (Transformers)     │
└────────────────────┘
```

---

## 3. Principais Funcionalidades

| Funcionalidade | Descrição | Fonte |
|---|---|---|
| Busca por nome | Encontra receitas pelo nome digitado | `base` |
| Busca por ingredientes | Encontra receitas com os ingredientes citados | `base` |
| Busca por categoria | Filtra por sobremesa, lanche, bebida, etc. | `base` |
| Receita aleatória | Sugere uma receita ao acaso | `base` |
| Guia passo a passo | Conduz o usuário por cada etapa da receita | `base` |
| Perguntas de culinária | Responde dúvidas técnicas via FAQ ou LLM | `base-faq` / `llm` |
| Filtro de escopo | Recusa perguntas fora do tema culinário | `fora-escopo` |
| Estatísticas em tempo real | Mostra % base vs LLM e tempo médio de resposta | `/stats` |

---

## 4. Tecnologias Utilizadas

```
Backend
├── Python 3.10+
├── Flask 3.0.3          — servidor web e API REST
├── NLTK 3.8.1           — tokenização, lematização, stopwords
├── scikit-learn ≥1.3    — Naive Bayes, TF-IDF, similaridade de cosseno
├── Transformers 4.45+   — carregamento e inferência do LLM
└── PyTorch 2.2+         — backend do modelo (CUDA ou CPU)

Frontend
├── HTML5 + CSS3         — interface dark theme, acento laranja
└── JavaScript (vanilla) — gerenciamento de múltiplos chats em sessão
```

---

## 5. Código — Principais Trechos

### 5.1 Pré-processamento NLP (`nlp.py`)

O texto do usuário passa por normalização unicode, tokenização, remoção de stopwords e lematização antes de qualquer classificação.

```python
# nlp.py
def normalizar(texto: str) -> str:
    """Unicode NFKD + minúsculas para lidar com acentos do PT-BR."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()

def tokenizar(texto: str, filtrar_stopwords: bool = True) -> list[str]:
    tokens = word_tokenize(normalizar(texto), language="portuguese")
    tokens = [t for t in tokens if t.isalpha()]       # remove pontuação
    if filtrar_stopwords:
        tokens = [t for t in tokens if t not in _stops]
    return [_lem.lemmatize(t) for t in tokens]          # lematização
```

### 5.2 Classificador Naive Bayes (`nlp.py`)

O classificador usa **171 exemplos de treino** em **15 classes de intenção**. A confiança é avaliada comparando a probabilidade máxima com a média: se não for 1,8× maior, o sistema declina para o LLM.

```python
# nlp.py — treinamento (executado na importação do módulo)
_vectorizer = CountVectorizer(binary=True, tokenizer=tokenizar)
_clf = MultinomialNB()

X = _vectorizer.fit_transform([t for t, _ in TREINO])
y = [label for _, label in TREINO]
_clf.fit(X, y)

def detectar_intencao(texto: str, fator: float = 1.8) -> tuple[str, bool]:
    v = _vectorizer.transform([texto]).toarray()[0]
    probs = _clf.predict_proba(v.reshape(1, -1))[0]
    idx = int(probs.argmax())
    confiante = bool(probs[idx] >= probs.mean() * fator)
    return _clf.classes_[idx], confiante
```

**Por que Naive Bayes?**  
- Treinamento instantâneo (< 1 ms)  
- Funciona bem com dados esparsos (bag-of-words)  
- Retorna probabilidades que permitem medir confiança

### 5.3 Busca FAQ com TF-IDF (`base.py`)

Perguntas frequentes são indexadas com **TF-IDF bigramas**. A similaridade de cosseno determina se a pergunta do usuário é próxima o suficiente de um item do FAQ (limiar: 0,62).

```python
# base.py
_vectorizer_faq = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_faq_matriz = _vectorizer_faq.fit_transform(_faq_perguntas)

LIMIAR_FAQ = 0.62

def buscar_faq(pergunta: str) -> Optional[Resultado]:
    vec = _vectorizer_faq.transform([_prep_faq(pergunta)])
    sims = cosine_similarity(vec, _faq_matriz)[0]
    idx = int(sims.argmax())
    score = float(sims[idx])
    if score >= LIMIAR_FAQ:
        return Resultado(_faq_respostas[idx], score)
    return None
```

**Exemplo de FAQ indexado:**
```python
("como substituir ovo na receita", "Você pode substituir 1 ovo por ..."),
("quanto tempo dura comida na geladeira", "Carnes cozidas duram até 3-4 dias ..."),
```

### 5.4 Máquina de Estados (`chatbot.py`)

O núcleo do chatbot é uma máquina com **3 estados** que controla o fluxo da conversa:

```python
# chatbot.py — estrutura de sessão por chat_id
sessao_padrao = {
    "estado": "inicio",        # inicio | sondagem | passo_a_passo
    "receita": None,           # receita selecionada
    "candidatas": [],          # lista de até 3 receitas encontradas
    "candidata_atual": 0,      # índice na lista durante sondagem
    "passo_atual": 0,          # passo atual no guia
    "historico": [],           # últimas 6 trocas para contexto do LLM
}
```

**Fluxo principal de decisão:**

```python
# chatbot.py — função _gerar (simplificada)
def _gerar(mensagem: str, sessao: dict) -> Resposta:
    t0 = time.monotonic()

    # 1. Estados especiais têm prioridade
    if sessao["estado"] == "passo_a_passo":
        return _handle_passo(mensagem, sessao, t0)
    if sessao["estado"] == "sondagem":
        return _handle_sondagem(mensagem, sessao, t0)

    # 2. Perguntas abertas vão direto para FAQ ou LLM
    if parece_pergunta(texto):
        return _tenta_faq(mensagem, t0) or _r_llm(mensagem, sessao, t0)

    # 3. Naive Bayes classifica a intenção
    intent, confiante = detectar_intencao(mensagem)
    if not confiante:
        # sem confiança: LLM reclassifica
        intent = llm.classificar(mensagem, INTENCOES, sessao["historico"])

    # 4. Roteamento por intenção
    return _rotear(intent, mensagem, sessao, t0)
```

### 5.5 Integração com LLM (`llm.py`)

O modelo **Qwen2.5-3B-Instruct** é carregado em background no startup para não bloquear o servidor. Ele desempenha **três papéis distintos**:

```python
# llm.py — os três papéis do LLM

# PAPEL 1: Classificador de fallback (quando Naive Bayes não tem confiança)
def classificar(mensagem: str, intencoes: list, historico: list) -> str:
    prompt = f"Classifique em uma das intenções: {intencoes}\nMensagem: {mensagem}\nResposta (só o nome):"
    return _inferir(prompt, max_new_tokens=20).strip()

# PAPEL 2: Guarda de escopo (detecta perguntas fora do tema)
def dentro_do_escopo(mensagem: str, historico: list) -> bool:
    resp = _inferir(f"É sobre culinária? SIM ou NAO:\n{mensagem}", max_new_tokens=5)
    return "NAO" not in resp.upper()  # dúvida = permissão (evita falsos bloqueios)

# PAPEL 3: Geração de resposta aberta
def responder(pergunta: str, historico: list, contexto: str = "") -> str:
    msgs = _montar_historico(historico) + [{"role": "user", "content": pergunta}]
    return _pipe(msgs, max_new_tokens=400, temperature=0.3, top_p=0.85)
```

**System Prompt do LLM:**
```
Você é o ChefBot, um assistente conversacional em Português Brasileiro
ESPECIALIZADO EXCLUSIVAMENTE EM CULINÁRIA.
REGRA ABSOLUTA: Responda APENAS sobre cozinha e gastronomia.
Recuse outros tópicos com: "Desculpe, eu só consigo ajudar com culinária..."
```

### 5.6 API REST (`app.py`)

Interface HTTP simples com 4 endpoints:

```python
# app.py
@app.post("/chat")
def chat():
    dados = request.get_json()
    mensagem = dados.get("mensagem", "").strip()
    chat_id = dados.get("chat_id", "default")
    resposta = chatbot.responder(mensagem, chat_id)
    return jsonify(resposta)   # {resposta, fonte, tempo_ms}

@app.get("/stats")
def estatisticas():
    return jsonify(chatbot.stats())
    # {total, pct_base, pct_llm, tempo_medio_base_ms, tempo_medio_llm_ms, llm_status}
```

---

## 6. Fluxo de Conversa — Exemplo Completo

```
Usuário: "receitas com frango"
ChefBot: Encontrei 3 receitas com frango:
         1. Frango Assado com Ervas
         2. Estrogonofe de Frango  
         3. Frango à Parmegiana
         Qual você quer saber mais?        [fonte: base]

Usuário: "a segunda"
ChefBot: Estrogonofe de Frango
         Tempo: 35 min | Porções: 4 | Dificuldade: Fácil
         Quer que eu te guie passo a passo?  [fonte: base]

Usuário: "sim"
ChefBot: Ingredientes: 500g frango, 1 cebola...
         — Passo 1 de 6 —
         Tempere o frango com sal e pimenta...
         Diga "pronto" quando terminar.      [fonte: base]

Usuário: "como saber se o frango está cozido?"
ChefBot: O frango está cozido quando a temperatura
         interna atinge 74°C. Sem termômetro, corte
         a parte mais grossa: sem rosa e sucos claros. [fonte: base-faq]

Usuário: "pronto"
ChefBot: — Passo 2 de 6 —
         Refogue a cebola na manteiga...     [fonte: base]
```

---

## 7. Dataset de Receitas

O banco de dados contém **110 receitas brasileiras** estruturadas:

```python
# receitas.py — schema de cada receita
{
    "nome": "Brigadeiro Tradicional",
    "ingredientes": ["1 lata de leite condensado", "1 colher de manteiga", ...],
    "categoria": "sobremesa",       # prato principal | sobremesa | lanche | bebida | acompanhamento
    "tempo": "30 minutos",
    "porcoes": 20,
    "dificuldade": "facil",         # facil | medio | dificil
    "instrucoes": [
        "Misture o leite condensado e a manteiga numa panela.",
        "Cozinhe em fogo baixo, mexendo sempre...",
        ...
    ]
}
```

**Distribuição por categoria:**

| Categoria | Exemplos |
|---|---|
| Prato Principal | Feijoada, Moqueca, Estrogonofe, Arroz Carreteiro |
| Sobremesa | Brigadeiro, Pudim, Bolo de Cenoura, Torta de Chocolate |
| Lanche | Coxinha, Pão de Queijo, Tapioca, Pastel |
| Bebida | Limonada, Vitamina, Caipirinha, Café Coado |
| Acompanhamento | Farofa, Arroz, Feijão, Purê de Batata |

---

## 8. Métricas e Desempenho

O endpoint `/stats` retorna métricas em tempo real:

```json
{
  "total": 47,
  "pct_base": 78.7,
  "pct_llm": 21.3,
  "tempo_medio_base_ms": 12.4,
  "tempo_medio_llm_ms": 3840.0,
  "llm_status": "pronto (CUDA · RTX 3060)"
}
```

**Interpretação esperada:**
- ~80% das respostas vêm da base (rápido, < 15 ms)
- ~20% usam o LLM (lento, ~3-5 s em GPU)
- O design híbrido garante boa experiência mesmo sem GPU dedicada

---

## 9. Como Executar o Projeto

```bash
# 1. Clonar e instalar dependências
git clone https://github.com/cozta1/chatbotreceitas
cd ChatBotReceitas
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Iniciar o servidor
python app.py
# → Acesse http://127.0.0.1:5000

# 3. Rodar testes manuais (sem LLM)
python testar.py --rapido

# 4. Rodar todos os testes (requer modelo carregado)
python testar.py
```

**Primeira execução:** o modelo Qwen2.5-3B (~5 GB) é baixado automaticamente do Hugging Face.

---

## 10. Pontos de Destaque Acadêmico

### Escolhas de Design

| Decisão | Justificativa |
|---|---|
| Naive Bayes como classificador | Alta velocidade, bom desempenho com dados esparsos, probabilidades interpretáveis |
| TF-IDF bigramas para FAQ | Captura co-ocorrências ("leite condensado") que unigramas perderiam |
| Threshold de confiança (1,8×) | Evita que o classificador "force" uma intenção quando o texto é ambíguo |
| Fast-path de vocabulário culinário | Evita falso-bloqueio de perguntas legítimas pelo guard LLM |
| LLM em fp16 | Reduz uso de VRAM de ~12 GB (fp32) para ~6 GB sem perda significativa de qualidade |
| Preload assíncrono do LLM | O servidor responde imediatamente; o modelo carrega em background |

### Limitações Conhecidas

- O LLM em CPU é muito lento (~30-60 s por resposta) — GPU é recomendada
- O classificador Naive Bayes pode errar em frases muito curtas ou ambíguas
- A base de FAQ tem apenas 24 entradas; perguntas muito específicas caem no LLM
- Sem persistência: o histórico de conversa é perdido ao reiniciar o servidor

### Possíveis Melhorias Futuras

- Persistência de sessão com Redis ou banco de dados
- Expansão do FAQ com embeddings semânticos (SBERT)
- Fine-tuning do LLM no domínio culinário brasileiro
- Interface com voz (speech-to-text / TTS)
- Avaliação automática com métricas BLEU/ROUGE

---

*Trabalho desenvolvido para a disciplina de Inteligência Artificial e Machine Learning.*
