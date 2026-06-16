from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm

W, H = A4
LARANJA = colors.HexColor("#E87C2B")
CINZA_ESCURO = colors.HexColor("#1E1E1E")
CINZA_MEDIO = colors.HexColor("#2D2D2D")
CINZA_CLARO = colors.HexColor("#444444")
BRANCO = colors.white
CINZA_TEXTO = colors.HexColor("#CCCCCC")
VERDE = colors.HexColor("#4CAF82")
AZUL = colors.HexColor("#5B9BD5")

def cabecalho(c, titulo, pagina, total):
    c.setFillColor(CINZA_ESCURO)
    c.rect(0, H - 1.6*cm, W, 1.6*cm, fill=True, stroke=False)
    c.setFillColor(LARANJA)
    c.rect(0, H - 1.6*cm, 0.4*cm, 1.6*cm, fill=True, stroke=False)
    c.setFillColor(BRANCO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.8*cm, H - 1.1*cm, titulo)
    c.setFont("Helvetica", 8)
    c.setFillColor(CINZA_TEXTO)
    c.drawRightString(W - 0.7*cm, H - 1.1*cm, f"ChefBot  ·  {pagina}/{total}")

def rodape(c):
    c.setFillColor(CINZA_ESCURO)
    c.rect(0, 0, W, 0.8*cm, fill=True, stroke=False)
    c.setFillColor(CINZA_TEXTO)
    c.setFont("Helvetica", 7)
    c.drawCentredString(W/2, 0.25*cm, "Chatbot de Receitas  ·  IA & Machine Learning")

def caixa_codigo(c, x, y, largura, altura):
    c.setFillColor(CINZA_ESCURO)
    c.setStrokeColor(LARANJA)
    c.roundRect(x, y, largura, altura, 4, fill=True, stroke=True)

def texto_codigo(c, linhas, x, y, tamanho=7.5):
    c.setFont("Courier", tamanho)
    espaco = tamanho * 1.45
    for i, linha in enumerate(linhas):
        cor = BRANCO
        stripped = linha.strip()
        if stripped.startswith("#"):
            cor = colors.HexColor("#6A9955")
        elif stripped.startswith("def ") or stripped.startswith("class "):
            cor = colors.HexColor("#DCDCAA")
        elif any(stripped.startswith(k) for k in ("return", "if ", "for ", "import")):
            cor = colors.HexColor("#C586C0")
        elif '"""' in stripped or stripped.startswith('"') or stripped.startswith("'"):
            cor = colors.HexColor("#CE9178")
        c.setFillColor(cor)
        c.drawString(x, y - i * espaco, linha)

def bullet(c, texto, x, y, tamanho=9.5, cor=BRANCO):
    c.setFillColor(LARANJA)
    c.setFont("Helvetica-Bold", tamanho)
    c.drawString(x, y, "•")
    c.setFillColor(cor)
    c.setFont("Helvetica", tamanho)
    c.drawString(x + 0.4*cm, y, texto)

def tag(c, texto, x, y, bg=LARANJA, fg=BRANCO):
    c.setFont("Helvetica-Bold", 7.5)
    largura = c.stringWidth(texto, "Helvetica-Bold", 7.5) + 10
    c.setFillColor(bg)
    c.roundRect(x, y - 2, largura, 12, 3, fill=True, stroke=False)
    c.setFillColor(fg)
    c.drawString(x + 5, y + 1, texto)
    return largura + 6

# ---------- GERAÇÃO ----------

pdf = canvas.Canvas("/home/user/ChatBotReceitas/APRESENTACAO_ChefBot.pdf", pagesize=A4)
TOTAL = 8

# =========================================================
# SLIDE 1 — CAPA
# =========================================================
pdf.setFillColor(CINZA_ESCURO)
pdf.rect(0, 0, W, H, fill=True, stroke=False)

pdf.setFillColor(LARANJA)
pdf.rect(0, H*0.42, W, 0.25*cm, fill=True, stroke=False)
pdf.rect(0, H*0.42 - 0.15*cm, 3*cm, 0.15*cm, fill=True, stroke=False)

pdf.setFillColor(BRANCO)
pdf.setFont("Helvetica-Bold", 38)
pdf.drawCentredString(W/2, H*0.58, "ChefBot")

pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 13)
pdf.drawCentredString(W/2, H*0.52, "Chatbot de Receitas com Arquitetura Híbrida")

pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 9.5)
pdf.drawCentredString(W/2, H*0.46, "NLP Clássico  +  Large Language Model  |  Python · Flask · NLTK · Transformers")

pdf.setFillColor(CINZA_CLARO)
pdf.rect(2*cm, H*0.32, W - 4*cm, 0.05*cm, fill=True, stroke=False)

itens = ["Disciplina: Inteligência Artificial e Machine Learning",
         "Trabalho Final de Semestre"]
for i, it in enumerate(itens):
    pdf.setFillColor(CINZA_TEXTO)
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(W/2, H*0.285 - i*0.5*cm, it)

pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 8)
pdf.drawCentredString(W/2, 1.2*cm, "~10 minutos de apresentação")

pdf.showPage()

# =========================================================
# SLIDE 2 — O QUE É O PROJETO
# =========================================================
cabecalho(pdf, "O que é o projeto?", 1, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "O que é o ChefBot?")

Y -= 0.5*cm
pdf.setFillColor(CINZA_CLARO)
pdf.rect(1.5*cm, Y, W - 3*cm, 0.04*cm, fill=True, stroke=False)

Y -= 0.7*cm
bullets_o_que = [
    "Chatbot conversacional especializado em culinária brasileira",
    "Responde perguntas, busca receitas e guia o usuário passo a passo",
    "Combina NLP clássico (rápido) com LLM (inteligente) na mesma pipeline",
    "Interface web simples, API REST e suporte a múltiplas conversas",
]
for b in bullets_o_que:
    bullet(pdf, b, 1.8*cm, Y, tamanho=10)
    Y -= 0.65*cm

Y -= 0.4*cm
pdf.setFillColor(CINZA_MEDIO)
pdf.roundRect(1.5*cm, Y - 2.2*cm, W - 3*cm, 2.5*cm, 6, fill=True, stroke=False)
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 9)
pdf.drawString(2*cm, Y - 0.3*cm, "Exemplo de interação:")
linhas_ex = [
    '  Usuário → "receitas com frango"',
    '  ChefBot → Lista 3 receitas encontradas',
    '  Usuário → "a primeira"',
    '  ChefBot → Detalhes + "quer guia passo a passo?"',
    '  Usuário → "sim"  →  ChefBot guia cada etapa até o prato ficar pronto',
]
pdf.setFont("Helvetica", 9)
for i, l in enumerate(linhas_ex):
    cor = LARANJA if "Usuário" in l else colors.HexColor("#7EC8A0")
    pdf.setFillColor(cor)
    pdf.drawString(2*cm, Y - 0.7*cm - i * 0.42*cm, l)

pdf.showPage()

# =========================================================
# SLIDE 3 — ARQUITETURA
# =========================================================
cabecalho(pdf, "Arquitetura do Sistema", 2, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Como o sistema decide a resposta?")

# Caixas do fluxo
BOX_W = 4.8*cm
BOX_H = 1.1*cm
GAP = 0.6*cm
START_X = (W - (3 * BOX_W + 2 * GAP)) / 2
BY = H - 5.5*cm

def caixa_fluxo(c, x, y, texto, bg, linha2=""):
    c.setFillColor(bg)
    c.roundRect(x, y, BOX_W, BOX_H, 5, fill=True, stroke=False)
    c.setFillColor(BRANCO)
    c.setFont("Helvetica-Bold", 8.5)
    cy = y + BOX_H/2 + (3 if not linha2 else 5)
    c.drawCentredString(x + BOX_W/2, cy, texto)
    if linha2:
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#DDDDDD"))
        c.drawCentredString(x + BOX_W/2, y + BOX_H/2 - 5, linha2)

def seta(c, x1, y1, x2, y2):
    c.setStrokeColor(LARANJA)
    c.setLineWidth(1.5)
    c.line(x1, y1, x2, y2)
    c.setFillColor(LARANJA)
    c.setStrokeColor(LARANJA)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    tam = 6
    px1 = x2 - tam * math.cos(ang - 0.4)
    py1 = y2 - tam * math.sin(ang - 0.4)
    px2 = x2 - tam * math.cos(ang + 0.4)
    py2 = y2 - tam * math.sin(ang + 0.4)
    p = pdf.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(px1, py1)
    p.lineTo(px2, py2)
    p.close()
    pdf.drawPath(p, fill=True, stroke=False)

caixa_fluxo(pdf, START_X, BY, "Mensagem do Usuário", colors.HexColor("#3A3A6A"))
seta(pdf, START_X + BOX_W, BY + BOX_H/2, START_X + BOX_W + GAP, BY + BOX_H/2)
caixa_fluxo(pdf, START_X + BOX_W + GAP, BY, "Naive Bayes", colors.HexColor("#2E5C3E"), "NLP clássico")
seta(pdf, START_X + 2*(BOX_W + GAP), BY + BOX_H/2, START_X + 2*(BOX_W + GAP) + GAP, BY + BOX_H/2)
caixa_fluxo(pdf, START_X + 2*(BOX_W + GAP), BY, "Base de Conhecimento", colors.HexColor("#3A5A2E"), "110 receitas + FAQ")

# Ramo LLM
LLM_Y = BY - 2.2*cm
caixa_fluxo(pdf, START_X + BOX_W + GAP, LLM_Y, "LLM Fallback", colors.HexColor("#5C3A2E"), "Qwen2.5-3B")
pdf.setStrokeColor(LARANJA)
pdf.setLineWidth(1.2)
MX = START_X + BOX_W + GAP + BOX_W/2
pdf.line(MX, BY, MX, LLM_Y + BOX_H)
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 7.5)
pdf.drawCentredString(MX - 0.8*cm, LLM_Y + BOX_H + 0.35*cm, "sem confiança")

# Legenda
Y = LLM_Y - 1.2*cm
pares = [
    (colors.HexColor("#2E5C3E"), "Naive Bayes — classifica intenções em < 5 ms"),
    (colors.HexColor("#3A5A2E"), "Base KB — 110 receitas + 24 FAQs (TF-IDF)"),
    (colors.HexColor("#5C3A2E"), "LLM Fallback — perguntas abertas, ~3 s em GPU"),
]
for bg, texto in pares:
    pdf.setFillColor(bg)
    pdf.roundRect(1.5*cm, Y, 0.4*cm, 0.3*cm, 2, fill=True, stroke=False)
    pdf.setFillColor(BRANCO)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(2.1*cm, Y + 0.05*cm, texto)
    Y -= 0.5*cm

# Estado machine note
Y -= 0.3*cm
pdf.setFillColor(colors.HexColor("#333355"))
pdf.roundRect(1.5*cm, Y - 0.7*cm, W - 3*cm, 1.1*cm, 5, fill=True, stroke=False)
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 8.5)
pdf.drawString(1.9*cm, Y - 0.1*cm, "Máquina de estados:")
pdf.setFillColor(BRANCO)
pdf.setFont("Helvetica", 8.5)
pdf.drawString(1.9*cm, Y - 0.5*cm, "inicio  →  sondagem (lista de receitas)  →  passo_a_passo (guia)")

pdf.showPage()

# =========================================================
# SLIDE 4 — FUNCIONALIDADES
# =========================================================
cabecalho(pdf, "Funcionalidades Principais", 3, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "O que o ChefBot sabe fazer?")

Y -= 0.7*cm
items = [
    ("Busca por nome", "\"receita de feijoada\"", VERDE),
    ("Busca por ingredientes", "\"tenho frango e batata\"", VERDE),
    ("Busca por categoria", "\"quero uma sobremesa\"", VERDE),
    ("Receita aleatória", "\"me surpreenda\"", VERDE),
    ("Guia passo a passo", "guia cada etapa até o fim", AZUL),
    ("Perguntas de culinária", "\"como amaciar carne?\" → FAQ ou LLM", AZUL),
    ("Filtro de escopo", "recusa perguntas fora da culinária", colors.HexColor("#C05050")),
    ("Estatísticas", "mostra % base vs LLM em tempo real", LARANJA),
]
for func, desc, cor in items:
    pdf.setFillColor(cor)
    pdf.roundRect(1.5*cm, Y - 0.05*cm, 0.28*cm, 0.28*cm, 2, fill=True, stroke=False)
    pdf.setFillColor(BRANCO)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(2.0*cm, Y, func)
    pdf.setFillColor(CINZA_TEXTO)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(7.0*cm, Y, f"→  {desc}")
    Y -= 0.58*cm

Y -= 0.4*cm
pdf.setFillColor(CINZA_MEDIO)
pdf.roundRect(1.5*cm, Y - 1.0*cm, W - 3*cm, 1.3*cm, 5, fill=True, stroke=False)
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 8.5)
pdf.drawString(1.9*cm, Y - 0.15*cm, "Tipos de fonte de resposta rastreados:")
xp = 1.9*cm
for t, bg in [("base", VERDE), ("base-faq", AZUL), ("llm", colors.HexColor("#C07030")), ("fora-escopo", colors.HexColor("#C05050"))]:
    xp += tag(pdf, t, xp, Y - 0.6*cm, bg=bg) + 0.1*cm

pdf.showPage()

# =========================================================
# SLIDE 5 — CÓDIGO: NLP + NAIVE BAYES
# =========================================================
cabecalho(pdf, "Código — NLP e Classificador", 4, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Classificação de Intenções (nlp.py)")

Y -= 0.55*cm
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 9)
pdf.drawString(1.5*cm, Y, "O texto do usuário é normalizado, tokenizado e classificado com Naive Bayes.")

Y -= 0.6*cm
caixa_codigo(pdf, 1.2*cm, Y - 4.1*cm, W - 2.4*cm, 4.4*cm)
codigo1 = [
    "# 1. Normaliza: remove acentos, converte para minúsculas",
    "def normalizar(texto):",
    '    return unicodedata.normalize("NFKD", texto)',
    '           .encode("ascii", "ignore").decode().lower()',
    "",
    "# 2. Tokeniza: remove pontuação, stopwords e lematiza",
    "def tokenizar(texto):",
    "    tokens = word_tokenize(normalizar(texto), language='portuguese')",
    "    tokens = [t for t in tokens if t.isalpha()]",
    "    return [lemmatizer.lemmatize(t) for t in tokens if t not in stops]",
    "",
    "# 3. Classificador Naive Bayes — 171 exemplos, 15 intenções",
    "def detectar_intencao(texto, fator=1.8):",
    "    probs = clf.predict_proba(vectorizer.transform([texto]))[0]",
    "    idx = probs.argmax()",
    "    confiante = probs[idx] >= probs.mean() * fator  # threshold 1.8x",
    "    return clf.classes_[idx], confiante",
]
texto_codigo(pdf, codigo1, 1.5*cm, Y - 0.35*cm, tamanho=7.5)

Y -= 4.7*cm
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 8.5)
txt = "Se confiança < threshold 1,8×, o LLM reclassifica a intenção como fallback."
pdf.drawString(1.5*cm, Y, txt)

Y -= 0.55*cm
bullet(pdf, "Naive Bayes: treinamento < 1 ms, resposta < 5 ms", 1.8*cm, Y, tamanho=9)
Y -= 0.45*cm
bullet(pdf, "Threshold evita classificações forçadas em textos ambíguos", 1.8*cm, Y, tamanho=9)
Y -= 0.45*cm
bullet(pdf, "171 exemplos de treino, 15 classes de intenção", 1.8*cm, Y, tamanho=9)

pdf.showPage()

# =========================================================
# SLIDE 6 — CÓDIGO: FAQ TF-IDF + LLM
# =========================================================
cabecalho(pdf, "Código — FAQ e LLM", 5, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Busca FAQ (TF-IDF) e Fallback LLM")

Y -= 0.55*cm
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 9)
pdf.drawString(1.5*cm, Y, "Perguntas técnicas de culinária vão ao FAQ; o LLM responde o que o FAQ não cobre.")

Y -= 0.65*cm
caixa_codigo(pdf, 1.2*cm, Y - 3.0*cm, W - 2.4*cm, 3.3*cm)
codigo2 = [
    "# FAQ: 24 pares pergunta-resposta indexados com TF-IDF bigramas",
    "vectorizer_faq = TfidfVectorizer(ngram_range=(1, 2))",
    "faq_matriz = vectorizer_faq.fit_transform(perguntas_faq)",
    "",
    "def buscar_faq(pergunta):",
    "    vec = vectorizer_faq.transform([pergunta])",
    "    sims = cosine_similarity(vec, faq_matriz)[0]",
    "    score = sims.max()",
    "    if score >= 0.62:            # limiar de similaridade",
    "        return respostas_faq[sims.argmax()]",
    "    return None                  # sem match → vai para o LLM",
]
texto_codigo(pdf, codigo2, 1.5*cm, Y - 0.35*cm, tamanho=7.8)

Y -= 3.7*cm
caixa_codigo(pdf, 1.2*cm, Y - 2.5*cm, W - 2.4*cm, 2.8*cm)
codigo3 = [
    "# LLM: Qwen2.5-3B-Instruct — 3 papéis distintos",
    "def classificar(msg):   # Papel 1: reclassifica intenção",
    "    return inferir(f'Classifique: {INTENCOES}\\n{msg}', max_tokens=20)",
    "",
    "def dentro_do_escopo(msg):   # Papel 2: guarda de escopo",
    "    resp = inferir(f'É culinária? SIM ou NAO:\\n{msg}', max_tokens=5)",
    "    return 'NAO' not in resp.upper()  # dúvida = permitir",
    "",
    "def responder(pergunta, historico):  # Papel 3: resposta aberta",
    "    return pipeline(msgs, max_new_tokens=400, temperature=0.3)",
]
texto_codigo(pdf, codigo3, 1.5*cm, Y - 0.35*cm, tamanho=7.8)

pdf.showPage()

# =========================================================
# SLIDE 7 — DATASET + API
# =========================================================
cabecalho(pdf, "Dataset e API REST", 6, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Dados e Interface")

# Colunas
COL = W/2 - 1.2*cm
Y -= 0.6*cm

# Coluna esquerda: dataset
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica-Bold", 10)
pdf.drawString(1.5*cm, Y, "Base de Receitas (receitas.py)")
Y -= 0.45*cm

caixa_codigo(pdf, 1.2*cm, Y - 3.4*cm, COL, 3.7*cm)
schema = [
    "{",
    '  "nome": "Brigadeiro",',
    '  "categoria": "sobremesa",',
    '  "tempo": "30 minutos",',
    '  "dificuldade": "facil",',
    '  "porcoes": 20,',
    '  "ingredientes": [...],',
    '  "instrucoes": [...]',
    "}",
]
texto_codigo(pdf, schema, 1.5*cm, Y - 0.4*cm, tamanho=7.5)

YL = Y
Y2 = YL
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica-Bold", 10)
pdf.drawString(W/2 + 0.3*cm, Y2, "API REST (app.py)")
Y2 -= 0.45*cm

caixa_codigo(pdf, W/2, Y2 - 3.4*cm, COL, 3.7*cm)
api_linhas = [
    "GET  /",
    "     Interface web do chatbot",
    "",
    "POST /chat",
    "  { mensagem, chat_id }",
    "  → { resposta, fonte, tempo_ms }",
    "",
    "GET  /stats",
    "  → { pct_base, pct_llm,",
    "      tempo_base_ms, llm_status }",
    "",
    "POST /chat/reset",
    "     Limpa sessão do chat",
]
texto_codigo(pdf, api_linhas, W/2 + 0.35*cm, Y2 - 0.4*cm, tamanho=7.5)

Y = YL - 4.1*cm
pdf.setFillColor(BRANCO)
pdf.setFont("Helvetica-Bold", 9.5)
pdf.drawString(1.5*cm, Y, "110 receitas brasileiras:")
Y -= 0.5*cm

cats = [
    ("Prato Principal", "Feijoada, Moqueca, Estrogonofe"),
    ("Sobremesa", "Brigadeiro, Pudim, Bolo de Cenoura"),
    ("Lanche", "Coxinha, Pão de Queijo, Tapioca"),
    ("Bebida", "Caipirinha, Limonada, Vitamina"),
    ("Acompanhamento", "Arroz, Feijão, Farofa, Purê"),
]
for cat, ex in cats:
    pdf.setFillColor(LARANJA)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(1.8*cm, Y, cat)
    pdf.setFillColor(CINZA_TEXTO)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(5.5*cm, Y, f"→  {ex}")
    Y -= 0.42*cm

pdf.showPage()

# =========================================================
# SLIDE 8 — COMO RODAR + CONCLUSÃO
# =========================================================
cabecalho(pdf, "Como Executar e Conclusão", 7, TOTAL)
rodape(pdf)

Y = H - 3.2*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Como Executar")

Y -= 0.6*cm
caixa_codigo(pdf, 1.2*cm, Y - 2.8*cm, W - 2.4*cm, 3.0*cm)
cmds = [
    "# Instalar dependências",
    "pip install -r requirements.txt",
    "",
    "# Iniciar o servidor",
    "python app.py    →  http://127.0.0.1:5000",
    "",
    "# Testar sem LLM (rápido)",
    "python testar.py --rapido",
    "",
    "# 1ª execução: baixa modelo Qwen2.5-3B (~5 GB) automaticamente",
]
texto_codigo(pdf, cmds, 1.5*cm, Y - 0.35*cm, tamanho=8)

Y -= 3.5*cm
pdf.setFillColor(LARANJA)
pdf.setFont("Helvetica-Bold", 15)
pdf.drawString(1.5*cm, Y, "Conclusão")
Y -= 0.6*cm

pontos = [
    ("Objetivo", "Chatbot culinário robusto com dois agentes colaborativos"),
    ("NLP Clássico", "Naive Bayes + TF-IDF garantem respostas rápidas (~10 ms)"),
    ("LLM", "Qwen2.5-3B cobre perguntas abertas (~3 s em GPU)"),
    ("Resultado", "~80 % das respostas via base (sem LLM), latência baixa"),
    ("Aprendizado", "Arquitetura híbrida: eficiência + inteligência sem GPU dedicada"),
]
for titulo, descricao in pontos:
    pdf.setFillColor(LARANJA)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(1.8*cm, Y, f"{titulo}:")
    pdf.setFillColor(BRANCO)
    pdf.setFont("Helvetica", 9.5)
    pdf.drawString(4.8*cm, Y, descricao)
    Y -= 0.55*cm

Y -= 0.3*cm
pdf.setFillColor(CINZA_MEDIO)
pdf.roundRect(1.5*cm, Y - 0.8*cm, W - 3*cm, 1.1*cm, 5, fill=True, stroke=False)
pdf.setFillColor(CINZA_TEXTO)
pdf.setFont("Helvetica", 9)
pdf.drawCentredString(W/2, Y - 0.1*cm, "Repositório:  github.com/cozta1/chatbotreceitas")
pdf.setFont("Helvetica-Bold", 9)
pdf.setFillColor(LARANJA)
pdf.drawCentredString(W/2, Y - 0.52*cm, "Obrigado!")

pdf.showPage()
pdf.save()
print("PDF gerado: APRESENTACAO_ChefBot.pdf")
