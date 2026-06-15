import random
import time

import base
import llm
from receitas import RECEITAS
from nlp import detectar_intencao, normalizar, tokenizar


# Mensagens e sinais
SAUDACOES = [
    "Ola! Sou o ChefBot. Diga 'ajuda' para ver o que faco. O que vamos cozinhar?",
    "Fala! ChefBot na area. Quer ajuda? Digite 'ajuda'.",
    "Boa! Especialista em receitas brasileiras. O que vai preparar hoje?",
]
DESPEDIDAS = ["Tchau! Bom apetite!", "Ate mais!", "Ate logo, bom prato!"]
AGRADECIMENTOS = ["De nada!", "Por nada, qualquer duvida e so chamar.", "Disponha!"]
AJUDA = (
    "Posso te ajudar com:\n"
    "  - INGREDIENTES: 'tenho frango, alho e cebola'\n"
    "  - NOME: 'receita de feijoada'\n"
    "  - CATEGORIA: prato principal, sobremesa, lanche, bebida, acompanhamento\n"
    "  - ALEATORIA: 'me surpreenda'\n"
    "  - DUVIDAS: tecnicas, substituicoes, conservacao (uso a base ou consulto o LLM)\n"
    "  - PASSO A PASSO: apos escolher, 'pronto' avanca e 'manda tudo' entrega completa"
)
ENCORAJAMENTOS = ["Otimo!", "Muito bem!", "Perfeito!", "Mandou bem!"]

SINAIS = {
    "cancelar": ("cancelar", "outra receita", "esquece", "desisti", "recomecar"),
    "completo": ("manda tudo", "receita completa", "quero tudo", "manda direto"),
    "confirmar": ("pronto", "feito", "ja fiz", "proximo", "continua",
                  "terminei", "ok feito"),
}
SIM = ("sim", "claro", "pode", "vamos", "ok", "bora", "topo", "aceito")
NAO = ("nao", "nop", "outro", "outra", "muda", "passa", "proxima", "proximo")
INTERROGATIVAS = (
    "como ", "qual ", "quais ", "porque ", "por que ", "o que ", "oq ",
    "o q ", "para que ", "quando ", "onde ", "quanto ", "quem ",
    "me explica", "me explique", "explique ", "explica ",
    "me fale", "fale sobre", "me conte", "conte sobre",
    "me de uma resposta", "me da uma resposta", "continue dissertando",
    "continue falando", "me ajuda a entender",
)

# Intencoes que o Agente 1 (LLM classificador) pode devolver
INTENCOES = [
    "saudacao", "despedida", "agradecimento", "ajuda", "receita_aleatoria",
    "buscar_por_nome", "buscar_por_ingredientes",
    "categoria:sobremesa", "categoria:prato principal", "categoria:lanche",
    "categoria:bebida", "categoria:acompanhamento",
]

# Intencoes que representam uma NOVA busca de receita (usadas para escapar da
# sondagem quando o usuario muda de ideia em vez de escolher um numero).
BUSCA_INTENTS = {"receita_aleatoria", "buscar_por_nome", "buscar_por_ingredientes"}

ESTATISTICAS = {"total": 0, "base": 0, "llm": 0, "tempo_base_ms": 0.0, "tempo_llm_ms": 0.0}


# Sessao
def nova_sessao():
    return {"estado": "inicio", "receita": None, "candidatas": [],
            "candidata_atual": 0, "pergunta_tipo": None, "passo_atual": 0,
            "historico": []}


def resetar(s):
    # historico nao e zerado, mantem o contexto da conversa
    s.update(estado="inicio", receita=None, candidatas=[], candidata_atual=0,
             pergunta_tipo=None, passo_atual=0)


def contem(texto, lista):
    return any(p in texto for p in lista)


def parece_pergunta(t):
    return t.endswith("?") or t.startswith(INTERROGATIVAS)


ORDINAIS = {"primeira": 1, "primeiro": 1, "segunda": 2, "segundo": 2,
            "terceira": 3, "terceiro": 3, "ultima": -1, "ultimo": -1}


def _extrair_numero(texto, n):
    # Extrai a escolha (1..n) de frases como 'quero a 1' ou 'a segunda'. Retorna None se nenhum numero valido aparecer.
    for palavra, num in ORDINAIS.items():
        if palavra in texto:
            return n if num == -1 else (num if num <= n else None)
    for tok in texto.split():
        if tok.isdigit() and 1 <= int(tok) <= n:
            return int(tok)
    return None


def _r_base(texto, t0, marcador="base"):
    dt = (time.time() - t0) * 1000
    ESTATISTICAS["base"] += 1
    ESTATISTICAS["total"] += 1
    ESTATISTICAS["tempo_base_ms"] += dt
    return {"texto": texto, "fonte": marcador, "tempo_ms": round(dt, 1)}


def _r_llm(pergunta, sessao, contexto=""):
    t0 = time.time()
    historico = sessao.get("historico", [])
    # guarda de escopo: so respondemos sobre culinaria/alimentacao.
    # fast-path: se a mensagem encosta no vocabulario culinario da base, e do
    # tema e nao consultamos o guardiao LLM (evita falso positivo do 3B).
    # so bloqueia quando vocabulario E guardiao concordam que esta fora.
    if not base.parece_culinario(pergunta) and not llm.dentro_do_escopo(pergunta, historico):
        return _r_base(llm.FORA_DO_ESCOPO, t0, "fora-escopo")
    texto = llm.responder(pergunta, historico=historico, contexto=contexto)
    dt = (time.time() - t0) * 1000
    ESTATISTICAS["llm"] += 1
    ESTATISTICAS["total"] += 1
    ESTATISTICAS["tempo_llm_ms"] += dt
    return {"texto": texto, "fonte": "llm", "tempo_ms": round(dt, 1)}



# Fluxo (sondagem -> passo a passo -> conclusao)

# Modelos de introducao da lista (estaticos, reutilizaveis, sem custo de LLM)
INTRO_PADRAO = "Encontrei estas opcoes:"
INTRO_NOME = "Encontrei estas receitas pra voce:"
INTRO_INGREDIENTES = "Com esses ingredientes da pra fazer alguma destas:"
INTRO_ALEATORIA = "Separei estas receitas pra te surpreender:"
CATEGORIA_ALVO = {
    "sobremesa": "uma sobremesa", "prato principal": "um prato principal",
    "lanche": "um lanche", "bebida": "uma bebida",
    "acompanhamento": "um acompanhamento",
}


def intro_categoria(cat):
    return (f"Para {CATEGORIA_ALVO.get(cat, 'isso')} voce pode fazer "
            "alguma das receitas que tenho a seguir:")


def _formatar_opcoes(cands, intro=INTRO_PADRAO):
    linhas = "\n".join(
        f"{i+1}. **{r['nome']}** "
        f"({r['tempo']}, {r.get('dificuldade','?').title()})"
        for i, r in enumerate(cands))
    return (intro + "\n\n" + linhas +
            "\n\nQual voce quer? Diga o numero (ex.: 'quero a receita 1').")


def iniciar_sondagem(candidatas, s, intro=INTRO_PADRAO):
    s.update(candidatas=list(candidatas), candidata_atual=0,
             pergunta_tipo="escolha", estado="sondagem")
    return _formatar_opcoes(s["candidatas"], intro)


def cancelar(s):
    resetar(s)
    return "Cancelei. O que voce gostaria de cozinhar agora?"


def _fechamento(s):
    # Mensagem de encerramento (sem pedir feedback) + reseta a sessao.
    resetar(s)
    return "Espero que tenha ficado bom!\n\nO que quer fazer agora?"


def entregar_completa(s):
    receita = base.formatar_receita(s["receita"])
    return f"Aqui esta a receita completa:{receita}\n\n{_fechamento(s)}"


def mostrar_passo(s):
    i, total = s["passo_atual"], len(s["receita"]["instrucoes"])
    passo = f"**Passo {i+1} de {total}**\n{s['receita']['instrucoes'][i]}"
    if i + 1 >= total:  # ultimo passo: encerra junto, sem pedir confirmacao
        return f"{passo}\n\n{_fechamento(s)}"
    return f"{passo}\n\nQuando terminar, e so dizer \"pronto\"."


def proximo_passo(s):
    s["passo_atual"] += 1
    return random.choice(ENCORAJAMENTOS) + "\n\n" + mostrar_passo(s)


def _quer_nova_busca(texto, mensagem, s):
    """Durante a sondagem, detecta se o usuario pediu OUTRA coisa em vez de
    escolher um numero (ex.: 'quero algo para beber'). So troca de fluxo com
    intencao de busca confiante e quando nao ha escolha valida na mensagem."""
    if _extrair_numero(texto, len(s["candidatas"])) is not None:
        return False  # e uma escolha, deixa a sondagem tratar
    if contem(texto, SIM) or contem(texto, NAO):
        return False  # sim/nao pertencem ao fluxo da sondagem
    intent, conf = detectar_intencao(mensagem)
    return conf and (intent in BUSCA_INTENTS or intent.startswith("categoria:"))


def resposta_sondagem(texto, s):
    tipo = s["pergunta_tipo"]
    sim, nao = contem(texto, SIM), contem(texto, NAO)

    if tipo == "escolha":
        escolha = _extrair_numero(texto, len(s["candidatas"]))
        if escolha is None:
            if nao and not sim:
                s["estado"] = "inicio"
                return "Sem problema! O que mais posso fazer?"
            return ("Nao entendi qual receita. Diga o numero, "
                    "ex.: 'quero a 1' ou 'me mostra a 2'.")
        s["candidata_atual"] = escolha - 1
        r = s["candidatas"][escolha - 1]
        s["receita"] = r
        s["pergunta_tipo"] = "confirmar_guia"
        return (f"{r['descricao']}\n\n"
                f"**{r['nome']}** - {r['tempo']} | {r.get('dificuldade','?').title()}\n\n"
                "Quer que eu te guie no passo a passo? (ou diga 'outra' pra ver a "
                "lista, ou 'manda tudo' pra receita completa)")

    r = s["receita"] or s["candidatas"][s["candidata_atual"]]

    if tipo == "confirmar_guia":
        if nao and not sim:  # nao quer o guia -> volta pra lista pra escolher outra
            s["pergunta_tipo"] = "escolha"
            return "Sem problema! " + _formatar_opcoes(s["candidatas"])
        lista = "\n".join(f"- {i.capitalize()}" for i in r["ingredientes"])
        s["estado"] = "passo_a_passo"
        s["passo_atual"] = 0
        return (f"**Ingredientes - {r['nome']}**\n{lista}\n\n"
                "Bora cozinhar! Te guio passo a passo - diga \"pronto\" para "
                "avancar (ou \"manda tudo\" pra ver a receita inteira).\n\n"
                + mostrar_passo(s))

    return _formatar_opcoes(s["candidatas"])


def _rotear(intent, mensagem, sessao, t0):
    # Mapeia uma intencao resolvida para uma resposta. None = nao resolveu (caller segue para o fluxo nome -> ingredientes -> FAQ -> LLM).
    if intent == "saudacao": return _r_base(random.choice(SAUDACOES), t0)
    if intent == "despedida": return _r_base(random.choice(DESPEDIDAS), t0)
    if intent == "agradecimento": return _r_base(random.choice(AGRADECIMENTOS), t0)
    if intent == "ajuda": return _r_base(AJUDA, t0)
    if intent == "receita_aleatoria":
        return _r_base(iniciar_sondagem(random.sample(RECEITAS, 3), sessao,
                                        INTRO_ALEATORIA), t0)
    if intent == "buscar_por_nome":
        cands, intro = base.buscar_por_nome(mensagem), INTRO_NOME
    elif intent.startswith("categoria:"):
        cat = intent.split(":", 1)[1]
        recs = base.buscar_por_categoria(cat)
        cands = random.sample(recs, min(3, len(recs))) if recs else []
        intro = intro_categoria(cat)
    elif intent == "buscar_por_ingredientes":
        cands, intro = base.buscar_por_ingredientes(tokenizar(mensagem)), INTRO_INGREDIENTES
    else:
        return None
    return _r_base(iniciar_sondagem(cands, sessao, intro), t0) if cands else None


def _tenta_faq(mensagem, t0):
    res = base.buscar_faq(mensagem)
    if res is None:
        return None
    return _r_base(f"{res.dados}\n\n[FAQ, sim {res.confianca:.2f}]", t0, "base-faq")


def gerar_resposta(mensagem, sessao):
    resp = _gerar(mensagem, sessao)
    # registra no historico (limite de ~12 trocas para nao estourar contexto)
    hist = sessao.setdefault("historico", [])
    hist.append({"user": mensagem, "bot": resp["texto"]})
    sessao["historico"] = hist[-12:]
    return resp


def _gerar(mensagem, sessao):
    if not mensagem or not mensagem.strip():
        return _r_base("Nao entendi. Pode reformular?", time.time())

    t0 = time.time()
    texto = normalizar(mensagem)
    estado = sessao["estado"]

    #passo a passo
    if estado == "passo_a_passo":
        if contem(texto, SINAIS["cancelar"]):
            return _r_base(cancelar(sessao), t0)
        if contem(texto, SINAIS["completo"]):
            return _r_base(entregar_completa(sessao), t0)
        if contem(texto, SINAIS["confirmar"]):
            return _r_base(proximo_passo(sessao), t0)
        intent, conf = detectar_intencao(mensagem)
        if conf and intent == "despedida":
            resetar(sessao)
            return _r_base(random.choice(DESPEDIDAS), t0)
        if conf and intent == "ajuda":
            return _r_base(f"Cozinhando '{sessao['receita']['nome']}'. "
                           "Use 'pronto', 'manda tudo' ou 'cancelar'.", t0)
        return _r_base("Diz 'pronto' quando terminar.\n\n" + mostrar_passo(sessao), t0)

    #sondagem
    if estado == "sondagem":
        if contem(texto, SINAIS["cancelar"]):
            return _r_base(cancelar(sessao), t0)
        if contem(texto, SINAIS["completo"]):
            sessao["receita"] = sessao["candidatas"][sessao["candidata_atual"]]
            return _r_base(entregar_completa(sessao), t0)
        if parece_pergunta(texto):
            resetar(sessao)  # escape: usuario perguntou algo culinario
        elif _quer_nova_busca(texto, mensagem, sessao):
            resetar(sessao)  # escape: usuario pediu outra coisa (cai no inicio)
        else:
            return _r_base(resposta_sondagem(texto, sessao), t0)

    #inicio
    if contem(texto, SINAIS["cancelar"]):
        return _r_base("Nao ha nada para cancelar agora. O que quer cozinhar?", t0)

    # perguntas culinarias claras vao direto pra FAQ/LLM
    if parece_pergunta(texto):
        return _tenta_faq(mensagem, t0) or _r_llm(mensagem, sessao)

    # Naive Bayes resolve os casos confiantes; se nao, o Agente 1 (LLM) tenta
    intent, conf = detectar_intencao(mensagem)
    if not conf:
        intent = llm.classificar(mensagem, INTENCOES, sessao.get("historico"))
    if intent:
        r = _rotear(intent, mensagem, sessao, t0)
        if r:
            return r

    #nome -> ingredientes -> FAQ -> LLM
    cands = base.buscar_por_nome(mensagem) or base.buscar_por_ingredientes(tokenizar(mensagem))
    if cands:
        return _r_base(iniciar_sondagem(cands, sessao), t0)

    faq = _tenta_faq(mensagem, t0)
    if faq:
        return faq

    relacionadas = base.buscar_por_ingredientes(tokenizar(mensagem))[:2]
    contexto = ""
    if relacionadas:
        contexto = "Receitas relacionadas:\n" + "\n".join(
            base.resumo_receita(r) for r in relacionadas)
    return _r_llm(mensagem, sessao, contexto=contexto)


#estatisticas
def stats():
    s = ESTATISTICAS
    total = s["total"] or 1
    return {
        "total": s["total"], "base": s["base"], "llm": s["llm"],
        "pct_base": round(100 * s["base"] / total, 1),
        "pct_llm": round(100 * s["llm"] / total, 1),
        "tempo_medio_base_ms": round(s["tempo_base_ms"] / max(s["base"], 1), 1),
        "tempo_medio_llm_ms": round(s["tempo_llm_ms"] / max(s["llm"], 1), 1),
        "llm_status": llm.status(),
    }
