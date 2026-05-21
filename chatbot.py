"""ChefBot — orquestrador hibrido (KB -> LLM fallback).

Fluxo no estado inicio:
  pergunta culinaria (interrogativa)? -> FAQ TF-IDF -> LLM
  intent confiavel?                   -> handler especifico
  fallback                            -> nome -> ingredientes -> FAQ -> LLM
"""
import random
import time

import kb
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
SUGESTOES_FIM = ["Que tal uma sobremesa?", "Vamos pra um lanche?", "Que tal algo para beber?"]

POSITIVOS = ("gostei", "otimo", "delicioso", "perfeito", "amei", "adorei",
             "ficou bom", "ficou gostoso", "ficou otimo", "ficou show",
             "deu certo", "maravilhoso", "excelente")
NEGATIVOS = ("nao gostei", "ficou ruim", "ficou horrivel", "horrivel",
             "horrivel", "pessimo", "ruim", "errei", "queimou",
             "deu errado", "estragou", "passou do ponto")

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

ESTATISTICAS = {"total": 0, "base": 0, "llm": 0,
                "tempo_base_ms": 0.0, "tempo_llm_ms": 0.0}


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


def _r_base(texto, t0, marcador="base"):
    dt = (time.time() - t0) * 1000
    ESTATISTICAS["base"] += 1
    ESTATISTICAS["total"] += 1
    ESTATISTICAS["tempo_base_ms"] += dt
    return {"texto": texto, "fonte": marcador, "tempo_ms": round(dt, 1)}


def _r_llm(pergunta, sessao, contexto=""):
    t0 = time.time()
    historico = sessao.get("historico", [])
    texto = llm.responder(pergunta, historico=historico, contexto=contexto)
    dt = (time.time() - t0) * 1000
    ESTATISTICAS["llm"] += 1
    ESTATISTICAS["total"] += 1
    ESTATISTICAS["tempo_llm_ms"] += dt
    return {"texto": texto, "fonte": "llm", "tempo_ms": round(dt, 1)}



# Fluxo (sondagem -> passo a passo -> conclusao)

def _formatar_sugestao(r):
    return (f"Que tal '{r['nome']}'?\n"
            f"Tempo: {r['tempo']} | Dificuldade: {r.get('dificuldade','?').title()}\n\n"
            f"Voce toparia fazer essa receita?")


def iniciar_sondagem(candidatas, s):
    s.update(candidatas=list(candidatas), candidata_atual=0,
             pergunta_tipo="sugestao", estado="sondagem")
    return _formatar_sugestao(s["candidatas"][0])


def cancelar(s):
    resetar(s)
    return "Cancelei. O que voce gostaria de cozinhar agora?"


def entregar_completa(s):
    s["estado"] = "conclusao"
    return f"Aqui esta a receita completa:{kb.formatar_receita(s['receita'])}\n\nMe diz quando terminar!"


def mostrar_passo(s):
    i, total = s["passo_atual"], len(s["receita"]["instrucoes"])
    return f"[ Passo {i+1}/{total} ]\n  >> {s['receita']['instrucoes'][i]}\n\nMe avisa quando terminar!"


def proximo_passo(s):
    s["passo_atual"] += 1
    if s["passo_atual"] >= len(s["receita"]["instrucoes"]):
        s["estado"] = "conclusao"
        return f"Concluido! '{s['receita']['nome']}' esta pronto!\n\nComo ficou?"
    return random.choice(ENCORAJAMENTOS) + "\n\n" + mostrar_passo(s)


def resposta_sondagem(texto, s):
    tipo = s["pergunta_tipo"]
    sim, nao = contem(texto, SIM), contem(texto, NAO)
    r = s["candidatas"][s["candidata_atual"]]

    if tipo == "sugestao":
        if nao and not sim:
            restantes = len(s["candidatas"]) - s["candidata_atual"] - 1
            if restantes <= 0:
                s["estado"] = "inicio"
                return "Era a ultima opcao! Quer pedir algo diferente?"
            s["pergunta_tipo"] = "confirmar_proxima"
            return f"Tudo bem! Tenho mais {restantes} opcao(oes). Quer ver a proxima?"
        if not sim:
            return "Quer fazer essa? Diz 'sim' ou 'nao'."
        s["pergunta_tipo"] = "confirmar_ingredientes"
        return f"Otimo! Te passo a lista de ingredientes do '{r['nome']}'?"

    if tipo == "confirmar_proxima":
        if nao and not sim:
            s["estado"] = "inicio"
            return "Sem problema! O que mais posso fazer?"
        if not sim:
            return "Sim ou nao?"
        s["candidata_atual"] += 1
        s["pergunta_tipo"] = "sugestao"
        return _formatar_sugestao(s["candidatas"][s["candidata_atual"]])

    if tipo == "confirmar_ingredientes":
        if nao and not sim:
            s["estado"] = "inicio"
            return "Tudo bem! Se quiser outra, e so chamar."
        s["receita"] = r
        lista = "\n".join(f" - {i.capitalize()}" for i in r["ingredientes"])
        s["pergunta_tipo"] = "confirmar_passos"
        return f"Ingredientes do '{r['nome']}':\n\n{lista}\n\nJa tem tudo? Comeco o passo a passo?"

    if tipo == "confirmar_passos":
        if nao and not sim:
            s["estado"] = "inicio"
            s["receita"] = None
            return "Quando tiver tudo, me chame!"
        s["estado"] = "passo_a_passo"
        s["passo_atual"] = 0
        return ("Vamos la! 'pronto' avanca, 'manda tudo' entrega completa.\n\n"
                + mostrar_passo(s))

    return _formatar_sugestao(r)


def processar_conclusao(msg, s):
    texto = normalizar(msg)
    nome = s["receita"]["nome"] if s["receita"] else "prato"
    sug = random.choice(SUGESTOES_FIM)
    resetar(s)
    if contem(texto, POSITIVOS):
        m = f"Que otimo! '{nome}' ficou uma delicia!"
    elif contem(texto, NEGATIVOS):
        m = "Que pena! Mas faz parte do aprendizado."
    else:
        m = "Espero que tenha ficado bom!"
    return f"{m}\n\n{sug}"



def _tenta_faq(mensagem, t0):
    res = kb.buscar_faq(mensagem)
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
        else:
            return _r_base(resposta_sondagem(texto, sessao), t0)

    #conclusao
    if sessao["estado"] == "conclusao":
        return _r_base(processar_conclusao(mensagem, sessao), t0)

    #inicio
    if contem(texto, SINAIS["cancelar"]):
        return _r_base("Nao ha nada para cancelar agora. O que quer cozinhar?", t0)

    # perguntas culinarias claras vao direto pra FAQ/LLM
    if parece_pergunta(texto):
        return _tenta_faq(mensagem, t0) or _r_llm(mensagem, sessao)

    intent, conf = detectar_intencao(mensagem)
    if conf:
        if intent == "saudacao": return _r_base(random.choice(SAUDACOES), t0)
        if intent == "despedida": return _r_base(random.choice(DESPEDIDAS), t0)
        if intent == "agradecimento": return _r_base(random.choice(AGRADECIMENTOS), t0)
        if intent == "ajuda": return _r_base(AJUDA, t0)
        if intent == "receita_aleatoria":
            return _r_base(iniciar_sondagem(random.sample(RECEITAS, 3), sessao), t0)
        if intent == "buscar_por_nome":
            cands = kb.buscar_por_nome(mensagem)
            if cands:
                return _r_base(iniciar_sondagem(cands, sessao), t0)
        elif intent.startswith("categoria:"):
            recs = kb.buscar_por_categoria(intent.split(":", 1)[1])
            if recs:
                amostra = random.sample(recs, min(3, len(recs)))
                return _r_base(iniciar_sondagem(amostra, sessao), t0)
        elif intent == "buscar_por_ingredientes":
            cands = kb.buscar_por_ingredientes(tokenizar(mensagem))
            if cands:
                return _r_base(iniciar_sondagem(cands, sessao), t0)

    #nome -> ingredientes -> FAQ -> LLM
    cands = kb.buscar_por_nome(mensagem) or kb.buscar_por_ingredientes(tokenizar(mensagem))
    if cands:
        return _r_base(iniciar_sondagem(cands, sessao), t0)

    faq = _tenta_faq(mensagem, t0)
    if faq:
        return faq

    relacionadas = kb.buscar_por_ingredientes(tokenizar(mensagem))[:2]
    contexto = ""
    if relacionadas:
        contexto = "Receitas relacionadas:\n" + "\n".join(
            kb.resumo_receita(r) for r in relacionadas)
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
