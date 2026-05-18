import random, string, unicodedata
import nltk, numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from receitas import RECEITAS
from dados import (
    DADOS_TREINO, SAUDACOES, DESPEDIDAS, AGRADECIMENTOS, AJUDA,
    SINAIS_CONFIRMACAO, SINAIS_COMPLETO, SINAIS_CANCELAR,
    POSITIVOS, NEGATIVOS, ENCORAJAMENTOS,
)

for p in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(p, quiet=True)

lematizador = WordNetLemmatizer()
stopwords_pt = set(stopwords.words('portuguese'))

def normalizar(texto):
    sem_acento = unicodedata.normalize('NFKD', texto.lower())
    return ''.join(c for c in sem_acento if not unicodedata.combining(c))

def tokenizar(texto, filtrar_sw=False):
    tokens = [t for t in word_tokenize(normalizar(texto), language='portuguese')
              if t not in string.punctuation and len(t) > 1]
    if filtrar_sw:
        tokens = [t for t in tokens if t not in stopwords_pt]
    return [lematizador.lemmatize(t, pos='v') for t in tokens]

# Classificador
tokens_treino = [tokenizar(f) for f, _ in DADOS_TREINO]
labels_treino = [l for _, l in DADOS_TREINO]
vocabulario = sorted(set(p for f in tokens_treino for p in f))
indice = {p: i for i, p in enumerate(vocabulario)}

def vetor(tokens):
    v = np.zeros(len(vocabulario))
    for t in tokens:
        if t in indice: v[indice[t]] = 1
    return v

clf = MultinomialNB()
clf.fit(np.array([vetor(t) for t in tokens_treino]), labels_treino)

def detectar_intencao(tokens):
    if not tokens: return "buscar_por_ingredientes"
    v = vetor(tokens).reshape(1, -1)
    probs = clf.predict_proba(v)[0]
    return clf.predict(v)[0] if probs.max() >= probs.mean() * 1.5 else "buscar_por_ingredientes"

# Busca
GENERICAS = {"receita","receitas","fazer","preparar","cozinhar","quero","uma","tem","como","busca","mostra"}

def buscar_por_nome(texto):
    palavras = [p for p in normalizar(texto).split() if len(p) > 2 and p not in GENERICAS]
    if not palavras: return []
    res = []
    for r in RECEITAS:
        partes = [p for p in normalizar(r["nome"]).split() if len(p) > 2]
        score = sum(1 for p in partes if any(p in u or u in p for u in palavras))
        if score: res.append((r, score))
    res.sort(key=lambda x: -x[1])
    return [r for r, _ in res[:3]]

def buscar_por_categoria(cat):
    return [r for r in RECEITAS if r["categoria"] == cat]

def buscar_por_ingredientes(palavras):
    res = []
    for r in RECEITAS:
        ings = " ".join(normalizar(i) for i in r["ingredientes"])
        matches = sum(1 for p in palavras if len(p) > 2 and p in ings)
        if matches: res.append((r, matches))
    res.sort(key=lambda x: -x[1])
    return [r for r, _ in res[:3]]

# Sessao
def nova_sessao():
    return {"estado":"inicio", "receita":None, "candidatas":[], "candidata_atual":0,
            "pergunta_tipo":None, "passo_atual":0, "historico":[]}

def resetar(sessao):
    sessao.update(estado="inicio", receita=None, candidatas=[], candidata_atual=0,
                  pergunta_tipo=None, passo_atual=0)

# Formatacao e passos
def formatar_receita(r):
    linha = "=" * 46
    ings = "\n".join(f"- {i.capitalize()}" for i in r["ingredientes"])
    passos = "\n".join(f"{i+1}. {p}" for i, p in enumerate(r["instrucoes"]))
    return (f"\n{linha}\n {r['nome'].upper()}\n{linha}\n"
            f"Tempo: {r['tempo']} | Porcoes: {r['porcoes']} | Dificuldade: {r.get('dificuldade','?').title()}\n\n"
            f"INGREDIENTES:\n{ings}\n\n  MODO DE PREPARO:\n{passos}\n{linha}")

def mostrar_passo(s):
    i = s["passo_atual"]
    return f"[ Passo {i+1} de {len(s['receita']['instrucoes'])} ]\n  >> {s['receita']['instrucoes'][i]}\n\nMe avisa quando terminar!"

def proximo_passo(s):
    s["passo_atual"] += 1
    if s["passo_atual"] >= len(s["receita"]["instrucoes"]):
        s["estado"] = "conclusao"
        return f"Ultimo passo concluido! Parabens, '{s['receita']['nome']}' esta pronto!\n\nComo ficou o prato? Deu tudo certo?"
    return random.choice(ENCORAJAMENTOS) + "\n\n" + mostrar_passo(s)

def entregar_completa(s):
    s["estado"] = "conclusao"
    return f"Claro! Aqui esta a receita completa:{formatar_receita(s['receita'])}\n\nMe diz quando terminar de preparar, tudo bem?"

def cancelar(s):
    resetar(s)
    return "Sem problema, cancelei essa receita! O que voce gostaria de cozinhar agora?"

def iniciar_busca(candidatas, s):
    s.update(candidatas=list(candidatas), candidata_atual=0, pergunta_tipo=None, estado="sondagem")
    return sugerir_receita(s)


PALAVRAS_SIM = ["sim","claro","pode","vamos","ok","legal","bora","topo","aceito","adorei","perfeito","gostei","combinado","show"]
PALAVRAS_NAO = ["nao","noa","nop","prefiro","outro","outra","muda","diferente","nope","passa","proxima","proximo"]


def sugerir_receita(s):
    r = s["candidatas"][s["candidata_atual"]]
    s["pergunta_tipo"] = "sugestao"
    return (f"Que tal '{r['nome']}'?\nTempo: {r['tempo']} | Dificuldade: {r.get('dificuldade','?').title()}\n\n"
            f"Voce toparia fazer essa receita?")

MSG_OPCOES = ("Posso te ajudar de outras formas:\n"
              "  - Diga os ingredientes que voce tem e sugiro algo\n"
              "  - Peca uma categoria: prato principal, sobremesa, lanche...\n"
              "  - Ou peca uma receita aleatoria!\n\nO que voce prefere?")

def resposta_sondagem(texto, s):
    tipo = s["pergunta_tipo"]
    sim = any(p in texto for p in PALAVRAS_SIM)
    nao = any(p in texto for p in PALAVRAS_NAO)

    if tipo == "sugestao":
        if nao and not sim:
            restantes = len(s["candidatas"]) - s["candidata_atual"] - 1
            if restantes <= 0:
                s["estado"] = "inicio"
                return f"Poxa, era a ultima opcao que eu tinha!\n\n{MSG_OPCOES}"
            s["pergunta_tipo"] = "confirmar_proxima"
            return f"Tudo bem! Tenho mais {restantes} {'opcoes' if restantes > 1 else 'opcao'} guardada. Quer ver a proxima?"
        if not sim:
            return "Nao entendi! Quer fazer essa receita? Pode dizer 'sim' ou 'nao'."
        s["pergunta_tipo"] = "confirmar_ingredientes"
        r = s["candidatas"][s["candidata_atual"]]
        return f"Otima escolha! Posso te passar a lista de ingredientes do '{r['nome']}'?"

    if tipo == "confirmar_proxima":
        if nao and not sim:
            s["estado"] = "inicio"
            return f"Sem problema! {MSG_OPCOES}"
        if not sim:
            return "Quer ver a proxima opcao? Pode dizer 'sim' ou 'nao'."
        s["candidata_atual"] += 1
        return sugerir_receita(s)

    if tipo == "confirmar_ingredientes":
        if nao and not sim:
            s["estado"] = "inicio"
            return "Tudo bem! Se quiser outra receita, e so chamar. O que mais posso fazer por voce?"
        r = s["candidatas"][s["candidata_atual"]]
        s["receita"] = r
        lista = "\n".join(f" - {i.capitalize()}" for i in r["ingredientes"])
        s["pergunta_tipo"] = "confirmar_passos"
        return f"Aqui estao os ingredientes do '{r['nome']}':\n\n{lista}\n\nJa tem tudo? Posso comecar o passo a passo?"

    if tipo == "confirmar_passos":
        if nao and not sim:
            s["estado"] = "inicio"
            s["receita"] = None
            return "Sem problema! Quando tiver os ingredientes, e so me chamar.\n\nQuer que eu sugira outra receita?"
        s["estado"] = "passo_a_passo"
        s["passo_atual"] = 0
        return "Vamos la! Diga 'pronto' ao terminar cada passo, ou 'manda tudo' para a receita completa.\n\n" + mostrar_passo(s)

    return sugerir_receita(s)

# Conclusao
SUGESTOES_FIM = [
    "Que tal preparar uma sobremesa agora?", "O que acha de um lanche rapido?",
    "Agora vamos preparar uma sobremesa?", "Que tal algo para beber?",
    "Que tal partir para um prato principal?", "O que acha de um acompanhamento?",
]

def processar_conclusao(msg, s):
    texto = normalizar(msg)
    nome = s["receita"]["nome"] if s["receita"] else "prato"
    sug = random.choice(SUGESTOES_FIM)
    resetar(s)
    if any(p in texto for p in POSITIVOS):
        m = random.choice([f"Que otimo! '{nome}' ficou uma delicia!", f"Arrasou! '{nome}' feito com sucesso!", "Excelente! Tenho certeza que ficou gostoso."])
    elif any(p in texto for p in NEGATIVOS):
        m = random.choice(["Que pena! Mas faz parte do aprendizado.", "Nao desanime! Cozinhar se aprende com pratica."])
    else:
        m = random.choice(["Espero que tenha ficado otimo!", "Que bom cozinhar com voce!", "Obrigado por cozinhar comigo!"])
    return f"{m}\n\n{sug}"

# Resposta
def gerar_resposta(mensagem, sessao):
    tokens = tokenizar(mensagem, filtrar_sw=True)
    texto = normalizar(mensagem)
    estado = sessao["estado"]

    if estado == "passo_a_passo":
        if any(s in texto for s in SINAIS_CANCELAR): return cancelar(sessao)
        if any(s in texto for s in SINAIS_COMPLETO): return entregar_completa(sessao)
        if any(s in texto for s in SINAIS_CONFIRMACAO): return proximo_passo(sessao)
        if tokens:
            intent = detectar_intencao(tokens)
            if intent == "despedida":
                resetar(sessao)
                return random.choice(DESPEDIDAS)
            if intent == "agradecimento":
                return random.choice(AGRADECIMENTOS) + f"\n\nMas ainda estamos no passo {sessao['passo_atual']+1}! Pronto para continuar?"
            if intent == "ajuda":
                return (f"Estamos cozinhando '{sessao['receita']['nome']}'!\n"
                        "  - 'pronto' para o proximo passo\n  - 'manda tudo' para a receita completa\n"
                        "  - 'cancelar' para escolher outra\n\nQuer continuar?")
        return "Hmm, nao entendi! Diz 'pronto' quando terminar este passo.\n\n" + mostrar_passo(sessao)

    if estado == "sondagem":
        if any(s in texto for s in SINAIS_CANCELAR): return cancelar(sessao)
        if any(s in texto for s in SINAIS_COMPLETO):
            sessao["receita"] = sessao["candidatas"][sessao["candidata_atual"]]
            return entregar_completa(sessao)
        return resposta_sondagem(texto, sessao)

    if estado == "conclusao":
        return processar_conclusao(mensagem, sessao)

    # estado inicio
    if not tokens:
        return "Nao entendi. Tente listar ingredientes ou use 'ajuda'. O que voce gostaria de fazer?"
    if any(s in texto for s in SINAIS_CANCELAR):
        return "Nao ha nada para cancelar agora. O que voce gostaria de cozinhar?"

    intent = detectar_intencao(tokens)

    if intent == "saudacao": return random.choice(SAUDACOES)
    if intent == "despedida": return random.choice(DESPEDIDAS)
    if intent == "agradecimento": return random.choice(AGRADECIMENTOS)
    if intent == "ajuda": return AJUDA

    if intent == "buscar_por_nome":
        cands = buscar_por_nome(mensagem)
        if cands: return iniciar_busca(cands, sessao)
        return "Nao encontrei essa receita. Quer tentar outro nome ou categoria?"

    if intent == "receita_aleatoria":
        return iniciar_busca(random.sample(RECEITAS, 3), sessao)

    if intent.startswith("categoria:"):
        cat = intent.split(":", 1)[1]
        recs = buscar_por_categoria(cat)
        if not recs: return f"Nao encontrei receitas na categoria '{cat}'. Quer tentar outra?"
        return iniciar_busca(random.sample(recs, min(3, len(recs))), sessao)

    # fallback: tenta nome, depois ingredientes
    cands = buscar_por_nome(mensagem) or buscar_por_ingredientes(tokens)
    if cands: return iniciar_busca(cands, sessao)

    if any(p in texto for p in ["receita","cozinhar","fazer","comer","algo","sugestao"]):
        return iniciar_busca(random.sample(RECEITAS, 3), sessao)

    return ("Nao entendi bem o que voce quer.\n"
            "Tente listar ingredientes como 'Tenho frango e arroz', ou peca uma categoria.\n\n"
            "Posso te ajudar de outra forma?")

def main():
    sessao = nova_sessao()
    print("CHEFBOT - Seu Assistente de Receitas\nDigite 'sair' para encerrar")
    print(f"\nChefBot: {AJUDA}\n")
    while True:
        entrada = input("Voce: ").strip()
        if not entrada: continue
        if entrada.lower() in ('sair', 'exit', 'quit'):
            print("\nChefBot: Tchau! Bom apetite!")
            break
        resposta = gerar_resposta(entrada, sessao)
        sessao["historico"].append((entrada, resposta))
        print(f"\nChefBot: {resposta}\n")

if __name__ == '__main__':
    main()
