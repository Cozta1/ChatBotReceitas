import re
import random
import string
import unicodedata

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from receitas import RECEITAS
from dados import (
    DADOS_TREINO, SAUDACOES, DESPEDIDAS, AGRADECIMENTOS, AJUDA,
    SINAIS_CONFIRMACAO, SINAIS_COMPLETO, SINAIS_CANCELAR,
    POSITIVOS, NEGATIVOS, ENCORAJAMENTOS,
)

for pacote in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(pacote, quiet=True)

lematizador  = WordNetLemmatizer()
stopwords_pt = set(stopwords.words('portuguese'))

def normalizar(texto):
    sem_acento = unicodedata.normalize('NFKD', texto.lower())
    return ''.join(c for c in sem_acento if not unicodedata.combining(c))


def tokenizar(texto, filtrar_stopwords=False):
    tokens = [t for t in word_tokenize(normalizar(texto), language='portuguese')
              if t not in string.punctuation and len(t) > 1]
    if filtrar_stopwords:
        tokens = [t for t in tokens if t not in stopwords_pt]
    return [lematizador.lemmatize(t, pos='v') for t in tokens]

frases_treino = [" ".join(tokenizar(frase)) for frase, _ in DADOS_TREINO]
labels_treino = [label for _, label in DADOS_TREINO]

vetorizador   = CountVectorizer()
classificador = MultinomialNB()
classificador.fit(vetorizador.fit_transform(frases_treino), labels_treino)


def detectar_intencao(tokens):
    texto = " ".join(tokens)
    if not texto.strip():
        return "buscar_por_ingredientes"
    vetor = vetorizador.transform([texto])
    probs = classificador.predict_proba(vetor)[0]
    if probs.max() >= probs.mean() * 1.5:
        return classificador.predict(vetor)[0]
    return "buscar_por_ingredientes"

def ingredientes_batem(ing_a, lista_ings):
    partes_a = [p for p in ing_a.split() if len(p) > 2]
    if not partes_a:
        return False
    for ing_b in lista_ings:
        for parte_b in [p for p in ing_b.split() if len(p) > 2]:
            if any(parte_b in pa or pa in parte_b for pa in partes_a):
                return True
    return False


def extrair_ingredientes(tokens):
    texto = " ".join(tokens)
    encontrados = []
    for receita in RECEITAS:
        for ing in receita["ingredientes"]:
            partes = [p for p in normalizar(ing).split() if len(p) > 2]
            if any(p in texto for p in partes) and ing not in encontrados:
                encontrados.append(ing)
    return encontrados


def buscar_receitas(ingredientes):
    ings_norm = [normalizar(i) for i in ingredientes]
    resultados = []
    for receita in RECEITAS:
        ings_r = [normalizar(i) for i in receita["ingredientes"]]
        match_r = sum(1 for i in ings_r if ingredientes_batem(i, ings_norm))
        if not match_r:
            continue
        match_u = sum(1 for iu in ings_norm if ingredientes_batem(iu, ings_r))
        cob_r = match_r / len(ings_r)
        cob_u = match_u / len(ings_norm) if ings_norm else 0
        score  = (2 * cob_r * cob_u / (cob_r + cob_u)) if (cob_r + cob_u) else 0
        resultados.append((receita, score, match_r, match_u))
    # prioriza quantos ingredientes do usuario sao usados, depois o score geral
    resultados.sort(key=lambda x: (-x[3], -x[1], -x[2]))
    return resultados[:3]


PALAVRAS_GENERICAS = {"receita", "receitas", "fazer", "preparar", "cozinhar", "quero", "uma", "tem", "como", "busca", "mostra"}

def buscar_por_nome(texto):
    palavras = [p for p in normalizar(texto).split() if len(p) > 2 and p not in PALAVRAS_GENERICAS]
    if not palavras:
        return []
    resultados = []
    for receita in RECEITAS:
        palavras_nome = [p for p in normalizar(receita["nome"]).split() if len(p) > 2]
        score = sum(1 for p in palavras_nome if any(p in u or u in p for u in palavras))
        if score:
            resultados.append((receita, score))
    resultados.sort(key=lambda x: -x[1])
    return [r for r, _ in resultados[:3]]


def buscar_por_categoria(categoria):
    return [r for r in RECEITAS if r["categoria"] == categoria]


def extrair_excluidos(texto):
    texto_norm = normalizar(texto)
    for padrao in [r'\bsem\s+(.+)', r'nao gosto de\s+(.+)', r'nao como\s+(.+)']:
        match = re.search(padrao, texto_norm)
        if match:
            itens = re.split(r'\s+e\s+|\s+nem\s+|,\s*', match.group(1))
            return [item.strip() for item in itens if len(item.strip()) > 2]
    return []


def receita_tem_excluido(receita, excluidos):
    for ing in receita["ingredientes"]:
        partes_ing = [p for p in normalizar(ing).split() if len(p) > 2]
        for exc in excluidos:
            partes_exc = [p for p in normalizar(exc).split() if len(p) > 2]
            for pe in partes_exc:
                for pi in partes_ing:
                    if pe in pi or pi in pe:
                        return True
    return False


def formatar_receita(receita):
    linha = "=" * 46
    dif = receita.get("dificuldade", "?").title()
    ings = "\n".join(f"- {i.capitalize()}" for i in receita["ingredientes"])
    passos = "\n".join(f"{i+1}. {p}" for i, p in enumerate(receita["instrucoes"]))
    return (
        f"\n{linha}\n {receita['nome'].upper()}\n{linha}\n"
        f"Tempo: {receita['tempo']} | Porcoes: {receita['porcoes']} | Dificuldade: {dif}\n\n"
        f"INGREDIENTES:\n{ings}\n\n  MODO DE PREPARO:\n{passos}\n{linha}"
    )


def nova_sessao():
    return {
        "estado": "inicio",
        "receita": None,
        "candidatas": [],
        "candidata_atual": 0,
        "pergunta_tipo": None,
        "passo_atual": 0,
        "historico": [],
    }


def resetar_sessao(sessao):
    sessao.update(estado="inicio", receita=None, candidatas=[], candidata_atual=0, pergunta_tipo=None, passo_atual=0)


def mostrar_passo(sessao):
    receita = sessao["receita"]
    i = sessao["passo_atual"]
    total = len(receita["instrucoes"])
    return f"[ Passo {i+1} de {total} ]\n  >> {receita['instrucoes'][i]}\n\nMe avisa quando terminar!"


def proximo_passo(sessao):
    sessao["passo_atual"] += 1
    receita = sessao["receita"]
    if sessao["passo_atual"] >= len(receita["instrucoes"]):
        sessao["estado"] = "conclusao"
        return f"Ultimo passo concluido! Parabens, '{receita['nome']}' esta pronto!\n\nComo ficou o prato? Deu tudo certo?"
    return random.choice(ENCORAJAMENTOS) + "\n\n" + mostrar_passo(sessao)


def entregar_receita_completa(sessao):
    sessao["estado"] = "conclusao"
    return f"Claro! Aqui esta a receita completa:{formatar_receita(sessao['receita'])}\n\nMe diz quando terminar de preparar, tudo bem?"


def cancelar_receita(sessao):
    resetar_sessao(sessao)
    return "Sem problema, cancelei essa receita! O que voce gostaria de cozinhar agora?"


SUGESTOES_POS_RECEITA = [
    "Que tal preparar uma sobremesa agora?",
    "O que acha de um lanche rapido na sequencia?",
    "Agora vamos preparar uma sobremesa?",
    "Que tal algo para beber para acompanhar?",
    "Que tal partir para um prato principal agora?",
    "O que acha de preparar um acompanhamento?",
]

def processar_conclusao(mensagem, sessao):
    texto_norm = normalizar(mensagem)
    nome = sessao["receita"]["nome"] if sessao["receita"] else "prato"
    sugestao = random.choice(SUGESTOES_POS_RECEITA)
    resetar_sessao(sessao)

    if any(p in texto_norm for p in POSITIVOS):
        return random.choice([
            f"Que otimo! Fico feliz que '{nome}' ficou uma delicia!\n\n{sugestao}",
            f"Arrasou na cozinha! '{nome}' feito com sucesso!\n\n{sugestao}",
            f"Excelente! Tenho certeza que ficou gostoso.\n\n{sugestao}",
        ])
    if any(p in texto_norm for p in NEGATIVOS):
        return random.choice([
            f"Que pena! Mas faz parte do aprendizado.\n\n{sugestao}",
            f"Nao desanime! Cozinhar e uma arte que se aprende com a pratica.\n\n{sugestao}",
        ])
    return random.choice([
        f"Espero que tenha ficado otimo!\n\n{sugestao}",
        f"Que bom cozinhar junto com voce!\n\n{sugestao}",
        f"Obrigado por cozinhar comigo!\n\n{sugestao}",
    ])

def sugerir_receita(sessao):
    indice = sessao["candidata_atual"]
    receita = sessao["candidatas"][indice]
    dif = receita.get("dificuldade", "?").title()
    sessao["pergunta_tipo"] = "sugestao"

    return (
        f"Que tal '{receita['nome']}'?\n"
        f"Tempo: {receita['tempo']} | Dificuldade: {dif}\n\nVoce toparia fazer essa receita?"
    )


def resposta_sondagem(mensagem, texto_norm, sessao):
    tipo = sessao["pergunta_tipo"]

    disse_sim = any(p in texto_norm for p in [
        "sim", "claro", "pode", "vamos", "ok", "legal", "bora",
        "topo", "aceito", "adorei", "perfeito", "gostei", "combinado", "show",
    ])
    disse_nao = any(p in texto_norm for p in [
        "nao", "noa", "nop", "prefiro", "outro", "outra", "muda", "diferente",
        "nope", "passa", "proxima", "proximo",
    ])

    if tipo == "sugestao":
        if disse_nao and not disse_sim:
            restantes = len(sessao["candidatas"]) - sessao["candidata_atual"] - 1
            if restantes <= 0:
                sessao["estado"] = "inicio"
                return (
                    "Poxa, era essa a ultima opcao que eu tinha!\n\n"
                    "Posso te ajudar de outras formas:\n"
                    "  - Diga os ingredientes que voce tem e sugiro algo\n"
                    "  - Peca uma categoria: prato principal, sobremesa, lanche...\n"
                    "  - Ou peÃ§a uma receita aleatoria!\n\n"
                    "O que voce prefere?"
                )
            sessao["pergunta_tipo"] = "confirmar_proxima"
            return f"Tudo bem! Tenho mais {restantes} {'opcoes' if restantes > 1 else 'opcao'} guardada. Quer ver a proxima?"
        if not disse_sim:
            return "Nao entendi! Quer fazer essa receita? Pode dizer 'sim' ou 'nao'."
        sessao["pergunta_tipo"] = "confirmar_ingredientes"
        receita = sessao["candidatas"][sessao["candidata_atual"]]
        return f"Otima escolha! Posso te passar a lista de ingredientes do '{receita['nome']}'?"

    if tipo == "confirmar_proxima":
        if disse_nao and not disse_sim:
            sessao["estado"] = "inicio"
            return (
                "Sem problema! Posso te ajudar de outras formas:\n"
                "  - Me diga os ingredientes que voce tem\n"
                "  - Peça uma categoria: prato principal, sobremesa, lanche...\n"
                "  - Ou peça uma receita aleatoria!\n\n"
                "O que voce quer fazer?"
            )
        if not disse_sim:
            return "Quer ver a proxima opcao? Pode dizer 'sim' ou 'nao'."
        sessao["candidata_atual"] += 1
        sessao["pergunta_tipo"] = None
        return sugerir_receita(sessao)

    if tipo == "confirmar_ingredientes":
        if disse_nao and not disse_sim:
            sessao["estado"] = "inicio"
            return "Tudo bem! Se quiser outra receita, e so chamar. O que mais posso fazer por voce?"
        receita = sessao["candidatas"][sessao["candidata_atual"]]
        sessao["receita"] = receita
        lista_ings = "\n".join(f" - {i.capitalize()}" for i in receita["ingredientes"])
        sessao["pergunta_tipo"] = "confirmar_passos"
        return f"Aqui estao os ingredientes do '{receita['nome']}':\n\n{lista_ings}\n\nJa tem tudo isso? Posso comecar com o passo a passo?"

    if tipo == "confirmar_passos":
        if disse_nao and not disse_sim:
            sessao["estado"] = "inicio"
            sessao["receita"] = None
            return "Sem problema! Quando tiver os ingredientes, e so me chamar.\n\nQuer que eu sugira alguma receita com o que voce tem agora?"
        sessao["estado"] = "passo_a_passo"
        sessao["passo_atual"] = 0
        return "Vamos la! Vou te guiar passo a passo.\nDiga 'pronto' ao terminar cada passo, ou 'manda tudo' para ver a receita completa.\n\n" + mostrar_passo(sessao)

    return sugerir_receita(sessao)


def iniciar_busca(candidatas, sessao):
    sessao.update(candidatas=list(candidatas), candidata_atual=0, pergunta_tipo=None, estado="sondagem")
    return sugerir_receita(sessao)


def gerar_resposta(mensagem, sessao):
    tokens = tokenizar(mensagem, filtrar_stopwords=True)
    texto_norm = normalizar(mensagem)
    estado = sessao["estado"]

    if estado == "passo_a_passo":
        if any(s in texto_norm for s in SINAIS_CANCELAR):
            return cancelar_receita(sessao)
        if any(s in texto_norm for s in SINAIS_COMPLETO):
            return entregar_receita_completa(sessao)
        if any(s in texto_norm for s in SINAIS_CONFIRMACAO):
            return proximo_passo(sessao)
        if tokens:
            intencao = detectar_intencao(tokens)
            if intencao == "despedida":
                sessao.update(estado="inicio", receita=None)
                return random.choice(DESPEDIDAS)
            if intencao == "agradecimento":
                return random.choice(AGRADECIMENTOS) + f"\n\nMas ainda estamos no passo {sessao['passo_atual'] + 1}! Pronto para continuar?"
            if intencao == "ajuda":
                return (
                    f"Estamos cozinhando '{sessao['receita']['nome']}'!\n"
                    "  - Diga 'pronto' para ir ao proximo passo\n"
                    "  - Diga 'manda tudo' para ver a receita completa\n"
                    "  - Diga 'cancelar' para cancelar e escolher outra\n\n"
                    "Quer continuar de onde paramos?"
                )
        return "Hmm, nao entendi! Diz 'pronto' quando terminar este passo, pode ser?\n\n" + mostrar_passo(sessao)

    if estado == "sondagem":
        if any(s in texto_norm for s in SINAIS_CANCELAR):
            return cancelar_receita(sessao)
        if any(s in texto_norm for s in SINAIS_COMPLETO):
            sessao["receita"] = sessao["candidatas"][sessao["candidata_atual"]]
            return entregar_receita_completa(sessao)
        return resposta_sondagem(mensagem, texto_norm, sessao)

    if estado == "conclusao":
        return processar_conclusao(mensagem, sessao)
    

    if not tokens:
        return "Nao entendi. Tente listar ingredientes ou use 'ajuda'. O que voce gostaria de fazer?"

    if any(s in texto_norm for s in SINAIS_CANCELAR):
        return "Nao ha nada para cancelar agora. O que voce gostaria de cozinhar?"

    gatilhos_sem = ["sem ", "nao gosto de ", "nao como "]
    if any(g in texto_norm for g in gatilhos_sem):
        excluidos = extrair_excluidos(mensagem)
        if excluidos:
            receitas_filtradas = [r for r in RECEITAS if not receita_tem_excluido(r, excluidos)]
            if not receitas_filtradas:
                return f"Nao encontrei receitas sem '{', '.join(excluidos)}'. Quer tentar remover outra restricao?"
            return iniciar_busca(random.sample(receitas_filtradas, min(3, len(receitas_filtradas))), sessao)

    intencao = detectar_intencao(tokens)

    if intencao == "saudacao": return random.choice(SAUDACOES)
    if intencao == "despedida": return random.choice(DESPEDIDAS)
    if intencao == "agradecimento": return random.choice(AGRADECIMENTOS)
    if intencao == "ajuda": return AJUDA

    if intencao == "buscar_por_nome":
        candidatas = buscar_por_nome(mensagem)
        if candidatas:
            return iniciar_busca(candidatas, sessao)
        return "Nao encontrei receita com esse nome. Quer tentar com outro nome ou outra categoria?"

    if intencao == "receita_aleatoria":
        return iniciar_busca(random.sample(RECEITAS, 3), sessao)

    if intencao.startswith("categoria:"):
        categoria = intencao.split(":", 1)[1]
        receitas_cat = buscar_por_categoria(categoria)
        if not receitas_cat:
            return f"Nao encontrei receitas na categoria '{categoria}'. Quer tentar outra categoria?"
        return iniciar_busca(random.sample(receitas_cat, min(3, len(receitas_cat))), sessao)

    candidatas_nome = buscar_por_nome(mensagem)
    if candidatas_nome:
        return iniciar_busca(candidatas_nome, sessao)

    ingredientes = extrair_ingredientes(tokens)
    if ingredientes:
        resultados = buscar_receitas(ingredientes)
        if resultados:
            return iniciar_busca([r for r, *_ in resultados], sessao)

    palavras_genericas = {"receita", "receitas", "cozinhar", "fazer", "comer", "algo", "sugestao", "preparo"}
    tokens_uteis = [t for t in tokens if t not in palavras_genericas]
    if tokens_uteis:
        resultados = buscar_receitas(tokens_uteis)
        if resultados:
            return iniciar_busca([r for r, *_ in resultados], sessao)

    pedidos_genericos = ["receita", "cozinhar", "fazer", "comer", "algo", "sugestao"]
    if any(p in texto_norm for p in pedidos_genericos):
        return iniciar_busca(random.sample(RECEITAS, 3), sessao)

    return (
        "Nao entendi bem o que voce quer.\n"
        "Tente listar ingredientes como 'Tenho frango e arroz', ou diga uma categoria como 'quero um lanche'.\n\n"
        "Posso te ajudar de outra forma?"
    )

def main():
    sessao = nova_sessao()
    print("CHEFBOT - Seu Assistente de Receitas")
    print("Digite 'sair' para encerrar")
    print(f"\nChefBot: {AJUDA}\n")

    while True:
        entrada = input("Voce: ").strip()
        if not entrada:
            continue
        if entrada.lower() in ('sair', 'exit', 'quit'):
            print("\nChefBot: Tchau! Bom apetite!")
            break
        resposta = gerar_resposta(entrada, sessao)
        sessao["historico"].append((entrada, resposta))
        print(f"\nChefBot: {resposta}\n")


if __name__ == '__main__':
    main()