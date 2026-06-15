# Base de conhecimento: receitas + FAQ culinaria + busca TF-IDF.
from __future__ import annotations
import random
import unicodedata
from typing import NamedTuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from receitas import RECEITAS

LIMIAR_FAQ = 0.62  # similaridade minima para a FAQ responder (abaixo disso -> LLM)

FAQ: list[tuple[str, str]] = [
    ("como substituir manteiga por oleo em bolo",
     "Use **3/4 da quantidade** de oleo em relacao a manteiga:\n"
     "- **Exemplo:** 100 g de manteiga = 75 ml de oleo\n"
     "- Prefira oleos neutros (girassol, canola ou milho)\n"
     "- A massa costuma ficar mais umida e fofinha"),
    ("como descongelar carne com seguranca",
     "O jeito seguro e descongelar **na geladeira**, de um dia para o outro. Com pressa:\n"
     "- **Micro-ondas:** funcao descongelar, cozinhando logo em seguida\n"
     "- **Agua fria:** carne em saco vedado e submerso, trocando a agua a cada 30 min\n"
     "- **Nunca** descongele em temperatura ambiente (favorece bacterias)"),
    ("quanto tempo posso guardar carne na geladeira",
     "Sempre em recipiente fechado e a no maximo **4 graus**:\n"
     "- **Carne crua:** 1 a 2 dias\n"
     "- **Carne cozida:** 3 a 4 dias\n"
     "- **Congelada:** 3 a 6 meses (muda a textura, mas continua segura)"),
    ("como conservar ovo",
     "- Guarde **na geladeira**, na embalagem original\n"
     "- Mantenha **longe da porta**, onde a temperatura varia\n"
     "- Duram cerca de **4 a 5 semanas** apos a data de embalagem\n"
     "- **Teste de frescor:** na agua, o ovo bom afunda; se boia, descarte"),
    ("para que serve folha de louro",
     "Aromatiza pratos de cozimento longo: **feijao, ensopados, molho de tomate e caldos**.\n"
     "- Use **1 a 2 folhas** por panela\n"
     "- Adicione no inicio do cozimento\n"
     "- **Retire antes de servir** (a folha nao se come)"),
    ("como saber se o oleo esta na temperatura para fritar",
     "O ponto ideal fica entre **170 e 180 graus**. Sem termometro:\n"
     "- **Teste do palito:** mergulhe um palito de madeira; quando formar bolhinhas ao redor, esta no ponto\n"
     "- **Teste do pao:** um cubo de pao doura em cerca de 40 segundos\n"
     "- Oleo quente demais queima por fora e deixa cru por dentro"),
    ("como evitar que o arroz empape",
     "- **Refogue** o arroz no oleo com alho antes da agua\n"
     "- Use a proporcao **1 de arroz para 2 de agua** fervente\n"
     "- Cozinhe **tampado, em fogo baixo, sem mexer**\n"
     "- Solte com um garfo no final"),
    ("qual a diferenca entre creme de leite fresco e uht",
     "- **Fresco (geladeira):** ~35% de gordura, pode virar chantilly\n"
     "- **UHT (caixinha):** ~20% de gordura, mais estavel para molhos e sobremesas que nao precisam aerar\n"
     "- Para bater **chantilly**, so o fresco bem gelado funciona"),
    ("posso substituir fermento quimico por bicarbonato",
     "Da para substituir, mas com ajuste:\n"
     "- **1 colher de cha de fermento** = 1/4 de colher de cha de bicarbonato + 1/2 de cremor tartaro\n"
     "- O bicarbonato puro precisa de um **ingrediente acido** (limao, iogurte) para agir\n"
     "- Em excesso, deixa **gosto amargo**"),
    ("como deixar bife macio",
     "- Retire da geladeira **20 min antes** de grelhar\n"
     "- **Seque bem** com papel toalha\n"
     "- Tempere com sal **so na hora** de levar a frigideira\n"
     "- Use a frigideira **bem quente** e nao vire o bife mais de uma vez por lado\n"
     "- Deixe **descansar 5 min** antes de cortar"),
    ("como fazer arroz soltinho",
     "- Refogue o arroz no alho ate ficar **branquinho**\n"
     "- Adicione **agua fervente** na proporcao 2:1\n"
     "- Tampe e cozinhe em **fogo baixo, sem mexer**\n"
     "- Ao secar, desligue e **solte com um garfo**"),
    ("quanto tempo cozinhar feijao na pressao",
     "Contando **apos pegar pressao**, em fogo medio:\n"
     "- **Feijao carioca:** 20 a 25 min\n"
     "- **Feijao preto:** 30 a 35 min\n"
     "- Deixe de molho por pelo menos **4 horas** antes: cozinha mais rapido e fica mais leve"),
    ("como tirar o amargor da berinjela",
     "- Corte a berinjela em fatias ou cubos\n"
     "- Polvilhe **sal** e deixe descansar **30 min**\n"
     "- O sal **puxa a agua amarga**\n"
     "- Enxague e seque bem antes de cozinhar"),
    ("posso congelar molho de tomate",
     "Pode, sim:\n"
     "- **Esfrie completamente** antes de congelar\n"
     "- Use pote ou saco proprio, com **pouco ar**\n"
     "- Congele por ate **3 meses**\n"
     "- Descongele na geladeira ou direto na panela em fogo baixo"),
    ("equivalencia xicara em gramas farinha",
     "Medidas da **farinha de trigo**:\n"
     "- **1 xicara** ~ 120 g\n"
     "- **1 colher de sopa** ~ 8 g\n"
     "- Peneire e nivele sem apertar para nao errar a quantidade"),
    ("equivalencia xicara em gramas acucar",
     "Medidas do **acucar refinado**:\n"
     "- **1 xicara** ~ 180 g\n"
     "- **1 colher de sopa** ~ 12 g\n"
     "- O mascavo pesa um pouco menos por ser mais umido"),
    ("como saber se o bolo esta assado",
     "Faca o **teste do palito**, espetando no centro do bolo:\n"
     "- Saiu **limpo** ou com migalhas secas: esta pronto\n"
     "- Saiu com **massa crua**: asse mais 5 min e teste de novo\n"
     "- Evite abrir o forno na primeira metade do tempo para nao murchar"),
    ("o que fazer quando o arroz queima",
     "- Desligue o fogo na hora\n"
     "- Transfira a parte boa para outra panela **sem raspar o fundo**\n"
     "- Coloque uma **fatia de pao** por cima por ~5 min: ela absorve o cheiro de queimado\n"
     "- Nao mexa o fundo queimado para o gosto nao se espalhar"),
    ("como conservar salsinha fresca",
     "- Lave e **seque muito bem**\n"
     "- Embrulhe em papel toalha e guarde em pote fechado na geladeira (dura ~1 semana)\n"
     "- Para longa duracao: **pique e congele** em forma de gelo com um fio de azeite"),
    ("como amaciar carne dura",
     "Opcoes que funcionam:\n"
     "- **Marinada** com suco de abacaxi, mamao ou kiwi (enzimas naturais)\n"
     "- **Vinagre ou limao** na marinada\n"
     "- **Bater** a carne com martelo de cozinha\n"
     "- **Cozimento longo em fogo baixo** (ensopados) derrete o colageno"),
    ("com o que posso substituir ovo no bolo",
     "Para **cada ovo**, escolha uma opcao:\n"
     "- 1/2 **banana** amassada\n"
     "- 1/4 de xicara de **iogurte** natural\n"
     "- 1/4 de xicara de **pure de maca**\n"
     "- 1 colher de sopa de **linhaca ou chia** moida + 3 de agua (descanse 5 min ate virar gel)\n"
     "- Funcionam bem em bolos e muffins; a massa pode ficar um pouco mais umida"),
    ("com o que substituir leite no bolo",
     "Troque pela **mesma quantidade** de:\n"
     "- **Bebida vegetal** (aveia, amendoa ou soja)\n"
     "- **Agua + 1 colher de sopa de manteiga** derretida por xicara (repoe a gordura)\n"
     "- **Suco de laranja**, em receitas que combinam (perfuma a massa)"),
    ("com o que substituir acucar no bolo",
     "- **Acucar mascavo ou demerara:** mesma medida, deixa a massa mais umida\n"
     "- **Adocante de forno:** siga a conversao do rotulo\n"
     "- **Banana ou tamara** amassadas: adocam e dao umidade; reduza um pouco os liquidos da receita"),
]


def normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


#FAQ via TF-IDF
class Resultado(NamedTuple):
    dados: str
    confianca: float


def _prep_faq(texto: str) -> str:
    # tokeniza (remove stopwords + lematiza) para o TF-IDF casar parafrases:
    # 'como faco para deixar o arroz soltinho' ~ 'como fazer arroz soltinho'.
    from nlp import tokenizar
    return " ".join(tokenizar(texto)) or normalizar(texto)


_faq_perguntas = [_prep_faq(p) for p, _ in FAQ]
_faq_respostas = [r for _, r in FAQ]
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_faq_matriz = _vectorizer.fit_transform(_faq_perguntas)


def buscar_faq(pergunta: str) -> Optional[Resultado]:
    vec = _vectorizer.transform([_prep_faq(pergunta)])
    sims = cosine_similarity(vec, _faq_matriz)[0]
    idx = sims.argmax()
    score = float(sims[idx])
    if score >= LIMIAR_FAQ:
        return Resultado(_faq_respostas[idx], score)
    return None


#Busca de receitas
GENERICAS = {"receita", "receitas", "fazer", "preparar", "cozinhar", "quero",
             "uma", "tem", "como", "busca", "mostra", "que", "dia"}


def buscar_por_nome(texto: str) -> list[dict]:
    palavras = [p for p in normalizar(texto).split() if len(p) > 2 and p not in GENERICAS]
    if not palavras:
        return []
    res = []
    for r in RECEITAS:
        partes = [p for p in normalizar(r["nome"]).split() if len(p) > 2]
        score = sum(1 for p in partes if any(p in u or u in p for u in palavras))
        if score:
            res.append((r, score))
    res.sort(key=lambda x: -x[1])
    return [r for r, _ in res[:3]]


def buscar_por_categoria(categoria: str) -> list[dict]:
    cat = normalizar(categoria)
    return [r for r in RECEITAS if cat in normalizar(r["categoria"])]


def buscar_por_ingredientes(tokens: list[str]) -> list[dict]:
    res = []
    for r in RECEITAS:
        ings = " ".join(normalizar(i) for i in r["ingredientes"])
        matches = sum(1 for p in tokens if len(p) > 2 and p in ings)
        if matches:
            res.append((r, matches))
    res.sort(key=lambda x: -x[1])
    return [r for r, _ in res[:3]]


def receita_aleatoria() -> dict:
    return random.choice(RECEITAS)


# Filtro de tema: vocabulario culinario montado a partir da propria base
# (nomes, ingredientes, categorias, FAQ) + termos gerais de cozinha. Serve de
# "fast-path": se a mensagem encosta nesse vocabulario, e culinaria e nao
# precisa consultar o guardiao LLM (que erra em modelo pequeno).
_TERMOS_CULINARIOS = {
    "receita", "receitas", "cozinhar", "cozinha", "fazer", "preparar", "preparo",
    "comida", "comer", "prato", "ingrediente", "ingredientes", "tempero",
    "temperar", "assar", "fritar", "refogar", "cozer", "ferver", "grelhar",
    "forno", "fogao", "panela", "frigideira", "massa", "molho", "caldo",
    "doce", "salgado", "sobremesa", "lanche", "bebida", "acompanhamento",
    "almoco", "jantar", "cafe", "sabor", "saboroso", "delicioso", "gostoso",
    "porcao", "porcoes", "xicara", "colher", "grama", "gramas", "litro",
    "substituir", "substituicao", "conservar", "congelar", "descongelar",
    "geladeira", "freezer", "fresco", "marinar", "amaciar", "dourar",
    "picar", "bater", "misturar", "mexer", "acucar", "sal", "oleo", "azeite",
    "manteiga", "farinha", "ovo", "ovos", "leite", "agua", "fermento",
    "beber", "tomar", "suco", "vitamina", "drink", "cafe", "cha",
}

# Interrogativos/genericos que vazam das perguntas da FAQ (nao sao stopword no
# NLTK) e gerariam falso positivo. Removidos do vocabulario culinario.
_GENERICOS_FORA = {
    "quanto", "quantos", "qual", "quais", "quando", "onde", "quem", "porque",
    "como", "fazer", "ter", "dia", "ano", "hoje", "agora", "coisa", "algo",
    "usar", "saber", "diferenca", "servir", "serve",
}


def _vocab_culinario() -> set[str]:
    # tokenizar (NLP) remove stopwords/pontuacao e lematiza, evitando que
    # palavras genericas das perguntas da FAQ (qual, como, para...) poluam o
    # vocabulario e gerem falso positivo. Os dois lados usam o mesmo tokenizer.
    from nlp import tokenizar
    vocab = set()
    for termo in _TERMOS_CULINARIOS:
        vocab.update(tokenizar(termo))
    for r in RECEITAS:
        vocab.update(tokenizar(r["nome"]))
        vocab.update(tokenizar(r.get("categoria", "")))
        for ing in r["ingredientes"]:
            vocab.update(tokenizar(ing))
    for pergunta, _ in FAQ:
        vocab.update(tokenizar(pergunta))
    for generico in _GENERICOS_FORA:
        vocab.difference_update(tokenizar(generico))
    return vocab


VOCAB_CULINARIO = _vocab_culinario()


def parece_culinario(texto: str) -> bool:
    """True se a mensagem encosta no vocabulario culinario da base.

    Usado como fast-path do filtro de escopo: evita bloquear perguntas de
    cozinha legitimas por uma falha do guardiao LLM. Nao bloqueia nada por si
    so (so confirma que e do tema)."""
    from nlp import tokenizar
    return any(t in VOCAB_CULINARIO for t in tokenizar(texto))


#formatacao
def formatar_receita(r: dict) -> str:
    ings = "\n".join(f"- {i.capitalize()}" for i in r["ingredientes"])
    passos = "\n".join(f"{n}. {p}" for n, p in enumerate(r["instrucoes"], 1))
    return (f"\n**{r['nome']}**\n"
            f"Tempo: {r['tempo']}  |  Porcoes: {r['porcoes']}  |  "
            f"Dificuldade: {r.get('dificuldade','?').title()}\n\n"
            f"**Ingredientes**\n{ings}\n\n"
            f"**Modo de preparo**\n{passos}")


def resumo_receita(r: dict) -> str:
    return (f"'{r['nome']}' — {r.get('categoria','?')}, "
            f"{r['tempo']}, dificuldade {r.get('dificuldade','?')}.")
