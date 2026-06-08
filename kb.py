"""Base de conhecimento: receitas + FAQ culinaria + busca TF-IDF."""
from __future__ import annotations
import random
import unicodedata
from typing import NamedTuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from receitas import RECEITAS

LIMIAR_FAQ = 0.70  # similaridade minima para a FAQ responder (abaixo disso -> LLM)

FAQ: list[tuple[str, str]] = [
    ("como substituir manteiga por oleo em bolo",
     "Use 3/4 da quantidade de oleo em relacao a manteiga. Ex.: 100g de manteiga = 75ml de oleo. Prefira oleos neutros como girassol ou canola."),
    ("como descongelar carne com seguranca",
     "O ideal e descongelar na geladeira (de um dia para o outro). Em emergencia, use o microondas na funcao descongelar ou um saco vedado submerso em agua fria, trocando a agua a cada 30 minutos. Nunca descongele em temperatura ambiente."),
    ("quanto tempo posso guardar carne na geladeira",
     "Carne crua: 1 a 2 dias. Carne cozida: 3 a 4 dias. Sempre em recipiente fechado e a no maximo 4 graus."),
    ("como conservar ovo",
     "Mantenha os ovos na geladeira, na embalagem original, longe da porta (zona de variacao de temperatura). Duram cerca de 4 a 5 semanas a partir da data de embalagem."),
    ("para que serve folha de louro",
     "A folha de louro aromatiza pratos longos como feijao, ensopados e molhos de tomate. Use 1 ou 2 folhas por panela e retire antes de servir."),
    ("como saber se o oleo esta na temperatura para fritar",
     "Coloque um palito de fosforo no oleo frio e ligue o fogo. Quando o palito acender, o oleo esta entre 170 e 180 graus, ideal para fritar."),
    ("como evitar que o arroz empape",
     "Refogue o arroz no oleo com alho antes de adicionar a agua, use a proporcao 1 medida de arroz para 2 de agua fervente, e nao mexa durante o cozimento."),
    ("qual a diferenca entre creme de leite fresco e uht",
     "O creme de leite fresco tem mais gordura (35%) e pode ser batido em chantilly. O UHT (caixinha) tem menos gordura (20%), e mais estavel para molhos e sobremesas que nao precisam aerar."),
    ("posso substituir fermento quimico por bicarbonato",
     "Sim, mas com cuidado. 1 colher de cha de fermento quimico equivale a 1/4 de colher de cha de bicarbonato + 1/2 colher de cha de cremor tartaro. Bicarbonato puro deixa gosto se em excesso."),
    ("como deixar bife macio",
     "Retire da geladeira 20 min antes, seque bem com papel toalha, tempere com sal so na hora, e use uma frigideira bem quente. Nao vire o bife mais que uma vez por lado."),
    ("como fazer arroz soltinho",
     "Refogue o arroz no alho ate ficar branquinho, adicione agua fervente na proporcao 2:1, tampe e cozinhe em fogo baixo sem mexer. Solte com garfo no final."),
    ("quanto tempo cozinhar feijao na pressao",
     "Feijao carioca: 20 a 25 minutos apos pegar pressao. Feijao preto: 30 a 35 minutos. Sempre deixar de molho por pelo menos 4 horas antes."),
    ("como tirar o amargor da berinjela",
     "Corte a berinjela, polvilhe sal e deixe descansar por 30 minutos. O sal puxa a agua amarga. Depois enxague e seque bem antes de cozinhar."),
    ("posso congelar molho de tomate",
     "Sim. Esfrie completamente, coloque em pote ou saco de congelamento com pouco ar, e congele por ate 3 meses. Descongele na geladeira."),
    ("equivalencia xicara em gramas farinha",
     "1 xicara de farinha de trigo = aproximadamente 120 gramas. 1 colher de sopa = 8 gramas. Sempre peneirar para medir corretamente."),
    ("equivalencia xicara em gramas acucar",
     "1 xicara de acucar refinado = aproximadamente 180 gramas. 1 colher de sopa = 12 gramas."),
    ("como saber se o bolo esta assado",
     "Espete um palito no centro do bolo. Se sair limpo ou com migalhas secas, esta pronto. Se sair com massa crua, asse mais 5 minutos e teste de novo."),
    ("o que fazer quando o arroz queima",
     "Desligue o fogo, transfira a parte boa para outra panela sem raspar o fundo, e coloque uma fatia de pao por cima. O pao absorve o cheiro de queimado em 5 minutos."),
    ("como conservar salsinha fresca",
     "Lave, seque bem, embrulhe em papel toalha e guarde em pote fechado na geladeira. Dura cerca de 1 semana. Tambem pode picar e congelar em forma de gelo com azeite."),
    ("como amaciar carne dura",
     "Marinada com suco de abacaxi, mamao ou kiwi (enzimas), vinagre, ou simplesmente bater a carne com martelo. Cozinhar em fogo baixo por tempo longo tambem amacia colageno."),
]


def normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


#FAQ via TF-IDF
class Resultado(NamedTuple):
    dados: str
    confianca: float


_faq_perguntas = [normalizar(p) for p, _ in FAQ]
_faq_respostas = [r for _, r in FAQ]
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_faq_matriz = _vectorizer.fit_transform(_faq_perguntas)


def buscar_faq(pergunta: str) -> Optional[Resultado]:
    vec = _vectorizer.transform([normalizar(pergunta)])
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


#formatacao
def formatar_receita(r: dict) -> str:
    linha = "=" * 46
    ings = "\n".join(f"- {i.capitalize()}" for i in r["ingredientes"])
    passos = "\n".join(f"{i+1}. {p}" for i, p in enumerate(r["instrucoes"]))
    return (f"\n{linha}\n {r['nome'].upper()}\n{linha}\n"
            f"Tempo: {r['tempo']} | Porcoes: {r['porcoes']} | "
            f"Dificuldade: {r.get('dificuldade','?').title()}\n\n"
            f"INGREDIENTES:\n{ings}\n\nMODO DE PREPARO:\n{passos}\n{linha}")


def resumo_receita(r: dict) -> str:
    return (f"'{r['nome']}' — {r.get('categoria','?')}, "
            f"{r['tempo']}, dificuldade {r.get('dificuldade','?')}.")
