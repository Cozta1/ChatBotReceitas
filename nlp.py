"""NLP: tokenizacao NLTK + classificador Naive Bayes de intencao.

Confianca segue o criterio do trabalho 1: max(probs) >= mean(probs) * 1.5.
"""
import string
import unicodedata

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

_lemma = WordNetLemmatizer()
_stop_pt = set(stopwords.words("portuguese"))


def normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def tokenizar(texto: str, filtrar_stopwords: bool = True) -> list[str]:
    tokens = [
        t for t in word_tokenize(normalizar(texto), language="portuguese")
        if t not in string.punctuation and len(t) > 1
    ]
    if filtrar_stopwords:
        tokens = [t for t in tokens if t not in _stop_pt]
    return [_lemma.lemmatize(t, pos="v") for t in tokens]


# Dataset de treino herdado e expandido a partir do trabalho v1

TREINO: list[tuple[str, str]] = [
    # saudacao
    ("oi", "saudacao"), ("ola", "saudacao"), ("hey", "saudacao"), ("eai", "saudacao"),
    ("bom dia", "saudacao"), ("boa tarde", "saudacao"), ("boa noite", "saudacao"),
    ("salve", "saudacao"), ("fala", "saudacao"), ("hello", "saudacao"),
    ("tudo bem", "saudacao"), ("como vai", "saudacao"),
    ("ola tudo bom", "saudacao"), ("eai tudo certo", "saudacao"),

    # despedida
    ("tchau", "despedida"), ("ate mais", "despedida"), ("ate logo", "despedida"),
    ("bye", "despedida"), ("flw", "despedida"), ("falou", "despedida"),
    ("adeus", "despedida"), ("xau", "despedida"), ("vou indo", "despedida"),
    ("vou embora", "despedida"), ("vou sair agora", "despedida"),

    # agradecimento
    ("obrigado", "agradecimento"), ("obrigada", "agradecimento"), ("valeu", "agradecimento"),
    ("brigado", "agradecimento"), ("brigada", "agradecimento"), ("thanks", "agradecimento"),
    ("muito obrigado", "agradecimento"), ("grato", "agradecimento"),
    ("legal valeu", "agradecimento"), ("agradeco", "agradecimento"), ("gratidao", "agradecimento"),

    # ajuda
    ("ajuda", "ajuda"), ("help", "ajuda"), ("como funciona", "ajuda"),
    ("o que voce faz", "ajuda"), ("comandos", "ajuda"), ("menu", "ajuda"),
    ("quais opcoes", "ajuda"), ("como usar", "ajuda"), ("o que posso fazer", "ajuda"),
    ("me ajuda", "ajuda"), ("preciso de ajuda", "ajuda"), ("nao sei usar", "ajuda"),

    # receita_aleatoria
    ("receita aleatoria", "receita_aleatoria"), ("qualquer receita", "receita_aleatoria"),
    ("receita surpresa", "receita_aleatoria"), ("sugestao de receita", "receita_aleatoria"),
    ("sugira uma receita", "receita_aleatoria"), ("sorteia uma receita", "receita_aleatoria"),
    ("me surpreenda", "receita_aleatoria"), ("escolha por mim", "receita_aleatoria"),
    ("me da uma receita", "receita_aleatoria"), ("random", "receita_aleatoria"),
    ("surpresa", "receita_aleatoria"), ("aleatoria", "receita_aleatoria"),
    ("sorteia", "receita_aleatoria"), ("sugere algo", "receita_aleatoria"),
    ("quero uma receita", "receita_aleatoria"), ("me sugere algo", "receita_aleatoria"),
    ("o que voce sugere", "receita_aleatoria"), ("me da uma sugestao", "receita_aleatoria"),
    ("tem alguma receita", "receita_aleatoria"), ("quero fazer algo", "receita_aleatoria"),

    # categoria:sobremesa
    ("quero uma sobremesa", "categoria:sobremesa"), ("receita de sobremesa", "categoria:sobremesa"),
    ("tem algum doce", "categoria:sobremesa"), ("receita de bolo", "categoria:sobremesa"),
    ("quero um pudim", "categoria:sobremesa"), ("algo doce", "categoria:sobremesa"),
    ("brigadeiro", "categoria:sobremesa"), ("sobremesa", "categoria:sobremesa"),
    ("torta", "categoria:sobremesa"), ("receita doce", "categoria:sobremesa"),
    ("sobremesas faceis", "categoria:sobremesa"), ("sobremesa facil", "categoria:sobremesa"),

    # categoria:prato principal
    ("receita de almoco", "categoria:prato principal"),
    ("o que fazer para almoco", "categoria:prato principal"),
    ("prato principal", "categoria:prato principal"),
    ("receita para jantar", "categoria:prato principal"),
    ("comida", "categoria:prato principal"), ("almoco", "categoria:prato principal"),
    ("jantar", "categoria:prato principal"), ("refeicao", "categoria:prato principal"),
    ("prato do dia", "categoria:prato principal"), ("janta", "categoria:prato principal"),
    ("prato quente", "categoria:prato principal"),
    ("quero algo para comer", "categoria:prato principal"),
    ("me da um prato", "categoria:prato principal"),

    # categoria:lanche
    ("receita de lanche", "categoria:lanche"), ("quero um salgado", "categoria:lanche"),
    ("tem tapioca", "categoria:lanche"), ("receita de coxinha", "categoria:lanche"),
    ("lanche rapido", "categoria:lanche"), ("lanche", "categoria:lanche"),
    ("salgadinho", "categoria:lanche"), ("sanduiche", "categoria:lanche"),
    ("petisco", "categoria:lanche"), ("coxinha", "categoria:lanche"),
    ("pao de queijo", "categoria:lanche"),

    # categoria:bebida
    ("receita de suco", "categoria:bebida"), ("quero uma bebida", "categoria:bebida"),
    ("tem drink", "categoria:bebida"), ("vitamina de fruta", "categoria:bebida"),
    ("receita de limonada", "categoria:bebida"), ("bebida", "categoria:bebida"),
    ("suco", "categoria:bebida"), ("vitamina", "categoria:bebida"),
    ("limonada", "categoria:bebida"), ("algo para beber", "categoria:bebida"),
    ("quero beber algo", "categoria:bebida"), ("tomar uma bebida", "categoria:bebida"),

    # categoria:acompanhamento
    ("receita de farofa", "categoria:acompanhamento"),
    ("quero um acompanhamento", "categoria:acompanhamento"),
    ("tem salada", "categoria:acompanhamento"),
    ("receita de pure", "categoria:acompanhamento"),
    ("acompanhamento", "categoria:acompanhamento"), ("farofa", "categoria:acompanhamento"),
    ("salada", "categoria:acompanhamento"), ("pure", "categoria:acompanhamento"),
    ("vinagrete", "categoria:acompanhamento"),

    # buscar_por_ingredientes
    ("tenho frango", "buscar_por_ingredientes"),
    ("tenho ovos e queijo", "buscar_por_ingredientes"),
    ("o que fazer com batata", "buscar_por_ingredientes"),
    ("tenho leite condensado", "buscar_por_ingredientes"),
    ("posso fazer com frango e alho", "buscar_por_ingredientes"),
    ("o que posso cozinhar com arroz", "buscar_por_ingredientes"),
    ("tenho cebola e alho em casa", "buscar_por_ingredientes"),
    ("o que fazer com macarrao", "buscar_por_ingredientes"),
    ("tenho banana e ovo", "buscar_por_ingredientes"),
    ("usar frango e batata", "buscar_por_ingredientes"),
    ("preparar algo com ovo", "buscar_por_ingredientes"),
    ("tenho bacon e linguica", "buscar_por_ingredientes"),
    ("tenho frango cebola alho", "buscar_por_ingredientes"),
    ("o que preparo com esses ingredientes", "buscar_por_ingredientes"),
    ("receita com frango", "buscar_por_ingredientes"),
    ("fazer com leite e ovo", "buscar_por_ingredientes"),

    # buscar_por_nome
    ("receita de feijoada", "buscar_por_nome"),
    ("como fazer estrogonofe", "buscar_por_nome"),
    ("quero fazer frango xadrez", "buscar_por_nome"),
    ("tem receita de bolo de cenoura", "buscar_por_nome"),
    ("busca feijoada", "buscar_por_nome"),
    ("me mostra a receita de arroz", "buscar_por_nome"),
    ("quero a receita de coxinha", "buscar_por_nome"),
    ("como se faz limonada", "buscar_por_nome"),
    ("procura receita de pudim", "buscar_por_nome"),
    ("me da a receita de moqueca", "buscar_por_nome"),
    ("como preparar feijao tropeiro", "buscar_por_nome"),

    # confirmacao_passo (passo a passo)
    ("pronto", "confirmacao_passo"), ("ja fiz", "confirmacao_passo"),
    ("feito", "confirmacao_passo"), ("pode continuar", "confirmacao_passo"),
    ("proximo", "confirmacao_passo"), ("proximo passo", "confirmacao_passo"),
    ("continua", "confirmacao_passo"), ("ok feito", "confirmacao_passo"),
    ("terminei esse passo", "confirmacao_passo"),
    ("segue", "confirmacao_passo"), ("vamos pro proximo", "confirmacao_passo"),

    # pedir_receita_completa
    ("manda tudo", "pedir_receita_completa"),
    ("quero a receita inteira", "pedir_receita_completa"),
    ("pode mandar tudo", "pedir_receita_completa"),
    ("receita completa", "pedir_receita_completa"),
    ("quero ver tudo de uma vez", "pedir_receita_completa"),
    ("nao precisa passo a passo", "pedir_receita_completa"),
    ("manda direto", "pedir_receita_completa"),

    # pergunta_culinaria -> KB(FAQ) ou LLM
    ("como substituir manteiga por oleo", "pergunta_culinaria"),
    ("quanto tempo guardar carne na geladeira", "pergunta_culinaria"),
    ("como descongelar carne", "pergunta_culinaria"),
    ("posso congelar molho de tomate", "pergunta_culinaria"),
    ("para que serve folha de louro", "pergunta_culinaria"),
    ("qual a diferenca entre creme de leite fresco e uht", "pergunta_culinaria"),
    ("como saber se o bolo esta assado", "pergunta_culinaria"),
    ("equivalencia de xicara em gramas", "pergunta_culinaria"),
    ("como amaciar carne", "pergunta_culinaria"),
    ("como conservar salsinha", "pergunta_culinaria"),
    ("qual a diferenca entre cozinhar a vapor e refogar", "pergunta_culinaria"),
    ("porque o pao de queijo cresce", "pergunta_culinaria"),
    ("como fazer molho bechamel", "pergunta_culinaria"),
    ("qual farinha usar para pizza", "pergunta_culinaria"),
]

_tokens_treino = [tokenizar(f) for f, _ in TREINO]
_labels = [l for _, l in TREINO]
_vocab = sorted({t for toks in _tokens_treino for t in toks})
_indice = {p: i for i, p in enumerate(_vocab)}


def _vetorizar(tokens: list[str]) -> np.ndarray:
    v = np.zeros(len(_vocab))
    for t in tokens:
        if t in _indice:
            v[_indice[t]] = 1
    return v


_clf = MultinomialNB()
_clf.fit(np.array([_vetorizar(t) for t in _tokens_treino]), _labels)


def detectar_intencao(texto: str, fator: float = 1.8) -> tuple[str, bool]:
    """Retorna (intencao, confiante). `confiante=False` -> caller usa fallback.

    Confianca exige que (a) algum token bata no vocabulario treinado e
    (b) probs.max() >= probs.mean() * fator. Sem (a) o classifier
    responderia so com priors -> retornavamos saudacao/etc. errado.
    """
    tokens = tokenizar(texto)
    if not tokens:
        return ("", False)
    v = _vetorizar(tokens)
    if v.sum() == 0:  # nenhum token do input esta no vocab
        return ("", False)
    probs = _clf.predict_proba(v.reshape(1, -1))[0]
    idx = int(probs.argmax())
    return (_clf.classes_[idx], bool(probs[idx] >= probs.mean() * fator))
