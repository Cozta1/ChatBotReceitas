from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Dataset simples
frases = [
    "oi", "olá", "bom dia",
    "tchau", "adeus", "até logo",
    "como você está", "tudo bem", "está bem",
    "qual é o seu nome", "quem é você", "como se chama"
]

intencoes = [
    "saudacao", "saudacao", "saudacao",
    "despedida", "despedida", "despedida",
    "sentimento", "sentimento", "sentimento",
    "nome", "nome", "nome"
]

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Pré-processar as frases (tokenização + lematização)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return " ".join(lemas)

frases_proc = [preprocess(f) for f in frases]

# Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases_proc)

# Treinamento
modelo = MultinomialNB()
modelo.fit(X, intencoes)

# Teste
entrada = "e aí, tudo certo?"
entrada_proc = preprocess(entrada)
entrada_vector = vectorizer.transform([entrada_proc])
intencao_prevista = modelo.predict(entrada_vector)[0]

print(f"Frase: '{entrada}'\nIntenção prevista: {intencao_prevista}")
