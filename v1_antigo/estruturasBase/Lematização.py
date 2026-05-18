import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Lista de palavras para lematizar
palavras = ["correndo", "melhores", "carros", "comendo", "dados"]

# Lematização (assumindo substantivos, mas pode mudar para verbos com pos='v')
lemmas = [lemmatizer.lemmatize(palavra, pos='v') for palavra in palavras]
print("Lemas:", lemmas)
