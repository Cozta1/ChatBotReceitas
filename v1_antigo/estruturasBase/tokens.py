import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Exemplo de frase
frase = "Olá! Como você está hoje?"

# Tokenização
tokens = word_tokenize(frase.lower())
print("Tokens:", tokens)
