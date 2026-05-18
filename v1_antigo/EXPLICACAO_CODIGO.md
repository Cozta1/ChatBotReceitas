# Explicação Completa do ChefBot — Código Linha por Linha

Documento técnico explicando o porquê de cada escolha de desenvolvimento e como cada parte do código funciona.

---

## 1. IMPORTS E SETUP INICIAL

```python
import random
import string
import unicodedata
```

- **`random`** — biblioteca padrão do Python para escolhas aleatórias. Usamos para sortear receitas, mensagens variadas, sugestões, etc. Sem aleatoriedade, o bot sempre daria a mesma resposta.
- **`string`** — contém constantes úteis como `string.punctuation` (todos os sinais de pontuação: `.,;!?` etc). Usamos para filtrar pontuação ao tokenizar.
- **`unicodedata`** — biblioteca padrão que manipula caracteres Unicode. Usamos especificamente para **remover acentos** (ex: "maçã" → "maca"). Isso é crucial porque o usuário pode digitar "não", "nao", "Não" — tudo precisa ser tratado como igual.

```python
import nltk
import numpy as np
```

- **`nltk`** (Natural Language Toolkit) — a biblioteca mais popular de PLN (Processamento de Linguagem Natural) em Python. Usamos para tokenização, stopwords e lematização.
- **`numpy`** — biblioteca de computação numérica. O `scikit-learn` (nosso classificador) trabalha com arrays do numpy, não com listas Python. Precisamos dele para montar os vetores de treino.

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
```

- **`word_tokenize`** — função que quebra uma string em uma lista de palavras, já tratando casos como contrações, pontuação junto de palavras, etc. Mais robusto do que simplesmente dar `split()`.
- **`stopwords`** — palavras muito comuns e sem carga semântica ("de", "o", "a", "para", "com"). Removê-las faz o classificador focar no que importa. Por exemplo: "eu quero **uma receita de feijoada**" → "quero receita feijoada".
- **`WordNetLemmatizer`** — reduz palavras à forma base ("cozinhando" → "cozinhar", "fazendo" → "fazer"). Isso agrupa variações da mesma palavra, dando ao classificador menos trabalho.
- **`MultinomialNB`** — o classificador Naive Bayes Multinomial do scikit-learn. É o padrão para classificação de texto.

```python
from receitas import RECEITAS
from dados import (DADOS_TREINO, SAUDACOES, ...)
```

Separamos os dados estáticos em arquivos diferentes (`receitas.py` e `dados.py`) para manter o código limpo. O código do bot fica em um arquivo, os dados em outro.

```python
for pacote in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(pacote, quiet=True)
```

O NLTK precisa baixar alguns pacotes de dados na primeira execução. Esse loop garante que tudo esteja disponível antes de começar. O `quiet=True` evita poluir o terminal.

```python
lematizador = WordNetLemmatizer()
stopwords_pt = set(stopwords.words('portuguese'))
```

Criamos as instâncias uma única vez (a nível de módulo) para reusar em todas as chamadas. Usar `set` em vez de `list` para as stopwords é uma otimização: verificar se uma palavra está num `set` é O(1) (instantâneo), numa lista é O(n).

---

## 2. PROCESSAMENTO DE TEXTO

### `normalizar(texto)`

```python
def normalizar(texto):
    sem_acento = unicodedata.normalize('NFKD', texto.lower())
    return ''.join(c for c in sem_acento if not unicodedata.combining(c))
```

**O que faz:** deixa tudo minúsculo e remove acentos.

**Como funciona:**
1. `texto.lower()` — "Não Quero" → "não quero"
2. `unicodedata.normalize('NFKD', ...)` — decompõe caracteres acentuados. O "ã" vira "a" + um caractere invisível de til.
3. `''.join(c for c in sem_acento if not unicodedata.combining(c))` — filtra os caracteres "combinantes" (os acentos decompostos), sobrando só as letras simples.

**Por que:** usuários digitam de formas diferentes ("não", "nao", "Não", "NÃO") e o bot precisa tratar como a mesma coisa.

### `tokenizar(texto, filtrar_stopwords=False)`

```python
def tokenizar(texto, filtrar_stopwords=False):
    tokens = [t for t in word_tokenize(normalizar(texto), language='portuguese')
              if t not in string.punctuation and len(t) > 1]
    if filtrar_stopwords:
        tokens = [t for t in tokens if t not in stopwords_pt]
    return [lematizador.lemmatize(t, pos='v') for t in tokens]
```

**O que faz:** transforma o texto bruto em uma lista de palavras limpas.

**Linha por linha:**
1. `word_tokenize(normalizar(texto), language='portuguese')` — primeiro normaliza, depois quebra em palavras usando o tokenizador do NLTK em modo português.
2. `if t not in string.punctuation and len(t) > 1` — remove sinais de pontuação e palavras de 1 caractere (normalmente lixo).
3. `if filtrar_stopwords: ...` — opcional. Quando queremos classificar a intenção, removemos stopwords. Quando queremos buscar ingredientes, mantemos (porque "sem" e "não" têm significado).
4. `lematizador.lemmatize(t, pos='v')` — reduz à forma base considerando como verbo (`pos='v'`). "cozinhando" → "cozinhar".

---

## 3. CLASSIFICADOR NAIVE BAYES

Essa é a parte mais importante teoricamente.

### O que é Naive Bayes?

É um algoritmo de **classificação supervisionada** baseado no **Teorema de Bayes**. A fórmula é:

```
P(classe | palavras) = P(palavras | classe) × P(classe) / P(palavras)
```

Em português: "A probabilidade de ser uma certa intenção dado as palavras da frase é igual à probabilidade daquelas palavras aparecerem naquela intenção, vezes a probabilidade geral daquela intenção, dividido pela probabilidade das palavras em geral."

**Por que "Naive" (ingênuo)?** Porque ele assume que as palavras são **independentes entre si** — o que é falso na linguagem natural, mas na prática funciona muito bem.

**Por que Multinomial?** Porque modela a contagem/presença de palavras. É o padrão para texto.

### Preparação dos dados

```python
tokens_treino = [tokenizar(frase) for frase, _ in DADOS_TREINO]
labels_treino = [label for _, label in DADOS_TREINO]
```

`DADOS_TREINO` é uma lista de tuplas `(frase, intenção)`, tipo:
```python
[("oi", "saudacao"), ("quero uma sobremesa", "categoria:sobremesa"), ...]
```

Separamos em duas listas paralelas: uma com os tokens (entrada) e outra com os rótulos (saída esperada).

### Construção do vocabulário

```python
vocabulario = sorted(set(palavra for frase in tokens_treino for palavra in frase))
indice_palavra = {palavra: i for i, palavra in enumerate(vocabulario)}
```

**O que é:** o "dicionário" de todas as palavras que o bot conhece, tiradas dos exemplos de treino.

- `set(...)` — garante que cada palavra aparece só uma vez
- `sorted(...)` — ordena alfabeticamente (para ter reprodutibilidade)
- `indice_palavra` — mapa de palavra → posição no vocabulário. Ex: `{"ajuda": 0, "arroz": 1, "bebida": 2, ...}`

### `texto_para_vetor(tokens)`

```python
def texto_para_vetor(tokens):
    vetor = np.zeros(len(vocabulario))
    for token in tokens:
        if token in indice_palavra:
            vetor[indice_palavra[token]] = 1
    return vetor
```

**O que faz:** transforma uma lista de palavras num **vetor numérico** que o classificador entende.

**Exemplo prático:** se o vocabulário é `["ajuda", "arroz", "bebida", "feijoada"]` e a frase é `"quero bebida"`, o vetor fica `[0, 0, 1, 0]` — só a posição da palavra "bebida" tem 1.

Isso é chamado de **bag-of-words binário**: representa a presença (1) ou ausência (0) de cada palavra, ignorando ordem.

### Treinamento

```python
classificador = MultinomialNB()
classificador.fit(np.array([texto_para_vetor(t) for t in tokens_treino]), labels_treino)
```

1. Cria uma instância do MultinomialNB
2. `.fit(X, y)` — treina com:
   - `X` = matriz onde cada linha é uma frase vetorizada
   - `y` = lista de rótulos (intenções) correspondentes

Durante o `fit`, o classificador calcula as probabilidades: "dado que a classe é `saudacao`, qual a probabilidade de ver a palavra 'oi'?" — e guarda isso para usar na hora de prever.

### `detectar_intencao(tokens)`

```python
def detectar_intencao(tokens):
    if not tokens:
        return "buscar_por_ingredientes"
    vetor = texto_para_vetor(tokens).reshape(1, -1)
    probs = classificador.predict_proba(vetor)[0]
    if probs.max() >= probs.mean() * 1.5:
        return classificador.predict(vetor)[0]
    return "buscar_por_ingredientes"
```

**Linha por linha:**

1. Se não há tokens, assume busca por ingredientes como fallback.
2. `.reshape(1, -1)` — o scikit-learn espera uma **matriz 2D**, mesmo que seja uma única amostra. O reshape transforma o vetor 1D em uma "matriz de 1 linha".
3. `predict_proba(vetor)` — retorna as probabilidades de cada classe. Ex: `[0.6, 0.1, 0.05, ...]`
4. **Threshold de confiança** (`probs.max() >= probs.mean() * 1.5`): só aceita a classificação se a melhor opção for pelo menos 1,5× a média de todas. Isso evita que o bot "chute" quando a frase é ambígua.
5. Se passa do threshold, usa a predição. Senão, volta ao fallback de buscar ingredientes.

---

## 4. BUSCA DE RECEITAS

### `buscar_por_nome(texto)`

```python
def buscar_por_nome(texto):
    palavras = [p for p in normalizar(texto).split() if len(p) > 2 and p not in PALAVRAS_GENERICAS]
    if not palavras:
        return []
    resultados = []
    for receita in RECEITAS:
        partes_nome = [p for p in normalizar(receita["nome"]).split() if len(p) > 2]
        score = sum(1 for p in partes_nome if any(p in u or u in p for u in palavras))
        if score:
            resultados.append((receita, score))
    resultados.sort(key=lambda x: -x[1])
    return [r for r, _ in resultados[:3]]
```

**O que faz:** procura receitas cujo nome contenha as palavras da frase.

**Detalhes importantes:**
- Filtra palavras genéricas como "receita", "fazer", "quero" — elas poluiriam o match.
- Filtra palavras com menos de 3 letras (geralmente artigos/preposições).
- Para cada receita, calcula um **score** baseado em quantas palavras do nome batem com as da frase.
- `any(p in u or u in p for u in palavras)` — comparação bidirecional: "feijão" bate com "feijoada" e vice-versa (uma string contém a outra).
- Retorna os top 3 resultados.

### `buscar_por_categoria(categoria)`

```python
def buscar_por_categoria(categoria):
    return [r for r in RECEITAS if r["categoria"] == categoria]
```

Bem direto: filtra receitas que têm a categoria exata. As categorias são definidas em cada receita: `"sobremesa"`, `"lanche"`, `"bebida"`, etc.

### `buscar_por_ingredientes(palavras)`

```python
def buscar_por_ingredientes(palavras):
    resultados = []
    for receita in RECEITAS:
        texto_ings = " ".join(normalizar(i) for i in receita["ingredientes"])
        matches = sum(1 for p in palavras if len(p) > 2 and p in texto_ings)
        if matches:
            resultados.append((receita, matches))
    resultados.sort(key=lambda x: -x[1])
    return [r for r, _ in resultados[:3]]
```

**O que faz:** conta quantas palavras do usuário aparecem nos ingredientes da receita.

- `" ".join(...)` — junta todos os ingredientes numa única string para buscar com `in`.
- `sum(1 for p in palavras if ...)` — conta quantas palavras batem (um contador simples).
- Ordena do maior para o menor e retorna top 3.

---

## 5. SESSÃO (MÁQUINA DE ESTADOS)

### `nova_sessao()` e `resetar(sessao)`

```python
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
```

A **sessão** é um dicionário que guarda tudo sobre o estado da conversa. É a "memória" do bot.

**Campos:**
- `estado` — em qual fase da conversa estamos (`inicio`, `sondagem`, `passo_a_passo`, `conclusao`)
- `receita` — a receita escolhida (quando já passou da sondagem)
- `candidatas` — lista de receitas sendo oferecidas ao usuário
- `candidata_atual` — índice da candidata sendo mostrada agora
- `pergunta_tipo` — sub-estado dentro de "sondagem" (`sugestao`, `confirmar_proxima`, `confirmar_ingredientes`, `confirmar_passos`)
- `passo_atual` — em qual instrução do preparo estamos
- `historico` — log das mensagens trocadas

### Por que máquina de estados?

Um chatbot precisa **lembrar o que veio antes**. Quando o usuário diz "sim", o bot precisa saber a qual pergunta ele está respondendo. A máquina de estados resolve isso: cada mensagem é interpretada no **contexto do estado atual**.

**Fluxo principal:**
```
inicio → sondagem → passo_a_passo → conclusao
```

**Sub-fluxo dentro de sondagem:**
```
sugestao → confirmar_proxima → confirmar_ingredientes → confirmar_passos
```

---

## 6. FORMATAÇÃO E PASSOS

### `formatar_receita(receita)`

Monta o texto completo da receita com cabeçalho, ingredientes numerados e passos. Usa **f-strings** (formatação moderna do Python) para inserir os dados dinamicamente.

### `mostrar_passo`, `proximo_passo`, `entregar_completa`

Funções auxiliares do modo passo a passo. `proximo_passo` incrementa o contador e verifica se a receita acabou (mudando o estado para "conclusão").

---

## 7. DETECÇÃO DE SIM/NÃO

```python
PALAVRAS_SIM = ["sim", "claro", "pode", "vamos", "ok", "legal", ...]
PALAVRAS_NAO = ["nao", "noa", "nop", "prefiro", "outro", ...]
```

Listas simples usadas para detectar confirmação/rejeição com `any(p in texto for p in PALAVRAS_X)`. Inclui variações como "noa" (typo comum de "não").

**Por que não usar o NB aqui?** Porque essas são respostas curtas e diretas. Regras simples são mais confiáveis do que machine learning para casos tão específicos.

---

## 8. FLUXO DE SONDAGEM

### `sugerir_receita(sessao)` e `resposta_sondagem(texto, sessao)`

Essa é a parte que controla a "negociação" entre o bot e o usuário:

1. **Bot sugere uma receita** → usuário aceita ou recusa
2. **Se recusa** → bot pergunta se quer ver a próxima opção
3. **Se aceita** → bot pergunta se pode mostrar os ingredientes
4. **Se aceita** → bot pergunta se pode começar o passo a passo
5. **Se aceita** → entra no modo passo a passo

Cada etapa é um valor diferente de `pergunta_tipo`. A função `resposta_sondagem` lê esse valor e processa a resposta do usuário conforme o contexto.

**Design importante:** o bot **nunca avança sem confirmação explícita**. Isso evita situações em que o usuário é "empurrado" para uma receita que não queria.

---

## 9. RESPOSTA PRINCIPAL

### `gerar_resposta(mensagem, sessao)`

A função mais importante — o "cérebro" do bot. Segue esta lógica em cascata:

```python
tokens = tokenizar(mensagem, filtrar_stopwords=True)
texto = normalizar(mensagem)
estado = sessao["estado"]
```

Primeiro prepara três versões da mensagem: tokens (para o NB), texto normalizado (para buscar substrings) e o estado atual.

**Então verifica o estado:**

1. **Se estiver no `passo_a_passo`** — só processa comandos de navegação (pronto, manda tudo, cancelar)
2. **Se estiver em `sondagem`** — delega para `resposta_sondagem`
3. **Se estiver em `conclusao`** — delega para `processar_conclusao`
4. **Se estiver em `inicio`** — faz a classificação completa:
   - Verifica intenções de conversa (saudação, despedida, etc.)
   - Se for `buscar_por_nome`, chama a função correspondente
   - Se for `receita_aleatoria`, sorteia 3 receitas
   - Se for `categoria:X`, filtra pela categoria
   - **Fallback em cascata:** tenta buscar por nome → por ingredientes → pedido genérico → mensagem de erro

### Por que verificações por regras *antes* do NB?

O NB é bom para casos gerais, mas pode errar em casos específicos. Por exemplo, "cancelar" deve sempre cancelar, independente do que o classificador pense. As regras explícitas (`SINAIS_CANCELAR`, `SINAIS_COMPLETO`) garantem comportamento previsível.

---

## 10. RESUMO DAS FERRAMENTAS E POR QUÊ

| Ferramenta | Para quê |
|---|---|
| **NLTK** | Processar linguagem natural — tokenizar, remover stopwords, lematizar |
| **scikit-learn (MultinomialNB)** | Classificar a intenção do usuário (saudação, busca, categoria, etc.) |
| **numpy** | Manipular os vetores de treino em formato que o sklearn aceita |
| **unicodedata** | Remover acentos para tratar "não" = "nao" |
| **Regras explícitas (listas)** | Casos específicos onde ML seria desnecessário ou errático |
| **Dicionário de sessão** | Memória da conversa — máquina de estados |
| **Busca em cascata** | Fallbacks em camadas, garantindo sempre uma resposta razoável |

---

## PONTOS DE DEFESA PARA O PROFESSOR

### 1. "Por que Naive Bayes e não algo mais moderno?"
NB é rápido, interpretável, funciona bem com poucos dados de treino e é o padrão para classificação de texto. Em um domínio restrito (receitas), não precisamos de redes neurais.

### 2. "Por que MultinomialNB e não outros tipos?"
Multinomial é feito para texto (contagem/presença de palavras). GaussianNB assume distribuição contínua — errado para texto. BernoulliNB tem viés estranho quando o vocabulário é pequeno porque modela ausência de palavras.

### 3. "Por que misturar ML com regras?"
Para casos críticos e específicos (cancelar, confirmar passo), regras são mais confiáveis. Para intenções abertas (saudação, busca), o ML generaliza melhor. É uma abordagem **híbrida** e prática.

### 4. "Por que uma máquina de estados?"
Porque o significado de uma mensagem depende do contexto. "Sim" depois de "quer ver os ingredientes?" é diferente de "sim" depois de "terminou o passo?". Sem estado, o bot seria incoerente.

### 5. "Por que a busca tem tantos fallbacks em cascata?"
Para maximizar a chance de dar uma resposta útil. Se a classificação do NB falhar, ainda tentamos busca por nome; se falhar, por ingredientes; se falhar, sugerimos receitas aleatórias. Nunca deixamos o usuário sem resposta.
