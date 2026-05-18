# Guia de Apresentação — ChefBot

Roteiro para apresentar cada função do bot ao vivo. Use como cola durante a explicação.

---

## ABERTURA (30 segundos)

> "O ChefBot é um chatbot de receitas brasileiras que entende linguagem natural. Combina **Naive Bayes** para classificar intenções com **regras explícitas** e uma **máquina de estados** para manter o contexto da conversa. Vou passar pelas funções principais."

---

## BLOCO 1 — PROCESSAMENTO DE TEXTO

### `normalizar(texto)`
**O que dizer:** "Essa função padroniza o texto: deixa minúsculo e remove acentos. Sem isso, 'Não', 'não' e 'nao' seriam coisas diferentes para o bot."

**Demo mental:** `"Não Quero"` → `"nao quero"`

---

### `tokenizar(texto, filtrar_stopwords)`
**O que dizer:** "Quebra a frase em palavras úteis. Tem três etapas:
1. Tokeniza com NLTK (inteligente, lida com pontuação)
2. Remove stopwords se pedido (palavras como 'de', 'o', 'para')
3. Lematiza — reduz à raiz ('cozinhando' vira 'cozinhar')"

**Por que duas versões (com/sem stopwords)?**
"Para classificar intenção, removo stopwords. Para buscar ingredientes, mantenho — porque palavras como 'sem' têm significado."

---

## BLOCO 2 — CLASSIFICADOR NAIVE BAYES

### Preparação do vocabulário
**O que dizer:** "Pego todas as palavras únicas dos exemplos de treino e crio um índice — cada palavra ganha uma posição."

### `texto_para_vetor(tokens)`
**O que dizer:** "Transforma a frase em um vetor binário. Para cada palavra do vocabulário, marco 1 se está na frase, 0 se não está. Isso é o **bag-of-words** — o classificador não entende texto, só números."

**Exemplo:**
- Vocabulário: `[ajuda, arroz, bebida, feijoada]`
- Frase: `"quero bebida"`
- Vetor: `[0, 0, 1, 0]`

### Treinamento
**O que dizer:** "Uso o **MultinomialNB** porque é o padrão para classificação de texto. Ele aprende, durante o `fit`, quais palavras aparecem mais em cada classe."

### `detectar_intencao(tokens)`
**O que dizer:** "Aqui acontece a mágica. Pego o vetor da frase, peço as probabilidades de cada classe, e só aceito a classificação se a melhor opção for pelo menos 1,5× a média de todas. Isso evita que o bot 'chute' em frases ambíguas — se não tem certeza, cai no fallback de buscar ingredientes."

**Pergunta provável do professor:** "Por que esse threshold?"
**Resposta:** "Porque o NB sempre dá uma resposta, mesmo quando a confiança é baixa. O threshold é uma trava de segurança."

---

## BLOCO 3 — BUSCA DE RECEITAS

### `buscar_por_nome(texto)`
**O que dizer:** "Procura receitas pelo nome. Filtro palavras genéricas como 'receita' e 'fazer' que poluiriam o match. Para cada receita, conto quantas palavras do nome aparecem na frase."

**Detalhe:** "Uso comparação bidirecional — `p in u or u in p` — para que 'feijão' bata com 'feijoada'."

### `buscar_por_categoria(categoria)`
**O que dizer:** "Bem direto: filtra pela categoria exata. Cada receita tem um campo 'categoria' definido nos dados."

### `buscar_por_ingredientes(palavras)`
**O que dizer:** "Junto todos os ingredientes da receita numa string só, e conto quantas palavras do usuário aparecem ali. Score simples mas eficaz. Retorna os top 3."

---

## BLOCO 4 — SESSÃO E MÁQUINA DE ESTADOS

### `nova_sessao()`
**O que dizer:** "A sessão é a memória do bot. É um dicionário com tudo que ele precisa lembrar: o estado atual, a receita escolhida, as candidatas, em qual passo está. Sem isso, o bot esqueceria tudo a cada mensagem."

### Os estados
**O que dizer e desenhar no quadro:**

```
inicio  →  sondagem  →  passo_a_passo  →  conclusao
              ↓
        sub-fluxo:
        sugestao → confirmar_proxima → confirmar_ingredientes → confirmar_passos
```

**Por que:** "O significado de uma mensagem depende do contexto. 'Sim' depois de 'quer os ingredientes?' é diferente de 'sim' depois de 'terminou o passo?'."

---

## BLOCO 5 — FLUXO DE SONDAGEM

### `sugerir_receita(sessao)`
**O que dizer:** "Pega a candidata atual e formata a sugestão. Define `pergunta_tipo = 'sugestao'` para indicar que está esperando um sim/não."

### `resposta_sondagem(texto, sessao)`
**O que dizer:** "Essa função controla a 'negociação' com o usuário. Tem 4 sub-estados:
1. **sugestao** — usuário aceita ou rejeita a receita
2. **confirmar_proxima** — se rejeitou, pergunto se quer ver outra
3. **confirmar_ingredientes** — pergunto se posso mostrar os ingredientes
4. **confirmar_passos** — pergunto se posso começar o passo a passo

O bot **nunca avança sem confirmação explícita**. Esse design evita empurrar o usuário para uma receita que ele não quer."

---

## BLOCO 6 — PASSO A PASSO

### `mostrar_passo(sessao)`
**O que dizer:** "Mostra a instrução atual com numeração."

### `proximo_passo(sessao)`
**O que dizer:** "Incrementa o contador. Se chegou ao fim das instruções, muda o estado para 'conclusao' e parabeniza o usuário."

### `entregar_completa(sessao)`
**O que dizer:** "Atalho — se o usuário disser 'manda tudo', pula o passo a passo e mostra a receita inteira de uma vez."

---

## BLOCO 7 — RESPOSTA PRINCIPAL

### `gerar_resposta(mensagem, sessao)`
**O que dizer:** "Essa é a função que o frontend chama. Funciona como um roteador baseado no estado:

1. **passo a passo:** só processa comandos de navegação
2. **sondagem:** delega para `resposta_sondagem`
3. **conclusao:** delega para `processar_conclusao`
4. **inicio:** classifica a intenção e despacha para a busca correta"

**Mostre a cascata de fallbacks no estado inicio:**
"Se o NB classifica como saudação → responde saudação. Se classifica como categoria → busca por categoria. Se nada bate, tento buscar por nome, depois por ingredientes, depois sorteio receitas aleatórias. Nunca deixo o usuário sem resposta."

---

## BLOCO 8 — REGRAS EXPLÍCITAS

### `SINAIS_CANCELAR`, `SINAIS_COMPLETO`, `SINAIS_CONFIRMACAO`
**O que dizer:** "Listas de palavras-chave verificadas **antes** do classificador. Para casos críticos como cancelar ou confirmar um passo, regras são mais confiáveis que ML. É a abordagem híbrida: ML para casos abertos, regras para casos fechados."

### `PALAVRAS_SIM` e `PALAVRAS_NAO`
**O que dizer:** "Mesma ideia para detectar sim/não nas perguntas de confirmação. Inclui variações como 'noa' (typo de 'não')."

---

## ENCERRAMENTO (30 segundos)

> "Resumindo: o ChefBot combina três técnicas. **Naive Bayes** para classificar intenções abertas; **regras explícitas** para casos críticos onde precisão é mais importante que flexibilidade; e uma **máquina de estados** para manter o contexto da conversa. Cada parte tem seu papel — usar só ML não funcionaria, usar só regras seria rígido demais."

---

## PERGUNTAS PROVÁVEIS E RESPOSTAS

| Pergunta | Resposta curta |
|---|---|
| Por que Naive Bayes? | Rápido, interpretável, funciona com poucos dados, padrão para classificação de texto |
| Por que MultinomialNB e não Gaussian/Bernoulli? | Multinomial é feito para texto. Gaussian assume distribuição contínua, Bernoulli tem viés com vocabulário pequeno |
| Por que misturar ML com regras? | Regras são confiáveis para casos críticos; ML é flexível para casos abertos |
| Por que máquina de estados? | Mensagens dependem do contexto. "Sim" significa coisas diferentes em momentos diferentes |
| Por que tantos fallbacks na busca? | Para nunca deixar o usuário sem resposta útil |
| Como o bot lida com erros de digitação? | Normalização (acentos), comparação por substring, lista expandida com typos comuns |
| Por que a sessão é global no Flask? | Projeto acadêmico, único usuário. Em produção seria por cookie/banco de dados |

---

## ROTEIRO MÍNIMO (5 MIN — VERSÃO COMPRIMIDA)

Se tiver pouco tempo, foca só nesses 5 pontos:

1. **`normalizar` + `tokenizar`** — preparação do texto (1 min)
2. **`texto_para_vetor` + `detectar_intencao`** — o classificador NB (1,5 min)
3. **`nova_sessao` + estados** — máquina de estados (1 min)
4. **`gerar_resposta`** — função principal e cascata de fallbacks (1 min)
5. **Regras explícitas vs ML** — abordagem híbrida (30 seg)
