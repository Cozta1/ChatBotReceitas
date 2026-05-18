# 🍳 ChefBot - Chatbot de Receitas

## Descrição

O **ChefBot** é um chatbot temático desenvolvido em Python que auxilia usuários na busca de receitas de forma interativa.
O usuário pode informar ingredientes, nomes de pratos ou categorias, e o sistema retorna receitas compatíveis.

---

## 🚀 Funcionalidades

* 🔎 Busca de receitas por ingredientes
* 📖 Busca por nome da receita
* 🎲 Sugestão de receita aleatória
* 🧠 Classificação de intenção com NLP (NLTK + Naive Bayes)
* 💬 Interface web interativa

---

## 🛠️ Tecnologias utilizadas

* Python
* Flask
* NLTK (Natural Language Toolkit)
* Scikit-learn
* HTML, CSS e JavaScript

---

## 🧠 Como funciona

O chatbot utiliza técnicas de Processamento de Linguagem Natural (NLP) para interpretar a entrada do usuário:

* Tokenização de texto
* Remoção de stopwords
* Lematização
* Classificação de intenção com Naive Bayes

Com base na intenção identificada, o sistema realiza buscas nas receitas cadastradas e retorna os resultados mais relevantes.

---

## ▶️ Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/Cozta1/ChatBotReceitas.git
cd ChatBotReceitas
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
```

### 3. Ativar o ambiente virtual

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Executar o projeto

```bash
python app.py
```

### 6. Acessar no navegador

```
http://127.0.0.1:5000
```

---

## 📁 Estrutura do projeto

```
projeto/
│
├── app.py
├── chatbot.py
├── receitas.py
├── dados.py
│
├── templates/
│   └── index.html
│
└── static/
    └── script.js
```

---

## 👥 Integrantes

* Gabriel Krepker
* Gabriel Monteiro
* Gustavo Lopes
* João Victor da Costa
* Rafael Lima Henriques

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos.
