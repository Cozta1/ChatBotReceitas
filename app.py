from flask import Flask, render_template, request, jsonify, session
from chatbot import gerar_resposta, nova_sessao

app = Flask(__name__)
app.secret_key = 'segredo'  # necessário pro session

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    mensagem = data.get('mensagem')

    if 'chat_sessao' not in session:
        session['chat_sessao'] = nova_sessao()

    sessao = session['chat_sessao']

    resposta = gerar_resposta(mensagem, sessao)

    sessao["historico"].append((mensagem, resposta))

    session['chat_sessao'] = sessao

    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    app.run(debug=True)