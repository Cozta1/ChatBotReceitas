from flask import Flask, render_template, request, jsonify
from chatbot import gerar_resposta, nova_sessao

app = Flask(__name__)

sessoes = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    mensagem = data.get('mensagem', '')
    chat_id  = data.get('chat_id', 'default')

    if chat_id not in sessoes:
        sessoes[chat_id] = nova_sessao()

    resposta = gerar_resposta(mensagem, sessoes[chat_id])
    return jsonify({'resposta': resposta})

@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    """Remove a sessão de um chat do servidor."""
    chat_id = request.json.get('chat_id')
    if chat_id and chat_id in sessoes:
        del sessoes[chat_id]
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(debug=True)