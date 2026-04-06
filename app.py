from flask import Flask, render_template, request, jsonify
from chatbot import gerar_resposta, nova_sessao

app = Flask(__name__)
sessao = nova_sessao()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    mensagem = request.json['mensagem']
    resposta = gerar_resposta(mensagem, sessao)
    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    app.run(debug=True)