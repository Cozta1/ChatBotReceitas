from flask import Flask, render_template, request, jsonify

from chatbot import gerar_resposta, nova_sessao, stats
import llm

app = Flask(__name__)
sessoes: dict[str, dict] = {}

llm.precarregar_async()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    mensagem = data.get("mensagem", "")
    chat_id = data.get("chat_id", "default")
    if chat_id not in sessoes:
        sessoes[chat_id] = nova_sessao()
    resp = gerar_resposta(mensagem, sessoes[chat_id])
    return jsonify({
        "resposta": resp["texto"],
        "fonte": resp["fonte"],
        "tempo_ms": resp["tempo_ms"],
    })


@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    chat_id = (request.json or {}).get("chat_id")
    if chat_id and chat_id in sessoes:
        del sessoes[chat_id]
    return jsonify({"ok": True})


@app.route("/stats")
def stats_endpoint():
    return jsonify(stats())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
