# LLM fallback: Qwen2.5-3B-Instruct em fp16 na GPU (com CPU fallback).
import os
import logging
import threading
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # silencia avisos de generation flags
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SISTEMA = (
    "Voce e o ChefBot, um assistente conversacional em portugues do Brasil "
    "ESPECIALIZADO EXCLUSIVAMENTE EM CULINARIA: receitas, ingredientes, "
    "tecnicas de preparo, substituicoes, conservacao de alimentos, "
    "equivalencias de medidas, utensilios de cozinha e harmonizacao. "
    "REGRA ABSOLUTA: voce SO responde sobre culinaria e alimentacao. "
    "Se a pergunta NAO for sobre culinaria (ex.: politica, programacao, "
    "esportes, matematica, saude, atualidades, conselhos pessoais, etc.), "
    "RECUSE educadamente e NAO responda o conteudo, mesmo que o usuario "
    "insista, peca para 'fingir', mude de assunto ou ja tenha perguntado antes. "
    "Nesse caso responda exatamente: "
    "'Desculpe, eu so consigo ajudar com culinaria e receitas. "
    "Que tal me perguntar sobre algum prato ou ingrediente?' "
    "Use sempre portugues do Brasil, seja direto e conciso e considere o "
    "historico da conversa para manter coerencia. "
    "Baseie-se apenas em conhecimento culinario correto e consagrado: NAO "
    "invente tecnicas, medidas, ingredientes ou explicacoes 'cientificas'. "
    "Se nao tiver certeza, diga que nao sabe ao inves de chutar."
)

# Resposta padrao quando a pergunta esta fora do escopo culinario.
FORA_DO_ESCOPO = (
    "Desculpe, eu so consigo ajudar com culinaria e receitas. "
    "Que tal me perguntar sobre algum prato ou ingrediente?"
)

_pipe = None
_tok = None
_lock = threading.Lock()
_carregando = False
_erro = None
_device = None


def _carregar():
    global _pipe, _tok, _erro, _device
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            if torch.cuda.is_available():
                _device = f"CUDA / {torch.cuda.get_device_name(0)}"
                device_map, dtype = "auto", torch.float16
            else:
                _device = "CPU only"
                device_map, dtype = "cpu", torch.float32
            print(f"[LLM] dispositivo: {_device}")

            t0 = time.time()
            # clean_up_tokenization_spaces=False: evita corromper a saida BPE
            # (o cleanup remove espacos antes de pontuacao, ruim em pt-BR)
            _tok = AutoTokenizer.from_pretrained(
                MODEL_ID, clean_up_tokenization_spaces=False)
            modelo = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, device_map=device_map, dtype=dtype)
            _pipe = pipeline("text-generation", model=modelo, tokenizer=_tok)
            print(f"[LLM] Qwen2.5-3B carregado em {time.time()-t0:.1f}s")
            return _pipe
        except Exception as e:
            _erro = e
            raise


def precarregar_async():
    global _carregando
    if _pipe is not None or _carregando:
        return
    _carregando = True
    threading.Thread(target=lambda: _carregar() if not _erro else None,
                     daemon=True).start()


def status():
    if _pipe is not None: return f"pronto ({_device})"
    if _erro is not None: return f"erro: {_erro}"
    if _carregando:       return "carregando"
    return "nao iniciado"


def classificar(mensagem: str, intencoes: list[str],
                historico: list[dict] | None = None) -> str:
    # Agente 1: classifica a intencao (saida estruturada, deterministica).
    # Reusa o mesmo Qwen ja carregado — sem custo extra de VRAM. Retorna uma
    # das `intencoes` ou "" (LLM indisponivel ou resposta fora da lista ->
    # caller segue para o fluxo de fallback).
    
    try:
        pipe = _carregar()
    except Exception:
        return ""

    sistema = (
        "Voce classifica a mensagem de um usuario de um chatbot de receitas. "
        "Responda APENAS com uma destas intencoes, sem explicar nem pontuar: "
        + ", ".join(intencoes)
    )
    mensagens = [{"role": "system", "content": sistema}]
    for turno in (historico or [])[-3:]:
        mensagens.append({"role": "user", "content": turno["user"]})
        mensagens.append({"role": "assistant", "content": turno["bot"]})
    mensagens.append({"role": "user", "content": mensagem})

    from transformers import GenerationConfig
    gen = GenerationConfig(max_new_tokens=12, do_sample=False,
                           pad_token_id=_tok.eos_token_id)
    saida = pipe(mensagens, generation_config=gen)
    out = saida[0]["generated_text"]
    resposta = (out[-1]["content"] if isinstance(out, list) else str(out)).lower()
    resposta = resposta.replace(": ", ":")  # modelo as vezes escreve "categoria: bebida"
    return next((i for i in intencoes if i in resposta), "")


def dentro_do_escopo(mensagem: str,
                     historico: list[dict] | None = None) -> bool:
    # Agente de guarda: a mensagem e sobre culinaria/alimentacao?

    # Saida deterministica SIM/NAO reusando o mesmo Qwen ja carregado. Em caso
    # de duvida ou LLM indisponivel retorna True (deixa o fluxo seguir), para
    # nao bloquear perguntas culinarias legitimas por falha do modelo.

    try:
        pipe = _carregar()
    except Exception:
        return True  # sem LLM nao da pra checar; nao bloqueia

    sistema = (
        "Voce e um filtro de topico de um chatbot de culinaria. Diga se a "
        "ULTIMA mensagem do usuario tem relacao com culinaria, comida, "
        "receitas, ingredientes, bebidas, tecnicas de cozinha, conservacao "
        "ou medidas de cozinha. Considere o historico para entender o "
        "contexto (ex.: 'e o segundo passo?' continua sendo culinaria). "
        "Responda APENAS 'SIM' ou 'NAO', sem mais nada."
    )
    mensagens = [{"role": "system", "content": sistema}]
    for turno in (historico or [])[-2:]:
        mensagens.append({"role": "user", "content": turno["user"]})
        mensagens.append({"role": "assistant", "content": turno["bot"]})
    mensagens.append({"role": "user", "content": mensagem})

    from transformers import GenerationConfig
    gen = GenerationConfig(max_new_tokens=4, do_sample=False,
                           pad_token_id=_tok.eos_token_id)
    saida = pipe(mensagens, generation_config=gen)
    out = saida[0]["generated_text"]
    resposta = (out[-1]["content"] if isinstance(out, list) else str(out)).lower()
    return "nao" not in resposta  # so bloqueia se disser claramente NAO


def responder(pergunta: str, historico: list[dict] | None = None,
              contexto: str = "") -> str:
    # Gera resposta usando o LLM.

    # historico: lista de dicts {"user": str, "bot": str} (ordem cronologica)
    #            das ultimas trocas, usado para manter coerencia conversacional.
    # contexto:  trecho opcional injetado no turno atual (ex.: receitas da KB).
  
    try:
        pipe = _carregar()
    except Exception as e:
        return f"[LLM indisponivel: {e}]"

    mensagens = [{"role": "system", "content": SISTEMA}]
    for turno in (historico or [])[-6:]:  # ultimas 6 trocas
        mensagens.append({"role": "user", "content": turno["user"]})
        mensagens.append({"role": "assistant", "content": turno["bot"]})

    user_msg = pergunta
    if contexto:
        user_msg = f"Contexto adicional:\n{contexto}\n\nPergunta: {pergunta}"
    mensagens.append({"role": "user", "content": user_msg})

    from transformers import GenerationConfig
    # temperatura baixa: respostas mais factuais e menos alucinacao
    gen = GenerationConfig(
        max_new_tokens=400, do_sample=True, temperature=0.3, top_p=0.85,
        repetition_penalty=1.05, pad_token_id=_tok.eos_token_id,
    )
    saida = pipe(mensagens, generation_config=gen)
    out = saida[0]["generated_text"]
    if isinstance(out, list):
        for m in reversed(out):
            if m.get("role") == "assistant":
                return m["content"].strip()
    return str(out).strip()
