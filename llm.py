"""LLM fallback: Qwen2.5-3B-Instruct em fp16 na GPU (com CPU fallback)."""
import os
import logging
import threading
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SISTEMA = (
    "Voce e o ChefBot, um assistente conversacional em portugues do Brasil. "
    "Sua especialidade e culinaria (tecnicas, substituicoes, conservacao, "
    "equivalencias de medidas), mas voce tambem pode responder duvidas gerais "
    "quando o usuario pedir. Use sempre portugues do Brasil, seja direto e "
    "considere o historico da conversa para manter coerencia."
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
            _tok = AutoTokenizer.from_pretrained(MODEL_ID)
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


def responder(pergunta: str, historico: list[dict] | None = None,
              contexto: str = "") -> str:
    """Gera resposta usando o LLM.

    historico: lista de dicts {"user": str, "bot": str} (ordem cronologica)
               das ultimas trocas — usado para manter coerencia conversacional.
    contexto:  trecho opcional injetado no turno atual (ex.: receitas da KB).
    """
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
    gen = GenerationConfig(
        max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05, pad_token_id=_tok.eos_token_id,
    )
    saida = pipe(mensagens, generation_config=gen)
    out = saida[0]["generated_text"]
    if isinstance(out, list):
        for m in reversed(out):
            if m.get("role") == "assistant":
                return m["content"].strip()
    return str(out).strip()
