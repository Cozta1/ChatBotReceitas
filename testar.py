"""Suite de testes manual: exercita todos os fluxos e mede fonte + tempo.

Uso:
    python testar.py            # roda tudo e imprime tabela + resumo
    python testar.py --rapido   # pula cenarios que dependem do LLM (sem GPU)

Cada cenario e uma conversa (lista de mensagens) numa sessao isolada, para
exercitar tambem a state machine (sondagem -> passo a passo -> conclusao).
A coluna `fonte` mostra de onde veio a resposta:
    base      -> regra/handler deterministico
    base-faq  -> FAQ (TF-IDF)
    llm       -> LLM (Agente 2, fallback de texto livre)
"""
import sys

from chatbot import gerar_resposta, nova_sessao, stats

PULAR_LLM = "--rapido" in sys.argv

# (titulo, [mensagens], usa_llm) — usa_llm marca cenarios que so fazem
# sentido com o modelo carregado (pulados no modo --rapido).
CENARIOS = [
    ("Saudacao / ajuda / agradecimento / despedida",
     ["oi", "ajuda", "valeu", "tchau"], False),

    ("Busca por NOME (lista -> escolhe -> descricao -> guia 'sim' -> passos)",
     ["receita de feijoada", "quero a 1", "sim", "pronto", "pronto"], False),

    ("Escolha por ordinal ('a segunda') -> 'outra' volta pra lista",
     ["quero uma sobremesa", "a segunda", "outra", "a 1", "manda tudo"], False),

    ("Busca por INGREDIENTES (escolhe por numero)",
     ["tenho frango, alho e cebola", "a 1", "manda tudo"], False),

    ("CATEGORIA: sobremesa",
     ["quero uma sobremesa", "a 1", "sim", "manda tudo"], False),

    ("CATEGORIA: bebida",
     ["algo para beber"], False),

    ("CATEGORIA: lanche / acompanhamento / prato principal",
     ["receita de lanche", "cancelar", "tem salada", "cancelar",
      "o que fazer para o almoco"], False),

    ("Receita ALEATORIA + cancelar",
     ["me surpreenda", "cancelar"], False),

    ("Conclusao com feedback POSITIVO e NEGATIVO",
     ["receita de limonada", "a 1", "sim", "manda tudo", "ficou otimo",
      "receita de limonada", "a 1", "sim", "manda tudo", "ficou ruim"], False),

    ("FAQ (perguntas culinarias que batem na base)",
     ["como descongelar carne com seguranca",
      "posso congelar molho de tomate?",
      "quanto tempo posso guardar carne na geladeira?",
      "como deixar o bife macio?"], False),

    ("LLM (perguntas culinarias FORA da base)",
     ["como fazer horchata, bebida mexicana de arroz",
      "o que e maturacao a seco de carne?",
      "como fazer sushi em casa?"], True),

    ("LLM (pergunta geral, fora de culinaria)",
     ["quem pintou a mona lisa?"], True),

    ("AGENTE 1 (frases ambiguas: Naive Bayes inseguro -> LLM classifica)",
     ["to a fim de algo gelado pra tomar",
      "me ve um docinho qualquer ai"], True),
]


def main():
    print(f"{'#':>3}  {'fonte':9}  {'ms':>7}  conversa")
    print("-" * 80)
    n = 0
    for titulo, mensagens, usa_llm in CENARIOS:
        if usa_llm and PULAR_LLM:
            print(f"\n[PULADO --rapido] {titulo}")
            continue
        print(f"\n>>> {titulo}")
        s = nova_sessao()
        for msg in mensagens:
            n += 1
            r = gerar_resposta(msg, s)
            print(f"{n:>3}  {r['fonte']:9}  {r['tempo_ms']:>7}  "
                  f"[{msg}] -> {r['texto'][:60].replace(chr(10), ' ')}")

    print("\n" + "=" * 80)
    print("RESUMO")
    for k, v in stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
