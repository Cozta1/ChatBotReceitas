"""dados.py — Dados estaticos do ChefBot."""

DADOS_TREINO = [
    # saudacao
    ("oi","saudacao"), ("ola","saudacao"), ("hey","saudacao"), ("eai","saudacao"),
    ("bom dia","saudacao"), ("boa tarde","saudacao"), ("boa noite","saudacao"),
    ("salve","saudacao"), ("fala","saudacao"), ("hello","saudacao"),
    ("tudo bem","saudacao"), ("como vai","saudacao"),
    ("ola tudo bom","saudacao"), ("eai tudo certo","saudacao"),
    # despedida
    ("tchau","despedida"), ("ate mais","despedida"), ("ate logo","despedida"),
    ("bye","despedida"), ("flw","despedida"), ("falou","despedida"),
    ("adeus","despedida"), ("xau","despedida"), ("vou indo","despedida"),
    ("vou embora","despedida"), ("vou sair agora","despedida"),
    # agradecimento
    ("obrigado","agradecimento"), ("obrigada","agradecimento"), ("valeu","agradecimento"),
    ("brigado","agradecimento"), ("brigada","agradecimento"), ("thanks","agradecimento"),
    ("muito obrigado","agradecimento"), ("grato","agradecimento"),
    ("legal valeu","agradecimento"), ("agradeco","agradecimento"), ("gratidao","agradecimento"),
    # ajuda
    ("ajuda","ajuda"), ("help","ajuda"), ("como funciona","ajuda"),
    ("o que voce faz","ajuda"), ("comandos","ajuda"), ("menu","ajuda"),
    ("quais opcoes","ajuda"), ("como usar","ajuda"), ("o que posso fazer","ajuda"),
    ("me ajuda","ajuda"), ("preciso de ajuda","ajuda"), ("nao sei usar","ajuda"),
    # receita_aleatoria
    ("receita aleatoria","receita_aleatoria"), ("qualquer receita","receita_aleatoria"),
    ("receita surpresa","receita_aleatoria"), ("sugestao de receita","receita_aleatoria"),
    ("sugira uma receita","receita_aleatoria"), ("sorteia uma receita","receita_aleatoria"),
    ("me surpreenda","receita_aleatoria"), ("escolha por mim","receita_aleatoria"),
    ("me da uma receita","receita_aleatoria"), ("random","receita_aleatoria"),
    ("surpresa","receita_aleatoria"), ("aleatoria","receita_aleatoria"),
    ("sorteia","receita_aleatoria"), ("sugere algo","receita_aleatoria"),
    ("qualquer coisa para comer","receita_aleatoria"), ("quero uma receita","receita_aleatoria"),
    ("me sugere algo","receita_aleatoria"), ("o que voce sugere","receita_aleatoria"),
    ("me da uma sugestao","receita_aleatoria"), ("tem alguma receita","receita_aleatoria"),
    ("quero fazer algo","receita_aleatoria"), ("o que posso fazer","receita_aleatoria"),
    # categoria:sobremesa
    ("quero uma sobremesa","categoria:sobremesa"), ("receita de sobremesa","categoria:sobremesa"),
    ("tem algum doce","categoria:sobremesa"), ("receita de bolo","categoria:sobremesa"),
    ("quero um pudim","categoria:sobremesa"), ("algo doce","categoria:sobremesa"),
    ("brigadeiro","categoria:sobremesa"), ("sobremesa","categoria:sobremesa"),
    ("torta","categoria:sobremesa"), ("receita doce","categoria:sobremesa"),
    ("doce para sobremesa","categoria:sobremesa"),
    # categoria:prato principal
    ("receita de almoco","categoria:prato principal"), ("o que fazer para almoco","categoria:prato principal"),
    ("prato principal","categoria:prato principal"), ("receita para jantar","categoria:prato principal"),
    ("comida","categoria:prato principal"), ("almoco","categoria:prato principal"),
    ("jantar","categoria:prato principal"), ("refeicao","categoria:prato principal"),
    ("prato do dia","categoria:prato principal"), ("janta","categoria:prato principal"),
    ("prato quente","categoria:prato principal"), ("quero um almoco","categoria:prato principal"),
    ("quero algo para jantar","categoria:prato principal"), ("o que tem para o almoco","categoria:prato principal"),
    ("quero algo para comer","categoria:prato principal"), ("me da um prato","categoria:prato principal"),
    # categoria:lanche
    ("receita de lanche","categoria:lanche"), ("quero um salgado","categoria:lanche"),
    ("tem tapioca","categoria:lanche"), ("receita de coxinha","categoria:lanche"),
    ("lanche rapido","categoria:lanche"), ("lanche","categoria:lanche"),
    ("salgadinho","categoria:lanche"), ("sanduiche","categoria:lanche"),
    ("petisco","categoria:lanche"), ("coxinha","categoria:lanche"),
    ("pao de queijo","categoria:lanche"),
    # categoria:bebida
    ("receita de suco","categoria:bebida"), ("quero uma bebida","categoria:bebida"),
    ("tem drink","categoria:bebida"), ("vitamina de fruta","categoria:bebida"),
    ("receita de limonada","categoria:bebida"), ("bebida","categoria:bebida"),
    ("suco","categoria:bebida"), ("vitamina","categoria:bebida"),
    ("limonada","categoria:bebida"), ("quentao","categoria:bebida"),
    ("algo para beber","categoria:bebida"),
    # categoria:acompanhamento
    ("receita de farofa","categoria:acompanhamento"), ("quero um acompanhamento","categoria:acompanhamento"),
    ("tem salada","categoria:acompanhamento"), ("receita de pure","categoria:acompanhamento"),
    ("acompanhamento","categoria:acompanhamento"), ("farofa","categoria:acompanhamento"),
    ("salada","categoria:acompanhamento"), ("pure","categoria:acompanhamento"),
    ("vinagrete","categoria:acompanhamento"), ("acompanha o prato","categoria:acompanhamento"),
    # buscar_por_ingredientes
    ("tenho frango","buscar_por_ingredientes"), ("tenho ovos e queijo","buscar_por_ingredientes"),
    ("o que fazer com batata","buscar_por_ingredientes"), ("tenho leite condensado","buscar_por_ingredientes"),
    ("posso fazer com frango e alho","buscar_por_ingredientes"),
    ("o que posso cozinhar com arroz","buscar_por_ingredientes"),
    ("tenho cebola e alho em casa","buscar_por_ingredientes"),
    ("o que fazer com macarrao","buscar_por_ingredientes"),
    ("tenho banana e ovo","buscar_por_ingredientes"), ("usar frango e batata","buscar_por_ingredientes"),
    ("preparar algo com ovo","buscar_por_ingredientes"), ("tenho bacon e linguica","buscar_por_ingredientes"),
    ("como usar feijao preto","buscar_por_ingredientes"),
    ("tenho frango cebola alho","buscar_por_ingredientes"),
    ("o que preparo com esses ingredientes","buscar_por_ingredientes"),
    ("tenho so esses ingredientes","buscar_por_ingredientes"),
    ("receita com frango","buscar_por_ingredientes"), ("fazer com leite e ovo","buscar_por_ingredientes"),
    # buscar_por_nome
    ("receita de feijoada","buscar_por_nome"), ("como fazer estrogonofe","buscar_por_nome"),
    ("quero fazer frango xadrez","buscar_por_nome"), ("tem receita de bolo de cenoura","buscar_por_nome"),
    ("busca feijoada","buscar_por_nome"), ("me mostra a receita de arroz","buscar_por_nome"),
    ("quero a receita de coxinha","buscar_por_nome"), ("como se faz limonada","buscar_por_nome"),
    ("procura receita de pudim","buscar_por_nome"), ("me da a receita de moqueca","buscar_por_nome"),
    ("como preparar feijao tropeiro","buscar_por_nome"),
    # confirmacao_passo
    ("pronto","confirmacao_passo"), ("ja fiz","confirmacao_passo"),
    ("feito","confirmacao_passo"), ("pode continuar","confirmacao_passo"),
    ("proximo","confirmacao_passo"), ("proximo passo","confirmacao_passo"),
    ("continua","confirmacao_passo"), ("ok feito","confirmacao_passo"),
    ("fiz esse","confirmacao_passo"), ("terminei esse passo","confirmacao_passo"),
    ("to pronto","confirmacao_passo"), ("pode ir","confirmacao_passo"),
    ("segue","confirmacao_passo"), ("vamos pro proximo","confirmacao_passo"),
    ("ja terminei","confirmacao_passo"),
    # sem_ingrediente
    ("quero uma receita sem carne","sem_ingrediente"),
    ("algo sem batata","sem_ingrediente"),
    ("receita sem ovos","sem_ingrediente"),
    ("nao quero frango","sem_ingrediente"),
    ("sem gluten","sem_ingrediente"),
    ("receita sem leite","sem_ingrediente"),
    ("nao gosto de cebola","sem_ingrediente"),
    ("algo sem queijo","sem_ingrediente"),
    ("receita sem carne vermelha","sem_ingrediente"),
    ("pode ser sem bacon","sem_ingrediente"),
    ("nao como carne","sem_ingrediente"),
    ("receita vegetariana","sem_ingrediente"),
    ("quero algo sem alho","sem_ingrediente"),
    ("nao tenho e nao quero frango","sem_ingrediente"),
    ("sem peixe por favor","sem_ingrediente"),
    # pedir_receita_completa
    ("manda tudo","pedir_receita_completa"), ("quero a receita inteira","pedir_receita_completa"),
    ("me da o preparo completo","pedir_receita_completa"), ("pode mandar tudo","pedir_receita_completa"),
    ("prefiro ver tudo","pedir_receita_completa"), ("me manda completo","pedir_receita_completa"),
    ("receita completa","pedir_receita_completa"), ("quero ver tudo de uma vez","pedir_receita_completa"),
    ("nao precisa passo a passo","pedir_receita_completa"),
    ("me da tudo","pedir_receita_completa"), ("manda direto","pedir_receita_completa"),
]

# Deteccao direta por texto (bypass NB)
SINAIS_CONFIRMACAO = [
    "pronto", "feito", "ja fiz", "pode continuar", "proximo", "proximo passo",
    "continua", "continuar", "fiz", "terminei", "concluido", "pode ir",
    "segue", "to pronto", "sim feito", "ok feito", "ja ta", "feito isso",
    "pode", "sim", "ok", "vamos", "vai", "ja", "next",
]
SINAIS_CANCELAR = [
    "cancelar", "outra receita", "quero outra", "muda a receita",
    "esquece essa", "nao quero mais", "recomecar", "comecar de novo",
    "desisti", "muda", "para essa", "esquece",
]

SINAIS_COMPLETO = [
    "manda tudo", "receita completa", "preparo completo", "quero tudo",
    "pode mandar tudo", "me manda completo", "quero ver tudo", "me da tudo",
    "tudo de uma vez", "receita inteira", "nao precisa passo a passo",
    "manda direto", "modo de preparo completo",
]

# Feedback da conclusao
POSITIVOS = [
    "gostei", "otimo", "delicioso", "perfeito", "amei", "incrivel",
    "ficou bom", "ficou bem", "excelente", "deu certo", "ficou gostoso",
    "ficou uma delicia", "ficou show", "ficou incrivel",
]
NEGATIVOS = [
    "nao gostei", "ficou ruim", "nao deu", "errei", "queimou",
    "nao ficou", "nao consegui", "deu errado",
]
ENCORAJAMENTOS = ["Otimo!", "Muito bem!", "Perfeito!", "Mandou bem!", "Continue assim!"]

# Respostas variadas
SAUDACOES = [
    ("Ola! Sou o ChefBot, seu assistente culinario!\n"
     "Me diga quais ingredientes você tem ou peça uma receita por categoria.\n\n"
     "O que vamos cozinhar hoje?"),
    ("Fala! Sou o ChefBot, pronto para cozinhar com você!\n"
     "Me conta o que tem na geladeira ou escolha uma categoria de receita.\n\n"
     "O que vamos preparar?"),
    ("Boa! ChefBot aqui, especialista em receitas brasileiras!\n"
     "Me diga os ingredientes que você tem ou o que quer cozinhar.\n\n"
     "Me conta, o que tem por ai?"),
]
DESPEDIDAS = [
    "Tchau! Bom apetite e ate a proxima!",
    "Ate mais! Espero ter ajudado na cozinha!",
    "Ate logo! Que o seu prato fique uma delicia!",
]
AGRADECIMENTOS = [
    "De nada! Espero que a receita fique otima! Quer mais alguma coisa?",
    "Por nada! Qualquer duvida sobre receitas, e so chamar! Posso ajudar em mais alguma coisa?",
    "Disponha! Quer mais alguma receita?",
]
AJUDA = (
    "Veja o que posso fazer por voce:\n\n"
    "  - INGREDIENTES: Me diga o que voce tem e sugiro receitas.\n"
    "    Exemplo: 'Tenho frango, alho e cebola'\n\n"
    "  - NOME: Busque uma receita pelo nome.\n"
    "    Exemplo: 'Receita de feijoada' ou 'Como fazer estrogonofe'\n\n"
    "  - CATEGORIA: Peca receitas de uma categoria.\n"
    "    Opcoes: prato principal, sobremesa, lanche, bebida, acompanhamento\n\n"
    "  - ALEATORIA: Peca uma receita surpresa.\n"
    "    Exemplo: 'Me da uma receita aleatoria'\n\n"
    "  - PASSO A PASSO: Apos escolher a receita, te guio instrucao por instrucao.\n"
    "    Diga 'pronto' para avancar, ou 'manda tudo' para ver a receita completa.\n\n"
    "O que prefere fazer?"
)
