"""AP7 UniAssist: compara classificação por regras e por Machine Learning (TF-IDF + LinearSVC).
Camadas: domínio, dados, classificadores, avaliação, apresentação e aplicação (main)."""
import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PASTA = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. DOMÍNIO
# ---------------------------------------------------------------------------
REVISAO_HUMANA = "Revisão humana"

# Palavras-chave de cada setor, usadas pelo baseline.
REGRAS = {
    "Financeiro": ["boleto", "mensalidade", "cobrança", "pagamento", "fatura"],
    "Matrícula": ["matrícula", "rematrícula", "disciplina", "turma", "trancamento"],
    "Estágio": ["estágio", "termo", "supervisor", "concedente", "convênio"],
    "Suporte": ["senha", "login", "portal", "acesso", "aplicativo", "erro"],
}


@dataclass(frozen=True)
class Mensagem:
    """Mensagem de estudante; a categoria só existe nos dados rotulados."""

    texto: str
    categoria: str = ""


# ---------------------------------------------------------------------------
# 2. DADOS
# ---------------------------------------------------------------------------
def ler_dataset(arquivo):
    with (PASTA / arquivo).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def carregar_mensagens(arquivo):
    """Lê o CSV rotulado (texto,categoria) e devolve uma lista de Mensagem."""
    return [Mensagem(r["texto"], r["categoria"]) for r in ler_dataset(arquivo)]


def carregar_desafios(arquivo):
    """Lê o CSV de desafios (id,texto) e devolve pares (id, Mensagem)."""
    return [(r["id"], Mensagem(r["texto"])) for r in ler_dataset(arquivo)]


# ---------------------------------------------------------------------------
# 3. CLASSIFICADORES
# ---------------------------------------------------------------------------
def classificar_regras(texto):
    """Conta as palavras-chave de cada setor e devolve o setor com mais pontos.
    Sem nenhuma palavra ou com empate no topo, devolve 'Revisão humana'."""
    texto = texto.casefold()
    pontos = {
        categoria: sum(1 for termo in termos if termo in texto)
        for categoria, termos in REGRAS.items()
    }
    maximo = max(pontos.values())
    if maximo == 0:
        return REVISAO_HUMANA
    vencedores = [c for c, p in pontos.items() if p == maximo]
    return vencedores[0] if len(vencedores) == 1 else REVISAO_HUMANA


def criar_pipeline(ngram_range=(1, 2)):
    """TF-IDF vira texto em números e o LinearSVC aprende a separar os setores.
    No Pipeline, o TF-IDF aprende só com o treino, sem vazamento do teste."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, lowercase=True)),
        ("svm", LinearSVC(random_state=42)),
    ])


def margem_de_confianca(modelo, texto):
    """Diferença entre as duas maiores notas do SVM; margem baixa = modelo em dúvida."""
    notas = sorted(modelo.decision_function([texto])[0], reverse=True)
    return notas[0] - notas[1]


# ---------------------------------------------------------------------------
# 4. AVALIAÇÃO
# ---------------------------------------------------------------------------
def avaliar_modelo(modelo, textos, reais, classes):
    """Prevê o teste e devolve acurácia, relatório por classe e matriz de confusão."""
    previstos = modelo.predict(textos)
    return {
        "acuracia": accuracy_score(reais, previstos),
        "relatorio": classification_report(reais, previstos, labels=classes, zero_division=0),
        "matriz": confusion_matrix(reais, previstos, labels=classes),
    }


# ---------------------------------------------------------------------------
# 5. APRESENTAÇÃO (único lugar com print)
# ---------------------------------------------------------------------------
def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def imprimir_dados(mensagens):
    contagem = {}
    for m in mensagens:
        contagem[m.categoria] = contagem.get(m.categoria, 0) + 1
    print(f"Total de mensagens: {len(mensagens)}")
    print(f"Categorias ({len(contagem)}):")
    for categoria, qtd in sorted(contagem.items()):
        print(f"  {categoria}: {qtd} exemplos")


def imprimir_matriz(matriz, classes):
    print("Linhas = classe real | Colunas = classe prevista")
    print(f"Ordem das classes: {classes}")
    largura = max(len(c) for c in classes) + 2
    print(" " * largura + "".join(c.rjust(largura) for c in classes))
    for classe, linha in zip(classes, matriz):
        print(classe.ljust(largura) + "".join(str(v).rjust(largura) for v in linha))


def imprimir_modelo(nome, resultado, classes):
    titulo(nome)
    print(f"Acurácia no teste: {resultado['acuracia']:.4f}\n")
    print(resultado["relatorio"])
    imprimir_matriz(resultado["matriz"], classes)


def imprimir_comparacao(total, acertos_regras, revisao_regras, acuracia_ml):
    titulo("COMPARAÇÃO NO MESMO CONJUNTO DE TESTE")
    print(f"Regras: {acertos_regras}/{total} corretas = {acertos_regras / total:.4f}")
    print(f"        ({revisao_regras} mensagens foram para 'Revisão humana' e contam como erro)")
    print(f"ML (TF-IDF + LinearSVC, ngram (1,2)): acurácia = {acuracia_ml:.4f}")


def imprimir_desafios(linhas):
    titulo("MENSAGENS DESAFIO (sem resposta correta conhecida)")
    for id_, texto, regras, ml, margem in linhas:
        print(f"{id_}: {texto}")
        print(f"     Regras: {regras}")
        print(f"     ML    : {ml}  (margem de confiança = {margem:.2f})\n")


def imprimir_experimento(res_bigramas, res_unigramas):
    titulo("EXPERIMENTO: ngram_range (1,2) x (1,1)")
    print(f"{'Configuração':<22}{'Acurácia no teste':>20}")
    print(f"{'(1,2) uni + bigramas':<22}{res_bigramas['acuracia']:>20.4f}")
    print(f"{'(1,1) só unigramas':<22}{res_unigramas['acuracia']:>20.4f}")
    print(
        "\nAtenção: o teste tem só 20 mensagens. Escolher o vencedor olhando para ele\n"
        "faz a escolha se ajustar ao teste, e a acurácia deixa de ser uma estimativa\n"
        "honesta. O correto é decidir com validação cruzada no treino e usar o teste\n"
        "uma única vez, no final."
    )


# ---------------------------------------------------------------------------
# 6. APLICAÇÃO
# ---------------------------------------------------------------------------
def main():
    # Passo 1: ler os dados e conferir as categorias.
    mensagens = carregar_mensagens("mensagens.csv")
    textos = [m.texto for m in mensagens]
    rotulos = [m.categoria for m in mensagens]
    classes = sorted(set(rotulos))
    titulo("DADOS")
    imprimir_dados(mensagens)

    # Passo 3: divisão estratificada, 75% treino e 25% teste.
    textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos
    )
    print(f"\nTreino: {len(textos_treino)} mensagens | Teste: {len(textos_teste)} mensagens")

    # Passo 4: treinar só com o treino e avaliar no teste.
    modelo = criar_pipeline(ngram_range=(1, 2))
    modelo.fit(textos_treino, rotulos_treino)
    res_ml = avaliar_modelo(modelo, textos_teste, rotulos_teste, classes)
    imprimir_modelo("MACHINE LEARNING: TF-IDF + LinearSVC", res_ml, classes)

    # Passos 2 e 5: baseline por regras no MESMO teste.
    previstos_regras = [classificar_regras(t) for t in textos_teste]
    acertos = sum(1 for p, r in zip(previstos_regras, rotulos_teste) if p == r)
    revisao = previstos_regras.count(REVISAO_HUMANA)
    imprimir_comparacao(len(textos_teste), acertos, revisao, res_ml["acuracia"])

    # Passo 6: os dois métodos nas 8 mensagens desafio.
    linhas = []
    for id_, m in carregar_desafios("desafios.csv"):
        linhas.append((
            id_,
            m.texto,
            classificar_regras(m.texto),
            modelo.predict([m.texto])[0],
            margem_de_confianca(modelo, m.texto),
        ))
    imprimir_desafios(linhas)

    # Passo 7: mesmo treino e teste, só mudando para unigramas.
    modelo_uni = criar_pipeline(ngram_range=(1, 1))
    modelo_uni.fit(textos_treino, rotulos_treino)
    res_uni = avaliar_modelo(modelo_uni, textos_teste, rotulos_teste, classes)
    imprimir_experimento(res_ml, res_uni)


if __name__ == "__main__":
    main()
