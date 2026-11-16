# Parecer técnico — AP7

## Resultados

Mesmo conjunto de teste para os dois métodos: 20 mensagens (25% de 80), divisão estratificada com `random_state=42`.

| Método | Acurácia no teste | Observações |
|---|---:|---|
| Regras | 0,80 (16/20) | As 4 mensagens restantes foram para "Revisão humana" e contam como não acerto. Nenhuma foi encaminhada ao setor errado. |
| TF-IDF + LinearSVC | 0,70 (14/20) | Sempre escolhe um setor, então os 6 erros foram encaminhamentos errados. Pior classe: Financeiro (recall 0,40, f1 0,44). |

## Matriz de confusão

Modelo TF-IDF + LinearSVC, `ngram_range=(1, 2)`. Linhas = classe real, colunas = classe prevista.

```text
Ordem das classes: ['Estágio', 'Financeiro', 'Matrícula', 'Suporte']
                 Estágio  Financeiro   Matrícula     Suporte
Estágio                5           0           0           0
Financeiro             1           2           1           1
Matrícula              0           2           3           0
Suporte                1           0           0           4
```

Financeiro e Matrícula são as classes mais confundidas: 2 mensagens de Matrícula foram previstas como Financeiro e 1 de Financeiro como Matrícula.

## Mensagens ambíguas (pelo menos 3)

| ID | Regras | Modelo | O que dificultou? | Política proposta |
|---|---|---|---|---|
| C01 "Não posso acessar o boleto porque esqueci minha senha" | Revisão humana | Suporte (margem 0,09) | Cita dois setores: "boleto" (Financeiro) e "senha" (Suporte). As regras empatam. O modelo escolheu um, mas com margem quase nula. | Quando houver empate ou margem baixa, encaminhar para revisão humana. O problema real é a senha (Suporte), mas o pedido final é o boleto (Financeiro). |
| C02 "Preciso quitar a parcela do semestre" | Revisão humana | Financeiro (margem 0,55) | Vocabulário novo: "quitar" e "parcela" não estão na lista de regras. | Aqui o modelo acertou. Aceitar se a margem for alta e registrar para revisar a lista de palavras. |
| C06 "Paguei a taxa mas a rematrícula continua pendente" | Matrícula | Matrícula (margem 0,87) | Mistura pagamento (Financeiro) e rematrícula (Matrícula). As regras só viram "rematrícula", pois "paguei" não está na lista. Os dois concordaram, mas a confiança pode ser falsa. | Mensagem que cita dois setores deve avisar os dois, mesmo quando o método dá uma resposta só. |
| C08 "Preciso saber onde registrar as horas da empresa" | Revisão humana | Matrícula (margem 0,01) | Parece ser Estágio, mas não usa nenhuma palavra típica. O modelo errou com margem praticamente zero. | Margem abaixo de um limite (ex.: 0,3) vai para atendente, mesmo com resposta do modelo. |
| C05 "O sistema caiu ao enviar o relatório de estágio" | Estágio | Estágio (margem 0,89) | "Sistema caiu" é problema técnico (Suporte); "estágio" é o assunto. Os dois métodos escolheram Estágio. | Sem a revisão de uma pessoa, não dá para saber qual dos dois setores é o certo. |

## Experimento com unigramas

| Configuração | Acurácia no teste |
|---|---:|
| `ngram_range=(1, 2)` unigramas + bigramas | 0,70 |
| `ngram_range=(1, 1)` só unigramas | 0,70 |

A acurácia foi igual. Com 20 mensagens de teste, cada mensagem vale 5 pontos percentuais, então uma diferença pequena seria só ruído. Escolher a configuração olhando o teste é limitado: a escolha se ajusta ao teste e ele deixa de medir como o modelo vai se sair com mensagens novas. O correto é comparar com validação cruzada no treino e usar o teste uma única vez no final.

## Parecer final (até 15 linhas)

1. As regras acertaram 80% e o modelo 70% no mesmo teste, mas as medidas não são comparáveis diretamente.
2. As regras erram "para o lado seguro": quando não têm certeza, mandam para uma pessoa. O modelo sempre decide.
3. O modelo tem poucos dados (60 mensagens de treino), e as classes Financeiro e Matrícula se confundem.
4. Com 20 mensagens de teste, a diferença de 10 pontos pode ser acaso.
5. As regras dependem de a pessoa escrever as palavras exatas e falham com sinônimos (C02, C07, C08).
6. O modelo generaliza melhor para palavras novas (C02), mas erra com confiança quando o assunto foge do treino (C08).
7. Nas mensagens que citam dois setores (C01, C06), nenhum dos métodos entrega os dois.
8. A saída do modelo não é decisão infalível. Margem baixa deve ser lida como dúvida.
9. Política proposta: regras em empate ou sem correspondência vão para uma pessoa.
10. Para o modelo, margem abaixo de um limite (ex.: 0,3) também vai para uma pessoa.
11. Mensagem com termos de dois setores deve ser encaminhada aos dois, ou ao atendente.
12. Mensagens revisadas por pessoas devem voltar como novos exemplos de treino.
13. Para decidir entre os métodos de verdade, é preciso mais dados e validação cruzada, não só este teste.
