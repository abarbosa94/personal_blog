# Checkpoint da revisão manual v1 e rerun v2

Data: 2026-07-27.

Este checkpoint preserva as primeiras 30 decisões humanas da amostra formal e
compara os 29 casos marcados como falha com um rerun PDF-only. O rerun não
consulta OpenAlex e não sobrescreve os resultados originais.

## Congelamento v1

| Medida | Valor |
| --- | ---: |
| Itens totais na fila | 100 |
| Revisados | 30 |
| Pass | 1 |
| Fail | 29 |
| Defer | 0 |

SHA-256 da fila congelada:
`2aa2f0e424e38f0d1712a0d1d7027ba0accd9b0aabfeaa993b823fccb455263b`.

O arquivo local está em
`artifacts/manual-review/v1-30/review-queue.csv` e permanece ignorado pelo Git
por conter o estado de trabalho da revisão.

## Mudanças avaliadas no v2

- tentativas HTTP com backoff exponencial;
- cache local de respostas bem-sucedidas;
- diagnósticos por estágio, distinguindo erro, ausência e sucesso;
- identificação do bloco de afiliações imediatamente antes de
  `Correspondence to`;
- reparo de palavras hifenizadas entre linhas;
- extração determinística de países explícitos e normalização ISO alpha-2;
- modo `--pdf-only` para isolar a melhoria sem uma chave OpenAlex.

## Resultado

| Medida | v1 | v2 |
| --- | ---: | ---: |
| Correspondência exata de países | 0/29 | 24/29 |
| Recall de países esperados | 8,9% | 86,7% |
| PDFs baixados com sucesso | não observável | 29/29 |
| País extra no resultado final | — | 0 |

SHA-256:

- entrada dos 29 casos:
  `915fee85cb31ffe53e65d4ba748c5f584a826c54df22ac30e7a268ab372ee45d`;
- saída v2:
  `6ff8e1f70559bf134f87963aaea845e543e67f26969531fe51acc9d4119a4cb6`;
- comparação:
  `324125e407efaf9a84d91ebba85216ae7c4159df6878d566d247e6b4c93361a3`.

## Casos residuais

Cinco casos não atingiram a correspondência exata:

- `icml:guo25r`: faltam CN e US;
- `icml:nikulin25a`: falta RU;
- `neurips:dd1fef536655685898a6602bfbf16857`: falta DE;
- `neurips:f3f607e4c13bd1cb8885de44b4ec45b7`: falta CN;
- `icml:huang25z`: falta CH; US foi recuperado.

Esses PDFs identificam organizações, mas não fornecem todos os países em texto
explícito. O próximo experimento deve testar segmentação de organizações e
resolução ROR. Um modelo NER geral não será aceito apenas por recuperar mais
entidades: ele deve ser comparado com as anotações congeladas e não pode reduzir
a precisão de países.

## Piloto spaCy + ROR

O piloto usou spaCy `3.8.14` e o modelo oficial `en_core_web_sm` `3.8.0`.
Listas de afiliações numeradas foram segmentadas antes do NER; candidatos foram
aceitos somente quando o ROR retornou `chosen:true`.

| Medida | Resultado |
| --- | ---: |
| Casos residuais | 5 |
| Exatos antes do piloto | 0/5 |
| Exatos após o piloto | 5/5 |
| Casos com país extra | 0 |
| Gate local | PASS |

O SHA-256 do relatório local
`artifacts/manual-review/v1-30/spacy-ner-pilot.json` é
`4ae7531c78f1979892d87e19bb1cf033b52f4f5d13377829459f312df62007ab`.

O resultado aceita a abordagem para avaliação ampliada, mas não para produção.
O NER também propôs entidades irrelevantes em blocos não numerados, como o
título do artigo e siglas do método; elas não produziram países porque o ROR não
as marcou como escolhidas. Os 70 itens restantes são necessários para estimar
falsos positivos fora deste conjunto dirigido por falhas.

## Limitações

As primeiras 30 linhas foram priorizadas por falha e incerteza; portanto, a taxa
de falha observada não estima a qualidade dos 100 itens. A comparação mede apenas
os casos já revisados e não valida os 70 restantes.
