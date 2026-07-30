# Amostra de revisão v2 dos 70 casos restantes

Data: 2026-07-27.

Este checkpoint aplica o pipeline candidato v2 somente aos 70 registros ainda
não revisados. As 30 decisões v1 permanecem congeladas e não foram alteradas.

## Pipeline candidato

O v2 preserva correspondências e afiliações OpenAlex do v1 e acrescenta:

- nova delimitação do bloco de afiliações do PDF;
- reparo de hifens entre linhas;
- países explícitos normalizados;
- segmentação de listas numeradas;
- candidatos `ORG` do spaCy `en_core_web_sm` 3.8.0;
- resolução ROR aceita apenas com `chosen:true`;
- diagnósticos e cache HTTP.

Este pipeline ainda é candidato. A amostra humana mede especialmente falsos
países introduzidos pelo NER e pelo enriquecimento ROR.

## Resultado automático nos 70

| Medida | v1 | v2 |
| --- | ---: | ---: |
| Registros com país | 53/70 | 65/70 |
| Registros que receberam país adicional | — | 26 |
| Menções de país adicionadas | — | 43 |
| Erros de estágio registrados | não observável | 0 |

## Amostra humana

A seleção usa amostragem aleatória determinística de 35 dos 70 registros, seed
`20250727`. A ordem da interface pode priorizar sinais de incerteza, mas a
composição da amostra não muda.

| Dimensão | Contagem |
| --- | ---: |
| Total | 35 |
| ICML | 12 |
| NeurIPS | 23 |
| `fallback_only` | 15 |
| `mixed_sources` | 9 |
| `automatic_pass` | 11 |
| Sobreposição com os 30 já revisados | 0 |

SHA-256:

- população v1 intocada:
  `da3d3fe7336f15e8af66f2688806262a466f2c7e2afca99b732f2fee861a26eb`;
- população v2:
  `538874be20cdc8884658422727cba8e27f9c70dc56a1c833a0fb8e9fc453d119`;
- amostra JSONL:
  `104eb3d2028af31aad7b038907b44f95bcc2e8eee97bab0c0178febd6e4731c5`;
- fila de revisão CSV:
  `acb2bf38fef38dac3176cfa99ae5d92e59392fde272c41279b16b4cc1e3170ca`.

## Arquivos locais

- `data/processed/2025-unreviewed-70-v1.jsonl`;
- `data/processed/2025-unreviewed-70-v2.jsonl`;
- `data/processed/2025-unreviewed-v2-review-sample-35.jsonl`;
- `data/processed/2025-unreviewed-v2-review-sample-35.csv`.

Todos permanecem ignorados pelo Git. Continue sempre no CSV v2 de 35 itens; não
reabra a fila v1 de 100 itens para esta etapa.
