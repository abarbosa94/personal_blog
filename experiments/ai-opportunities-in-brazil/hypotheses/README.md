# Registro de hipóteses

Este diretório é a fonte de verdade das hipóteses do experimento. O notebook do
post pode consumir e resumir estes arquivos, mas não deve manter uma versão
paralela das hipóteses.

## Arquivos

- `registry.csv`: uma linha por hipótese, com alegação, contra-hipótese,
  mecanismo e critérios de decisão.
- `indicators.csv`: relação entre hipóteses e indicadores candidatos ou
  aprovados.
- `evidence.csv`: registro cumulativo das observações que sustentam, desafiam ou
  contextualizam uma hipótese.

## Estados de uma hipótese

| Estado | Significado |
| --- | --- |
| `proposed` | A alegação foi registrada, mas seu escopo ainda está aberto |
| `scoped` | Escopo, mecanismo e contra-hipótese foram definidos |
| `measurable` | Indicadores, comparadores e critérios foram aprovados |
| `collecting` | A coleta de dados está em andamento |
| `tested` | Os indicadores foram calculados e as tentativas de refutação executadas |
| `classified` | A conclusão e a incerteza foram registradas |
| `retired` | A hipótese foi substituída ou deixou de ser relevante |

## Classificações

O campo `classification` permanece vazio até a conclusão do teste. Os valores
permitidos são:

- `revealed_advantage`;
- `emerging_advantage`;
- `potential_advantage`;
- `inconclusive`;
- `no_advantage_demonstrated`.

## Regras de atualização

1. A alegação e a contra-hipótese devem ser registradas antes da análise.
2. Uma hipótese só passa para `measurable` depois da aprovação de seus
   indicadores, comparadores e critérios.
3. Não se reescreve silenciosamente uma hipótese depois de observar os dados.
   Uma mudança substantiva deve gerar uma nova hipótese ou ser documentada no
   histórico do Git.
4. Evidências favoráveis e contrárias são registradas em `evidence.csv`.
5. `absence_of_data` não equivale a evidência contrária.
6. Uma classificação deve citar as evidências usadas e registrar limitações.

## Relações entre os arquivos

```text
registry.csv
    hypothesis_id
         │
         ├── indicators.csv
         │       indicator_id
         │
         └── evidence.csv
                 indicator_id (quando aplicável)
```
