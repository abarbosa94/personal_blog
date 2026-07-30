# Auditoria inicial das fontes do baseline

Data da auditoria: 2026-07-27.

## Pergunta de auditoria

As fontes candidatas permitem medir, com definições auditáveis, tanto a
amplitude do ecossistema nacional quanto sua presença observada na fronteira de
IA?

## Resultado

| Fonte | Papel | Cobertura observada | Decisão | Condição principal |
| --- | --- | --- | --- | --- |
| Stanford Global AI Vibrancy Tool | Amplitude do ecossistema | 66 países, 2017–2024 | Aceitação condicional | Vincular regras da edição 2025 aos arquivos atuais |
| Epoch AI Models | Modelos relevantes e compute de treinamento | 3.572 modelos | Aceitação com condições | Não confundir país da organização, treinamento e captura de valor |
| Epoch AI Data Centers | Maiores projetos conhecidos | 74 sites, 8 países | Aceitação com condições | Separar atual, anunciado e projetado; não tratar como censo |
| Epoch AI GPU Clusters | Clusters publicamente documentados | 482 clusters, 36 países | Aceitação com condições | Filtrar operação, certeza e duplicidades |

## O que as fontes podem sustentar

- O Stanford HAI pode descrever a amplitude relativa do ecossistema e suas
  dimensões, sujeito à auditoria das transformações.
- O Epoch pode mostrar presença observada entre modelos, clusters e grandes
  data centers documentados publicamente.
- A combinação oferece triangulação por lentes diferentes.

## O que as fontes não podem sustentar

- Um score único Stanford + Epoch.
- A inferência de que ausência no Epoch significa inexistência no país.
- Atribuição de infraestrutura a um país usando somente a sede da organização.
- Comparação de capacidade operacional com projetos anunciados sem distinção.
- Conclusões sobre publicação científica usando a base de modelos do Epoch.

## Bloqueios antes de olhar o resultado brasileiro

1. congelar versões e checksums;
2. resolver a documentação metodológica da edição 2025 do Stanford;
3. definir grupos de comparação sem observar a posição brasileira;
4. definir regras de missingness, certeza, status e duplicidade;
5. registrar critérios de suporte e contestação da hipótese baseline.

## Próximo artefato

Criar `analysis-specs/H-BASE-001.md` com o contrato analítico. Ele deve declarar
unidades de análise, períodos, grupos de comparação, filtros, normalizações,
tratamento de ausências e tabelas esperadas antes da primeira análise por país.

