# Contrato analítico — H-BASE-001

## Objetivo

Construir um baseline contemporâneo comparável entre países, inspirado nos
indicadores do Global AI Vibrancy Tool, sem apresentar a extensão como uma
edição oficial do Stanford HAI.

## Painéis temporais

| Painel | Período | Interpretação |
| --- | --- | --- |
| HAI oficial | 2017–2024 | Histórico publicado no Global AI Vibrancy Tool |
| Contemporâneo anual | 2025 | Ano completo, somente para indicadores equivalentes disponíveis |
| Contemporâneo parcial | 2026 YTD | Estoques, snapshots ou fluxos parciais até a data de corte |

Valores de 2026 YTD não serão comparados como se fossem totais anuais de 2025.

## Regra de seleção

Um indicador entra no painel contemporâneo somente quando:

1. sua definição pode ser reproduzida ou obtida da mesma organização;
2. há cobertura internacional suficiente para os grupos de comparação;
3. o período e a data de corte podem ser registrados;
4. o dado pode ser auditado por um leitor;
5. sua inclusão foi decidida antes de observar o valor brasileiro.

Indicadores indisponíveis serão removidos. Eles não serão imputados,
extrapolados ou substituídos silenciosamente por proxies.

## Painel de comparação congelado

O painel de 16 países foi congelado na versão 1 em
`analysis-specs/country-comparison-panel.csv`: Brasil; Estados Unidos e China;
Argentina, Chile, Colômbia e México; Índia, Indonésia, África do Sul e
Turquia; Emirados Árabes Unidos; Canadá, Reino Unido, França e Alemanha.
Alterações posteriores exigem nova versão e justificativa anterior à
recomputação dos resultados.

## Indicadores aprovados

### Painel de 2025

| Indicador | Fonte | Comparabilidade |
| --- | --- | --- |
| Notable AI Models | Epoch AI | Exata dentro da taxonomia do Epoch |
| Academia–Industry Model Production Concentration | Epoch AI | Exata dentro da classificação organizacional do Epoch |
| Total AI Private Investment | AI Index 2026/Quid | Agregado oficial de 2025 |
| Newly Funded AI Companies | AI Index 2026/Quid | Agregado oficial de 2025 |
| Accepted Papers at Selected AI Conferences | OpenAlex | Indicador adicional do estudo |
| Accepted Papers on RAI Topics | AI Index/OpenAlex | Adaptação explícita para trabalhos aceitos/publicados |
| Supercomputers | TOP500 | Snapshot |
| Compute Capacity (Rmax) | TOP500 | Snapshot |

### Painel de 2026 YTD

| Indicador | Data de corte |
| --- | --- |
| Notable AI Models | Data da extração do Epoch |
| Academia–Industry Model Production Concentration | Data da extração do Epoch |
| Accepted Papers at Selected AI Conferences | Data da extração do OpenAlex |
| Accepted Papers on RAI Topics | Data da extração do OpenAlex |
| Supercomputers | Lista TOP500 de junho de 2026 |
| Compute Capacity (Rmax) | Lista TOP500 de junho de 2026 |

Publicações de 2026 serão rotuladas como YTD e acompanhadas pela data de
indexação. Elas não serão comparadas diretamente com o total anual de 2025.
O agregado de 2026 também não será usado enquanto parte dos venues anuais ainda
não tiver ocorrido. Serão mostradas contagens por venue e o status
`realizada`, `a realizar` ou `não ocorre neste ano`.

## Publicações em conferências de IA

O universo principal será composto por venues reconhecidos de IA geral,
aprendizado de máquina, visão computacional, processamento de linguagem e
mineração de dados:

- AAAI e IJCAI;
- NeurIPS, ICML e ICLR;
- CVPR, ICCV e ECCV;
- ACL e EMNLP;
- KDD.

Cada venue será identificado pelo registro versionado em
`data/sources/conference-venues.csv`. A fonte oficial define o universo de
trabalhos; o OpenAlex é usado para enriquecer afiliações. Workshops, arXiv,
editoriais, erratas e duplicatas serão excluídos.

### Atribuição geográfica e organizacional

Serão produzidas duas lentes independentes:

1. **por país:** um trabalho conta para cada país representado nas afiliações
   dos autores;
2. **por organização:** um trabalho conta uma vez para cada instituição
   representada, preservando o tipo de instituição informado pelo OpenAlex.

A análise principal usará contagem completa. Uma análise de sensibilidade usará
contagem fracionária por país e por organização. Por isso, a soma das contagens
completas entre países pode ser maior que o total de trabalhos únicos.

Empresas serão identificadas pela classificação institucional do OpenAlex e
auditadas manualmente para as organizações com maior número de trabalhos.

## Responsible AI — extensão adiada

Responsible AI não integra mais o baseline inicial. A tentativa de adaptar o
indicador para trabalhos aceitos/publicados foi executada e preservada, mas os
classificadores temáticos não passaram o gate cego pós-freeze. O melhor
classificador local testado obteve 60,0% de precisão e 65,5% de recall em 120
papers revisados sem acesso às predições.

O indicador não será usado em contagens por país, comparações ou conclusões do
post inicial. Os universos, abstracts, rótulos manuais, contratos e resultados
negativos permanecem congelados como uma extensão metodológica futura. Sua
eventual retomada exigirá um novo método e um novo gate independente; não é
condição para concluir `H-BASE-001`.

## Scores e pesos

Não será calculado um Vibrancy Score para 2025 ou 2026. A redistribuição dos
pesos após remover indicadores e pilares produziria um construto diferente do
índice oficial.

Serão apresentados valores absolutos, variantes per capita quando houver
denominador compatível, posição no grupo, cobertura, missingness e mudanças
somente entre observações semanticamente equivalentes.

## Regra para snapshots

Para TOP500, serão preservadas as observações de junho e novembro de 2025. O
resumo anual usará novembro; junho permanecerá disponível para mostrar variação.

## Checklist de fechamento

- [x] Baixar os arquivos do AI Index 2026 relevantes para economia.
- [x] Resolver e registrar os IDs OpenAlex de cada conferência.
- [x] Registrar como adiada a extensão de Responsible AI após falha do gate
  independente; ela não bloqueia o baseline inicial.
- [x] Fixar a data de indexação e medir a cobertura de afiliações.
- [x] Fixar o universo de modelos e as categorias organizacionais do Epoch.
- [x] Definir os grupos de comparação.
- [x] Registrar data de corte, URL, licença e SHA-256 dos arquivos publicados.
- [x] Verificar cobertura antes de calcular resultados.
