# Checkpoint para retomada

Última atualização: 2026-07-30.

## Plano staged 1–9 concluído

O notebook original foi continuado e passou pelo gate de publicação:

- `H-BASE-001` foi contestada pelo baseline de sete indicadores;
- P&D e infraestrutura foram classificados como mistos, ciência e demanda
  como inconclusivas;
- finanças, jurídico e modelos em português permaneceram vantagens candidatas;
- `H-CONV-001` recebeu suporte provisório e descritivo;
- `H-MAIN-001` permanece inconclusiva porque nenhum caso demonstrou,
  simultaneamente, diferenciação, resultado e captura local comparável;
- Responsible AI continua fora da análise inicial;
- três figuras foram inspecionadas, 122 testes passaram e o notebook foi
  renderizado com Quarto 1.9.38;
- o post está com `draft: false`.

Os artefatos de fechamento estão em
`artifacts/analysis/baseline-seven-indicator-evidence-matrix.*`,
`artifacts/analysis/world-bank-factor-context.*` e
`artifacts/analysis/hypothesis-assessments.*`. Extensões futuras não alteram
esse fechamento sem um novo contrato e uma nova versão do post.

## Checkpoint atual do baseline pós-conferências

O pool final das sete venues foi concluído e não há jobs de conferências
pendentes. Todas passaram o gate de 90%; os resultados e a sensibilidade estão
em `artifacts/analysis/conference-presence-2025-seven-venues-*`.

O lote seguinte do plano mestre foi iniciado:

- o snapshot Epoch AI de 2026-07-29 foi congelado em
  `data/raw/external/epoch-ai/2026-07-29/all_ai_models.csv`, SHA-256
  `2E4E18BF5630D47FD60343860D597166A18BB337929B50B77FBB1EB900E2CF1A`;
- os indicadores de modelos notáveis e composição academia–indústria foram
  implementados para 2025 e 2026 YTD, sob
  `analysis-specs/epoch-models-contract.md`;
- o Epoch contém 103 modelos notáveis em 2025; nenhum está atribuído a uma
  organização brasileira. Isso é evidência contrária de presença observada na
  produção de modelos de fronteira, não prova de ausência de capacidade;
- as listas oficiais TOP500 de junho/novembro de 2025 e junho de 2026 foram
  congeladas, cinco páginas e 500 ranks únicos por snapshot;
- os artefatos `artifacts/analysis/top500-*.csv` reportam sistemas e soma de
  Rmax. Em novembro de 2025, o Brasil tem 10 sistemas e 124,52 PFlop/s;
- 108 testes passam;
- o Wikidata foi congelado como fallback terciário para análise institucional
  em `analysis-specs/wikidata-institution-fallback.md`. Ele não deve
  sobrescrever localização explícita/ROR nem converter sede de multinacional
  em localização de autor.

Os dois agregados econômicos do AI Index 2026/Quid foram concluídos. A extensão
temática de Responsible AI foi testada, falhou o gate independente e foi
retirada do baseline inicial.

Os dois agregados econômicos foram concluídos em
`artifacts/analysis/ai-index-economy-2025.csv`, sob o contrato
`analysis-specs/ai-index-economy-2025-contract.md`. O Brasil não aparece no
top 15 publicado em nenhuma das duas figuras. Portanto, os resultados são
censurados, não zeros: investimento privado em IA abaixo de US$ 0,970 bilhão e
menos de 33 empresas de IA recém-financiadas em 2025. A inconsistência entre o
CSV e o gráfico oficial da Figura 4.2.9 foi preservada e a seção transversal
foi extraída do PDF oficial. Após esse lote, 110 testes passam.

O codebook HAI de 24 de setembro de 2025 também foi congelado. Ele confirma as
quatro dimensões temáticas de Responsible AI: privacidade/governança de dados,
transparência/explicabilidade, segurança/safety e fairness. O estudo mantém a
adaptação já congelada de submissões para trabalhos aceitos/publicados em
AAAI, AIES, FAccT, ICLR, ICML e NeurIPS.

Em 2026-07-30 foram registradas três hipóteses posteriores, ainda não testadas:
`H-SECTOR-FIN-001` (Nubank e Itaú), `H-SECTOR-LEGAL-001` (Jusbrasil) e
`H-CONV-MARITACA-001` (modelos locais em português). Elas não são conclusões do
baseline. Seus mecanismos, contra-hipóteses, indicadores candidatos e
guardrails estão no registro de hipóteses e em
`analysis-specs/sector-case-hypotheses.md`.

## Responsible AI — extensão encerrada no baseline inicial

A tentativa do oitavo indicador foi concluída como resultado metodológico
negativo. Ela não é mais tarefa ativa nem condição para o post inicial. Os
universos oficiais permanecem congelados:

- AIES 2025: 238 trabalhos de main track; 40 student abstracts do terceiro
  volume foram excluídos;
- FAccT 2025: 217 entradas no CSV oficial, sendo 206 archival e 11
  nonarchival. O relato do processo informa 218 aceitos, portanto existe uma
  divergência oficial de um registro a preservar.

A amostra determinística de 86 trabalhos foi validada com contexto de
abstract/full text. O resultado proposto contém 48 positivos e 38 negativos:
29/32 candidatos explícitos foram positivos, 13/24 casos ambíguos foram
positivos e 6/30 negativos do screen por título eram positivos. Portanto, o
classificador apenas por título falhou o gate, apesar da alta precisão entre
os candidatos explícitos. Os artefatos de decisão, contexto e auditoria estão
em `artifacts/analysis/responsible-ai-2025-validation-*`; sete decisões
limítrofes estão marcadas para eventual segunda revisão.

O próximo passo é congelar e aplicar um classificador por contexto de abstract
aos universos completos das seis venues. Não calcular o indicador por país
antes dessa aplicação e de seus checks de qualidade.

O contrato desse classificador foi congelado em
`analysis-specs/responsible-ai-2025-context-contract.md`. Na amostra de
calibração, a regra determinística obteve 48 verdadeiros positivos, dois falsos
positivos, nenhum falso negativo e 36 verdadeiros negativos (precisão de 96% e
recall de 100%). Como a própria amostra orientou o refinamento, esses números
não são uma validação externa e será feita auditoria pós-classificação.

A coleta resumível de abstracts oficiais dos seis universos está ativa com
oito workers. O checkpoint é
`artifacts/analysis/responsible-ai-2025-corpus-context.jsonl`, o PID está em
`artifacts/responsible-ai-2025-context.pid` e os logs usam o prefixo
`artifacts/responsible-ai-2025-context`. Erros são preservados e repetidos na
retomada; o gate exige pelo menos 95% de abstracts em cada venue.

O ICLR foi separado do coletor genérico porque os endpoints OpenReview
retornaram HTTP 403. O adaptador escalável usa o índice oficial único de 3.703
papers em `proceedings.iclr.cc`, reconcilia títulos e baixa somente as páginas
HTML leves de abstract com 20 workers. A coleta concluiu 3.703/3.703 abstracts;
quatro mudanças de título entre OpenReview e os proceedings foram documentadas
no adaptador e em
`artifacts/analysis/responsible-ai-2025-iclr-reconciliation.json`. O checkpoint
é `artifacts/analysis/responsible-ai-2025-iclr-context.jsonl`.

Por decisão explícita, os dez abstracts ausentes do NeurIPS foram aceitos como
limitação e não são tratados como negativos. O corpus consolidado contém
16.797 papers, 16.787 abstracts observados e 2.215 candidatos positivos pela
regra contextual. As coberturas são 100% em AAAI, AIES, FAccT, ICLR e ICML, e
99,83% no NeurIPS. Os artefatos estão em
`artifacts/analysis/responsible-ai-2025-corpus-classified.jsonl` e
`responsible-ai-2025-corpus-summary.csv`.

A auditoria pós-classificação determinística contém 158 papers: 15 positivos e
10 negativos por venue, mais oito positivos sustentados somente por padrões de
título adicionados após o screen original. Ela está ativa em
`artifacts/analysis/responsible-ai-2025-postclassification-audit.csv`. Não
agregar países antes de concluir esse gate.

A primeira auditoria pós-classificação falhou a regra original (85,7% de
precisão, 92,3% de recall). A revisão orientada pelos 21 erros adicionou
conceitos de incidentes, deepfakes não consensuais, red-teaming, bias audits,
estereótipos, disparidades demográficas e surveillance privacy, além de excluir
usos matemáticos/incidentais de explanation, discrimination, bias, fairness e
transparency. A regra revisada passa nos dois conjuntos já revisados: 93,8% de
precisão/recall nos 86 papers e 94,7% de precisão com 98,9% de recall nos 158
papers; 119 testes passam.

O corpus foi reclassificado em
`artifacts/analysis/responsible-ai-2025-corpus-classified-v2.jsonl`, com 2.186
candidatos positivos. Uma confirmação fresca, sem sobreposição com os 238
papers anteriores, foi congelada em
`artifacts/analysis/responsible-ai-2025-confirmation-sample.csv`: cinco
positivos e cinco negativos por venue, 60 papers no total. A revisão
independente dessa confirmação está ativa; não agregar países antes do seu
resultado.

O classificador local foi substituído por TF-IDF + LightGBM, sem venue como
feature. A avaliação nested sobre 298 papers únicos indicou 96,0% de precisão e
99,4% de recall, mas esse resultado era desenvolvimento porque as regras já
haviam sido informadas pelos rótulos. O gate cego pós-freeze de 120 papers foi
concluído e aberto de forma independente. O modelo falhou materialmente:
36 TP, 24 FP, 19 FN e 41 TN, com 60,0% de precisão e 65,5% de recall.
Nenhuma dimensão passou 90%/90%. Os artefatos de abertura estão em
`artifacts/analysis/responsible-ai-2025-lightgbm-v1-blind120-*`.

Portanto, `responsible-ai-lightgbm-v1` não pode ser congelado para inferência,
os 2.186 rótulos provisórios não podem alimentar um indicador por país e
nenhuma agregação RAI será feita nesta análise inicial. Esse fracasso também
demonstra que a amostra anterior havia produzido uma estimativa excessivamente
otimista. O baseline inicial passa a ter sete indicadores concluídos; o próximo
trabalho ativo é consolidá-los na matriz de evidências e continuar o notebook.

## Regra editorial do post

A etapa de redação não deve criar um texto separado nem substituir a abertura
existente. Ela deve continuar diretamente o notebook
`posts/2026-07-27-VantagensCompetitivas-Brasil-IA.ipynb`, preservando o tom já
estabelecido pelo autor: português brasileiro em primeira pessoa do plural,
explicação exploratória e acessível, perguntas que conduzem o argumento,
transições conversacionais, cautela metodológica sem excesso de jargão e
observações pessoais pontuais. Novas seções devem partir do baseline que já
começa no notebook e integrar tabelas, figuras e limitações à mesma narrativa.

Este arquivo registra o ponto exato de retomada do experimento. Nenhuma chave de
API é armazenada no repositório.

## Objetivo principal

O objetivo deste trabalho não é apenas recuperar países de afiliação em
conferências. O objetivo final é concluir e publicar o post
`posts/2026-07-27-VantagensCompetitivas-Brasil-IA.ipynb`, respondendo com
evidência reproduzível se o Brasil possui vantagens competitivas mensuráveis em
IA.

A hierarquia correta do trabalho é:

1. construir e auditar a base de evidências;
2. testar as hipóteses sobre P&D, ciência, infraestrutura, demanda,
   especialização setorial e conversão em valor;
3. comparar o Brasil com o painel congelado de 16 países usando indicadores
   absolutos, normalizados e temporais;
4. escrever o argumento, as tabelas e as figuras do post;
5. validar fontes, limitações e interpretações alternativas;
6. executar as revisões técnica e editorial e preparar a publicação.

O censo de países das conferências é apenas uma subtarefa do primeiro item:

`post -> base de evidências -> presença científica em 2025 -> países das afiliações em conferências`.

Essa subtarefa foi concluída sob o contrato v3. Todas as sete venues superam
90% de cobertura e a amplitude é 6,00 pontos percentuais. O pool primário
coloca o Brasil em 10º de 16 países, com share fracionário de 0,0640%; o pool
ponderado por papers também o coloca em 10º. O resultado não sustenta vantagem
ampla em presença nas conferências; sustenta presença pequena, segunda entre os
países latino-americanos do painel e parcialmente dependente de ICML. O brief
está em
`artifacts/analysis/conference-presence-2025-seven-venues-brief.md`.

Ao retomar o projeto, não tratar a conclusão do censo como conclusão do
experimento. Depois do censo, retornar ao plano completo em `plan.md` e à
especificação do baseline em `analysis-specs/H-BASE-001.md`.

## Estado atual do censo de conferências

- O reconciliador de afiliações V9 tornou-se o caminho padrão.
- O gate cego V9 em 25 trabalhos ICML obteve 25/25 conjuntos de países exatos
  e 36/36 rótulos de país corretos.
- Os gates de regressão preservam 125/125 trabalhos previamente revisados.
- O painel comparativo de 16 países está congelado na versão 1 em
  `analysis-specs/country-comparison-panel.csv`.
- A enumeração oficial de AAAI, ACL, ICML e NeurIPS 2025 está concluída.
- AAAI está concluída: 3.486 trabalhos, cobertura de país de 91,48%.
- ACL está concluída: 1.602 trabalhos, cobertura de país de 59,93%; o resultado
  final deve carregar uma advertência explícita de cobertura.
- O retry OpenAlex do ICML está concluído: 2.933/2.933 registros. A extração
  oficial por PDF também está concluída para 3.330/3.330 trabalhos. Ainda é
  necessário fundir as duas fontes e gerar a tabela final do ICML.
- O NeurIPS foi finalizado nesta versão do pipeline: 5.823 registros únicos,
  2.970 matches OpenAlex e 2.425 trabalhos com evidência de país (41,65%).
  Restaram três erros HTTP 429 (0,05% do universo), preservados como limitação.
- A cobertura baixa do NeurIPS continua sendo uma limitação analítica material.
  O diagnóstico e as regras de interpretação estão em
  `data/sources/neurips-2025-census-audit.md`.
- O ICML foi fundido em `data/processed/icml-2025-v9-census.jsonl`: 2.197/3.330
  trabalhos possuem evidência de país (65,98%).
- O comparativo das quatro conferências está em
  `artifacts/analysis/conference-presence-2025-four-venues.csv`. Ele cobre
  14.241 trabalhos, dos quais 8.771 possuem país (61,59%).
- A análise de sensibilidade de cobertura está concluída em
  `artifacts/analysis/conference-presence-2025-coverage-sensitivity.md` e no CSV
  de mesmo prefixo. A ordem China–Estados Unidos–Reino Unido é estável nas
  quatro venues sob contagem completa e fracionária; o rank do Brasil também
  não muda. Essa estabilidade vale para os registros observados e não elimina
  o viés possível da cobertura ausente.
- Todos os 96 testes passam após a correção dos títulos OpenAlex nulos, a busca
  exata e a implementação da concorrência limitada.

Contagens intermediárias são checkpoints operacionais, não resultados finais.
Antes de citar números, verificar os arquivos e processos locais.

## Próximas etapas

O escopo do indicador de conferências foi reaberto e congelado na versão 2 em
`analysis-specs/conference-pooled-2025-contract.md`. O resultado final deverá
incluir AAAI, ACL, ICML, NeurIPS, EMNLP, ICLR e KDD 2025 Applied Data Science
(ADS), o track industrial/aplicado. A estimativa principal será a
média de shares fracionários com peso igual por venue; o pool ponderado por
trabalho será secundário.

O pool somente será considerado comparavelmente observado se todas as venues
atingirem pelo menos 90% de cobertura (contrato v3), a amplitude entre
coberturas for no
máximo 15 pontos percentuais e erros sistemáticos de API ficarem abaixo de
0,5%. O comparativo atual de quatro venues volta a ser um checkpoint
diagnóstico, não o resultado final do indicador.

O EMNLP 2025 main já foi enumerado: 1.809 trabalhos de pesquisa; a entrada
`.0` de front matter foi excluída dos 1.810 itens listados pelo Anthology. O
arquivo é `data/raw/emnlp-2025.jsonl`, SHA-256
`0f5c59ef2d98945491db208ee523d858c9196e1f48031298bc452bc26734bb62`.

O ICLR 2025 foi enumerado via `openreview-py`: 3.703 trabalhos aceitos,
`data/raw/iclr-2025.jsonl`, SHA-256
`411e8e6f7fc5461534811f29db9189a4a6a426c739a55d904d410f1c1e343896`.
O censo base V9 roda com 8 workers e PID persistido em
`artifacts/iclr-2025-v9-census-base.pid`.

O KDD 2025 ADS foi enumerado na lista oficial: 155 trabalhos, sendo 92 do
ciclo de fevereiro e 63 do ciclo de agosto. O arquivo
`data/raw/kdd-ads-2025.jsonl` tem SHA-256
`bec3f202a20f1b557316206b5afcf64004bdb86c401af18ca8425028d21b7cb8`.
O censo base V9 roda com 8 workers e PID persistido em
`artifacts/kdd-ads-2025-v9-census-base.pid`.

## Execução ativa e bloqueios de 2026-07-29

Os passes OpenAlex de EMNLP e KDD ADS concluíram a enumeração, mas todas as
tentativas receberam `HTTP 429`; portanto, esses arquivos base têm cobertura
zero e não satisfazem o gate de erro de API. O passe equivalente de ICLR foi
interrompido de forma resumível em 1.450/3.703 para evitar chamadas inúteis.

A recuperação V9 por PDF oficial de EMNLP está ativa com 8 workers. O PID está
em `artifacts/emnlp-2025-v9-pdf-country.pid`, o checkpoint em
`data/processed/emnlp-2025-v9-pdf-country.jsonl` e os logs em
`artifacts/emnlp-2025-v9-pdf-country.*.log`.

KDD ADS não expõe URLs diretas de PDF na lista oficial. O estágio genérico de
PDF preservou os 155 registros, mas não adicionou países; será necessário um
adaptador específico para DOI/ACM ou para as afiliações publicadas na página
oficial. Depois de EMNLP, ICLR deve seguir diretamente pelo caminho
`--pdf-only`, sem repetir o passe OpenAlex enquanto persistirem os `429`.

O cliente HTTP agora respeita `Retry-After`, usa backoff exponencial com jitter
e permite limitar o atraso máximo. O novo passe ICLR está ativo com um worker
em `data/processed/iclr-2025-v9-openalex-backoff.jsonl`; PID e logs usam o
prefixo `artifacts/iclr-2025-v9-openalex-backoff`. Em 2026-07-29, OpenAlex
informou saldo diário e prepaid de USD 0, com reset à meia-noite UTC; portanto,
o processo aguarda o reset indicado pelo servidor.

O adaptador KDD ADS extraiu afiliações publicadas na página oficial para todos
os 155 DOIs: 331 strings únicas, nenhuma ausente. A resolução ROR/país está
ativa com 8 workers; saída em
`data/processed/kdd-ads-2025-v9-official-affiliations.jsonl`, com PID e logs
sob `artifacts/kdd-ads-2025-v9-official-affiliations`.

Após a compra de créditos OpenAlex, todos os 155 DOIs do KDD ADS foram
reconciliados e mesclados com as afiliações da página oficial. O artefato final
é `data/processed/kdd-ads-2025-v9-final.jsonl`: 149/155 papers com país,
96,13% de cobertura, superando a meta explícita de 140/155. O relatório está
em `artifacts/kdd-ads-2025-v9-final-quality.json`.

O comando `v2_pipeline augment` agora aceita `--workers` com fila limitada e
mantém um modelo spaCy/cliente de busca por worker. Os cinco jobs ativos foram
reiniciados de seus checkpoints com 20 workers totais: ICLR 2, EMNLP 4, ACL 3,
ICML 4 e NeurIPS 7. Cada job continua usando seu arquivo JSONL, PID e logs com
o mesmo prefixo; `--resume` evita recalcular checkpoints concluídos.

ACL, ICML, NeurIPS e EMNLP concluíram a recuperação e foram sobrepostos aos
censos completos. As coberturas provisórias são 96,19%, 95,56%, 91,38% e
92,65%, respectivamente. ICLR foi aceito com 3.702/3.703 registros; o único
paper restante foi deliberadamente omitido (0,027% do universo). Como o passe
OpenAlex trouxe país para apenas 79 papers, a recuperação por PDF oficial do
OpenReview está ativa com 8 workers, PID em
`artifacts/iclr-2025-v9-pdf-country.pid`.

O endpoint PDF do OpenReview impôs limite de 26 downloads por hora; aumentar
concorrência não elevaria throughput. O downloader preservou 25 PDFs válidos,
mas foi substituído por recuperação em lote via perfis OpenReview: IDs de
autores são obtidos das notas aceitas, perfis são buscados em lotes de até
1.000 e instituições ativas em 2025 são resolvidas com 20 workers ROR. O job
ativo usa PID/logs `artifacts/iclr-2025-v9-profile-affiliations*` e saída
`data/processed/iclr-2025-v9-profile-affiliations.jsonl`.

O V9 está congelado como método suficientemente validado. Os gates cegos e de
regressão já executados serão reutilizados como qualificação do método; não será
criada outra amostra cega para EMNLP, ICLR, KDD ADS ou para o pool. As novas venues ainda
devem passar checks automáticos de universo oficial, filtros de aceitação,
duplicatas, schema, checksum, cobertura e erros de API.

1. concluir o censo ICLR 2025 iniciado e enumerar o KDD 2025 ADS oficial;
2. elevar ACL, ICML e NeurIPS ao gate de cobertura congelado usando V9;
3. executar a recuperação de país de EMNLP e ICLR usando V9;
4. gerar pools com peso igual por venue e ponderado por trabalho;
5. executar sensibilidade leave-one-venue-out;
6. incorporar os resultados e limitações ao post;
4. documentar limitações, especialmente a cobertura menor do ACL;
5. preservar Responsible AI como extensão adiada, sem usá-la no baseline;
6. extrair os demais indicadores aprovados de 2025 e 2026 YTD;
7. retornar à análise comparativa e à escrita do post;
8. gerar tabelas e figuras reproduzíveis, revisar o notebook e preparar a
   publicação.

Novas iterações do reconciliador ficam suspensas, salvo se surgir um bloqueio
analítico ou uma falha nos gates congelados.

## Artefatos locais

Os arquivos abaixo são ignorados pelo Git, mas permanecem no computador:

| Arquivo | SHA-256 |
| --- | --- |
| `data/processed/aaai-2025-formal-sample.jsonl` | `459b3671c1d337b259fc818a4544f78f3397db2ef86fe53eace8425f060d6c7b` |
| `data/processed/acl-2025-formal-sample.jsonl` | `6dcd7b4327c3842b038f81797defe385b4c56262f83279f69848129e393f1e42` |
| `data/processed/icml-2025-formal-sample.jsonl` | `2580a52f6dd6ad3f4ea3c43fb494958f108f99d1843edb6739c4236882989a22` |
| `data/processed/neurips-2025-formal-sample.jsonl` | `e9091e77d5e13e3aa258e413406e7a749e51037a2f3ab51d7281658f05a19357` |

Os relatórios automáticos estão em
`artifacts/<venue>-2025-formal-sample-quality.json`.

## Como retomar

No diretório `experiments/ai-opportunities-in-brazil`:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
Get-FileHash data\processed\*-formal-sample.jsonl -Algorithm SHA256
```

Compare os hashes com a tabela acima. Se coincidirem, não repita a coleta das
amostras formais. Elas são artefatos históricos de validação; o trabalho ativo
usa os censos completos descritos em **Estado atual do censo de conferências**.

Depois:

1. verificar os processos e checkpoints locais antes de iniciar qualquer nova
   coleta;
2. concluir as **Próximas etapas** na ordem registrada acima;
3. atualizar este checkpoint ao terminar cada etapa material;
4. usar `plan.md` como plano mestre e não limitar a retomada ao pipeline de
   conferências.

Caso seja necessário consultar novamente as APIs, configure uma nova chave
somente na sessão:

```powershell
$env:OPENALEX_API_KEY = "<nova-chave>"
```

Remova-a ao terminar:

```powershell
Remove-Item Env:OPENALEX_API_KEY
```

## Referências para contexto

Leia, nesta ordem:

1. `RESUME.md`;
2. `analysis-specs/H-BASE-001.md`;
3. `data/sources/vertical-slice-results.md`;
4. `data/sources/conference-venue-audit.md`;
5. `RUNBOOK.md`;
6. `plan.md`.

Código-base no início desta execução: commit
`bea75f16f032034e12e4a20ce8e42ea38b5f293f`.
