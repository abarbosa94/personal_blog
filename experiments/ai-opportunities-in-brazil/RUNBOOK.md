# Runbook — presença em conferências de IA

## Objetivo

Executar de forma reproduzível a prova vertical de enumeração e reconciliação
de trabalhos de AAAI, ICML e ACL 2025, sem calcular resultados por país antes de
os critérios de qualidade serem satisfeitos.

## Escopo

| Venue | Universo oficial | Regra inicial |
| --- | --- | --- |
| AAAI 2025 | Metadados DOI da AAAI no Crossref, volume 39 | Artigos do container AAAI e volume 39 |
| ICML 2025 | PMLR volume 267 | Main proceedings |
| ACL 2025 | ACL Anthology `2025.acl-long` | Long papers; front matter excluído |
| NeurIPS 2025 | Proceedings oficiais, volume 38 | Main, datasets/benchmarks e position papers preservados por track |

O uso de Crossref para AAAI é uma contingência: o OJS bloqueou enumeração
automatizada durante o desenvolvimento. A contagem deverá ser comparada com o
volume oficial e com o OpenAlex antes de ser aprovada.

## Pré-requisitos

- Python 3.11 ou superior;
- acesso HTTPS às fontes;
- chave gratuita do OpenAlex para a etapa de reconciliação;
- PowerShell, Bash ou terminal equivalente.

O pipeline usa apenas a biblioteca padrão do Python em produção. A suíte de
testes usa pytest e pytest-bdd.

## Preparação

No diretório `experiments/ai-opportunities-in-brazil`:

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### Bash

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e ".[test]"
```

### PowerShell

```powershell
$env:PYTHONPATH = "src"
$env:OPENALEX_API_KEY = "<sua-chave>"
```

### Bash

```bash
export PYTHONPATH=src
export OPENALEX_API_KEY="<sua-chave>"
```

Não salve a chave no repositório. Obtenha-a em
https://openalex.org/settings/api.

## 1. Executar os testes offline

```bash
pytest -v
```

Resultado esperado: testes unitários e cenários BDD passam sem acesso à rede.
Os cenários legíveis estão em `tests/features/conference_pipeline.feature`.

## 2. Enumerar os proceedings

```bash
python -m conference_pipeline enumerate \
  --venue icml \
  --output data/raw/icml-2025.jsonl

python -m conference_pipeline enumerate \
  --venue acl \
  --output data/raw/acl-2025-long.jsonl

python -m conference_pipeline enumerate \
  --venue aaai \
  --output data/raw/aaai-2025.jsonl

python -m conference_pipeline enumerate \
  --venue neurips \
  --output data/raw/neurips-2025.jsonl
```

Cada linha contém um trabalho. Os arquivos brutos são ignorados pelo Git.

## 3. Criar uma amostra de reconciliação

Comece com uma amostra aleatória reproduzível de 50 trabalhos por venue:

```bash
python -m conference_pipeline reconcile \
  --input data/raw/icml-2025.jsonl \
  --output data/processed/icml-2025-sample.jsonl \
  --sample-size 50 \
  --seed 20250727
```

Repita para ACL, AAAI e NeurIPS. A reconciliação tenta DOI primeiro e título
como fallback, exigindo similaridade mínima de 0,95. Quando as instituições
estão ausentes, a cascata tenta XML GROBID, primeira página do PDF e resolução
da afiliação pelo ROR. Somente resultados `chosen:true` do ROR são aceitos
automaticamente.

`--limit` permanece disponível apenas para smoke tests que deliberadamente usam
os primeiros registros. Não o use para estimar a cobertura formal.

## 4. Auditar a amostra

Use como `official-total` o número oficial de trabalhos daquele escopo, não a
quantidade da amostra. Para auditar apenas uma amostra, o denominador oficial
deve ser o tamanho planejado da amostra:

```bash
python -m conference_pipeline audit \
  --input data/processed/icml-2025-sample.jsonl \
  --official-total 50 \
  --output artifacts/icml-2025-sample-quality.json
```

Critérios:

| Medida | Mínimo |
| --- | ---: |
| Enumeração do universo oficial | 95% |
| Reconciliação com OpenAlex | 90% |
| Trabalhos com pelo menos um país | 85% |

Uma amostra aprovada valida a reconciliação, mas não prova que o enumerador
cobriu o universo completo.

## 5. Auditar o universo completo

Antes de reconciliar tudo:

1. registre o total informado pela fonte oficial;
2. compare-o com as linhas enumeradas;
3. explique tracks e exclusões;
4. investigue diferenças superiores a 5%;
5. registre URL, data de acesso e SHA-256.

Somente depois execute `reconcile` sem `--limit`.

## Revisão humana da amostra formal

Siga a rubrica em `MANUAL-REVIEW.md`. Gere uma única fila com ICML e NeurIPS:

```powershell
python -m conference_pipeline.manual_review build `
  data/processed/icml-2025-formal-sample.jsonl `
  data/processed/neurips-2025-formal-sample.jsonl `
  --output data/processed/2025-formal-manual-review.csv
```

Abra a interface local e retome sempre o mesmo CSV:

```powershell
python -m conference_pipeline.manual_review serve `
  data/processed/2025-formal-manual-review.csv
```

A interface salva cada decisão atomicamente. Ela apresenta falhas automáticas
primeiro, mas mantém todos os trabalhos da amostra, incluindo controles que o
pipeline considera corretos. Use `defer` quando a evidência não sustentar uma
decisão e revise esses casos depois da primeira passagem.

Para congelar um checkpoint e preparar um rerun isolado:

```powershell
python -m conference_pipeline.review_analysis freeze `
  data/processed/2025-formal-manual-review.csv `
  --output-dir artifacts/manual-review/v1-30

python -m conference_pipeline.review_analysis targets `
  data/processed/2025-formal-manual-review.csv `
  data/processed/icml-2025-formal-sample.jsonl `
  data/processed/neurips-2025-formal-sample.jsonl `
  --paper-output data/processed/2025-reviewed-failures-v2-input.jsonl `
  --expectation-output data/processed/2025-reviewed-failures-expected.csv

python -m conference_pipeline reconcile `
  --input data/processed/2025-reviewed-failures-v2-input.jsonl `
  --output data/processed/2025-reviewed-failures-v2.jsonl `
  --pdf-only

python -m conference_pipeline.review_analysis compare `
  data/processed/2025-reviewed-failures-expected.csv `
  data/processed/2025-formal-manual-review.csv `
  data/processed/2025-reviewed-failures-v2.jsonl `
  --output artifacts/manual-review/v1-30/v2-comparison.json
```

`--pdf-only` mede o fallback sem consultar OpenAlex. Não interprete esse modo
como uma substituição do reconciliador completo.

### Piloto NER para casos residuais

Instale a dependência opcional e o modelo oficial usado no checkpoint:

```powershell
python -m pip install -e ".[ner]"
python -m pip install `
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Execute o piloto somente sobre os casos que o v2 determinístico não resolveu:

```powershell
python -m conference_pipeline.ner_pilot `
  data/processed/2025-reviewed-failures-v2.jsonl `
  data/processed/2025-reviewed-failures-expected.csv `
  --output artifacts/manual-review/v1-30/spacy-ner-pilot.json
```

O piloto segmenta listas numeradas, usa spaCy para candidatos `ORG` e aceita
países somente de respostas ROR marcadas como `chosen`. Ele permanece
experimental até ser validado nos 70 itens ainda não revisados.

### Revisar metade dos 70 casos restantes com v2

```powershell
python -m conference_pipeline.v2_pipeline build-input `
  data/processed/2025-formal-manual-review.csv `
  data/processed/icml-2025-formal-sample.jsonl `
  data/processed/neurips-2025-formal-sample.jsonl `
  --output data/processed/2025-unreviewed-70-v1.jsonl

python -m conference_pipeline.v2_pipeline augment `
  data/processed/2025-unreviewed-70-v1.jsonl `
  --output data/processed/2025-unreviewed-70-v2.jsonl

python -m conference_pipeline.v2_pipeline sample `
  data/processed/2025-unreviewed-70-v2.jsonl `
  --output data/processed/2025-unreviewed-v2-review-sample-35.jsonl `
  --size 35 `
  --seed 20250727

python -m conference_pipeline.manual_review build `
  data/processed/2025-unreviewed-v2-review-sample-35.jsonl `
  --output data/processed/2025-unreviewed-v2-review-sample-35.csv

python -m conference_pipeline.manual_review serve `
  data/processed/2025-unreviewed-v2-review-sample-35.csv
```

A seleção é aleatória e reproduzível antes da priorização visual da fila. Ela
não contém nenhum dos 30 casos já revisados.

## Artefatos

| Diretório | Conteúdo | Git |
| --- | --- | --- |
| `data/raw/` | Enumeração original normalizada | Ignorado |
| `data/processed/` | Trabalhos reconciliados | Ignorado |
| `artifacts/` | Relatórios de qualidade locais | Ignorado |
| `data/sources/` | Registro, metodologia e decisões | Versionado |

Para tornar um resultado público auditável, gere um manifesto sanitizado com:

- comando executado;
- commit do código;
- data e hora UTC;
- URLs;
- contagens;
- SHA-256 dos arquivos;
- taxas de qualidade;
- tracks incluídas e excluídas.

O manifesto ainda será implementado antes da coleta completa.

## Problemas conhecidos

### AAAI possui contagens divergentes

Na primeira execução, o filtro anual do Crossref recuperou 1.599 trabalhos do
volume 39, enquanto o OpenAlex registrou 3.485. O coletor passou a percorrer o
prefixo DOI completo e filtrar o volume localmente, recuperando 3.486. O item
excedente ainda deve ser classificado.

### OpenAlex não resolve todas as afiliações

No smoke test inicial, país estava disponível em 10/10 trabalhos da AAAI, 2/10
da ACL e 0/10 do ICML. Não trate campo ausente como ausência de colaboração.
Consulte `data/sources/vertical-slice-results.md` antes de executar a amostra
formal.

Não execute a amostra formal sem `OPENALEX_API_KEY`. Execuções repetidas sem
chave produziram reconciliação instável após o consumo da cota disponível.

### Fallback PDF e ROR

O parser de PDF é deliberadamente conservador. Ele procura candidatos no
front matter até a introdução e pode deixar afiliações sem resolver. O ROR
recomenda aceitar automaticamente apenas o resultado marcado como `chosen`;
resultados apenas ordenados por score permanecem para revisão.

### OpenAlex fragmenta venues

Não use o primeiro ID retornado por busca textual. O universo é definido pela
fonte oficial e o OpenAlex serve para enriquecimento.

### 2026 é parcial

Não agregue 2026 enquanto venues previstos para o segundo semestre ainda não
tiverem ocorrido. Apresente resultados por venue e status.

## Retomada

Ao retomar o experimento:

1. leia `RESUME.md`;
2. leia `analysis-specs/H-BASE-001.md`;
3. leia `data/sources/conference-venue-audit.md`;
4. execute os testes offline;
5. confira os hashes e o último manifesto público ou relatório local;
6. continue somente a partir da primeira etapa não aprovada.
