# Resultado da prova vertical

Data de execução: 2026-07-27.

## Enumeração

| Venue | Escopo | Trabalhos |
| --- | --- | ---: |
| AAAI 2025 | Volume 39 após paginação completa do prefixo DOI | 3.486 |
| ICML 2025 | PMLR volume 267 | 3.330 |
| ACL 2025 | `2025.acl-long`, sem front matter | 1.602 |
| NeurIPS 2025 | Proceedings oficiais, tracks preservados | 5.823 |

O OpenAlex contém 3.485 trabalhos associados ao volume 39 da AAAI. A diferença
de um registro permanece para classificação, mas a paginação corrigida resolveu
a perda substancial da estratégia inicial.

## Smoke test de reconciliação

Amostra não aleatória: os primeiros dez trabalhos enumerados de cada venue.
Ela valida o caminho técnico, não estima a cobertura definitiva.

| Venue | Reconciliados | Com país | Decisão |
| --- | ---: | ---: | --- |
| AAAI | 10/10 | 10/10 | Passou no smoke test inicial |
| ACL long, após PDF+ROR | 10/10 | 10/10 | Passou no smoke test |
| ICML, após PDF+ROR | Instável sem chave | 2/10 | Falhou cobertura geográfica |
| NeurIPS, após PDF+ROR | 6/10 | 5/10 | Falhou os gates, mas validou o fallback |

## Interpretação

DOI ou título são suficientes para encontrar muitos trabalhos, mas o OpenAlex
não fornece afiliações resolvidas de maneira uniforme entre venues. Nos
primeiros trabalhos de ACL e ICML, as afiliações textuais também estavam
frequentemente ausentes.

Portanto:

- o pipeline não pode inferir que um trabalho sem país no OpenAlex não possui
  afiliação;
- país e tipo de organização não serão calculados usando apenas o OpenAlex;
- será necessário testar extração das páginas oficiais, metadados estruturados
  ou PDFs, seguida de resolução institucional;
- o gate formal de 50 trabalhos por venue permanece pendente.

A cascata PDF+ROR elevou ACL de 20% para 100% de cobertura na amostra, NeurIPS
para 50% e ICML para 20%. Como a amostra tem apenas dez trabalhos e não é
aleatória, esses valores não devem ser generalizados.

Execuções sem `OPENALEX_API_KEY` também produziram taxas de reconciliação
variáveis quando a cota pública foi consumida. A amostra formal deverá usar uma
chave e registrar a resposta de erro separadamente de um verdadeiro
`não encontrado`.

## Próxima decisão técnica

Implementar uma cascata de afiliações:

1. instituições resolvidas pelo OpenAlex;
2. afiliações textuais da fonte oficial;
3. metadados estruturados ou texto inicial do PDF;
4. resolução da organização contra ROR/OpenAlex Institutions;
5. revisão manual dos casos ainda não resolvidos na amostra.

Essa cascata deve ser validada na amostra antes de expandir para os outros
venues.

## Amostra formal reproduzível

Em 2026-07-27, a cascata foi executada sobre uma amostra aleatória determinística
de 50 trabalhos por venue, com seed `20250727`.

| Venue | Reconciliados | Com país | Gate automático |
| --- | ---: | ---: | --- |
| AAAI | 50/50 | 48/50 | Passou |
| ACL | 50/50 | 45/50 | Passou |
| ICML | 23/50 | 10/50 | Falhou |
| NeurIPS | 26/50 | 47/50 | Falhou |

AAAI e ACL satisfizeram os gates automáticos. ICML ficou abaixo dos limites de
reconciliação e cobertura geográfica. NeurIPS obteve país em 94% da amostra por
meio do fallback, mas reconciliou apenas 52% dos trabalhos com OpenAlex. Esses
resultados ainda dependem de revisão manual e não autorizam a expansão para o
universo completo. O checkpoint e os hashes dos artefatos locais estão em
`RESUME.md`.
