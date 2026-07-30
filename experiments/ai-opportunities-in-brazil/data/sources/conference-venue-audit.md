# Auditoria inicial dos venues

Data: 2026-07-27.

## Resultado

A busca por nomes no OpenAlex não produz, sozinha, um registro confiável de
conferências. O OpenAlex mistura:

- séries completas;
- proceedings de uma edição específica;
- volumes individuais;
- journals com nomes semelhantes;
- workshops com a conferência principal.

Dos 14 venues candidatos, somente AAAI e AIES apresentaram um ID de série com
trabalhos de 2025–2026 na consulta inicial. Os IDs candidatos dos outros 12
retornaram cobertura zero nesse período, mesmo quando as páginas oficiais
confirmam que os proceedings existem.

Isso não significa ausência no OpenAlex. Significa que o container não está
representado por aquele ID de série.

## Decisão de arquitetura

O universo de trabalhos será enumerado pela fonte oficial de cada conferência:

- AAAI/AIES OJS;
- proceedings oficiais de IJCAI e NeurIPS;
- PMLR para ICML;
- OpenReview para ICLR;
- CVF Open Access para CVPR e ICCV;
- ECVA/Springer para ECCV;
- ACL Anthology para ACL e EMNLP;
- programa/proceedings oficiais de KDD e FAccT.

Depois da enumeração oficial, DOI, título e autores serão usados para reconciliar
os trabalhos com o OpenAlex. O OpenAlex fornecerá afiliações, países e tipos de
instituição quando houver correspondência.

Trabalhos não reconciliados continuarão no total oficial do venue, mas serão
marcados como sem atribuição geográfica ou organizacional até uma fonte
complementar resolver a afiliação.

## Regras de tracks

- O painel principal usará trabalhos de pesquisa do programa técnico principal.
- Workshops, tutoriais, editoriais, erratas e material não arquivístico serão
  excluídos.
- Industry tracks serão preservadas separadamente, pois são importantes para a
  análise de empresas, mas não serão misturadas silenciosamente ao main track.
- Findings, demos e student research workshops serão preservados como tabelas
  auxiliares quando existirem, sem entrar na contagem principal.
- Conferências bienais entram apenas nos anos em que ocorrerem. Ausência no ano
  alternado não será registrada como zero de produção.
- O total parcial de 2026 não será agregado enquanto conferências programadas
  para o segundo semestre ainda não tiverem ocorrido.

## Validações realizadas

- AAAI: `S4210191458`, com 8.463 trabalhos datados de 2025–2026 na consulta
  inicial.
- AIES: `S5407048695`, com 299 trabalhos datados de 2025–2026.
- NeurIPS 2025: a fonte oficial lista 5.823 trabalhos.
- ICML 2025: proceedings oficiais publicados como PMLR volume 267.
- ACL e EMNLP: a ACL Anthology oferece eventos e volumes identificáveis
  por IDs próprios.

## Prova vertical

Os primeiros enumeradores foram implementados e executados em 2026-07-27:

| Venue | Linhas enumeradas | Estado |
| --- | ---: | --- |
| ICML 2025, PMLR 267 | 3.330 | Enumeração executada; total oficial ainda deve ser fixado |
| ACL 2025, long papers | 1.602 | Consistente com as 1.603 entradas da página incluindo front matter |
| AAAI 2025, volume 39 | 3.486 após correção | Diferença de um registro em relação ao OpenAlex |

Para AAAI, o OpenAlex contém 3.485 trabalhos no volume 39. A primeira consulta
Crossref filtrada pelo ano retornou apenas 1.599. O coletor foi alterado para
percorrer o prefixo `10.1609` completo e aplicar o filtro de volume localmente,
chegando a 3.486. O registro excedente ainda deve ser classificado.

O smoke test de dez trabalhos encontrou cobertura de país de 100% para AAAI,
20% para ACL e 0% para ICML. Os resultados completos e sua interpretação estão
em `data/sources/vertical-slice-results.md`.

Após adicionar a cascata PDF+ROR, ACL chegou a 100%, NeurIPS a 50% e ICML a
20% na amostra de dez. Esses resultados são apenas smoke tests. A amostra
formal, com chave OpenAlex e 50 trabalhos por venue, permanece pendente.

## Exclusões de escopo

- NAACL foi removida por decisão de desenho porque é uma conferência regional
  da América do Norte. A exclusão não será interpretada como ausência de
  produção dos países participantes.

As contagens iniciais servem para validar o mecanismo de descoberta, não são
resultados do experimento. Ainda precisam de deduplicação e aplicação das regras
de tracks.

## Próximas verificações

- [ ] Confirmar cada URL marcada como `pending`.
- [ ] Implementar enumeradores por família de fonte oficial.
- [ ] Reconciliar uma amostra com OpenAlex por DOI e título.
- [ ] Medir a taxa de reconciliação e a cobertura de afiliações por venue.
- [ ] Revisar manualmente uma amostra de empresas e universidades.
- [ ] Congelar a data de corte para 2026 YTD.
