# OpenAlex

## Papel no experimento

O OpenAlex será usado para enriquecer trabalhos de conferências selecionadas de
IA e Responsible AI em 2025 e 2026 YTD com afiliações e tipos institucionais.
O universo de trabalhos será definido pelos proceedings oficiais de cada venue,
e não por uma busca geral no OpenAlex. Esse indicador é uma extensão própria do
estudo, não uma atualização oficial do Global AI Vibrancy Tool.

## Unidade observada

A unidade básica é um trabalho indexado em um venue previamente selecionado.
Países e organizações são atribuídos pelas afiliações dos autores.

Serão preservados o ID, título, data, venue, autores, afiliações, países, tipos
das instituições e data da extração.

## Regras

- Venues serão identificados pelo registro em
  `data/sources/conference-venues.csv`.
- O universo será enumerado nas fontes oficiais e reconciliado com OpenAlex por
  DOI e, como fallback auditável, título e autores.
- A contagem principal será completa: um trabalho colaborativo conta para cada
  país e instituição representados.
- A contagem fracionária será usada como análise de sensibilidade.
- A organização será classificada inicialmente pelo tipo institucional do
  OpenAlex.
- As organizações com maior contribuição serão auditadas manualmente.
- Trabalhos sem afiliação geográfica permanecerão no total de trabalhos únicos,
  mas não serão atribuídos a um país.
- A data de indexação acompanhará todo resultado de 2026 YTD.

## Cobertura inicial de venues

- IA geral: AAAI e IJCAI;
- aprendizado de máquina: NeurIPS, ICML e ICLR;
- visão computacional: CVPR, ICCV e ECCV;
- processamento de linguagem: ACL e EMNLP;
- mineração de dados: KDD.

Responsible AI preservará inicialmente as conferências usadas pelo HAI: AAAI,
AIES, FAccT, ICLR, ICML e NeurIPS. Além do venue, será necessária uma regra
temática validada em amostra.

## Licença e acesso

O OpenAlex oferece API e snapshot completo. Sua documentação declara o dataset
sob CC0. A API requer chave gratuita e possui limite diário; o snapshot público
é atualizado trimestralmente.

- Documentação: https://developers.openalex.org/
- API: https://api.openalex.org/
- Chave: https://openalex.org/settings/api

## Limitações

- A cobertura de trabalhos e afiliações não é uniforme entre venues e países.
- O tipo institucional pode conter classificações incorretas ou desatualizadas.
- Trabalhos recentes podem aparecer com atraso.
- País da afiliação não equivale à nacionalidade do pesquisador.
- Contagem completa favorece colaboração internacional; por isso, a análise
  fracionária é obrigatória como sensibilidade.

## Pendências

- [ ] Resolver e revisar os IDs dos venues.
- [ ] Medir cobertura de afiliações por venue e ano.
- [ ] Definir a data de corte de 2026.
- [ ] Validar a classificação das principais organizações.
- [ ] Definir e validar a regra temática de Responsible AI.
