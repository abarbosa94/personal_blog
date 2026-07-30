# Protocolo de revisão manual

Este protocolo aplica ao experimento os princípios de revisão humana usados
anteriormente no projeto e descritos no capítulo 10 de *AI Evals*. A unidade de
revisão é um trabalho da amostra formal, não uma afiliação isolada.

## Princípios

- A amostra formal aleatória permanece intacta. A fila muda apenas a ordem de
  apresentação para colocar falhas e incertezas primeiro.
- O revisor vê um trabalho por vez, com metadados oficiais, método automático,
  correspondência OpenAlex e afiliações lado a lado.
- A decisão inicial é simples: `pass`, `fail` ou `defer`.
- Falhas recebem códigos estruturados e uma nota que registre a evidência.
- Casos incertos devem ser adiados, não adivinhados.
- Decisões anteriores podem ser reabertas e corrigidas para controlar *criteria
  drift*.
- Novos padrões começam como `other` mais uma nota. Se recorrerem, a rubrica e
  as decisões anteriores devem ser atualizadas.
- Casos que representam um padrão corrigível podem ser marcados para o conjunto
  de regressão.

## Evidência mínima

Antes de concluir um item, compare:

1. título, autores, venue e ano na fonte oficial;
2. título e autoria do registro OpenAlex, quando houver;
3. instituições e países na primeira página do PDF;
4. identidade e país das organizações ROR, quando o fallback tiver sido usado.

`pass` significa que a correspondência bibliográfica e as afiliações registradas
são adequadas para agregação por país. Ausência de OpenAlex não é, por si só,
falha se o fallback PDF+ROR cobrir corretamente as afiliações.

## Códigos de falha

| Código | Regra operacional |
| --- | --- |
| `api_error` | A fonte não pôde ser consultada; não há evidência suficiente para distinguir erro de ausência. |
| `not_found` | A consulta funcionou, mas nenhum trabalho ou organização correspondente existe na fonte consultada. |
| `ambiguous_match` | Há dois ou mais candidatos plausíveis e a evidência disponível não decide entre eles. |
| `wrong_work` | O OpenAlex selecionou outro trabalho. |
| `missing_affiliation` | O trabalho está correto, mas nenhuma afiliação utilizável foi extraída. |
| `incomplete_affiliations` | Parte das instituições ou países do PDF está ausente. |
| `wrong_affiliation` | Uma afiliação foi atribuída ao trabalho incorretamente. |
| `ror_mismatch` | O fallback associou o texto de afiliação à organização ROR errada. |
| `other` | Padrão ainda não previsto; exige nota. |

`api_error` e `not_found` não devem ser inferidos apenas de um `openalex_id`
vazio. O reconciliador atual captura exceções de rede, portanto esses dois
códigos exigem evidência observada pelo revisor ou uma nova execução
instrumentada.

## Confiança e adjudicação

- `high`: a fonte oficial e a fonte reconciliada tornam a decisão direta;
- `medium`: a decisão é sustentada, mas depende de normalização ou inferência;
- `low`: a evidência é incompleta; prefira `defer`.

Itens adiados não entram como acerto ou erro. Depois da primeira passagem,
revise os adiados em conjunto e documente qualquer refinamento da rubrica antes
de adjudicá-los.

## Encadeamento com engenharia

Depois da revisão:

1. agrupe falhas por código, venue e método;
2. identifique padrões recorrentes;
3. transforme cada padrão corrigível marcado para regressão em fixture e
   cenário BDD;
4. corrija o reconciliador;
5. reexecute apenas os registros afetados;
6. preserve a decisão original e registre a nova versão do pipeline.

## Estratégia de extração

Países escritos explicitamente no bloco de afiliações são extraídos por regras
determinísticas e normalizados para ISO alpha-2 antes de depender do ROR. O
trecho termina antes de `Correspondence to`, `Abstract` ou `Introduction`, o que
evita interpretar o local da conferência no rodapé como afiliação.

NER estatístico pode ser avaliado futuramente para recuperar limites de nomes de
organizações. Ele não substitui a resolução ROR nem a regra explícita de países:
os PDFs contêm quebras de linha e nomes longos fora do domínio típico dos
modelos gerais, e uma entidade `ORG` ou `GPE` não estabelece por si só uma
afiliação autor-organização-país.
