# Metodologia

## Objetivo

Este experimento investiga a seguinte pergunta:

> Quais características do Brasil constituem vantagens competitivas no cenário
> global de inteligência artificial, quais são apenas vantagens potenciais e
> quais não encontram sustentação suficiente nos dados?

O estudo é exploratório e comparativo. Seu objetivo é organizar e testar
hipóteses com evidências observáveis, não demonstrar causalidade econômica.

## Definição operacional

Neste estudo, uma vantagem competitiva em IA é uma condição que permite ao
Brasil transformar recursos, capacidades ou características do seu mercado em
resultados relacionados à IA de maneira relativamente superior a países
comparáveis.

Para ser classificada como vantagem competitiva, a condição precisa:

1. ser relevante para o desenvolvimento ou a adoção de IA;
2. diferenciar o Brasil de um grupo de comparação previamente definido;
3. apresentar um mecanismo plausível de conversão em resultados;
4. possuir evidências observáveis dessa conversão;
5. permitir que parte relevante do valor produzido seja capturada no Brasil.

A durabilidade e a dificuldade de reprodução fortalecem a hipótese, mas **não**
serão presumidas apenas porque um recurso existe atualmente.

## Fundamentação

O desenho combina três perspectivas:

- O modelo de vantagem competitiva nacional de Michael Porter orienta a busca
  por condições dos fatores, condições da demanda, setores relacionados e de
  apoio, além da estratégia, estrutura e rivalidade das empresas.
- O Global Innovation Index da WIPO orienta a **separação** entre insumos de
  inovação e resultados de inovação.
- O Global AI Vibrancy Tool do Stanford HAI fornece um primeiro benchmark
  específico para comparar ecossistemas nacionais de IA.
- Os datasets do Epoch AI complementam o benchmark com evidências sobre modelos,
  GPU clusters e grandes data centers próximos da fronteira técnica.

Essas referências não oferecem, em conjunto, uma definição pronta de vantagem
competitiva nacional em IA. Essa definição operacional é uma adaptação
analítica deste estudo e deve ser apresentada dessa forma.

Stanford HAI e Epoch AI não serão somados em um novo índice. O primeiro será
usado para caracterizar ecossistemas nacionais; o segundo, para investigar
manifestações específicas da fronteira técnica. Resultados do Epoch serão
interpretados considerando que suas bases são não exaustivas e dependem de
informações publicamente disponíveis.

### Baseline mult fonte

O baseline não será sinônimo do ranking do Stanford HAI. Ele será um painel
mult fonte que descreve, pelo menos, a amplitude do ecossistema, sua presença na
fronteira, seus insumos, seus resultados e sua capacidade de captura de valor.

Stanford HAI, Epoch AI e outras fontes poderão fundamentar mais de uma hipótese.
Cada indicador continuará associado a uma hipótese principal, mas o campo
`baseline_inclusion` em `hypotheses/indicators.csv` registrará quando ele também
compõe o retrato inicial.

Fontes e indicadores semelhantes serão triangulados, não somados
automaticamente. Diferenças de definição, cobertura ou período serão
preservadas como limitações ou evidências divergentes. O estudo não criará um
novo score composto a partir dessas fontes.

### Extensão contemporânea

O baseline oficial do Global AI Vibrancy Tool termina em 2024. Para reduzir
essa defasagem, o estudo manterá um painel contemporâneo separado com dados de
2025 e sinais de 2026 disponíveis até a data de corte.

Essa extensão não será chamada de ranking ou índice do Stanford HAI. Somente
indicadores com definição equivalente, cobertura internacional e fonte
auditável serão mantidos. Indicadores proprietários, defasados ou cuja
classificação não possa ser reproduzida serão removidos da extensão, sem
imputação ou substituição silenciosa por proxies.

O estudo adicionará uma medida própria de presença em conferências selecionadas
de IA usando o OpenAlex. Ela será apresentada por país e por organização, com
contagem completa e uma análise de sensibilidade fracionária. Responsible AI
será preservada usando trabalhos aceitos ou publicados, e não submissões.

O ano de 2025 será tratado como período completo. Dados de 2026 serão
identificados como `YTD` ou snapshot. Fluxos parciais de 2026 não serão
comparados diretamente com totais anuais de 2025. As regras estão congeladas em
`analysis-specs/H-BASE-001.md`, e a auditoria por indicador está em
`data/sources/hai-contemporary-panel.csv`.

O catálogo de fontes está em `data/sources/registry.csv`, e as regras para
incorporar novas fontes estão em `data/sources/README.md`.

## Unidade de análise

A unidade principal é o ecossistema brasileiro de IA. Dependendo da hipótese,
a análise poderá observar setores específicos, mas a conclusão deve explicar
como o resultado setorial se relaciona com a competitividade do país.

Uma empresa brasileira bem-sucedida, isoladamente, não demonstra uma vantagem
nacional. Da mesma forma, a presença de empresas estrangeiras no Brasil não
implica captura doméstica de valor sem evidências adicionais.

## Desenvolvimento orientado por hipóteses

O trabalho seguirá uma abordagem de *hypothesis-driven development*. A análise
não começará pela procura de gráficos favoráveis ao Brasil. Cada investigação
começará com uma hipótese explícita, seu mecanismo esperado, as evidências que
poderiam sustentá-la e as evidências que poderiam enfraquecê-la.

As hipóteses vigentes estão em `hypotheses/registry.csv`. Seus indicadores são
mantidos em `hypotheses/indicators.csv`, e as observações coletadas são
registradas separadamente em `hypotheses/evidence.csv`. Essa separação preserva
a formulação original e reduz o risco de adaptá-la silenciosamente aos dados.

O andamento das fases está documentado em `plan.md`.

### Ciclo de uma hipótese

1. **Propor:** registrar a possível vantagem sem tratá-la como conclusão.
2. **Delimitar:** definir recurso, mecanismo, resultado e população analisada.
3. **Tornar mensurável:** escolher indicadores e comparadores antes de observar
   o resultado final.
4. **Testar:** coletar, processar e comparar os dados.
5. **Tentar refutar:** procurar explicações alternativas e evidências
   desfavoráveis.
6. **Classificar:** aplicar as regras de decisão e registrar a incerteza.
7. **Revisar:** atualizar ou abandonar a hipótese quando novas evidências
   justificarem a mudança.

### Contrato mínimo de uma hipótese

Cada hipótese deverá registrar:

| Campo | Pergunta |
| --- | --- |
| Identificador | Como a hipótese será referenciada? |
| Alegação | Qual vantagem está sendo proposta? |
| Pilar de Porter | Em qual condição competitiva ela se apoia? |
| Mecanismo | Como o recurso deveria produzir o resultado? |
| Insumos | Quais recursos ou capacidades são necessários? |
| Conversão | Como os insumos seriam transformados em aplicação de IA? |
| Resultados | Quais efeitos observáveis são esperados? |
| Captura de valor | Como o Brasil se beneficia do resultado? |
| Comparadores | Contra quais países ou grupos a hipótese será testada? |
| Indicadores | Quais medidas serão usadas? |
| Evidência favorável | O que aumentaria nossa confiança na hipótese? |
| Evidência contrária | O que reduziria nossa confiança na hipótese? |
| Limitações | O que os dados não permitem concluir? |
| Estado | Em qual etapa do ciclo a hipótese está? |

## Modelo de conversão

Cada hipótese será analisada como uma cadeia:

```text
recursos e condições
        ↓
capacidade de conversão
        ↓
resultados de IA
        ↓
captura de valor no Brasil
```

Um indicador de insumo não poderá ser apresentado sozinho como evidência
suficiente de vantagem competitiva.

| Nível | Exemplos de indicadores |
| --- | --- |
| Insumos | pesquisadores, energia, capital, dados e infraestrutura |
| Conversão | compute acessível, adoção empresarial e criação de empresas |
| Resultados | produtos, publicações, produtividade, patentes e exportações |
| Captura | empregos, renda, propriedade, impostos e reinvestimento doméstico |

## Comparações

Não existe um único grupo adequado para todas as perguntas. Cada hipótese deverá
justificar seus comparadores entre, pelo menos, estas perspectivas:

- fronteira tecnológica, como Estados Unidos e China;
- economias tecnologicamente avançadas;
- economias emergentes estruturalmente comparáveis;
- América Latina;
- países com população, renda ou estrutura produtiva semelhante.

Sempre que possível, a análise mostrará o valor do indicador, e não apenas a
posição em um ranking. Rankings agregados serão tratados como baseline, não como
veredito.

## Regras de classificação

Depois do teste, cada hipótese receberá uma classificação:

| Classificação | Regra |
| --- | --- |
| Vantagem revelada | Há diferenciação e resultados observáveis, com evidência de captura de valor |
| Vantagem emergente | Há resultados iniciais, mas a escala, a captura ou a persistência ainda são limitadas |
| Vantagem potencial | Há condições favoráveis, mas a conversão em resultados ainda não foi demonstrada |
| Hipótese inconclusiva | Os dados são insuficientes, incompatíveis ou contraditórios |
| Sem vantagem demonstrada | A comparação não mostra diferenciação relevante ou o mecanismo esperado não aparece |

Não será usado um limiar numérico universal. Quando uma hipótese exigir um
limiar, ele deverá ser justificado e registrado antes da comparação final.

## Evidência e incerteza

As fontes serão priorizadas nesta ordem:

1. dados oficiais e organismos multilaterais;
2. artigos acadêmicos e documentação metodológica;
3. bases setoriais com metodologia pública;
4. relatórios empresariais ou de consultorias;
5. estimativas e evidências anedóticas.

Uma fonte de menor prioridade poderá ser usada quando for a única disponível,
mas essa limitação deverá acompanhar a conclusão. Ausência de dados não será
interpretada automaticamente como ausência de capacidade.

Correlação, posição em ranking e casos individuais não serão descritos como
prova causal.

## Reprodutibilidade

Para cada conjunto de dados, o experimento deverá registrar:

- organização responsável;
- URL ou identificador persistente;
- versão ou data de acesso;
- período coberto;
- definição de cada indicador;
- transformações realizadas;
- valores ausentes;
- limitações conhecidas;
- licença ou condições de uso.

Dados brutos serão preservados separadamente dos dados processados. Tabelas e
figuras publicadas deverão ser geradas por código sempre que a licença da fonte
permitir.

## Uso de IA no estudo

Agentes de IA podem auxiliar na pesquisa, implementação, documentação e revisão.
Eles não são tratados como fonte factual. Toda fonte usada como evidência deverá
ser identificável, e a seleção das fontes, a verificação dos resultados e as
conclusões permanecem sob responsabilidade do autor.

## Limitações previstas

- Índices diferentes operacionalizam “IA” e “competitividade” de formas
  distintas.
- Alguns dados possuem defasagem temporal relevante.
- A cobertura internacional pode favorecer países com maior transparência.
- Pesos de índices agregados incorporam escolhas normativas.
- Resultados nacionais podem esconder diferenças regionais e setoriais.
- A captura de valor doméstica é mais difícil de medir do que os insumos.

Essas limitações deverão orientar a linguagem das conclusões e impedir que uma
análise exploratória seja apresentada como diagnóstico causal definitivo.
