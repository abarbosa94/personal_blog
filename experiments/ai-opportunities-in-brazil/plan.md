# Plano do experimento

Este plano segue a ordem lógica das hipóteses registradas em
`hypotheses/registry.csv`. A passagem entre fases depende dos critérios
metodológicos, não apenas da disponibilidade de um novo dataset.

## Fase 1 — Registro e governança

- [x] Definir vantagem competitiva em IA.
- [x] Documentar o desenvolvimento orientado por hipóteses.
- [x] Registrar a hipótese central e as hipóteses explicativas.
- [x] Registrar contra-hipóteses para reduzir viés de confirmação.
- [x] Criar os registros de indicadores e evidências.

Critério de saída: todas as hipóteses principais estão no estado `scoped`.

## Fase 2 — Baseline mult fonte (`H-BASE-001`)

- [x] Confirmar quais dados do Stanford HAI podem ser obtidos e reproduzidos.
- [x] Auditar os datasets AI Models, AI Data Centers e GPU Clusters do Epoch AI.
- [x] Manter um catálogo de fontes candidatas em `data/sources/registry.csv`.
- [ ] Identificar lacunas do baseline e buscar fontes adicionais com metodologia
  pública, sem escolhê-las pela direção do resultado brasileiro.
- [ ] Registrar versões, licenças, dicionários, cobertura e campos ausentes.
  Cobertura e esquema iniciais foram auditados; checksums e a licença de
  redistribuição do Stanford ainda estão pendentes.
- [x] Definir e congelar os grupos de comparação.
- [ ] Aprovar um painel de indicadores absolutos, normalizados e temporais.
- [ ] Definir quais indicadores entram no baseline e qual hipótese principal
  cada um fundamenta.
- [ ] Definir a análise de sensibilidade aos pesos.
- [ ] Definir regras de triangulação para indicadores semelhantes.
- [ ] Congelar os critérios de suporte e contestação.
- [x] Auditar quais indicadores do HAI podem ser estendidos para 2025 e 2026.
- [x] Congelar a separação entre o HAI histórico, 2025 completo e 2026 YTD em
  `analysis-specs/H-BASE-001.md`.
- [x] Extrair os sete indicadores aprovados para o baseline inicial de 2025.
- [x] Adiar Responsible AI para uma extensão metodológica: a unidade foi
  definida e os dados preservados, mas o classificador falhou o gate cego e o
  indicador não entra nas comparações ou conclusões iniciais.
- [x] Auditar os IDs OpenAlex candidatos e detectar fragmentação por edição.
- [x] Registrar as fontes oficiais e regras iniciais dos venues.
- [x] Implementar a enumeração oficial e reconciliação com OpenAlex.
- [x] Implementar a prova vertical para AAAI, ICML e ACL.
- [x] Persistir instruções reproduzíveis em `RUNBOOK.md`.
- [x] Corrigir a paginação e reduzir a divergência do AAAI a um registro.
- [ ] Classificar o registro excedente do AAAI.
- [x] Executar e auditar automaticamente amostras de reconciliação com OpenAlex.
- [x] Congelar o contrato v3 do pool de sete venues/tracks de 2025 em
  `analysis-specs/conference-pooled-2025-contract.md`.
- [x] Implementar e validar a enumeração de EMNLP 2025 main, ICLR 2025 aceitos
  e KDD 2025 ADS; os censos de país continuam em execução/recuperação.
- [x] Elevar AAAI, ACL, ICML, NeurIPS, EMNLP, ICLR e KDD ADS ao gate comum
  v3 de pelo menos 90% de cobertura.
- [x] Gerar o pool primário com peso igual por venue, o pool ponderado por
  trabalho e a sensibilidade leave-one-venue-out.
- [x] Congelar V9 como método qualificado pelos gates cegos e de regressão já
  executados; não repetir gate cego para as novas venues.
- [x] Executar smoke tests de dez trabalhos por venue.
- [x] Executar os fallbacks de afiliação na amostra formal de ACL e ICML.
- [x] Adicionar NeurIPS à prova vertical preservando seus tracks.
- [x] Implementar fallbacks GROBID, PDF e resolução ROR.
- [x] Executar a amostra formal com `OPENALEX_API_KEY`.
- [x] Auditar manualmente as falhas de ICML e NeurIPS e validar os fallbacks.
- [x] Revisar manualmente as correspondências ROR das amostras de validação.
- [x] Validar a regra temática de Responsible AI na amostra estratificada de
  86 trabalhos; o screen apenas por título falhou por encontrar 6 positivos
  entre 30 negativos amostrados.
- [x] Encerrar a tentativa inicial de classificador de Responsible AI e
  preservar seus artefatos e resultados negativos sem agregar países.
- [ ] Extrair os seis indicadores aprovados para 2026 YTD.
- [ ] Mover `H-BASE-001` para `measurable`.
- [ ] Coletar os dados e reproduzir o painel do baseline.
- [ ] Apresentar separadamente amplitude, presença na fronteira, insumos,
  resultados e captura de valor.
- [ ] Registrar evidências favoráveis, contrárias e limitações.
- [x] Congelar o Wikidata como fallback terciário para geografia institucional,
  preservando separadamente sede, origem, formação e país direto em
  `analysis-specs/wikidata-institution-fallback.md`.

Critério de saída: `H-BASE-001` está `tested`, e nenhuma conclusão depende
exclusivamente de um score agregado ou de uma única fonte.

## Fase 3 — Condições dos fatores

### P&D (`H-RD-001`)

- [ ] Separar volume, composição, estabilidade e eficiência.
- [ ] Definir indicadores gerais e específicos para IA.
- [ ] Testar investimento absoluto, relativo e por pesquisador.
- [ ] Comparar investimento empresarial e público.

### Ciência (`H-SCI-001`)

- [ ] Definir uma taxonomia reproduzível de publicações de IA.
- [ ] Separar volume, impacto, liderança e especialização.
- [ ] Normalizar por população, pesquisadores e investimento.
- [ ] Testar áreas em que o Brasil possa superar o resultado agregado.
- [ ] Não usar a base de modelos do Epoch como substituta de uma base
  bibliométrica.

### Infraestrutura (`H-INFRA-001`)

- [ ] Separar capacidade, adequação, acesso e dependência externa.
- [ ] Inventariar compute público, acadêmico, empresarial e cloud.
- [ ] Evitar tratar petaflops de HPC como proxy automática de compute para IA.
- [ ] Medir acesso e uso, não apenas capacidade nominal.
- [ ] Usar AI Data Centers e GPU Clusters do Epoch como evidência complementar.
- [ ] Separar infraestrutura operacional, em construção e anunciada.

Critério de saída: as três hipóteses estão `tested`, com evidência dos dois lados.

## Fase 4 — Vantagens candidatas

### Demanda (`H-DEMAND-001`)

- [ ] Medir adoção, escala e sofisticação da demanda.
- [ ] Verificar se o desenvolvimento e a captura de valor são locais.
- [ ] Comparar setores públicos e privados.

### Especialização setorial (`H-SECTOR-001`)

- [ ] Priorizar setores usando critérios definidos antes dos resultados.
- [ ] Criar sub-hipóteses apenas para setores com mecanismo mensurável.
- [x] Registrar, sem classificar, os casos financeiros Nubank/Itaú e o caso
  jurídico Jusbrasil em `analysis-specs/sector-case-hypotheses.md`.
- [ ] Comparar aplicações locais, importadas e exportadas.
- [ ] Avaliar agronegócio, finanças, energia, saúde, setor público,
  biodiversidade e clima sem presumir vantagem.

Critério de saída: vantagens candidatas estão classificadas ou explicitamente
inconclusivas.

## Fase 5 — Conversão e síntese

- [x] Mapear transições entre insumos, pesquisa, infraestrutura, aplicações,
  empresas, produtividade e captura de valor.
- [x] Usar AI Models do Epoch para testar presença em modelos notáveis ou de
  fronteira, sem interpretar ausência de registro como ausência de capacidade.
- [x] Registrar a Maritaca como hipótese de conversão e especialização
  linguística, sem tratá-la como evidência conclusiva.
- [x] Testar `H-CONV-001`; a evidência permaneceu inconclusiva para identificar
  a conexão entre as etapas como o principal gargalo.
- [x] Sintetizar as hipóteses filhas sem criar um score arbitrário.
- [x] Classificar `H-MAIN-001`.
- [x] Registrar explicações alternativas e incertezas remanescentes.

## Fase 6 — Publicação reproduzível

- [x] Continuar, sem reiniciar, o notebook existente
  `posts/2026-07-27-VantagensCompetitivas-Brasil-IA.ipynb`, preservando a voz,
  o ritmo e a estrutura argumentativa já estabelecidos pelo autor.
- [x] Gerar tabelas e figuras por código.
- [x] Manter o código reproduzível nos scripts ligados pelo notebook; o post
  não duplica células longas de coleta.
- [x] Citar datasets, versões, licenças e datas de acesso.
- [x] Mostrar resultados favoráveis, contrários e mistos.
- [x] Revisar se a conclusão responde à pergunta inicial.
- [x] Executar a revisão técnica e editorial do post.
- [x] Publicar um pacote compacto e verificável dos resultados finais.
- [x] Incorporar as correções priorizadas pela revisão e retirar
  `draft: true`.

## Fechamento do plano staged 1–9 — 2026-07-30

1. [x] Matriz dos sete indicadores congelada.
2. [x] Contrato de decisão e triangulação congelado.
3. [x] Tabelas e cinco figuras reproduzíveis geradas e inspecionadas.
4. [x] Baseline incorporado ao notebook na voz existente.
5. [x] P&D, ciência e infraestrutura testados; resultados misto,
   inconclusivo e misto, respectivamente.
6. [x] Demanda e casos de finanças, jurídico e português avaliados como
   candidatos, sem promovê-los a vantagens demonstradas.
7. [x] Cadeia de conversão analisada; a evidência não identificou um único
   gargalo principal.
8. [x] Hipótese central classificada como inconclusiva e `H-BASE-001`
   contestada, sem score agregado.
9. [x] Notebook analiticamente concluído, fontes e limitações revisadas,
   124 testes aprovados e renderização Quarto validada. As correções editoriais
   foram incorporadas e o post foi liberado para publicação em 31 de julho de
   2026.

Os itens amplos ainda não executados nas fases 2–5 são extensões futuras.
Resultados inconclusivos representam testes concluídos com evidência
insuficiente, e não uma autorização para preencher lacunas por inferência.
