# Auditoria da fonte: Stanford HAI

## Identificação

- Organização: Stanford Institute for Human-Centered Artificial Intelligence.
- Produto: Global AI Vibrancy Tool.
- Edição auditada: arquivos datados de 24 de setembro de 2025.
- Período observado nos dados: 2017–2024.
- Data de acesso: 2026-07-27.
- Página pública: https://hai.stanford.edu/ai-index/global-vibrancy-tool

## Arquivos públicos

| Arquivo | URL direta | Linhas | Papel |
| --- | --- | ---: | --- |
| Dados completos | https://d3i91vx6n7fixv.cloudfront.net/data/full_data_09.24.25.csv | 528 | Valores brutos, per capita e normalizados |
| Codebook | https://d3i91vx6n7fixv.cloudfront.net/data/codebook_09.24.25.csv | 74 | Definições, fontes, pilares e elegibilidade |
| Lista de variáveis | https://d3i91vx6n7fixv.cloudfront.net/data/variable_list_09.24.25.csv | 225 | Metadados usados pela interface |

Os checksums devem ser calculados no momento em que uma cópia imutável for
baixada para a análise. Os URLs acima apontam para arquivos versionados pelo
nome, mas isso não garante que o conteúdo remoto seja imutável.

## Cobertura e esquema

- 528 observações país-ano.
- 66 países distintos.
- Oito anos: 2017 a 2024.
- Sete pilares no `codebook`: R&D, Responsible AI, Economy, Talent, Policy and
  Governance, Public Opinion e Infrastructure.
- O arquivo contém valores brutos, variantes per capita e valores normalizados
  para exibição e cálculo.
- O painel atual informa que 36 países entram na comparação principal e que
  essa seleção exige pelo menos 70% de cobertura média nos três anos mais
  recentes.

## Divergência entre edições

O PDF metodológico publicado em 2024 descreve 42 indicadores e oito pilares,
incluindo Education e Diversity. Já a interface e os arquivos datados de 2025
usam sete pilares, e a página pública resume o ranking em 23 indicadores.

Consequências:

1. o PDF de 2024 não pode ser tratado como especificação completa da edição
   atual;
2. números de indicadores dependem do universo contado, pois o `codebook`
   também contém medidas auxiliares e variantes;
3. qualquer reprodução deve fixar a edição dos dados, o conjunto de indicadores
   e a regra de inclusão usada.

## Transformações que exigem controle

Segundo a metodologia publicada para a edição anterior:

- indicadores são normalizados por min-max entre países para a escala 0–100;
- valores ausentes podem ser imputados pela mediana anual entre países;
- quando um indicador não existe para nenhum país em um ano, ele é removido e
  seu peso é redistribuído;
- pilares e o agregado são médias ponderadas;
- os pesos editoriais foram definidos pela equipe do AI Index e podem ser
  alterados pelo usuário.

Essas regras devem ser confirmadas nas notas da edição 2025 antes de reproduzir
o agregado. Mesmo quando confirmadas, o experimento deve mostrar valores brutos
e pilares, não apenas o score.

## Limitações relevantes

- A composição do índice mudou entre edições.
- Normalização min-max torna o resultado relativo ao conjunto de países.
- Imputação e redistribuição de pesos afetam comparabilidade temporal.
- Fontes baseadas em plataformas, como LinkedIn, refletem também a cobertura da
  própria plataforma.
- Algumas fontes são mais completas em inglês.
- Um agregado alto pode esconder gargalos e um agregado baixo pode esconder
  especializações.

## Decisão de auditoria

**Aceitação condicional.**

Os dados e metadados atuais são publicamente acessíveis e têm cobertura adequada
para construir o baseline amplo. A reprodução do score agregado permanece
bloqueada até que as regras da edição 2025 sejam vinculadas a uma documentação
compatível com os arquivos de 24 de setembro de 2025. Análises de indicadores
brutos e cobertura podem começar antes disso.

## Critérios para encerrar a pendência

- [ ] Obter e arquivar as release notes da edição 2025.
- [ ] Confirmar normalização, imputação, pesos e regra de elegibilidade atuais.
- [ ] Explicar por que a página informa 23 indicadores enquanto o `codebook`
  contém um universo maior.
- [ ] Baixar uma cópia imutável dos três CSVs e registrar SHA-256.
- [ ] Medir ausência por país, ano e indicador antes de escolher comparações.

