# Epoch AI

## Papel no experimento

O Epoch AI será usado como fonte complementar de evidências sobre a fronteira
técnica de IA. Ele não substituirá o Stanford Global AI Vibrancy Tool e não será
combinado a ele em um score único.

| Fonte | Papel |
| --- | --- |
| Stanford HAI | Comparação ampla de ecossistemas nacionais de IA |
| Epoch AI | Modelos relevantes, compute de fronteira, GPU clusters e grandes data centers |

## Datasets candidatos

### AI Models

- URL: https://epoch.ai/data/ai-models
- URL direta: https://epoch.ai/data/all_ai_models.csv
- Formato: CSV e ZIP.
- Cobertura declarada: modelos de 1950 até o presente.
- Uso proposto: identificar modelos notáveis, de larga escala ou de fronteira
  associados a organizações e países.
- Auditoria: 3.572 registros; país da organização preenchido em 3.476,
  treinamento computacional em 1.398, confiança em 3.375 e data center de
  treinamento em apenas 67.

### AI Data Centers

- URL: https://epoch.ai/data/ai-data-centers
- URL direta: https://epoch.ai/data/data_centers/data_centers.zip
- Formato: ZIP.
- Uso proposto: estimar a presença e a escala de grandes projetos de
  infraestrutura especializada para IA.
- Auditoria: 74 sites em 8 países. Os 74 têm país, H100 equivalents e potência
  atual; 66 têm proprietário. O ZIP separa cadastro, cronologia, chips e
  elementos de refrigeração.

### GPU Clusters

- URL: https://epoch.ai/data
- URL direta: https://epoch.ai/data/gpu_clusters.csv
- Formato: CSV.
- Uso proposto: comparar clusters e supercomputadores relevantes para
  treinamento e inferência.
- Auditoria: 482 registros, 36 países observados, 460 valores de H100
  equivalents, 466 países preenchidos e 475 datas de início de operação.

## Licença

O Epoch AI declara seus datasets livres para uso, distribuição e reprodução
com atribuição, sob licença Creative Commons Attribution. A versão, a data de
acesso e a citação específica de cada dataset deverão acompanhar os dados
baixados.

## Limitações

- As bases são não exaustivas.
- A cobertura depende da disponibilidade pública de informações.
- Projetos privados ou pouco documentados podem não aparecer.
- País da sede da organização, localização do hardware e país que captura o
  valor são conceitos diferentes.
- Capacidade anunciada, em construção e operacional não deve ser somada sem
  distinção.
- Um supercomputador de HPC não deve ser tratado automaticamente como compute
  adequado para IA sem conhecer seus aceleradores e condições de acesso.
- Ausência de uma entidade brasileira no dataset não será tratada, sozinha,
  como evidência de ausência de capacidade.
- A base de data centers cobre os maiores sites conhecidos e não constitui um
  censo da infraestrutura de cada país.
- Modelos notáveis e de fronteira são categorias editoriais: o Epoch define
  fronteira como os dez modelos com maior treinamento computacional no momento
  do lançamento.
- Estimativas de treinamento têm graus de confiança distintos. Valores
  `Confident`, `Likely` e `Speculative` não devem ser agregados como observações
  igualmente precisas.
- O campo de data center de treinamento está preenchido em menos de 2% dos
  modelos auditados e não sustenta, sozinho, atribuição geográfica.

## Decisão de auditoria

**Aceitação com condições para os três datasets.**

Eles são apropriados para medir presença observada na fronteira, desde que cada
análise preserve estado operacional, incerteza, localização e definição da
unidade. Não são apropriados para provar ausência de capacidade nacional.

## Pendências antes da análise

- [x] Registrar os URLs diretos de download.
- [ ] Registrar data, versão e checksum de cada arquivo.
- [x] Inspecionar o esquema dos arquivos.
- [x] Verificar campos de país, organização e localização.
- [x] Medir a cobertura global dos principais campos.
- [ ] Definir filtros reproduzíveis para estado operacional e certeza.
- [ ] Definir como estimativas e observações serão apresentadas separadamente.
- [ ] Congelar a versão antes de calcular resultados por país.
