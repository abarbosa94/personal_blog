# Pacote público de evidências

Este diretório contém o snapshot compacto que sustenta os números do post
**O Brasil é bom em IA?**. O fechamento analítico foi
congelado em 30 de julho de 2026.

## Escopo

O pacote inclui:

- a matriz final dos sete indicadores, sem nota agregada;
- o painel congelado de 16 países;
- os resultados, a cobertura e a sensibilidade das sete conferências;
- as tabelas derivadas do Epoch AI, TOP500, AI Index/Quid e World Bank;
- as classificações finais das hipóteses;
- o registro de fontes e os checksums de todos os arquivos de evidência
  publicados aqui.

Responsible AI não integra o baseline. O classificador falhou no teste cego e
nenhum resultado por país dessa extensão foi incorporado ao pacote.

## Como auditar

`source-manifest.json` relaciona cada arquivo de evidência ao artefato local
que o originou e registra seu SHA-256. `SHA256SUMS` permite verificar
rapidamente a integridade desses arquivos. Os contratos de decisão ficam em
[`../analysis-specs`](../analysis-specs), e os scripts e testes em
[`../scripts`](../scripts) e [`../tests`](../tests).

Os arquivos de conferências registram um universo oficial de 19.908 trabalhos,
dos quais 19.907 foram enumerados. Todas as sete venues passaram o gate de 90%
de cobertura de país; a cobertura variou de 91,38% a 97,38%.

## Limite da reprodução pública

Os diretórios locais `data/raw`, `data/processed` e `artifacts` permanecem fora
do Git por volume, caches de rede e condições de redistribuição. Este pacote
permite auditar as tabelas finais, os metadados, as limitações e a identidade
dos resultados, mas não é um espelho dos dados brutos.

Para reconstruí-lo depois de regenerar os artefatos locais:

```powershell
python scripts/build_baseline_evidence_matrix.py
python scripts/build_hypothesis_assessments.py
python scripts/build_publication_bundle.py
pytest -q
```

O script do pacote copia apenas a lista explícita de arquivos compactos e
confirma que o SHA-256 da cópia é idêntico ao artefato de origem.
