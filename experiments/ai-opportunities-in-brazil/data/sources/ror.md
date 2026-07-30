# Research Organization Registry — ROR

## Papel

O ROR normaliza strings de afiliação extraídas de XML ou PDF em organizações,
países e tipos institucionais.

## Regra de aceitação

O pipeline usa `GET /v2/organizations?affiliation=...` e aceita
automaticamente somente o item marcado pelo serviço como `chosen:true`.

Não será selecionado:

- o primeiro resultado apenas por posição;
- um resultado apenas porque possui score alto;
- um resultado ambíguo sem `chosen:true`.

Casos não escolhidos permanecem como afiliação textual e entram na fila de
revisão.

## Fonte

- Documentação: https://ror.readme.io/docs/api-affiliation
- Endpoint: https://api.ror.org/v2/organizations
- Licença dos dados: CC0

## Limitações

- Uma linha de PDF pode conter mais de uma organização.
- A extração de texto pode quebrar uma afiliação em várias linhas.
- Nem toda empresa está registrada ou classificada da maneira esperada.
- Mesmo resultados `chosen` serão auditados na amostra formal.
