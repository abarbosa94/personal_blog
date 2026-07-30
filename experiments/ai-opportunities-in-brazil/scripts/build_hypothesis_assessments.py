"""Build the post-baseline hypothesis assessment tables without an aggregate score."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "analysis"
OUT_CSV = ANALYSIS / "hypothesis-assessments.csv"
OUT_MD = ANALYSIS / "hypothesis-assessments.md"


def read(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    wdi = read("world-bank-factor-context.csv")
    rd = {
        row["country_code"]: row
        for row in wdi
        if row["indicator_code"] == "GB.XPD.RSDV.GD.ZS" and row["value"]
    }
    rd_rank = 1 + sum(float(row["value"]) > float(rd["BR"]["value"]) for row in rd.values())
    assessments = [
        {
            "hypothesis_id": "H-BASE-001",
            "classification": "challenged",
            "claim": "O baseline brasileiro é superior ao esperado entre pares.",
            "evidence": "Quatro famílias não mostram superioridade robusta: fronteira e conversão desafiam; ciência desafia; infraestrutura é mista.",
            "limitation": "Painel contemporâneo selecionado, sem score agregado e sem inferência causal.",
        },
        {
            "hypothesis_id": "H-RD-001",
            "classification": "mixed",
            "claim": "P&D limita os resultados competitivos em IA.",
            "evidence": f"Dispêndio em P&D de {float(rd['BR']['value']):.2f}% do PIB ({rd['BR']['year']}), posição {rd_rank}/16: acima dos pares latino-americanos observados, abaixo da fronteira e de Turquia.",
            "limitation": "P&D geral, anos de referência desiguais e sem composição pública/empresarial específica para IA.",
        },
        {
            "hypothesis_id": "H-SCI-001",
            "classification": "inconclusive",
            "claim": "A ciência brasileira em IA fica abaixo dos pares após normalização.",
            "evidence": "Presença de 0,0640% e posição 10/16 nas sete conferências selecionadas desafiam liderança ampla.",
            "limitation": "Conferências selecionadas não medem toda a ciência, citações, liderança, especializações nem eficiência por pesquisador.",
        },
        {
            "hypothesis_id": "H-INFRA-001",
            "classification": "mixed",
            "claim": "Infraestrutura computacional limita o desenvolvimento competitivo.",
            "evidence": "Brasil tem 10 sistemas e 143,12 PFlop/s Rmax no TOP500 de jun. 2026, posição intermediária no painel.",
            "limitation": "HPC nominal não equivale a aceleradores adequados, acesso, custo ou uso produtivo em IA.",
        },
        {
            "hypothesis_id": "H-DEMAND-001",
            "classification": "inconclusive",
            "claim": "Escala e diversidade da demanda favorecem aplicações brasileiras.",
            "evidence": "População de 212,8 milhões e casos financeiros/jurídicos estabelecem escala e mecanismos plausíveis.",
            "limitation": "Não há comparação congelada de adoção, sofisticação, origem do fornecedor e captura local de valor.",
        },
        {
            "hypothesis_id": "H-SECTOR-FIN-001",
            "classification": "candidate",
            "claim": "Finanças convertem escala e dados locais em capacidade própria de IA.",
            "evidence": "Nubank declara nuFormer em crédito e Itaú documenta produtos generativos integrados a dados e modelos proprietários.",
            "limitation": "Declarações corporativas e resultados sem contrafactual internacional não provam vantagem nacional.",
        },
        {
            "hypothesis_id": "H-SECTOR-LEGAL-001",
            "classification": "candidate",
            "claim": "Dados e complexidade jurídica sustentam IA local diferenciada.",
            "evidence": "Jus IA combina acervo jurídico local e produto especializado, com métricas de uso declaradas.",
            "limitation": "Benchmark disponível é ligado ao fornecedor; faltam comparação independente, produtividade e captura financeira.",
        },
        {
            "hypothesis_id": "H-CONV-MARITACA-001",
            "classification": "candidate",
            "claim": "Especialização em português converte pesquisa local em modelos e valor.",
            "evidence": "A Maritaca desenvolve modelos em português e a relação com Jusbrasil mostra um caminho de modelo para aplicação vertical.",
            "limitation": "Faltam auditoria independente contemporânea, adoção comparável, receita, dependências e exportação.",
        },
        {
            "hypothesis_id": "H-CONV-001",
            "classification": "provisionally_supported",
            "claim": "O gargalo central está na conversão de insumos em produtos e valor local.",
            "evidence": "P&D e HPC intermediários coexistem com zero modelos notáveis observados, pequena presença nas conferências e ausência do top 15 econômico.",
            "limitation": "Associação descritiva; fontes têm universos distintos e não identificam causalmente as transições.",
        },
        {
            "hypothesis_id": "H-MAIN-001",
            "classification": "inconclusive",
            "claim": "O Brasil possui vantagens competitivas específicas em IA.",
            "evidence": "Há mecanismos e casos candidatos em finanças, jurídico e português, mas nenhuma vantagem completa passou os critérios de diferenciação, resultado e captura.",
            "limitation": "Conclusão válida para a evidência congelada; não equivale a afirmar ausência de capacidade ou oportunidade.",
        },
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assessments[0].keys())
        writer.writeheader()
        writer.writerows(assessments)
    lines = [
        "# Hypothesis assessments",
        "",
        "| Hypothesis | Classification | Evidence | Limitation |",
        "| --- | --- | --- | --- |",
    ]
    for row in assessments:
        lines.append(
            f"| {row['hypothesis_id']} | {row['classification']} | "
            f"{row['evidence']} | {row['limitation']} |"
        )
    lines += [
        "",
        "Classifications are kept separate. No aggregate score is calculated.",
        "A candidate has a plausible documented mechanism but has not met the",
        "pre-registered outcome and local-value-capture criteria.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
