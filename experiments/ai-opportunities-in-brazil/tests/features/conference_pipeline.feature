Feature: Build an auditable conference-publication dataset
  The experiment must preserve the official publication universe and expose
  decisions that affect country and organization comparisons.

  Scenario: Enumerate ICML papers from the official PMLR volume
    Given the ICML PMLR fixture
    When I enumerate the ICML main proceedings
    Then 2 papers are enumerated
    And the second ICML title is "A Second & Better Paper"
    And the second ICML paper has 2 authors

  Scenario: Exclude ACL front matter from the publication count
    Given the ACL Anthology fixture with front matter
    When I enumerate ACL long papers
    Then the ACL front matter is excluded
    And 2 ACL research papers are enumerated
    And every ACL paper has a derived DOI

  Scenario: Preserve NeurIPS tracks instead of silently merging them
    Given the NeurIPS proceedings fixture with two tracks
    When I enumerate NeurIPS papers
    Then 2 NeurIPS papers are enumerated
    And the NeurIPS tracks are "conference,datasets_and_benchmarks_track"

  Scenario: Count an international collaboration in two auditable ways
    Given a reconciled paper with affiliations in Brazil and the United States
    When I calculate country weights
    Then full counting assigns 1 to each country
    And fractional counting assigns 0.5 to each country

  Scenario: Reject a sample with insufficient country coverage
    Given 10 enumerated papers with countries available for 2 papers
    When I evaluate the publication quality gates
    Then country coverage is 0.2
    And the sample does not pass

  Scenario: Recover affiliation candidates from paper front matter
    Given PDF front matter with a university and a company
    When I extract affiliation candidates
    Then the university and company are preserved as 2 candidates
    And body text after the introduction is excluded

  Scenario: Reproduce a formal random sample
    Given 100 officially enumerated papers
    When I select 50 papers twice with seed 20250727
    Then both formal samples contain the same paper identifiers
    And the formal sample is not simply the first 50 papers

  Scenario: Prioritize manual review without biasing the formal sample
    Given a formal sample with an automatic failure and a clean control
    When I build the manual review queue
    Then the automatic failure appears before the clean control
    And both formal papers remain in the review queue

  Scenario: Recover explicit countries from a wrapped affiliation block
    Given a wrapped multinational affiliation block
    When I extract explicit affiliation countries
    Then the countries are "FR,GB,US"
    And the conference location is not an affiliation country
