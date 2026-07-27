Feature: Generate a Portuguese notebook that is safe for an author to review
  Tower+ should translate the prose of an English post without damaging the
  Markdown and Quarto syntax that makes the notebook render correctly.

  The generated notebook is not considered publication-ready. It must remain a
  draft, identify itself as the Portuguese translation, and link back to the
  English source so the language switcher can connect both rendered pages.

  Scenario: Technical Markdown can make a protection round trip without changing
    Given a source paragraph contains a Markdown link, inline code, math, and a citation
    When the generator replaces every protected construct with an internal token
    And the generator restores the original constructs without calling Tower+
    Then the complete restored paragraph exactly matches the source paragraph
    And the link, code, math, and citation are preserved byte for byte

  Scenario: A dropped placeholder does not corrupt a link or block the translation
    Given an English paragraph says "Read" immediately before a protected Markdown link
    And Tower+ cannot be trusted to copy an internal placeholder
    When the fallback separates translatable prose from protected Markdown
    And the fallback translates "Read" to "Leia"
    Then no internal placeholder is sent to Tower+ during the fallback
    And the result is "Leia [o artigo](https://example.com)."

  Scenario: An expanded link label cannot invent sections in the translated post
    Given an English paragraph contains the short link label "English"
    And Tower+ expands that label into a multiline article with Markdown headings
    When the fallback validates the translated link label
    Then the unsafe label translation is discarded
    And the original link "[English](https://example.com)" is retained
    And no new Markdown heading is introduced

  Scenario: A generated Portuguese notebook remains paired and unpublished
    Given source metadata describes a published English notebook
    When Portuguese metadata is generated for the source file "post.ipynb"
    Then the generated language is Brazilian Portuguese
    And its translation link points back to "post.ipynb"
    And it is marked as the secondary translation
    And it remains a draft until an author reviews it
