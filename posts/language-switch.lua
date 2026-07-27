local translation_target = nil
local current_language = "en"

local function text(value)
  if value == nil then
    return nil
  end

  return pandoc.utils.stringify(value)
end

local function rendered_target(target)
  return target
    :gsub("%.ipynb$", ".html")
    :gsub("%.qmd$", ".html")
    :gsub("%.md$", ".html")
end

function Meta(meta)
  translation_target = text(meta.translation)

  if translation_target == nil or translation_target == "" then
    return meta
  end

  current_language = (text(meta.lang) or "en"):lower()
  translation_target = rendered_target(translation_target)
  return meta
end

local function active_language(label, language)
  return pandoc.Span(
    { pandoc.Str(label) },
    pandoc.Attr(
      "",
      { "language-switcher-active" },
      {
        { "lang", language },
        { "aria-current", "page" }
      }
    )
  )
end

local function language_link(label, language, target)
  return pandoc.Link(
    { pandoc.Str(label) },
    target,
    "Read this post in " .. label,
    pandoc.Attr(
      "",
      { "language-switcher-link" },
      {
        { "lang", language },
        { "hreflang", language }
      }
    )
  )
end

function Pandoc(document)
  if translation_target == nil or translation_target == "" then
    return document
  end

  local portuguese
  local english
  if current_language:match("^pt") then
    portuguese = active_language("PT-BR", "pt-BR")
    english = language_link("EN", "en", translation_target)
  else
    portuguese = language_link("PT-BR", "pt-BR", translation_target)
    english = active_language("EN", "en")
  end

  local switcher = pandoc.Div(
    {
      pandoc.Plain({
        portuguese,
        pandoc.Space(),
        pandoc.Span(
          { pandoc.Str("|") },
          pandoc.Attr("", { "language-switcher-separator" })
        ),
        pandoc.Space(),
        english
      })
    },
    pandoc.Attr(
      "",
      { "language-switcher" },
      {
        { "role", "navigation" },
        { "aria-label", "Select post language" }
      }
    )
  )

  table.insert(document.blocks, 1, switcher)
  return document
end
