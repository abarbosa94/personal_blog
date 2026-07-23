local translation_target = nil
local translation_label = nil

local function text(value)
  if value == nil then
    return nil
  end

  return pandoc.utils.stringify(value)
end

function Meta(meta)
  translation_target = text(meta.translation)

  if translation_target == nil or translation_target == "" then
    return meta
  end

  local language = text(meta.lang) or "en"
  if language:lower():match("^pt") then
    translation_label = "🇺🇸 English"
  else
    translation_label = "🇧🇷 Português"
  end

  return meta
end

function Pandoc(document)
  if translation_target == nil or translation_target == "" then
    return document
  end

  local link = pandoc.Link(
    { pandoc.Str(translation_label) },
    translation_target,
    "Read this post in " .. translation_label:gsub("^[^ ]+ ", "")
  )
  local switcher = pandoc.Div(
    { pandoc.Para({ link }) },
    pandoc.Attr("", { "language-switcher" })
  )

  table.insert(document.blocks, 1, switcher)
  return document
end
