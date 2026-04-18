# Prompt Files

All model-facing prompt text is stored in this directory.

## File format

- Prompt files use YAML.
- `content` stores a single string prompt body.
- `parts` composes prompt blocks and supports `$ref` references.
- `lines` stores ordered list fragments (for constraints).
- `fields` stores schema-description mappings.

## Placeholders

Use Python `str.format()` placeholders for interpolated values, for example:

`{ticker}`, `{question}`, `{constraints}`.

## Naming

Use slash-based logical names that map to paths:

- `system/analyze_market` -> `prompts/system/analyze_market.yaml`
- `user/category_hints/sports` -> `prompts/user/category_hints/sports.yaml`

