{% if tags %}Choose the tags that best describe this song, from this list only:
{{ tags | join(", ") }}
{% else %}Choose 2 to 4 lowercase genre tags that best describe this song{% endif %}

Style prompt: {{ style_prompt }}

Lyrics:
```
{{ final_lyrics }}
```

Reply with only the tags, comma-separated on a single line, ordered from most to least relevant. Do not reply with anything else.
