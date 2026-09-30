Use this guide to write a song in {{ params.language }}:
```
{{ brainstorm }}
```

Write the song in the {{ params.language }} language, {{ params.script_note }}. Don't write a translation, and don't write romaji. Just write the song title and lyrics (with section headers [Intro], [Verse], [Chorus], [Outro], [Bridge], [Solo], [Instrumental] as appropriate. Do not write any comments on instrumentation or arrangement, just write the lyrics. Treat the brainstorm as a rough guide, but focus on writing a good song. Don't overdo it, focus on the most important parts of the theme.
{% if style_guide %}

{{ style_guide }}
{% endif %}
{% if structure_guide %}

{{ structure_guide }}
{% endif %}
