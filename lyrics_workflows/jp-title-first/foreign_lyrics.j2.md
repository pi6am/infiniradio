Use this guide to write a song in {{ params.language }}:
```
{{ brainstorm }}
```

Write the song in the {{ params.language }} language, {{ params.script_note }}. Don't write a translation, and don't write romaji. Just write the song title and lyrics (mark sections with headers from this list: [Intro], [Verse], [Chorus], [Outro], [Bridge], [Solo], [Instrumental] as appropriate, but they don't have to be in this order and you don't have to use all of them, just what makes sense for the song. Do not write any comments on instrumentation or arrangement, just write the lyrics. Treat the brainstorm as a rough guide, but focus on writing a good song. Don't overdo it, focus on the most important parts of the theme.
{% if params.style_guide %}

{{ params.style_guide }}
{% endif %}
{% if params.lyrics_guide %}

{{ params.lyrics_guide }}
{% endif %}
