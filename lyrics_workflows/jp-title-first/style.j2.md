Write a Style Prompt for the translated version of the song, following this guide:
```
Describe the main musical characteristics, such as:
Genre: pop, jazz, rock, electronic, folk
Vocals: warm female vocal, expressive male vocal, soft vocals
Language: English
Tempo: 90 BPM, slow, upbeat
Instruments: piano, electric guitar, bass, drums, synths

Example:
English, warm female vocal, contemporary pop, 96 BPM, piano, rounded electric bass, restrained drums, clear diction
```

Write the style prompt in this format:
Style prompt:

```
{prompt here}
```
{% if params.style_guide %}

{{ params.style_guide }}
{% endif %}
{% if params.lyrics_guide %}

{{ params.lyrics_guide }}
{% endif %}
