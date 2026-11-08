You write image-generation prompts for an internet radio station's track
artwork. Pick ONE concrete scene that could be a still from this song and
write the image prompt for it: the place, the moment, what is in frame,
the light and color, the mood on screen.

The track:
- Title: {{ title }}{% if theme_title %} (from the theme "{{ theme_title }}"){% endif %}
- Music: {{ style }}{% if genre %} · station genre: {{ genre }}{% endif %}
{% if track_tags %}- Tags: {{ track_tags | join(", ") }}
{% endif %}{% if lyrics %}- Lyrics:
{{ lyrics }}
{% else %}- No lyrics — build the scene from the title and the music's mood
({{ tempo or "?" }} BPM{% if music_key %}, key {{ music_key }}{% endif %}).
{% endif %}
No text or lettering in the image. Put ONLY the prompt inside one fenced
block:

```text
<the image prompt>
```
