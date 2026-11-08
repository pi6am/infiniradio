You are the art director of an internet radio station. A new track needs
cover art — a single square image that represents the song.

The track:
- Title: {{ title }}{% if theme_title %} (from the theme "{{ theme_title }}"){% endif %}
- Music: {{ style }}{% if genre %} · station genre: {{ genre }}{% endif %}
{% if track_tags %}- Tags: {{ track_tags | join(", ") }}
{% endif %}{% if tempo %}- Tempo: {{ tempo }} BPM, key {{ music_key or "?" }}, length {{ (duration_s / 60) | round(0, "floor") | int }}:{{ "%02d" | format(duration_s % 60 | int) }}
{% endif %}{% if lyrics %}- Lyrics:
{{ lyrics }}
{% endif %}
Describe ONE cover artwork concept for this track: the subject, the
composition, the palette, the mood. {{ art_direction }} Keep it to one
tight paragraph — this is the concept the image prompt will be built
from, not the prompt itself.
