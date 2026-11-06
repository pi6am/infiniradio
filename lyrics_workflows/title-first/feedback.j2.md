That attempt didn't work. Send the complete reply again, fixing these problems:
{% if failed("llm_error") %}- (The request itself failed before any reply — just answer the previous request again.)
{% endif %}{% if failed("truncated") %}- The reply was cut off by the output length limit before it finished. Write the complete reply, more concisely.
{% endif %}{% if failed("too_long") %}- The reply was too long. Tighten it: keep the structure, cut the excess.
{% endif %}{% if failed("too_short") %}- The reply was too short or incomplete. Include the complete response.
{% endif %}{% if failed("missing_any") %}- Include at least one section header on its own line, e.g. [Intro], [Verse], [Chorus], [Bridge], [Outro], [Instrumental] or [Solo].
{% endif %}{% if failed("missing_pattern") %}- Follow the requested output format exactly; a required marker was missing.
{% endif %}{% if failed("forbidden_present") %}- The reply included content it must not include:{% for f in failures %}{% if f.code == "forbidden_present" %} {{ f.detail }}.{% endif %}{% endfor %} Rewrite those lines entirely rather than tweaking them.
{% endif %}{% if extract_failed("song_name") %}- Include a line of the form: Name: `<English song name>`
{% endif %}{% if extract_failed("style_prompt") %}- Put the style prompt after the line "Style prompt:", inside a code fence.
{% endif %}
