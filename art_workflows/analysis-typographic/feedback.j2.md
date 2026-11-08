That attempt didn't work. Send the complete reply again, fixing these problems:
{% if failed("llm_error") %}- (The request itself failed before any reply — just answer the previous request again.)
{% endif %}{% if failed("truncated") %}- The reply was cut off by the output length limit before it finished. Write the complete reply, more concisely.
{% endif %}{% if failed("too_long") %}- The reply was too long. Keep the analysis to two tight paragraphs and the art prompt to one.
{% endif %}{% if failed("too_short") %}- The reply was too short or incomplete. Include the analysis AND the art prompt.
{% endif %}{% if failed("missing_pattern") %}- Follow the requested output format exactly; a required marker was missing.
{% endif %}{% if extract_failed("image_prompt") %}- End with the line "Prompt:" followed by the image prompt alone inside one fenced block — the analysis stays outside the fence.
{% endif %}
