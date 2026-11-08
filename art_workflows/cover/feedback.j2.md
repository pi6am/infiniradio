That attempt didn't work. Send the complete reply again, fixing these problems:
{% if failed("llm_error") %}- (The request itself failed before any reply — just answer the previous request again.)
{% endif %}{% if failed("truncated") %}- The reply was cut off by the output length limit before it finished. Write the complete reply, more concisely.
{% endif %}{% if failed("too_long") %}- The prompt was too long. Tighten it: keep the concrete visual detail, cut the excess.
{% endif %}{% if failed("too_short") %}- The prompt was too short or incomplete. Include the complete prompt.
{% endif %}{% if failed("missing_pattern") %}- Follow the requested output format exactly; a required marker was missing.
{% endif %}{% if extract_failed("image_prompt") %}- Put the final image prompt alone inside one fenced block marked ```text — no commentary before or after inside the fence.
{% endif %}
