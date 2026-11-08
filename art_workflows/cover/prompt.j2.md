Turn that concept into the final image-generation prompt. Write it as one
paragraph for an image model: subject and setting, composition and
framing, medium and style{% if params.get("medium") %} ({{ params.get("medium") }}){% endif %}, palette and lighting, mood. Concrete visual
language, no abstractions the model can't draw, no text or lettering in
the image.

Put ONLY the prompt inside one fenced block:

```text
<the image prompt>
```
