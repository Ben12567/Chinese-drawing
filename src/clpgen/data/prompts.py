from __future__ import annotations

from dataclasses import asdict

from clpgen.data.schema import PromptBundle


def _join(items: list[str] | None, sep: str = ", ", fallback: str = "none") -> str:
    values = [item.strip() for item in (items or []) if item and item.strip()]
    return sep.join(values) if values else fallback


def build_prompt_bundle(
    subject: str,
    season: str,
    weather: str,
    foreground: list[str] | None = None,
    midground: list[str] | None = None,
    background: list[str] | None = None,
    ink_tone: str = "light ink wash",
    palette: str = "ink wash",
    blankness: str = "ample blank space",
    mood: str = "serene",
    dense_caption_zh: str | None = None,
    dense_caption_en: str | None = None,
) -> PromptBundle:
    short_text = (
        f"{subject}, {season}, {weather}, foreground {_join(foreground)}, "
        f"{ink_tone}, {palette}, {blankness}"
    )
    structured_text = (
        f"Subject: {subject}; "
        f"Composition layers: foreground {_join(foreground)}, "
        f"midground {_join(midground)}, background {_join(background)}; "
        f"Ink tone: {ink_tone}; "
        f"Palette: {palette}; "
        f"Blankness and mood: {blankness}, {mood}; "
        f"Season and weather: {season}, {weather}."
    )
    dense_text = dense_caption_en or (
        f"A Chinese landscape painting depicting {subject} in {season} under {weather}, "
        f"with {_join(foreground)} in the foreground, {_join(midground)} in the midground, "
        f"and {_join(background)} in the background. The image uses {ink_tone} and {palette}, "
        f"preserves {blankness}, and conveys a {mood} atmosphere."
    )
    return PromptBundle(
        short_zh=short_text,
        structured_zh=structured_text,
        structured_en=structured_text,
        dense_zh=dense_caption_zh or dense_text,
        dense_en=dense_text,
    )


def bundle_to_dict(bundle: PromptBundle) -> dict[str, str]:
    return asdict(bundle)
