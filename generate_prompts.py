"""Create exact UTF-8 byte length benchmark prompts in ./prompt."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIZES = (100, 1000, 10000)
INTRO = "Read the story. Summarize it and answer the final question.\n\n"
QUESTION = "\n\nQuestion: What color was the brass key?"
STORY = (
    "Mara walked through the old town in no particular hurry. She passed a bakery and listened to a clock. ",
    "At the market, a mapmaker described a winding road beyond the orchard. Mara asked where it ended. ",
    "She stopped for tea, watched clouds cross the square, and remembered a rainy train ride from years ago. ",
    "A traveler spoke about a distant bridge, and Mara recalled a letter she had meant to send. ",
    "By evening she returned by a different lane, noticing lamps, bicycles, and the smell of fresh bread. ",
)


def make_prompt(size: int) -> str:
    if size not in SIZES:
        raise ValueError(f"unsupported prompt size: {size}")
    color = random.choice(("blue", "pink", "gold"))
    fact = f"The brass key was {color}."
    if size == 100:
        intro = "Summarize this story. "
        story = f"Mara wandered around town. {fact}"
        question = "\nWhat color was the key?"
    else:
        intro, question = INTRO, QUESTION
    available = size - len((intro + question).encode("utf-8"))
    if available < len(fact.encode("utf-8")):
        raise ValueError("prompt is too short for its fact")
    if size == 100:
        narrative = story + " " * (available - len(story))
    else:
        rng = random.Random(size)
        sections = []
        while len("".join(sections)) < available:
            sections.extend(rng.sample(STORY, len(STORY)))
        split = 0
        while split + 1 < len(sections) and len("".join(sections[:split + 1])) <= available // 2:
            split += 1
        before_text = "".join(sections[:split])
        narrative = before_text + fact + " "
        remaining = available - len(narrative)
        narrative += "".join(sections[split:])[:remaining]
        if not narrative.endswith(" "):
            narrative = narrative.rsplit(" ", 1)[0]
        narrative += " " * (available - len(narrative))
    result = intro + narrative + question
    assert len(result.encode("utf-8")) == size
    assert fact in result
    return result


def generate(directory: Path = ROOT / "prompt", overwrite: bool = False) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for size in SIZES:
        path = directory / f"{size}.txt"
        if not path.exists() or overwrite:
            path.write_text(make_prompt(size), encoding="utf-8", newline="")
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing prompts")
    args = parser.parse_args()
    for path in generate(overwrite=args.overwrite):
        print(f"Created {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
