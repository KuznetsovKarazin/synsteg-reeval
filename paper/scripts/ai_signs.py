"""Flag signs of AI writing (after Wikipedia:Signs of AI writing) in a LaTeX or text file."""
import re, sys

WORDS = r"""delve|delves|delving|tapestry|pivotal|underscore|underscores|underscoring|testament|intricate|intricacies|
showcase|showcases|showcasing|landscape|crucial|foster|fosters|fostering|garner|garnered|enhance|enhances|enhancing|
align with|aligns with|boast|boasts|vibrant|enduring|interplay|additionally|notably|meticulous|meticulously|valuable|
seamless|seamlessly|robust|comprehensive|holistic|leverage|leverages|leveraging|realm|paramount|multifaceted|nuanced|
navigate|navigating|unlock|unleash|elevate|empower|game-changer|cutting-edge|state-of-the-art|groundbreaking|remarkable|
significant advancement|key insight|key insights|in the realm of|ever-evolving|rapidly evolving|plays a vital role|
plays a crucial role|plays a significant role|stands as|serves as|it is important to note|it's important to note|
it is worth noting|worth noting|no discussion|in conclusion|in summary|overall,|moreover|furthermore|in addition,|
on the other hand|in contrast|ultimately|essentially|arguably|truly|indeed|novel|superior|unprecedented|
paves the way|pave the way|shed light|sheds light|a deeper understanding|broader implications|a wide range of|
a variety of|various|numerous|myriad|plethora|daunting|intriguing|captivat\w*|profound|deeply rooted|
lasting impact|watershed|turning point|rich history|commendable|noteworthy|bolster\w*|streamline\w*|harness\w*|
underpin\w*|spearhead\w*|endeavou?r\w*|facilitat\w*|utiliz\w*|utilis\w*"""
WORDS = re.compile(r"\b(" + WORDS.replace("\n", "") + r")\b", re.I)
PATTERNS = {
    "negative parallelism": re.compile(r"\bnot (only|just|merely)\b[^.]{0,80}\bbut\b|\bit'?s not\b[^.]{0,60}[;,] it'?s\b", re.I),
    "tailing -ing clause": re.compile(r",\s+(highlighting|emphasizing|underscoring|reflecting|ensuring|showcasing|demonstrating|illustrating|making|providing|allowing|enabling|leading|resulting|contributing|paving)\b", re.I),
    "em dash": re.compile(r"---|—|–(?!\d)"),
    "rule of three": re.compile(r"\b\w+, \w+,? and \w+\b"),
    "vague attribution": re.compile(r"\b(studies show|research shows|experts|observers|critics argue|it is widely|it is generally|some (researchers|authors|studies))\b", re.I),
    "chatbot residue": re.compile(r"\b(I hope this helps|certainly!|let me know|as an AI|as of my|knowledge cutoff|turn0search)\b", re.I),
    "bold label": re.compile(r"\\textbf\{[^}]{1,40}[.:]\}"),
    "hedge stack": re.compile(r"\b(may|might|could) (potentially|possibly)\b", re.I),
}


def strip_tex(line):
    line = re.sub(r"(?<!\\)%.*", "", line)
    return line


def main(path):
    total = 0
    for n, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = strip_tex(raw)
        hits = [f"word:{m.group(0)}" for m in WORDS.finditer(line)]
        for name, rx in PATTERNS.items():
            if name == "rule of three":
                continue
            hits += [f"{name}:{m.group(0)[:40]}" for m in rx.finditer(line)]
        if hits:
            total += len(hits)
            print(f"{path}:{n}: {'; '.join(hits)}")
    print(f"-- {total} hits")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
