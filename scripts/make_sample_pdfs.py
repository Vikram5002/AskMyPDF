"""
Generate the demo PDFs in /samples -- using only the Python standard library.

A PDF is just a text-like file of numbered "objects" (catalog, pages, fonts,
content streams) plus a cross-reference table giving each object's byte
offset. Writing that by hand keeps the project free of extra dependencies.

    python scripts/make_sample_pdfs.py
"""

import sys
import textwrap
from pathlib import Path
from typing import List

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

LINES_PER_PAGE = 48
WRAP_CHARS = 88


def _escape(line: str) -> bytes:
    """Encode one line for a PDF string literal (Windows-1252 covers é, è, à...)."""
    raw = line.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def build_pdf(pages: List[List[str]]) -> bytes:
    """Build a PDF where each page is a list of text lines (Helvetica 11pt)."""
    objects: List[bytes] = []  # objects[i] is PDF object number i + 1

    n_pages = len(pages)
    font_id = 3
    page_ids = [4 + 2 * i for i in range(n_pages)]      # page, then its content stream

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode()
    objects.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n_pages).encode() + b" >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    for pid, lines in zip(page_ids, pages):
        stream = b"BT /F1 11 Tf 14 TL 72 760 Td\n"
        for line in lines:
            stream += b"(" + _escape(line) + b") '\n"   # ' = move to next line and show text
        stream += b"ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return out


def paginate(text: str) -> List[List[str]]:
    """Word-wrap paragraphs and split the lines into pages."""
    lines: List[str] = []
    for para in textwrap.dedent(text).strip().split("\n\n"):
        lines.extend(textwrap.wrap(" ".join(para.split()), WRAP_CHARS) or [""])
        lines.append("")
    return [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

SHORT_EN = """
The Eiffel Tower

The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the
tower. Construction began in 1887 and the tower was completed in 1889, in time to serve
as the entrance to the 1889 World's Fair, which celebrated the centennial of the French
Revolution.

The tower is 330 metres tall, about the same height as an 81-storey building. When it
was finished it was the tallest man-made structure in the world, a title it held for
41 years until the Chrysler Building in New York City was completed in 1930.

The tower has three levels open to visitors, with restaurants on the first and second
levels. The top level's upper platform is 276 metres above the ground. Visitors can
climb the stairs to the second level or take one of the lifts. Around seven million
people visit the Eiffel Tower every year, making it one of the most visited paid
monuments in the world.
"""

SHORT_FR = """
La tour Eiffel

La tour Eiffel est une tour de fer puddlé située sur le Champ-de-Mars, à Paris, en
France. Elle porte le nom de l'ingénieur Gustave Eiffel, dont l'entreprise a conçu et
construit la tour. Sa construction a commencé en 1887 et elle a été achevée en 1889,
pour l'Exposition universelle qui célébrait le centenaire de la Révolution française.

La tour mesure 330 mètres de hauteur. À son achèvement, elle était la plus haute
structure construite par l'homme dans le monde, un titre qu'elle a conservé pendant
41 ans, jusqu'à la construction du Chrysler Building à New York en 1930.

Chaque année, environ sept millions de personnes visitent la tour Eiffel, ce qui en
fait l'un des monuments payants les plus visités au monde.
"""

LONG_EN = """
A Short History of Artificial Intelligence

1. The Question: Can Machines Think?

The idea of building machines that think is much older than the computer. Greek myths
told of bronze automatons, and in the seventeenth and eighteenth centuries inventors
built mechanical figures that could write, draw or play music. These devices followed
fixed mechanisms, however, and nobody seriously claimed they could reason. The modern
story of artificial intelligence begins in the twentieth century, when mathematicians
began to ask what it means to compute and whether reasoning itself could be reduced to
the manipulation of symbols.

The British mathematician Alan Turing gave the question its most famous form. In his
1950 paper "Computing Machinery and Intelligence", Turing proposed replacing the vague
question "Can machines think?" with a practical test he called the imitation game. In
the game, a human judge holds text conversations with a hidden person and a hidden
machine. If the judge cannot reliably tell which is which, the machine is said to have
passed. Today this is known as the Turing test. Turing also predicted that by the end of
the century people would speak of machines thinking without expecting to be
contradicted, and he answered many objections that critics still raise today.

2. The Birth of a Field

Artificial intelligence became an academic discipline at a summer workshop held at
Dartmouth College in 1956. The workshop was organised by John McCarthy, Marvin Minsky,
Nathaniel Rochester and Claude Shannon. The proposal for the workshop, written the year
before, stated that every aspect of learning or any other feature of intelligence can
in principle be so precisely described that a machine can be made to simulate it. John
McCarthy coined the term "artificial intelligence" for this proposal, and the name has
been used ever since.

The years after Dartmouth were full of optimism. Programs were written that proved
theorems in logic, solved algebra word problems and played checkers well enough to beat
their own authors. In 1958 the psychologist Frank Rosenblatt introduced the perceptron,
a simple model of a neuron that could learn to classify patterns by adjusting numerical
weights. Newspapers reported that such machines would soon walk, talk and be conscious
of their existence. Between 1964 and 1966 Joseph Weizenbaum at MIT wrote ELIZA, a
program that imitated a psychotherapist by rephrasing the user's own sentences as
questions. Although ELIZA understood nothing, many users became emotionally attached to
it, which surprised and worried its creator.

3. The First AI Winter

Early successes on small, carefully chosen problems did not scale to the real world.
Computers were slow, memory was expensive, and many problems turned out to require an
enormous amount of common-sense knowledge that nobody knew how to encode. In 1969 Minsky
and Seymour Papert published the book "Perceptrons", which showed that a single-layer
perceptron cannot learn even simple functions such as exclusive-or. Research on neural
networks slowed sharply for more than a decade.

In 1973 the Lighthill report, written for the British Science Research Council, judged
that AI had failed to achieve its grand objectives. Funding was cut in the United
Kingdom, and American agencies also became more cautious. This period of reduced
funding and interest, which lasted through most of the 1970s, is now called the first
AI winter.

4. Expert Systems and the Second Winter

AI returned to favour in the 1980s through expert systems. Instead of trying to create
general intelligence, an expert system captured the knowledge of human specialists in a
narrow domain as a large set of if-then rules. One of the most successful was XCON,
used by Digital Equipment Corporation to configure orders for its VAX computers; it was
reported to save the company tens of millions of dollars a year. Companies around the
world built expert systems, and specialised Lisp machines were sold to run them.

The boom did not last. Expert systems were expensive to build and maintain, could not
learn, and failed in strange ways when faced with situations their rules did not
cover. In 1987 the market for Lisp machines collapsed as cheaper desktop computers
became powerful enough to run the same software. Funding dried up again, and the late
1980s and early 1990s became known as the second AI winter.

5. The Return of Neural Networks

While symbolic AI struggled, a quieter revival of neural networks was under way. In
1986 David Rumelhart, Geoffrey Hinton and Ronald Williams published an influential paper
showing how the backpropagation algorithm could train networks with several layers of
neurons. Multi-layer networks could learn the exclusive-or function and much more,
answering the criticism made in "Perceptrons". During the 1990s Yann LeCun and his
colleagues used convolutional neural networks to read handwritten digits on bank
cheques, one of the first commercial uses of the technology.

Games provided some of the most public milestones. In 1997 IBM's chess computer Deep
Blue defeated the world champion Garry Kasparov in a six-game match. Deep Blue relied
mainly on specialised hardware that searched around two hundred million positions per
second rather than on learning, but its victory was seen around the world as a sign
that machines could outperform people at a task long associated with intelligence.

6. The Deep Learning Revolution

Three developments came together in the late 2000s: far larger datasets, much faster
hardware in the form of graphics processing units, and better training techniques. The
ImageNet dataset, containing over fourteen million labelled images, became the standard
benchmark for image recognition. In 2012 a deep convolutional network called AlexNet,
created by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton, won the ImageNet
competition with an error rate far lower than any previous system. The result convinced
most of the field that deep learning was the way forward.

In 2016 AlphaGo, a program developed by DeepMind, defeated the Go champion Lee Sedol by
four games to one in Seoul. Go has far more possible positions than chess, and many
experts had expected such a victory to be at least a decade away. AlphaGo combined deep
neural networks with tree search and learned partly by playing millions of games
against itself.

7. Transformers and Language Models

Language was one of the hardest problems for AI because meaning depends on long-range
context. Recurrent neural networks read text one word at a time, which made them slow to
train and forgetful over long passages. In 2017 researchers at Google published the
paper "Attention Is All You Need", which introduced the Transformer architecture. A
Transformer uses a mechanism called self-attention to let every word in a sentence look
directly at every other word, and it can be trained in parallel on very large amounts
of text.

In 2018 Google released BERT, a Transformer that was pre-trained on large amounts of
unlabelled text and could then be fine-tuned for tasks such as classification or
question answering. Fine-tuned on the Stanford Question Answering Dataset, known as
SQuAD, BERT answered reading-comprehension questions about Wikipedia passages more
accurately than the human baseline. The second version of the dataset, SQuAD 2.0, added
more than fifty thousand unanswerable questions so that models also had to learn when
the passage does not contain an answer. RoBERTa, released by Facebook AI in 2019,
improved on BERT by training longer on more data.

In 2020 OpenAI described GPT-3, a language model with 175 billion parameters that could
write essays, translate text and answer questions after seeing only a few examples. Two
years later, in November 2022, the release of ChatGPT brought conversational language
models to hundreds of millions of people and started a new wave of investment and
public debate about the future of artificial intelligence.

8. Looking Ahead

The history of AI has alternated between periods of excitement and disappointment.
Each wave produced lasting tools even when its biggest promises were not met: search
algorithms, rule-based systems, and statistical learning are all still in use. Modern
systems raise new questions about reliability, bias, energy use and the effect of
automation on work. Whatever comes next, the question Turing asked in 1950 remains a
useful guide: rather than arguing about definitions, we can judge machines by what they
are actually able to do.
"""

SAMPLES = {
    "sample_short.pdf": SHORT_EN,
    "sample_long.pdf": LONG_EN,
    "sample_french.pdf": SHORT_FR,
}


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    for name, text in SAMPLES.items():
        pages = paginate(text)
        (SAMPLES_DIR / name).write_bytes(build_pdf(pages))
        words = len(text.split())
        print(f"wrote samples/{name}: {len(pages)} page(s), {words} words")


if __name__ == "__main__":
    sys.exit(main())
