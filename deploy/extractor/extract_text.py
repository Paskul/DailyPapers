"""Fixed entrypoint of the paperwatch extractor image. Reads /input/source and writes UTF-8 text to stdout.

  PDF  -> poppler `pdftotext` in reading order. Pages are separated by form feeds (\\f), so page numbers survive for
          the passage ids.
  XML  -> (JATS, TEI, or other) the text of each content block on its own paragraph; headings are prefixed "## ".
          Documents that declare entities are refused (no entity expansion, no external DTDs).
  else -> exit status 2.

This runs inside the sandbox (no network, read-only root, one read-only input). It never takes arguments.
"""

import os
import sys

INPUT_PATH = "/input/source"
PDFTOTEXT_PATH = "/usr/bin/pdftotext"
SNIFF_BYTES = 4096
UNSUPPORTED_INPUT_EXIT_STATUS = 2
HEADING_TAGS = frozenset({"head", "title", "article-title"})
CONTENT_TAGS = HEADING_TAGS | frozenset(
    {
        "p",
        "caption",
        "label",
        "formula",
        "disp-formula",
        "td",
        "th",
        "item",
        "list-item",
        "note",
        "ref",
        "mixed-citation",
        "bibl",
    }
)


def main() -> int:
    with open(INPUT_PATH, "rb") as input_file:
        leading_bytes = input_file.read(SNIFF_BYTES)
    if leading_bytes.startswith(b"%PDF-"):
        # Replace this process: pdftotext streams straight to our stdout.
        os.execv(PDFTOTEXT_PATH, ["pdftotext", "-enc", "UTF-8", "-q", INPUT_PATH, "-"])  # noqa: S606  # fixed pdftotext path and arguments; the input path is fixed too
    if leading_bytes.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
        return write_xml_text()
    print("unsupported input: neither PDF nor XML", file=sys.stderr)
    return UNSUPPORTED_INPUT_EXIT_STATUS


def write_xml_text() -> int:
    import xml.etree.ElementTree as element_tree

    with open(INPUT_PATH, "rb") as input_file:
        document_bytes = input_file.read()
    if b"<!ENTITY" in document_bytes.upper():
        print("refused: XML declares entities", file=sys.stderr)
        return UNSUPPORTED_INPUT_EXIT_STATUS
    try:
        root_element = element_tree.fromstring(document_bytes)  # noqa: S314  # a DOCTYPE/ENTITY is refused before parsing; expat also limits entity expansion
    except element_tree.ParseError as parse_error:
        print(f"invalid XML: {parse_error}", file=sys.stderr)
        return UNSUPPORTED_INPUT_EXIT_STATUS
    output_blocks = []
    elements_to_visit = [root_element]
    while elements_to_visit:  # iterative depth-first walk, document order
        element = elements_to_visit.pop()
        local_tag = element.tag.rsplit("}", 1)[-1] if isinstance(element.tag, str) else ""
        if local_tag in CONTENT_TAGS:
            block_text = " ".join("".join(element.itertext()).split())
            if block_text:
                output_blocks.append(f"## {block_text}" if local_tag in HEADING_TAGS else block_text)
            continue  # the block's descendants are already in its text
        elements_to_visit.extend(reversed(list(element)))
    sys.stdout.write("\n\n".join(output_blocks) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
