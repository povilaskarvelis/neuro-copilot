"""
Utilities for exporting markdown reports to PDF.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape


try:  # pragma: no cover - exercised indirectly in environments with reportlab.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal environments.
    REPORTLAB_AVAILABLE = False

try:  # pragma: no cover - exercised indirectly when markdown backend is available.
    import markdown as markdown_lib

    MARKDOWN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal environments.
    MARKDOWN_AVAILABLE = False


CHROME_CANDIDATES = (
    "google-chrome",
    "chromium",
    "chromium-browser",
    "chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

HIGH_FIDELITY_CSS = """
@page {
  size: Letter;
  margin: 0.75in 0.75in 0.85in 0.75in;
}
html, body {
  margin: 0;
  padding: 0;
  background: #0b0f19;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.6;
  color: #e2e8f0;
  background: #0b0f19;
}
article.report {
  max-width: 100%;
}

/* ── Title / H1 ── */
h1 {
  font-size: 26pt;
  font-weight: 700;
  color: #ffffff;
  border-bottom: 3px solid #6366f1;
  padding-bottom: 0.35em;
  margin-top: 0;
  margin-bottom: 0.6em;
  line-height: 1.2;
  letter-spacing: -0.01em;
}

/* ── Section headings ── */
h2 {
  font-size: 14.5pt;
  font-weight: 700;
  color: #a5b4fc;
  margin-top: 1.5em;
  margin-bottom: 0.45em;
  padding-bottom: 0.2em;
  border-bottom: 1.5px solid #312e81;
  line-height: 1.25;
}
h3 {
  font-size: 11.5pt;
  font-weight: 600;
  color: #818cf8;
  margin-top: 1.1em;
  margin-bottom: 0.3em;
}

/* ── Research question callout block ── */
blockquote {
  margin: 1em 0;
  padding: 0.8em 1.1em;
  border-left: 4px solid #6366f1;
  background: #1e1b4b;
  color: #c7d2fe;
  border-radius: 0 8px 8px 0;
  font-size: 10.5pt;
}
blockquote strong {
  color: #e0e7ff;
}

/* ── Body text ── */
p {
  margin: 0.5em 0;
  color: #cbd5e1;
}

/* ── Lists ── */
ul, ol {
  margin: 0.5em 0 0.5em 1.4em;
  padding-left: 0.8em;
  color: #cbd5e1;
}
li {
  margin: 0.35em 0;
  line-height: 1.55;
}
li::marker {
  color: #818cf8;
}

/* ── Code ── */
pre {
  margin: 0.8em 0;
  padding: 0.8em 1em;
  background: #020617;
  color: #a5f3fc;
  border: 1px solid #1e293b;
  border-radius: 8px;
  font-size: 9pt;
  overflow: auto;
}
code {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: #1e293b;
  padding: 0.15em 0.4em;
  border-radius: 4px;
  color: #a5f3fc;
}
pre code {
  background: transparent;
  padding: 0;
  color: inherit;
}

/* ── Tables ── */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 9.5pt;
}
th {
  background: #312e81;
  color: #e0e7ff;
  font-weight: 600;
  padding: 0.5em 0.7em;
  text-align: left;
  border: 1px solid #3730a3;
}
td {
  border: 1px solid #1e293b;
  padding: 0.45em 0.65em;
  vertical-align: top;
  color: #cbd5e1;
}
tr:nth-child(even) td {
  background: #0f172a;
}
tr:nth-child(odd) td {
  background: #111827;
}

/* ── Horizontal rule ── */
hr {
  border: none;
  border-top: 1px solid #1e293b;
  margin: 1.3em 0;
}

/* ── Coverage / footer note ── */
em {
  color: #94a3b8;
  font-size: 9.5pt;
}

/* ── Links ── */
a {
  color: #818cf8;
  text-decoration: underline;
  word-break: break-all;
}
"""


def _heading_level(line: str) -> int:
    if line.startswith("### "):
        return 3
    if line.startswith("## "):
        return 2
    if line.startswith("# "):
        return 1
    return 0


def _escape_attr(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def _format_inline_markdown(text: str) -> str:
    """
    Convert common inline markdown to ReportLab Paragraph markup.
    """
    if not text:
        return ""

    placeholders: list[str] = []

    def _store(value: str) -> str:
        token = f"@@MD{len(placeholders)}@@"
        placeholders.append(value)
        return token

    def _replace_links(raw: str) -> str:
        pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")

        def repl(match: re.Match[str]) -> str:
            label = escape(match.group(1).strip())
            href = _escape_attr(match.group(2).strip())
            return _store(f'<font color="#818cf8"><u><link href="{href}">{label}</link></u></font>')

        return pattern.sub(repl, raw)

    def _replace_code(raw: str) -> str:
        pattern = re.compile(r"`([^`]+)`")

        def repl(match: re.Match[str]) -> str:
            value = escape(match.group(1).strip())
            return _store(f'<font name="Courier">{value}</font>')

        return pattern.sub(repl, raw)

    def _replace_bold(raw: str) -> str:
        pattern = re.compile(r"(\*\*|__)(.+?)\1")

        def repl(match: re.Match[str]) -> str:
            value = escape(match.group(2).strip())
            return _store(f"<b>{value}</b>")

        return pattern.sub(repl, raw)

    def _replace_italic(raw: str) -> str:
        # Require non-whitespace bounds inside emphasis to avoid treating list markers as italics.
        pattern = re.compile(
            r"(?<!\*)\*(?!\*)(\S(?:[^*]*?\S)?)\*(?!\*)|(?<!_)_(?!_)(\S(?:[^_]*?\S)?)_(?!_)"
        )

        def repl(match: re.Match[str]) -> str:
            value = match.group(1) if match.group(1) is not None else match.group(2)
            return _store(f"<i>{escape(value.strip())}</i>")

        return pattern.sub(repl, raw)

    transformed = text
    transformed = _replace_links(transformed)
    transformed = _replace_code(transformed)
    transformed = _replace_bold(transformed)
    transformed = _replace_italic(transformed)
    transformed = escape(transformed)

    for idx, value in enumerate(placeholders):
        transformed = transformed.replace(f"@@MD{idx}@@", value)

    return transformed


def _is_bold_only_line(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r"(\*\*|__)(.+?)\1:?", stripped))


def _extract_bold_line_text(line: str) -> str:
    stripped = line.strip()
    match = re.fullmatch(r"(\*\*|__)(.+?)\1(:?)", stripped)
    if not match:
        return stripped
    suffix = ":" if match.group(3) else ""
    return f"{match.group(2).strip()}{suffix}"


def _strip_blockquote_prefix(line: str) -> str:
    return re.sub(r"^>+\s*", "", line.strip())


def _is_markdown_table_divider(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{1,}:?", cell or "")) for cell in cells)


def _is_markdown_table_start(lines: list[str], start_idx: int) -> bool:
    if start_idx + 1 >= len(lines):
        return False
    current = lines[start_idx].strip()
    next_line = lines[start_idx + 1].strip()
    return "|" in current and not _is_markdown_table_divider(current) and _is_markdown_table_divider(next_line)


def _split_markdown_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _find_chrome_binary() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if os.path.isabs(candidate):
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _build_html_document(markdown_text: str, *, title: str) -> str:
    safe_title = escape(title or "AI Co-Scientist Report")
    if MARKDOWN_AVAILABLE:
        body_html = markdown_lib.markdown(
            markdown_text,
            extensions=[
                "extra",
                "sane_lists",
                "fenced_code",
                "tables",
                "nl2br",
            ],
        )
    else:
        body_html = f"<pre>{escape(markdown_text)}</pre>"
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <meta name=\"color-scheme\" content=\"dark\" />\n"
        f"  <title>{safe_title}</title>\n"
        f"  <style>{HIGH_FIDELITY_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"  <article class=\"report\">{body_html}</article>\n"
        "</body>\n"
        "</html>\n"
    )


def _write_markdown_pdf_chrome(markdown: str, output_path: Path, *, title: str) -> str | None:
    chrome_binary = _find_chrome_binary()
    if not chrome_binary:
        return "High-fidelity PDF export unavailable: Chrome/Chromium binary not found."
    if not MARKDOWN_AVAILABLE:
        return "High-fidelity PDF export unavailable: markdown package is not installed."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html_document(markdown, title=title)

    with tempfile.TemporaryDirectory(prefix="co_scientist_pdf_") as tmpdir:
        html_path = Path(tmpdir) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        file_uri = html_path.resolve().as_uri()

        command_variants = [
            [
                chrome_binary,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={str(output_path)}",
                file_uri,
            ],
            [
                chrome_binary,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={str(output_path)}",
                file_uri,
            ],
        ]

        last_error = "Unknown Chromium rendering error."
        for command in command_variants:
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return None

            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            last_error = detail[:300]

        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return f"High-fidelity PDF export failed (Chromium): {last_error}"


def _indent_width(whitespace: str) -> int:
    return len((whitespace or "").expandtabs(4))


def _match_list_item(line: str):
    return re.match(
        r"^(?P<indent>\s*)(?:(?P<num>\d+)\.\s+|(?P<bullet>[-*+])\s+)(?P<text>.+)$",
        line,
    )


def _extract_list_items(lines: list[str], start_idx: int) -> tuple[list[dict], int]:
    items: list[dict] = []
    idx = start_idx
    ordered_counters: dict[int, int] = {}

    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip():
            look = idx + 1
            while look < len(lines) and not lines[look].strip():
                look += 1
            if look < len(lines) and _match_list_item(lines[look]):
                idx = look
                continue
            break

        match = _match_list_item(raw)
        if not match:
            break

        indent_width = _indent_width(match.group("indent") or "")
        level = max(0, indent_width // 4)

        for depth in list(ordered_counters.keys()):
            if depth > level:
                del ordered_counters[depth]

        text_parts = [match.group("text").strip()]
        idx += 1

        while idx < len(lines):
            candidate = lines[idx]
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                look = idx + 1
                while look < len(lines) and not lines[look].strip():
                    look += 1
                if look >= len(lines):
                    idx = look
                    break

                next_match = _match_list_item(lines[look])
                if next_match:
                    next_level = max(0, _indent_width(next_match.group("indent") or "") // 4)
                    if next_level <= level:
                        idx = look
                        break
                    idx = look
                    break

                next_indent = _indent_width(re.match(r"^(\s*)", lines[look]).group(1))
                if next_indent > indent_width:
                    text_parts.append(lines[look].strip())
                    idx = look + 1
                    continue
                idx = look
                break

            next_match = _match_list_item(candidate)
            if next_match:
                next_level = max(0, _indent_width(next_match.group("indent") or "") // 4)
                if next_level <= level:
                    break
                break

            continuation_indent = _indent_width(re.match(r"^(\s*)", candidate).group(1))
            if continuation_indent > indent_width:
                text_parts.append(candidate_stripped)
                idx += 1
                continue
            break

        if match.group("num") is not None:
            source_number = int(match.group("num"))
            if level not in ordered_counters:
                ordered_counters[level] = max(0, source_number - 1)
            ordered_counters[level] += 1
            marker = f"{ordered_counters[level]}."
            kind = "ordered"
        else:
            marker = "•"
            kind = "unordered"

        items.append(
            {
                "kind": kind,
                "level": level,
                "marker": marker,
                "text": " ".join(text_parts).strip(),
            }
        )

    return items, idx


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "report_body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=4,
            textColor=colors.HexColor("#cbd5e1"),
        ),
        "h1": ParagraphStyle(
            "report_h1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
            textColor=colors.HexColor("#ffffff"),
        ),
        "h2": ParagraphStyle(
            "report_h2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceAfter=8,
            textColor=colors.HexColor("#a5b4fc"),
        ),
        "h3": ParagraphStyle(
            "report_h3",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            spaceAfter=6,
            textColor=colors.HexColor("#818cf8"),
        ),
        "code": ParagraphStyle(
            "report_code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=9,
            leading=12,
            leftIndent=10,
            rightIndent=10,
            spaceAfter=6,
            textColor=colors.HexColor("#a5f3fc"),
            backColor=colors.HexColor("#020617"),
        ),
        "quote": ParagraphStyle(
            "report_quote",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=13,
            leftIndent=14,
            rightIndent=8,
            textColor=colors.HexColor("#c7d2fe"),
            backColor=colors.HexColor("#1e1b4b"),
            borderPadding=6,
            spaceAfter=6,
        ),
    }


def _flush_paragraph(buffer: list[str], story: list, styles: dict[str, ParagraphStyle]) -> None:
    if not buffer:
        return
    text = " ".join(item.strip() for item in buffer if item.strip())
    buffer.clear()
    if not text:
        return
    story.append(Paragraph(_format_inline_markdown(text), styles["body"]))
    story.append(Spacer(1, 4))


def _add_list_block(lines: list[str], start_idx: int, story: list, styles: dict[str, ParagraphStyle]) -> int:
    items, idx = _extract_list_items(lines, start_idx)
    for item in items:
        level = int(item.get("level", 0))
        list_style = ParagraphStyle(
            f"report_list_{item.get('kind', 'unordered')}_{level}",
            parent=styles["body"],
            leftIndent=16 + (level * 14),
            firstLineIndent=-12,
            spaceAfter=3,
        )
        marker = str(item.get("marker", "•"))
        text_markup = _format_inline_markdown(str(item.get("text", "")))
        story.append(Paragraph(f"{escape(marker)} {text_markup}", list_style))
    if items:
        story.append(Spacer(1, 4))
    return idx


def _add_code_block(lines: list[str], start_idx: int, story: list, styles: dict[str, ParagraphStyle]) -> int:
    idx = start_idx + 1
    code_lines: list[str] = []
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith("```"):
            idx += 1
            break
        code_lines.append(lines[idx].rstrip())
        idx += 1
    code_text = "<br/>".join(escape(item) if item else "&nbsp;" for item in code_lines) or "&nbsp;"
    story.append(Paragraph(code_text, styles["code"]))
    story.append(Spacer(1, 4))
    return idx


def _add_blockquote(lines: list[str], start_idx: int, story: list, styles: dict[str, ParagraphStyle]) -> int:
    idx = start_idx
    chunks: list[str] = []
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped.startswith(">"):
            break
        cleaned = _strip_blockquote_prefix(stripped)
        if cleaned:
            chunks.append(cleaned)
        idx += 1
    quote_text = " ".join(chunks).strip() or "&nbsp;"
    story.append(Paragraph(_format_inline_markdown(quote_text), styles["quote"]))
    story.append(Spacer(1, 4))
    return idx


def _add_table(lines: list[str], start_idx: int, story: list, styles: dict[str, ParagraphStyle]) -> int:
    header = _split_markdown_table_row(lines[start_idx])
    idx = start_idx + 2  # Skip divider row.
    body_rows: list[list[str]] = []
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or "|" not in stripped:
            break
        if _is_markdown_table_divider(stripped):
            idx += 1
            continue
        body_rows.append(_split_markdown_table_row(stripped))
        idx += 1

    rows = [header] + body_rows
    max_cols = max((len(row) for row in rows), default=0)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    table_data = [
        [Paragraph(_format_inline_markdown(cell), styles["body"]) for cell in row]
        for row in normalized_rows
    ]

    table = Table(table_data, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#1e293b")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#111827"), colors.HexColor("#0f172a")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6))
    return idx


def _markdown_story(markdown: str) -> list:
    styles = _styles()
    story: list = []
    paragraph_buffer: list[str] = []
    lines = markdown.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            _flush_paragraph(paragraph_buffer, story, styles)
            idx += 1
            continue

        if stripped.startswith("```"):
            _flush_paragraph(paragraph_buffer, story, styles)
            idx = _add_code_block(lines, idx, story, styles)
            continue

        if stripped.startswith(">"):
            _flush_paragraph(paragraph_buffer, story, styles)
            idx = _add_blockquote(lines, idx, story, styles)
            continue

        if _is_markdown_table_start(lines, idx):
            _flush_paragraph(paragraph_buffer, story, styles)
            idx = _add_table(lines, idx, story, styles)
            continue

        heading_level = _heading_level(stripped)
        if heading_level:
            _flush_paragraph(paragraph_buffer, story, styles)
            heading_text = stripped[heading_level + 1 :].strip()
            style_key = f"h{heading_level}"
            story.append(Paragraph(_format_inline_markdown(heading_text), styles[style_key]))
            story.append(Spacer(1, 4))
            idx += 1
            continue

        if _is_bold_only_line(stripped):
            _flush_paragraph(paragraph_buffer, story, styles)
            heading_text = _extract_bold_line_text(stripped)
            story.append(Paragraph(_format_inline_markdown(heading_text), styles["h3"]))
            story.append(Spacer(1, 4))
            idx += 1
            continue

        if _match_list_item(lines[idx]):
            _flush_paragraph(paragraph_buffer, story, styles)
            idx = _add_list_block(lines, idx, story, styles)
            continue

        paragraph_buffer.append(stripped)
        idx += 1

    _flush_paragraph(paragraph_buffer, story, styles)
    if not story:
        story.append(Paragraph("No content.", styles["body"]))
    return story


def _write_markdown_pdf_legacy(markdown: str, output_path: Path, *, title: str = "AI Co-Scientist Report") -> str | None:
    """
    Export markdown text to PDF using the legacy ReportLab renderer.

    Returns:
        None on success, otherwise an error message.
    """
    if not REPORTLAB_AVAILABLE:
        return "PDF export unavailable: reportlab is not installed."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip HTML anchor tags inserted for Chrome internal links — they render
    # as literal text in the ReportLab path.
    clean_markdown = re.sub(r'<a\s[^>]*></a>', '', (markdown or "").strip())
    try:
        def _dark_bg_canvas(canvas, doc):
            """Draw a dark background on every page for the legacy renderer."""
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#0b0f19"))
            canvas.rect(0, 0, LETTER[0], LETTER[1], fill=True, stroke=False)
            canvas.restoreState()

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            title=title,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )
        story = _markdown_story(clean_markdown)
        doc.build(story, onFirstPage=_dark_bg_canvas, onLaterPages=_dark_bg_canvas)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return f"PDF export failed ({type(exc).__name__}): {exc}"

    return None


def write_markdown_pdf(markdown: str, output_path: Path, *, title: str = "AI Co-Scientist Report") -> str | None:
    """
    Export markdown text to PDF.

    Backend order:
    1) High-fidelity HTML/CSS via headless Chrome/Chromium
    2) Legacy ReportLab fallback
    """
    normalized = (markdown or "").strip()

    high_fidelity_error = _write_markdown_pdf_chrome(normalized, output_path, title=title)
    if high_fidelity_error is None:
        return None

    legacy_error = _write_markdown_pdf_legacy(normalized, output_path, title=title)
    if legacy_error is None:
        return None

    return f"{high_fidelity_error} Fallback renderer error: {legacy_error}"


def write_markdown_with_pdf(
    markdown_path: Path,
    content: str,
    *,
    title: str,
    enable_pdf: bool = True,
) -> tuple[str, str | None]:
    """
    Persist markdown to disk and optionally emit a sidecar PDF.

    Returns:
        (pdf_path, error). pdf_path is empty when PDF generation is disabled or failed.
    """
    normalized = (content or "").rstrip() + "\n"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(normalized, encoding="utf-8")
    if not enable_pdf:
        return "", "PDF export disabled by --no-pdf."
    pdf_path = markdown_path.with_suffix(".pdf")
    pdf_error = write_markdown_pdf(normalized, pdf_path, title=title)
    if pdf_error:
        return "", pdf_error
    return str(pdf_path), None
