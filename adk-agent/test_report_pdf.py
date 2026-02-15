from pathlib import Path

from report_pdf import (
    _build_html_document,
    _extract_list_items,
    _format_inline_markdown,
    _is_bold_only_line,
    _is_markdown_table_divider,
    _is_markdown_table_start,
    _strip_blockquote_prefix,
    write_markdown_pdf,
)
from run_acceptance_demo import _write_markdown_with_pdf


def test_write_markdown_pdf_generates_file_or_reports_missing_dependency(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    markdown = """
# Sample Report

## Findings
- Item one
- Item two

## Evidence
- PMID:12345678
""".strip()

    error = write_markdown_pdf(markdown, pdf_path, title="Sample")
    if error:
        assert "reportlab" in error.lower()
        assert not pdf_path.exists()
    else:
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0


def test_write_markdown_with_pdf_always_writes_markdown_file(tmp_path: Path):
    markdown_path = tmp_path / "report.md"
    pdf_file, pdf_error = _write_markdown_with_pdf(
        markdown_path,
        "## Report\n\n- evidence item",
        title="Integration Test",
    )

    assert markdown_path.exists()
    assert markdown_path.read_text(encoding="utf-8") == "## Report\n\n- evidence item\n"

    if pdf_error:
        assert pdf_file == ""
    else:
        assert pdf_file
        assert Path(pdf_file).exists()


def test_write_markdown_with_pdf_can_be_disabled(tmp_path: Path):
    markdown_path = tmp_path / "disabled.md"
    pdf_file, pdf_error = _write_markdown_with_pdf(
        markdown_path,
        "## Disabled PDF",
        title="Disabled",
        enable_pdf=False,
    )

    assert markdown_path.exists()
    assert markdown_path.read_text(encoding="utf-8") == "## Disabled PDF\n"
    assert pdf_file == ""
    assert pdf_error == "PDF export disabled by --no-pdf."


def test_inline_markdown_conversion_emits_reportlab_markup():
    line = "**Bold** and *italic* plus `code` and [OpenAlex](https://openalex.org)."
    rendered = _format_inline_markdown(line)
    assert "<b>Bold</b>" in rendered
    assert "<i>italic</i>" in rendered
    assert '<font name="Courier">code</font>' in rendered
    assert '<link href="https://openalex.org">OpenAlex</link>' in rendered


def test_inline_markdown_does_not_leak_placeholders_for_bullet_like_text():
    rendered = _format_inline_markdown("*   **Contributions:** 25 matched publications.")
    assert "@@MD" not in rendered
    assert "<b>Contributions:</b>" in rendered


def test_bold_only_line_detection():
    assert _is_bold_only_line("**Top Targets:**")
    assert _is_bold_only_line("__Limitations__")
    assert not _is_bold_only_line("**Top Targets:** some inline text")


def test_table_detection_helpers():
    assert _is_markdown_table_divider("| --- | :---: | ---: |")
    assert not _is_markdown_table_divider("| name | score |")

    lines = [
        "| Name | Score |",
        "| --- | ---: |",
        "| ESR1 | 95 |",
    ]
    assert _is_markdown_table_start(lines, 0)
    assert not _is_markdown_table_start(lines, 1)


def test_blockquote_prefix_strip():
    assert _strip_blockquote_prefix("> quoted line") == "quoted line"
    assert _strip_blockquote_prefix(">> nested quote") == "nested quote"


def test_write_markdown_pdf_with_table_and_blockquote(tmp_path: Path):
    pdf_path = tmp_path / "table_quote.pdf"
    markdown = """
## Summary
> This is a quoted insight with **bold emphasis**.

| Metric | Value |
| --- | ---: |
| Evidence IDs | 12 |
| Tool calls | 8 |
""".strip()

    error = write_markdown_pdf(markdown, pdf_path, title="Table+Quote")
    if error:
        assert "reportlab" in error.lower()
        assert not pdf_path.exists()
    else:
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0


def test_list_extraction_preserves_numbering_and_levels():
    lines = [
        "1. **Aarsland D**",
        "    * **Contributions:** 25 matched publications.",
        "    * **Key Publications:** PMID:32936544",
        "",
        "2. **Corlett PR**",
        "    * **Contributions:** 25 matched publications.",
        "",
        "3. **West ML**",
    ]
    items, next_idx = _extract_list_items(lines, 0)
    assert next_idx == len(lines)
    assert [item["marker"] for item in items] == ["1.", "•", "•", "2.", "•", "3."]
    assert [item["level"] for item in items] == [0, 1, 1, 0, 1, 0]


def test_list_extraction_sequential_when_source_uses_all_ones():
    lines = [
        "1. first",
        "1. second",
        "1. third",
    ]
    items, _ = _extract_list_items(lines, 0)
    assert [item["marker"] for item in items] == ["1.", "2.", "3."]


def test_high_fidelity_html_contains_semantic_markdown_elements():
    markdown = """
## Header
> quoted context

1. first
2. second

| A | B |
| --- | --- |
| 1 | 2 |
""".strip()
    html = _build_html_document(markdown, title="Fidelity")
    assert "<h2>Header</h2>" in html
    assert "<blockquote>" in html
    assert "<ol>" in html
    assert "<table>" in html
