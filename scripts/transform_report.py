# pip install python-docx
"""
Transform the full MRANC system research report (Word) to IEEEtran LaTeX.

Processes the entire document: all datasets in tables (clinical SOTA, cross-dataset
summary), hero clinical figures, appendix decompositions (clinical, seed, deap,
artifact_benchmark), references, and static manuscript sections.

Default input (first found):
  outputs/reports/clinical/MRANC_Final_Research_Report.docx
  outputs/reports/MRANC_Final_Research_Report.docx

Default output (submission manuscript; never overwrites .docx):
  docs/manuscript/MRANC_Final_Research_Report.tex

Figure assets are copied to figures/{dataset}/ with relative includegraphics paths.
Back matter order: Conclusion, Data Availability, BibTeX references, appendices.
Uses docs/manuscript/references.bib (copied beside each .tex output).
A build copy is also written to outputs/reports/latex/.

  py scripts/transform_report.py
  py scripts/transform_report.py --input path/to/report.docx --output path/to/report.tex
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from runtime import setup_src_path

setup_src_path()

from paths import (
    CLINICAL_FINAL_REPORT_PATH,
    CRITICAL_FIGURES_DATASET_DIRS,
    FINAL_REPORT_PATH,
    LATEX_REPORT_DIR,
    MANUSCRIPT_DIR,
    MANUSCRIPT_TEX_PATH,
    PROJECT_ROOT,
    SYSTEM_LATEX_ASSETS_DIR,
    SYSTEM_REPORT_TEX_PATH,
    resolve_critical_figures_dir,
    resolve_manuscript_figures_dir,
)

REFERENCES_BIB_PATH = MANUSCRIPT_DIR / "references.bib"

DATASETS_DEFAULT = ("clinical", "seed", "deap", "artifact_benchmark")

SOTA_TABLE_CAPTION = "Quantitative Performance Comparison Profiles"

SECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^abstract$", re.I), "abstract", ""),
    (re.compile(r"^1\.\s*Introduction", re.I), "section", "Introduction and Related Work"),
    (re.compile(r"^2\.\s*Methodology", re.I), "section", "Methodology"),
    (re.compile(r"^3\.\s*Results", re.I), "section", "Results"),
    (re.compile(r"^4\.\s*Discussion", re.I), "section", "Discussion and Limitations"),
    (re.compile(r"^5\.\s*Conclusion", re.I), "section", "Conclusion"),
    (re.compile(r"^Data Availability Statement$", re.I), "data_availability", ""),
    (re.compile(r"^references$", re.I), "references", ""),
    (
        re.compile(r"^Appendix:\s*Supplementary Multi-Channel Decompositions", re.I),
        "appendix_start",
        "Supplementary Multi-Channel Decompositions",
    ),
]

SECTION_LABELS: dict[str, str] = {
    "Introduction and Related Work": "sec:intro",
    "Methodology": "sec:methodology",
    "Results": "sec:results",
    "Discussion and Limitations": "sec:discussion",
    "Conclusion": "sec:conclusion",
}

BRACKET_CITE_PATTERN = re.compile(r"\[(\d+)\]")
URL_PATTERN = re.compile(r"https?://[^\s)]+")

SUBSECTION_PATTERN = re.compile(r"^(\d+\.\d+)\s+(.+)$")
APPENDIX_LETTER_PATTERN = re.compile(r"^Appendix\s+([A-Z])\.\s+(.+)$", re.I)
FIGURE_CAPTION_PATTERN = re.compile(r"^Figure\s+(\d+)\s*:\s*(.+)$", re.I | re.DOTALL)
TABLE_HEADING_PATTERN = re.compile(r"^Table\s+[IVXLC\d]+\.", re.I)
REFERENCE_LINE_PATTERN = re.compile(r"^\[(\d+)\]\s*(.+)$", re.S)
SOURCE_FILE_PATTERN = re.compile(r"Source:\s*(\S+\.png)", re.I)
REGENERATE_HINT_PATTERN = re.compile(r"^Regenerate figures and metrics:", re.I)

BIBITEM_KEYS = {
    "1": "makeig1997",
    "2": "zhang2022eegdenoisenet",
    "3": "shoeb2010chb",
}

DEFAULT_MANUSCRIPT_TITLE = (
    "Multi-Resolution Attention-Guided Neural Cleaner (MRANC) "
    "for Real-World Clinical EEG Denoising"
)

LATEX_PREAMBLE = (
    r"""\documentclass[journal,twocolumn]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{url}

\title{"""
    + DEFAULT_MANUSCRIPT_TITLE
    + r"""}
\author{%
\IEEEauthorblockN{Ghadeer Mostafa%
\thanks{Manuscript and software \copyright{} 2026 Ghadeer Mostafa. Licensed under CC BY-NC-SA 4.0.}}%
\IEEEauthorblockA{MRANC Research Project\\
GitHub: \url{https://github.com/GhadeerMostafa/EEG-MRANC}}%
}

\begin{document}
\maketitle
"""
)

LATEX_POSTAMBLE = r"""
\end{document}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform full MRANC system Word report to IEEEtran LaTeX"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input .docx (default: clinical or root MRANC_Final_Research_Report.docx)",
    )
    parser.add_argument("--output", type=str, default=str(MANUSCRIPT_TEX_PATH))
    parser.add_argument("--assets-dir", type=str, default=str(SYSTEM_LATEX_ASSETS_DIR))
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_default_input() -> Path | None:
    for candidate in (CLINICAL_FINAL_REPORT_PATH, FINAL_REPORT_PATH):
        if candidate.is_file():
            return candidate
    return None


def validate_output_path(input_path: Path, output_path: Path) -> str | None:
    """Return error message if output would clobber run_report Word artifact."""
    if output_path.suffix.lower() == ".docx":
        return "Output must be .tex, not .docx (run_report owns the Word file)."
    try:
        if output_path.resolve() == input_path.resolve():
            return "Output path must differ from input .docx."
    except OSError:
        pass
    protected = (
        CLINICAL_FINAL_REPORT_PATH.resolve(),
        FINAL_REPORT_PATH.resolve(),
    )
    try:
        out_resolved = output_path.resolve()
        for protected_path in protected:
            if out_resolved == protected_path:
                return (
                    f"Output cannot be {protected_path.name}; use "
                    f"{LATEX_REPORT_DIR / 'MRANC_Final_Research_Report.tex'}."
                )
    except OSError:
        pass
    return None


def escape_latex(text: str) -> str:
    if not text:
        return ""
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def normalize_unicode_for_latex(text: str) -> str:
    return (
        text.replace("\u2299", r"$\odot$")
        .replace("\u2014", "---")
        .replace("\u2013", "--")
        .replace("\u201c", "``")
        .replace("\u201d", "''")
    )


def prepare_latex_text(text: str) -> str:
    """Escape body text while converting [n] cites and bare URLs for LaTeX."""
    if not text:
        return ""
    text = normalize_unicode_for_latex(text)
    cite_tokens: list[tuple[str, str]] = []
    url_tokens: list[tuple[str, str]] = []

    def stash_cite(match: re.Match[str]) -> str:
        key = BIBITEM_KEYS.get(match.group(1), f"ref{match.group(1)}")
        token = f"@@CITE{len(cite_tokens)}@@"
        cite_tokens.append((token, rf"\cite{{{key}}}"))
        return token

    def stash_url(match: re.Match[str]) -> str:
        token = f"@@URL{len(url_tokens)}@@"
        url_tokens.append((token, match.group(0).rstrip(".,;")))
        return token

    staged = BRACKET_CITE_PATTERN.sub(stash_cite, text)
    staged = URL_PATTERN.sub(stash_url, staged)
    escaped = escape_latex(staged)
    for token, cmd in cite_tokens:
        escaped = escaped.replace(token, cmd)
    for token, url in url_tokens:
        escaped = escaped.replace(token, rf"\url{{{url}}}")
    return escaped


def emit_section(title: str) -> str:
    label = SECTION_LABELS.get(title)
    if label:
        return rf"\section{{{title}}}\label{{{label}}}"
    return rf"\section{{{title}}}"


def emit_bibliography_block() -> str:
    return (
        "% --- References Section ---\n"
        r"\bibliographystyle{IEEEtran}"
        + "\n"
        + r"\bibliography{references}"
        + "\n"
    )


def emit_data_availability_heading() -> str:
    return "% --- Data Availability Section ---\n" + r"\section*{Data Availability Statement}"


def paragraph_text(paragraph: Paragraph) -> str:
    parts: list[str] = []
    for run in paragraph.runs:
        parts.append(run.text)
    return "".join(parts).strip()


def is_italic_caption(paragraph: Paragraph) -> bool:
    if not paragraph.runs:
        return False
    styled = [run for run in paragraph.runs if run.text.strip()]
    if not styled:
        return False
    return all(run.italic for run in styled)


def iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:\\")
SUPPLEMENTARY_INVENTORY_PATTERN = re.compile(
    r"^Supplementary figure inventory:", re.I
)


PIPELINE_JARGON_PATTERN = re.compile(
    r"^(MRANC validation windows:|MRANC-only validation metrics|"
    r"Checkpoint:|evaluate_baseline_|live evaluation JSON|"
    r"critical_figures/\{\{dataset\}\})",
    re.I,
)


def should_skip_body_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if WINDOWS_PATH_PATTERN.search(stripped):
        return True
    if SUPPLEMENTARY_INVENTORY_PATTERN.match(stripped):
        return True
    if REGENERATE_HINT_PATTERN.match(stripped):
        return True
    if stripped.startswith("Full pipeline:"):
        return True
    if stripped.startswith("Metrics/figures only:"):
        return True
    if PIPELINE_JARGON_PATTERN.search(stripped):
        return True
    if "The following 30 figure(s)" in stripped and "{{dataset}}" in stripped:
        return True
    return False


def classify_heading(text: str, style_name: str = "") -> tuple[str, str] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    for pattern, kind, title in SECTION_PATTERNS:
        if pattern.match(cleaned):
            if kind == "section" and not title:
                return ("section", escape_latex(cleaned))
            if kind == "appendix_start":
                return ("appendix_start", escape_latex(title))
            return (kind, title)
    sub = SUBSECTION_PATTERN.match(cleaned)
    if sub:
        return ("subsection", escape_latex(sub.group(2).strip()))
    appendix_sub = APPENDIX_LETTER_PATTERN.match(cleaned)
    if appendix_sub:
        letter = appendix_sub.group(1).upper()
        label = appendix_sub.group(2).strip()
        return ("appendix_subsection", escape_latex(f"{letter}. {label}"))
    if TABLE_HEADING_PATTERN.match(cleaned):
        return ("table_heading", escape_latex(cleaned))
    if style_name == "Heading 2" and cleaned.lower().startswith("table "):
        return ("table_heading", escape_latex(cleaned))
    return None


def table_header_row(table: Table) -> list[str]:
    if not table.rows:
        return []
    return [cell.text.strip() for cell in table.rows[0].cells]


def is_sota_benchmark_table(table: Table) -> bool:
    header = " ".join(table_header_row(table)).lower()
    if "evaluation metric" in header and "mranc" in header:
        return True
    for row in table.rows[1:]:
        first = row.cells[0].text.strip().lower() if row.cells else ""
        if "reconstruction mse" in first and "clinical" in first:
            return True
    return False


def is_cross_dataset_summary_table(table: Table) -> bool:
    header = " ".join(table_header_row(table)).lower()
    return "dataset" in header and "recon" in header


def table_to_tabular(table: Table) -> str:
    lines: list[str] = []
    col_count = len(table.rows[0].cells) if table.rows else 0
    align = "l" + "c" * max(col_count - 1, 0)
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\toprule")
    for row_index, row in enumerate(table.rows):
        cells = [prepare_latex_text(cell.text.strip()) for cell in row.cells]
        line = " & ".join(cells) + r" \\"
        lines.append(line)
        if row_index == 0:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _table_label_for_caption(caption: str) -> str:
    lower = caption.lower()
    if "cross-dataset" in lower:
        return "tab:cross-dataset"
    if "comparison" in lower or "performance" in lower:
        return "tab:clinical-sota"
    slug = re.sub(r"[^a-z0-9]+", "-", lower).strip("-")[:40]
    return f"tab:{slug or 'data'}"


def emit_table_star(table: Table, caption: str) -> str:
    tabular = table_to_tabular(table)
    cap = escape_latex(caption)
    label = _table_label_for_caption(caption)
    return (
        r"\begin{table*}[!t]"
        + "\n"
        + r"\caption{"
        + cap
        + r"}\label{"
        + label
        + "}"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\small"
        + "\n"
        + tabular
        + "\n"
        + r"\end{table*}"
        + "\n"
    )


def extract_paragraph_image(
    paragraph: Paragraph,
    document: Document,
    assets_dir: Path,
    counter: list[int],
) -> Path | None:
    blips = paragraph._element.xpath(".//a:blip")
    if not blips:
        return None
    embed = blips[0].get(qn("r:embed"))
    if not embed:
        return None
    try:
        part = document.part.related_parts[embed]
    except KeyError:
        return None
    counter[0] += 1
    ext = "png"
    if hasattr(part, "content_type"):
        if "jpeg" in part.content_type or "jpg" in part.content_type:
            ext = "jpg"
        elif "png" in part.content_type:
            ext = "png"
    filename = f"fig_{counter[0]:03d}.{ext}"
    out_path = assets_dir / filename
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(part.blob)
    return out_path


def _dataset_for_figure(path: Path) -> str:
    for dataset, ds_dir in CRITICAL_FIGURES_DATASET_DIRS.items():
        try:
            path.relative_to(ds_dir)
            return dataset
        except ValueError:
            continue
    parent_name = path.parent.name
    if parent_name in DATASETS_DEFAULT:
        return parent_name
    return "misc"


def _sync_figure_for_submission(source: Path) -> Path:
    dataset = _dataset_for_figure(source)
    dest_dir = resolve_manuscript_figures_dir() / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    if not dest.is_file() or source.stat().st_mtime_ns > dest.stat().st_mtime_ns:
        shutil.copy2(source, dest)
    return dest


def resolve_figure_path(
    extracted: Path,
    caption_text: str,
    tex_dir: Path,
) -> str:
    source_path = extracted
    source_match = SOURCE_FILE_PATTERN.search(caption_text)
    if source_match:
        name = source_match.group(1)
        search_roots: list[Path] = []
        for dataset in DATASETS_DEFAULT:
            ds_dir = CRITICAL_FIGURES_DATASET_DIRS.get(dataset)
            if ds_dir is not None:
                search_roots.append(ds_dir)
        search_roots.extend(
            [
                resolve_critical_figures_dir(),
                PROJECT_ROOT / "artifacts" / "figures" / "legacy",
                PROJECT_ROOT / "outputs" / "figures",
                resolve_manuscript_figures_dir(),
            ]
        )
        for base in search_roots:
            if not base.is_dir():
                continue
            direct = base / name
            if direct.is_file():
                source_path = direct
                break
            found = list(base.rglob(name))
            if found:
                source_path = found[0]
                break
    synced = _sync_figure_for_submission(source_path)
    return _path_for_latex(synced, tex_dir)


def _path_for_latex(path: Path, tex_dir: Path) -> str:
    try:
        rel = path.relative_to(tex_dir)
        return str(rel).replace("\\", "/")
    except ValueError:
        rel_os = os.path.relpath(path, tex_dir)
        return rel_os.replace("\\", "/")


def _clean_figure_caption(caption: str) -> str:
    cap = FIGURE_CAPTION_PATTERN.sub(r"\2", caption.strip())
    cap = SOURCE_FILE_PATTERN.sub("", cap).strip()
    cap = re.sub(r"\s{2,}", " ", cap)
    return cap.rstrip(" .")


HERO_FIGURE_LABELS: dict[str, str] = {
    "clinical_eval_window0_Fp1_decomposition": "fig:clinical-fp1-w0",
    "clinical_eval_window0_Cz_decomposition": "fig:clinical-cz-w0",
    # Legacy filenames from older runs:
    "tuh_eval_window0_Fp1_decomposition": "fig:clinical-fp1-w0",
    "tuh_eval_window0_Cz_decomposition": "fig:clinical-cz-w0",
}


def _figure_label_from_path(image_path: str) -> str:
    stem = Path(image_path).stem
    if stem in HERO_FIGURE_LABELS:
        return HERO_FIGURE_LABELS[stem]
    return f"fig:{stem.replace('-', '_')}"


def emit_figure_star(image_path: str, caption: str) -> str:
    cap = escape_latex(_clean_figure_caption(caption))
    label = _figure_label_from_path(image_path)
    img = image_path.replace("\\", "/")
    return (
        r"\begin{figure*}[!t]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\includegraphics[width=\linewidth,keepaspectratio]{"
        + img
        + "}"
        + "\n"
        + r"\caption{"
        + cap
        + r"}\label{"
        + label
        + "}"
        + "\n"
        + r"\end{figure*}"
        + "\n"
    )


def parse_reference_line(text: str) -> tuple[str, str] | None:
    match = REFERENCE_LINE_PATTERN.match(text.strip())
    if not match:
        return None
    num = match.group(1)
    body = match.group(2).strip()
    key = BIBITEM_KEYS.get(num, f"ref{num}")
    return key, body


def sanitize_bibliography_text(text: str) -> str:
    out = text.replace("'", "''")
    return out


def references_to_bibliography(lines: list[str]) -> str:
    """Legacy inline bibliography (unused when references.bib is present)."""
    items: list[str] = []
    for line in lines:
        parsed = parse_reference_line(line)
        if parsed is None:
            continue
        key, body = parsed
        escaped = escape_latex(sanitize_bibliography_text(body))
        items.append(r"\bibitem{" + key + "} " + escaped)
    if not items:
        return ""
    inner = "\n".join(items)
    return (
        r"\begin{thebibliography}{10}"
        + "\n"
        + inner
        + "\n"
        + r"\end{thebibliography}"
        + "\n"
    )


def detect_title(paragraphs: list[Paragraph]) -> str | None:
    for para in paragraphs[:12]:
        text = paragraph_text(para)
        if not text:
            continue
        lower = text.lower()
        if lower == "abstract":
            break
        if "mranc" in lower and len(text) > 40:
            return text
        if para.style and para.style.name == "Title":
            return text
    return None


def table_caption_for_block(
    table: Table,
    pending_table_caption: str,
) -> str:
    if is_sota_benchmark_table(table):
        return SOTA_TABLE_CAPTION
    if is_cross_dataset_summary_table(table):
        if pending_table_caption and "cross-dataset" in pending_table_caption.lower():
            return pending_table_caption
        return "MRANC Cross-Dataset Summary"
    if pending_table_caption:
        return pending_table_caption
    return "Performance Comparison Table"


def transform_document(
    document: Document,
    output_tex: Path,
    assets_dir: Path,
) -> dict[str, int]:
    blocks = list(iter_block_items(document))
    paragraphs_only = [b for b in blocks if isinstance(b, Paragraph)]
    title_text = detect_title(paragraphs_only)

    body_lines: list[str] = []
    references_buffer: list[str] = []
    pending_table_caption = SOTA_TABLE_CAPTION
    pending_figure_image: Path | None = None

    in_abstract = False
    in_references = False
    in_data_availability = False
    bibliography_emitted = False
    appendix_mode = False
    title_emitted = False
    image_counter = [0]
    stats = {"tables": 0, "figures": 0, "appendix_subsections": 0}

    tex_dir = output_tex.parent

    def close_abstract() -> None:
        nonlocal in_abstract
        if in_abstract:
            body_lines.append(r"\end{abstract}")
            body_lines.append("")
            in_abstract = False

    def emit_bibliography_once() -> None:
        nonlocal bibliography_emitted, in_references
        if bibliography_emitted:
            in_references = False
            references_buffer.clear()
            return
        if REFERENCES_BIB_PATH.is_file():
            body_lines.append(emit_bibliography_block())
        elif references_buffer:
            body_lines.append(references_to_bibliography(references_buffer))
        bibliography_emitted = True
        references_buffer.clear()
        in_references = False

    def flush_references() -> None:
        emit_bibliography_once()

    for block in blocks:
        if isinstance(block, Table):
            if in_references:
                continue
            close_abstract()
            caption = table_caption_for_block(block, pending_table_caption)
            pending_table_caption = SOTA_TABLE_CAPTION
            body_lines.append(emit_table_star(block, caption))
            stats["tables"] += 1
            continue

        paragraph: Paragraph = block
        text = paragraph_text(paragraph)
        has_image = bool(paragraph._element.xpath(".//a:blip"))
        if not text and not has_image:
            continue

        image_path = extract_paragraph_image(
            paragraph, document, assets_dir, image_counter
        )
        if image_path is not None:
            pending_figure_image = image_path
            continue

        style_name = paragraph.style.name if paragraph.style else ""
        heading = classify_heading(text, style_name)
        if heading is not None:
            kind, title = heading
            if kind == "abstract":
                close_abstract()
                in_abstract = True
                body_lines.append(r"\begin{abstract}")
                continue
            if kind == "data_availability":
                close_abstract()
                in_data_availability = True
                body_lines.append(emit_data_availability_heading())
                body_lines.append("")
                continue
            if kind == "references":
                close_abstract()
                in_data_availability = False
                emit_bibliography_once()
                in_references = True
                continue
            if kind == "appendix_start":
                close_abstract()
                in_data_availability = False
                flush_references()
                appendix_mode = True
                body_lines.append("% --- Appendices Section ---")
                body_lines.append(r"\appendices")
                body_lines.append(
                    r"\section{" + title + r"}\label{sec:appendix}"
                )
                body_lines.append(
                    "The following figures supplement the main-text hero panels "
                    r"(Figs.~\ref{fig:clinical-fp1-w0} and~\ref{fig:clinical-cz-w0})."
                )
                body_lines.append("")
                continue
            if kind == "section":
                close_abstract()
                in_data_availability = False
                section_title = title if title else escape_latex(text)
                body_lines.append(emit_section(section_title))
                continue
            if kind == "subsection":
                close_abstract()
                in_data_availability = False
                body_lines.append(r"\subsection{" + title + "}")
                continue
            if kind == "appendix_subsection":
                appendix_title = title
                if ". " in appendix_title:
                    appendix_title = appendix_title.split(". ", 1)[1]
                body_lines.append(r"\subsection{" + appendix_title + "}")
                stats["appendix_subsections"] += 1
                continue
            if kind == "table_heading":
                pending_table_caption = text
                continue

        if in_references:
            references_buffer.append(text)
            continue

        if in_data_availability:
            body_lines.append(prepare_latex_text(text))
            body_lines.append("")
            continue

        cap_match = FIGURE_CAPTION_PATTERN.match(text)
        if cap_match and (is_italic_caption(paragraph) or cap_match):
            fig_num = cap_match.group(1)
            cap_body = cap_match.group(2).strip()
            full_caption = f"Figure {fig_num}: {cap_body}"
            if pending_figure_image is not None:
                rel_path = resolve_figure_path(
                    pending_figure_image, full_caption, tex_dir
                )
                body_lines.append(emit_figure_star(rel_path, full_caption))
                pending_figure_image = None
                stats["figures"] += 1
            else:
                body_lines.append(
                    r"% Figure "
                    + fig_num
                    + " caption without embedded image: "
                    + escape_latex(full_caption)
                )
            continue

        if in_abstract:
            body_lines.append(prepare_latex_text(text))
            body_lines.append("")
            continue

        if not title_emitted and title_text and text == title_text:
            title_emitted = True
            continue

        if should_skip_body_paragraph(text):
            continue

        body_lines.append(prepare_latex_text(text))
        body_lines.append("")

    close_abstract()
    if in_references or references_buffer:
        emit_bibliography_once()

    if pending_figure_image is not None:
        rel_path = resolve_figure_path(pending_figure_image, "", tex_dir)
        body_lines.append(
            emit_figure_star(rel_path, "Embedded figure from source document.")
        )
        stats["figures"] += 1

    preamble = LATEX_PREAMBLE
    if title_text:
        escaped_title = escape_latex(title_text)
        preamble = preamble.replace(
            r"\title{" + DEFAULT_MANUSCRIPT_TITLE + "}",
            r"\title{" + escaped_title + "}",
        )

    content = preamble + "\n".join(body_lines) + "\n" + LATEX_POSTAMBLE
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(content, encoding="utf-8", newline="\n")
    stats["appendix_mode"] = int(appendix_mode)
    return stats


def main() -> int:
    args = parse_args()
    if args.input:
        input_path = resolve_path(Path(args.input))
    else:
        resolved = resolve_default_input()
        if resolved is None:
            print(
                "ERROR: no MRANC_Final_Research_Report.docx found.",
                file=sys.stderr,
            )
            print(
                "Generate the full system report first:",
                file=sys.stderr,
            )
            print("  py scripts/run_report.py", file=sys.stderr)
            return 1
        input_path = resolved

    output_path = resolve_path(Path(args.output))
    assets_dir = resolve_path(Path(args.assets_dir))

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    path_error = validate_output_path(input_path, output_path)
    if path_error:
        print(f"ERROR: {path_error}", file=sys.stderr)
        return 1

    try:
        document = Document(str(input_path))
        stats = transform_document(document, output_path, assets_dir)
    except Exception as exc:
        print(f"ERROR: transformation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {output_path.resolve()}")
    print(f"  Source docx: {input_path.resolve()}")
    print(f"  Figure exports: {resolve_manuscript_figures_dir().resolve()}")
    print(f"  Assets: {assets_dir.resolve()}")
    print(f"  Tables: {stats.get('tables', 0)}")
    print(f"  Figures: {stats.get('figures', 0)}")
    print(f"  Appendix subsections: {stats.get('appendix_subsections', 0)}")

    if output_path.resolve() != SYSTEM_REPORT_TEX_PATH.resolve():
        SYSTEM_REPORT_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, SYSTEM_REPORT_TEX_PATH)
        print(f"  Build copy: {SYSTEM_REPORT_TEX_PATH.resolve()}")

    for tex_target in {output_path.resolve(), SYSTEM_REPORT_TEX_PATH.resolve()}:
        bib_target = tex_target.parent / "references.bib"
        if REFERENCES_BIB_PATH.is_file() and bib_target != REFERENCES_BIB_PATH.resolve():
            shutil.copy2(REFERENCES_BIB_PATH, bib_target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
