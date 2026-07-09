#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import HTTPError

PAPERS_DIR = Path(__file__).parent / "papers"


def fetch_pdf_text(arxiv_id: str) -> str:
    """Download PDF from arxiv and extract text with pymupdf (fitz)."""
    import pymupdf  # pip install pymupdf

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  [pdf] Downloading {pdf_url}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        req = Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
        try:
            with urlopen(req, timeout=60) as resp:
                tmp.write(resp.read())
        except HTTPError as e:
            return f"ERROR: HTTP {e.code} fetching {pdf_url}"

    try:
        doc = pymupdf.open(tmp_path)
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return "\n\n".join(pages)


def fetch_html_text(arxiv_id: str) -> str:
    """Fetch HTML version from arxiv and extract with trafilatura."""
    import trafilatura  # pip install trafilatura

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    print(f"  [html] Fetching {html_url}")

    req = Request(html_url, headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
    try:
        with urlopen(req, timeout=60) as resp:
            raw_html = resp.read()
    except HTTPError as e:
        return f"ERROR: HTTP {e.code} fetching {html_url}"

    text = trafilatura.extract(
        raw_html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=False,
        favor_recall=True,
    )
    return text or "ERROR: trafilatura returned empty extraction"


def process_paper(arxiv_id: str, method: str) -> None:
    """Fetch a single paper using the specified method."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return

    print(f"[{arxiv_id}] method={method}")

    if method in ("pdf", "both"):
        text = fetch_pdf_text(arxiv_id)
        out_path = PAPERS_DIR / f"{arxiv_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path} ({len(text):,} chars)")

    if method in ("html", "both"):
        text = fetch_html_text(arxiv_id)
        out_path = PAPERS_DIR / f"{arxiv_id}.html.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path} ({len(text):,} chars)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    method = sys.argv[1].lower()
    if method not in ("pdf", "html", "both"):
        print(f"Unknown method: {method}. Use: pdf, html, both")
        sys.exit(1)

    paper_ids = sys.argv[2:]
    print(f"Fetching {len(paper_ids)} paper(s) with method={method}\n")

    for pid in paper_ids:
        try:
            process_paper(pid, method)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
