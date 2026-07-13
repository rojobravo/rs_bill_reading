"""Standard PDF invoice extractor using pypdf and pdfplumber."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pypdf


@dataclass
class InvoiceData:
    """Structured representation of extracted invoice data."""

    raw_text: str = ""
    pages: list[str] = field(default_factory=list)
    tables: list[list[list[str | None]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Common invoice fields extracted via regex heuristics
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    vendor_name: str | None = None
    total_amount: str | None = None
    tax_amount: str | None = None
    subtotal: str | None = None


class StandardInvoiceExtractor:
    """
    Extracts invoice information from PDF files using two mainstream libraries:

    - **pypdf**: lightweight, fast text extraction and metadata reading.
    - **pdfplumber**: higher-fidelity extraction with table and layout support.

    Usage::

        extractor = StandardInvoiceExtractor("invoice.pdf")
        data = extractor.extract()
        print(data.invoice_number)
    """

    # Simple heuristic patterns — adjust per your invoice format
    _PATTERNS: dict[str, str] = {
        "invoice_number": r"(?i)invoice\s*(?:no\.?|number|#)[:\s]*([A-Z0-9\-\/]+)",
        "invoice_date": r"(?i)(?:invoice\s+date|date\s+of\s+invoice)[:\s]*([\d]{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        "due_date": r"(?i)(?:due\s+date|payment\s+due)[:\s]*([\d]{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        # Require a colon after bare "total" to avoid matching table column headers
        "total_amount": r"(?i)(?:total\s+(?:amount\s+)?due|amount\s+due|grand\s+total|(?<![a-z])total\s*:)[:\s]*\$?([\d,]+\.?\d*)",
        # Allow optional rate annotation, e.g. "VAT (23%):"
        "tax_amount": r"(?i)(?:tax|vat|gst)(?:\s*\([^)]+\))?\s*:\s*\$?([\d,]+\.?\d*)",
        "subtotal": r"(?i)(?:subtotal|sub-total|sub\s+total)\s*:\s*\$?([\d,]+\.?\d*)",
        # First line that looks like a company name (ends with a legal suffix)
        "vendor_name": r"(?m)^([A-Z][A-Za-z0-9 &\.,]+(?:Ltd\.?|LLC|Inc\.?|Corp\.?|GmbH|Solutions|Group|Services|Co\.?))",
    }

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> InvoiceData:
        """Run the full extraction pipeline and return an :class:`InvoiceData`."""
        data = InvoiceData()
        self._extract_with_pypdf(data)
        self._extract_with_pdfplumber(data)
        self._apply_heuristics(data)
        return data

    def extract_text_pypdf(self) -> str:
        """Return raw concatenated text extracted by pypdf."""
        pages: list[str] = []
        with pypdf.PdfReader(self.pdf_path) as reader:
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def extract_text_pdfplumber(self) -> tuple[str, list[list[list[str | None]]]]:
        """Return (full_text, tables) extracted by pdfplumber."""
        all_text: list[str] = []
        all_tables: list[list[list[str | None]]] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                all_text.append(page.extract_text() or "")
                all_tables.extend(page.extract_tables())
        return "\n".join(all_text), all_tables

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_with_pypdf(self, data: InvoiceData) -> None:
        with pypdf.PdfReader(self.pdf_path) as reader:
            data.metadata = dict(reader.metadata or {})
            data.pages = [page.extract_text() or "" for page in reader.pages]
        data.raw_text = "\n".join(data.pages)

    def _extract_with_pdfplumber(self, data: InvoiceData) -> None:
        """Overwrite raw_text with pdfplumber output (usually higher quality)."""
        with pdfplumber.open(self.pdf_path) as pdf:
            plumber_pages: list[str] = []
            for page in pdf.pages:
                plumber_pages.append(page.extract_text() or "")
                data.tables.extend(page.extract_tables())
            # Replace raw_text only if pdfplumber produced more content
            plumber_text = "\n".join(plumber_pages)
            if len(plumber_text) > len(data.raw_text):
                data.raw_text = plumber_text

    def _apply_heuristics(self, data: InvoiceData) -> None:
        """Apply regex patterns to populate structured fields."""
        for field_name, pattern in self._PATTERNS.items():
            match = re.search(pattern, data.raw_text)
            if match:
                setattr(data, field_name, match.group(1).strip())
