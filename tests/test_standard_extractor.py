"""Tests for StandardInvoiceExtractor."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rs_bill_reading.standard_extractor import InvoiceData, StandardInvoiceExtractor


SAMPLE_TEXT = textwrap.dedent(
    """\
    ACME Corp
    Invoice No. INV-2024-001
    Invoice Date: 01/15/2024
    Due Date: 02/15/2024

    Subtotal: $1,000.00
    Tax: $100.00
    Total: $1,100.00
    """
)


@pytest.fixture()
def mock_extractor(tmp_path: Path) -> StandardInvoiceExtractor:
    """Return an extractor whose PDF file actually exists (empty placeholder)."""
    pdf_file = tmp_path / "test_invoice.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")  # minimal content so Path.exists() passes
    return StandardInvoiceExtractor(pdf_file)


def test_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        StandardInvoiceExtractor("/nonexistent/path/invoice.pdf")


def test_heuristics_parse_invoice_number(mock_extractor: StandardInvoiceExtractor) -> None:
    data = InvoiceData(raw_text=SAMPLE_TEXT)
    mock_extractor._apply_heuristics(data)
    assert data.invoice_number == "INV-2024-001"


def test_heuristics_parse_dates(mock_extractor: StandardInvoiceExtractor) -> None:
    data = InvoiceData(raw_text=SAMPLE_TEXT)
    mock_extractor._apply_heuristics(data)
    assert data.invoice_date == "01/15/2024"
    assert data.due_date == "02/15/2024"


def test_heuristics_parse_amounts(mock_extractor: StandardInvoiceExtractor) -> None:
    data = InvoiceData(raw_text=SAMPLE_TEXT)
    mock_extractor._apply_heuristics(data)
    assert data.subtotal == "1,000.00"
    assert data.tax_amount == "100.00"
    assert data.total_amount == "1,100.00"


def test_heuristics_missing_fields_stay_none(mock_extractor: StandardInvoiceExtractor) -> None:
    data = InvoiceData(raw_text="No useful content here.")
    mock_extractor._apply_heuristics(data)
    assert data.invoice_number is None
    assert data.total_amount is None


@patch("rs_bill_reading.standard_extractor.pypdf.PdfReader")
@patch("rs_bill_reading.standard_extractor.pdfplumber.open")
def test_extract_calls_both_libraries(
    mock_plumber_open: MagicMock,
    mock_pypdf_reader: MagicMock,
    mock_extractor: StandardInvoiceExtractor,
) -> None:
    # Set up pypdf mock
    mock_page = MagicMock()
    mock_page.extract_text.return_value = SAMPLE_TEXT
    mock_reader_instance = MagicMock()
    mock_reader_instance.__enter__ = MagicMock(return_value=mock_reader_instance)
    mock_reader_instance.__exit__ = MagicMock(return_value=False)
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.metadata = {}
    mock_pypdf_reader.return_value = mock_reader_instance

    # Set up pdfplumber mock
    mock_plumber_page = MagicMock()
    mock_plumber_page.extract_text.return_value = SAMPLE_TEXT
    mock_plumber_page.extract_tables.return_value = []
    mock_plumber_instance = MagicMock()
    mock_plumber_instance.__enter__ = MagicMock(return_value=mock_plumber_instance)
    mock_plumber_instance.__exit__ = MagicMock(return_value=False)
    mock_plumber_instance.pages = [mock_plumber_page]
    mock_plumber_open.return_value = mock_plumber_instance

    result = mock_extractor.extract()

    assert mock_pypdf_reader.called
    assert mock_plumber_open.called
    assert result.invoice_number == "INV-2024-001"
