"""Tests for AIInvoiceExtractor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rs_bill_reading.ai_extractor import AIInvoiceExtractor, InvoiceFields


SAMPLE_LLM_RESPONSE: dict = {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "vendor_name": "ACME Corp",
    "vendor_address": "123 Main St, Springfield",
    "customer_name": "John Doe",
    "customer_address": "456 Oak Ave, Shelbyville",
    "line_items": [
        {"description": "Widget A", "quantity": 2, "unit_price": "500.00", "total": "1000.00"}
    ],
    "subtotal": "1000.00",
    "tax_amount": "100.00",
    "total_amount": "1100.00",
    "currency": "USD",
    "payment_terms": "Net 30",
    "notes": None,
}


@pytest.fixture()
def pdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "invoice.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def test_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        AIInvoiceExtractor("/nonexistent/invoice.pdf")


@patch("rs_bill_reading.ai_extractor.ChatOpenAI")
def test_extract_returns_invoice_fields(
    mock_chat_openai: MagicMock, pdf_path: Path
) -> None:
    # Mock the LLM chain result
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance

    extractor = AIInvoiceExtractor(pdf_path)

    # Replace the chain with a mock that returns our canned response
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = SAMPLE_LLM_RESPONSE
    extractor._chain = mock_chain

    # Mock PDF loading
    with patch.object(extractor, "load_documents") as mock_load:
        mock_doc = MagicMock()
        mock_doc.page_content = "Invoice No. INV-2024-001 ..."
        mock_load.return_value = [mock_doc]

        result = extractor.extract()

    assert isinstance(result, InvoiceFields)
    assert result.invoice_number == "INV-2024-001"
    assert result.total_amount == "1100.00"
    assert result.vendor_name == "ACME Corp"
    assert len(result.line_items) == 1


@patch("rs_bill_reading.ai_extractor.ChatOpenAI")
def test_extract_raw_returns_dict(
    mock_chat_openai: MagicMock, pdf_path: Path
) -> None:
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance

    extractor = AIInvoiceExtractor(pdf_path)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = SAMPLE_LLM_RESPONSE
    extractor._chain = mock_chain

    with patch.object(extractor, "load_documents") as mock_load:
        mock_doc = MagicMock()
        mock_doc.page_content = "Invoice text..."
        mock_load.return_value = [mock_doc]

        result = extractor.extract_raw()

    assert isinstance(result, dict)
    assert result["invoice_number"] == "INV-2024-001"


def test_invoice_fields_model() -> None:
    fields = InvoiceFields(**SAMPLE_LLM_RESPONSE)
    assert fields.currency == "USD"
    assert fields.notes is None
    assert fields.line_items[0]["description"] == "Widget A"
