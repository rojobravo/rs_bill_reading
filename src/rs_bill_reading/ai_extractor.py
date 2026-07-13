"""AI-powered PDF invoice extractor using LangChain."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class InvoiceFields(BaseModel):
    """Structured invoice fields extracted by the LLM."""

    invoice_number: str | None = Field(None, description="Invoice number or ID")
    invoice_date: str | None = Field(None, description="Date the invoice was issued")
    due_date: str | None = Field(None, description="Payment due date")
    vendor_name: str | None = Field(None, description="Name of the vendor / seller")
    vendor_address: str | None = Field(None, description="Vendor postal address")
    customer_name: str | None = Field(None, description="Name of the customer / buyer")
    customer_address: str | None = Field(None, description="Customer postal address")
    line_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of line items, each with keys: description, quantity, unit_price, total",
    )
    subtotal: str | None = Field(None, description="Subtotal before taxes")
    tax_amount: str | None = Field(None, description="Total tax / VAT / GST amount")
    total_amount: str | None = Field(None, description="Grand total amount due")
    currency: str | None = Field(None, description="Currency code, e.g. USD, EUR")
    payment_terms: str | None = Field(None, description="Payment terms, e.g. Net 30")
    notes: str | None = Field(None, description="Any additional notes or remarks")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert accountant specialized in reading and parsing invoices. "
                "Extract all relevant fields from the invoice text provided by the user. "
                "Return ONLY a valid JSON object that matches the requested schema. "
                "Use null for any field you cannot find."
            ),
        ),
        (
            "human",
            (
                "Extract invoice information from the following text and return it as JSON.\n\n"
                "Schema description:\n{schema}\n\n"
                "Invoice text:\n{invoice_text}"
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class AIInvoiceExtractor:
    """
    Extracts invoice information from a PDF using LangChain + an LLM.

    The default model is **gpt-4o-mini** (cheap, fast, high accuracy for
    structured-data tasks). Override via *model_name* or set the
    ``OPENAI_API_KEY`` environment variable before instantiating.

    Usage::

        extractor = AIInvoiceExtractor("invoice.pdf")
        result = extractor.extract()
        print(result.total_amount)

    You can also pass a custom LangChain chat model::

        from langchain_anthropic import ChatAnthropic
        extractor = AIInvoiceExtractor(
            "invoice.pdf",
            llm=ChatAnthropic(model="claude-3-5-haiku-latest"),
        )
    """

    def __init__(
        self,
        pdf_path: str | Path,
        *,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        llm: Any | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self._llm: Any = llm or ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        self._chain: Runnable = _EXTRACTION_PROMPT | self._llm | JsonOutputParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self) -> list[Document]:
        """Load the PDF with pdfplumber and return a list of LangChain :class:`Document` objects."""
        docs: list[Document] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(self.pdf_path), "page": i},
                ))
        return docs

    def extract(self) -> InvoiceFields:
        """
        Load the PDF, send its text to the LLM, and return a validated
        :class:`InvoiceFields` instance.
        """
        documents = self.load_documents()
        invoice_text = "\n\n".join(doc.page_content for doc in documents)
        schema_description = _build_schema_description()

        raw: dict[str, Any] = self._chain.invoke(
            {"invoice_text": invoice_text, "schema": schema_description}
        )
        return InvoiceFields(**raw)

    def extract_raw(self) -> dict[str, Any]:
        """Like :meth:`extract` but returns the raw dict from the LLM."""
        documents = self.load_documents()
        invoice_text = "\n\n".join(doc.page_content for doc in documents)
        schema_description = _build_schema_description()
        return self._chain.invoke(
            {"invoice_text": invoice_text, "schema": schema_description}
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_schema_description() -> str:
    """Build a human-readable field description from :class:`InvoiceFields`."""
    lines: list[str] = []
    for name, info in InvoiceFields.model_fields.items():
        desc = info.description or ""
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)
