"""
main.py — Compare standard vs AI-powered invoice extraction.

Usage:
    python main.py <path/to/invoice.pdf> [--approach standard|ai|both]

Examples:
    python main.py samples/invoice.pdf
    python main.py samples/invoice.pdf --approach standard
    python main.py samples/invoice.pdf --approach ai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env if present


def run_standard(pdf_path: Path) -> None:
    from rs_bill_reading import StandardInvoiceExtractor

    print("\n" + "=" * 60)
    print("APPROACH 1 — Standard PDF Extraction (pypdf + pdfplumber)")
    print("=" * 60)

    extractor = StandardInvoiceExtractor(pdf_path)
    data = extractor.extract()

    print(f"\n[Metadata]\n{json.dumps(data.metadata, indent=2, default=str)}")
    print(f"\n[Heuristic fields]")
    print(f"  Invoice number : {data.invoice_number}")
    print(f"  Invoice date   : {data.invoice_date}")
    print(f"  Due date       : {data.due_date}")
    print(f"  Vendor name    : {data.vendor_name}")
    print(f"  Subtotal       : {data.subtotal}")
    print(f"  Tax amount     : {data.tax_amount}")
    print(f"  Total amount   : {data.total_amount}")

    if data.tables:
        print(f"\n[Tables found: {len(data.tables)}]")
        for i, table in enumerate(data.tables, 1):
            print(f"  Table {i}: {len(table)} rows × {len(table[0]) if table else 0} cols")

    print(f"\n[Raw text excerpt — first 500 chars]\n{data.raw_text[:500]}")


def run_ai(pdf_path: Path) -> None:
    from rs_bill_reading import AIInvoiceExtractor

    print("\n" + "=" * 60)
    print("APPROACH 2 — AI Extraction (LangChain + OpenAI gpt-4o-mini)")
    print("=" * 60)

    extractor = AIInvoiceExtractor(pdf_path)
    result = extractor.extract()

    print(f"\n[Extracted invoice fields]")
    print(json.dumps(result.model_dump(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF invoice extractor demo")
    parser.add_argument("pdf", type=Path, help="Path to the PDF invoice")
    parser.add_argument(
        "--approach",
        choices=["standard", "ai", "both"],
        default="both",
        help="Which extraction approach to run (default: both)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: file not found — {args.pdf}", file=sys.stderr)
        sys.exit(1)

    if args.approach in ("standard", "both"):
        run_standard(args.pdf)

    if args.approach in ("ai", "both"):
        run_ai(args.pdf)


if __name__ == "__main__":
    main()
