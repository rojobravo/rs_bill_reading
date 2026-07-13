# rs_bill_reading

Python 3.13 project for extracting structured data from PDF invoices using two approaches:

| Approach | Libraries | Best for |
|---|---|---|
| **Standard** | `pypdf`, `pdfplumber` | Fast, offline, no API cost |
| **AI-powered** | `langchain`, `langchain-openai` | Complex/variable layouts, high accuracy |

## Project structure

```
rs_bill_reading/
├── src/rs_bill_reading/
│   ├── __init__.py
│   ├── standard_extractor.py   # pypdf + pdfplumber + regex heuristics
│   └── ai_extractor.py         # LangChain + OpenAI structured extraction
├── tests/
│   ├── test_standard_extractor.py
│   └── test_ai_extractor.py
├── samples/                    # place your test PDFs here
├── main.py                     # CLI demo comparing both approaches
├── pyproject.toml
└── .env.example
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install the package with dev dependencies
pip install -e ".[dev]"

# 3. Configure your OpenAI API key (required for AI approach)
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run both approaches on an invoice PDF
python main.py samples/invoice.pdf

# Standard extraction only (no API key needed)
python main.py samples/invoice.pdf --approach standard

# AI extraction only
python main.py samples/invoice.pdf --approach ai
```

### Programmatic usage

```python
from dotenv import load_dotenv
load_dotenv()

# --- Standard ---
from rs_bill_reading import StandardInvoiceExtractor

extractor = StandardInvoiceExtractor("invoice.pdf")
data = extractor.extract()
print(data.invoice_number, data.total_amount)

# --- AI-powered ---
from rs_bill_reading import AIInvoiceExtractor

extractor = AIInvoiceExtractor("invoice.pdf")          # uses gpt-4o-mini by default
result = extractor.extract()                            # returns InvoiceFields (Pydantic)
print(result.model_dump())
```

### Bring your own LLM

```python
from langchain_anthropic import ChatAnthropic
from rs_bill_reading import AIInvoiceExtractor

extractor = AIInvoiceExtractor(
    "invoice.pdf",
    llm=ChatAnthropic(model="claude-3-5-haiku-latest"),
)
result = extractor.extract()
```

## Running tests

```bash
pytest
```

## Key design decisions

- **`pypdf`** handles metadata and basic text; **`pdfplumber`** is preferred for body text and table extraction (better layout fidelity).
- **`pdfplumber`** is also used as the LangChain `PDFPlumberLoader` document source, keeping library usage consistent.
- The AI extractor uses `JsonOutputParser` with a detailed field-level prompt so the LLM returns structured data that maps directly to the `InvoiceFields` Pydantic model.
- `temperature=0.0` is set on the LLM to maximise determinism for data-extraction tasks.
