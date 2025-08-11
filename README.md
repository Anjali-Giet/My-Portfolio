## AISHE PDF Extractor

Extracts Basic Details, Officers and Address from the first two pages of an AISHE-style PDF to JSON.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python extract_aishe.py --pdf /absolute/path/to/file.pdf --out /workspace/aishe.json --pretty
```

If you omit `--out`, the JSON will be printed to stdout.

The output roughly follows the schema in `aishe_schema.json`.

### Notes
- The script first tries to extract text directly. If the page is a scanned image, it falls back to OCR.
- OCR accuracy depends on scan quality. If any field is missed, please share the PDF/text so we can refine the label patterns.