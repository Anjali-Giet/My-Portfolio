#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    lowered = value.lower()
    # Remove punctuation-like characters for fuzzy matching
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_text_pymupdf(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)
    # Try a layout-preserving mode first
    text = page.get_text("text")
    if text and len(text.strip()) > 100:
        return text
    # Fallback to blocks
    text = page.get_text("blocks")
    if isinstance(text, list):
        text = "\n".join(block[4] for block in text if isinstance(block, (list, tuple)) and len(block) >= 5)
    return text or ""


def render_page_image(doc: fitz.Document, page_index: int, dpi: int = 220) -> Image.Image:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def ocr_image_to_text(img: Image.Image) -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    # Convert to grayscale and apply slight sharpening for better OCR
    gray = img.convert("L")
    text = pytesseract.image_to_string(gray)
    return text


def cvt_pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def ensure_text(doc: fitz.Document, page_index: int) -> str:
    text = extract_text_pymupdf(doc, page_index)
    if len(normalize_text(text)) < 50:
        # Try OCR fallback
        img = render_page_image(doc, page_index)
        text = ocr_image_to_text(img)
    return text


def find_value_for_label(lines: List[str], label_candidates: List[str], start_at: int = 0) -> Tuple[Optional[str], int]:
    """
    Finds a value for any of the given label candidates. Returns (value, index_of_line_where_found or -1)
    Heuristic: If label and value are on same line, capture trailing text after label. Otherwise, take the next non-empty line(s)
    that do not look like section headers or new labels.
    """
    normalized_candidates = [normalize_text(c) for c in label_candidates]

    def looks_like_label(text: str) -> bool:
        t = normalize_text(text)
        # lines starting with digits and a dot or long all-caps words often indicate headers/labels
        if re.match(r"^\d+\s*\.", t):
            return True
        keyword_hits = sum(1 for c in normalized_candidates if c.split(" ")[0] in t)
        return keyword_hits > 0 and len(t) < 140

    for idx in range(start_at, len(lines)):
        line = lines[idx]
        norm_line = normalize_text(line)
        for cand in normalized_candidates:
            if cand in norm_line:
                # Same-line capture (text after the last occurrence of the candidate's last word)
                words = cand.split()
                if words:
                    last_word = words[-1]
                    # Find position of last occurrence of last word in original line to cut after it
                    pattern = re.compile(re.escape(words[-1]), re.IGNORECASE)
                    matches = list(pattern.finditer(line))
                    if matches:
                        cut_pos = matches[-1].end()
                        trailing = line[cut_pos:].strip().strip(':').strip()
                        if trailing:
                            return trailing, idx
                # Next non-empty line(s) capture
                # Concatenate up to 2 following lines if they don't look like labels
                collected: List[str] = []
                j = idx + 1
                while j < len(lines) and len(collected) < 2:
                    nxt = lines[j].strip()
                    if not nxt:
                        j += 1
                        continue
                    if looks_like_label(nxt):
                        break
                    collected.append(nxt)
                    j += 1
                value = " ".join(collected).strip()
                if value:
                    return value, idx
                return None, idx
    return None, -1


def parse_basic_details(page_text: str) -> Dict[str, str]:
    lines = [ln.rstrip() for ln in page_text.splitlines()]

    labels = {
        "aishe_code": ["AISHE Code"],
        "institution_name": ["Name of the Institution"],
        "year_of_establishment": ["Year of Establishment"],
        "status_prior_to_establishment": ["Status Prior to Establishment, if applicable"],
        "year_declared_university_or_ion": [
            "Year when declared University Institute of National Importance",
            "Year when declared University/ Institute of National Importance",
        ],
        "type_of_institution": ["Type of Institution"],
        "ownership_status": ["Ownership status of Institution"],
        "single_gender_institution": [
            "Whether the Institution is exclusively meant for students from one gender",
        ],
        "unnat_bharat_scheme_adopted": [
            "Whether Institute adopted any village under Unnat Bharat Scheme",
        ],
        "minority_managed_institution": ["Is this a Minority Managed Institution"],
        "minority_type": [
            "If Yes, Type of minority community managing the",
            "If Yes Type of minority community managing the",
        ],
        "has_ncc": [
            "Whether the Institution has the National Cadet Corps",
            "Whether the Institution has the National Cadet Corps (NCC)",
        ],
        "has_nss": [
            "Whether the Institution has the National Social Service",
            "Whether the Institution has the National Service Scheme",
        ],
        "is_specialized_university": ["Is it a Specialized University"],
        "specialization_type": ["If Yes Specialization Type"],
        "other_specialization": ["Other Specialisation", "Other Specialization"],
    }

    result: Dict[str, Optional[str]] = {}
    cursor = 0
    for key, candidates in labels.items():
        value, pos = find_value_for_label(lines, candidates, start_at=cursor)
        if pos != -1:
            cursor = pos
        result[key] = value

    # Best-effort post-process booleans and numerics
    def to_bool(val: Optional[str]) -> Optional[bool]:
        if val is None:
            return None
        v = normalize_text(val)
        if v in {"yes", "y"}:
            return True
        if v in {"no", "n"}:
            return False
        return None

    def to_int(val: Optional[str]) -> Optional[int]:
        if val is None:
            return None
        m = re.search(r"\d{4}", val)
        return int(m.group(0)) if m else None

    parsed = {
        "aishe_code": (result.get("aishe_code") or "").strip(),
        "institution_name": (result.get("institution_name") or "").strip(),
        "year_of_establishment": to_int(result.get("year_of_establishment")),
        "status_prior_to_establishment": (result.get("status_prior_to_establishment") or "").strip(),
        "year_declared_university_or_ion": to_int(result.get("year_declared_university_or_ion")),
        "institute_details": {
            "type_of_institution": (result.get("type_of_institution") or "").strip(),
            "ownership_status": (result.get("ownership_status") or "").strip(),
            "single_gender_institution": to_bool(result.get("single_gender_institution")),
            "unnat_bharat_scheme_adopted": to_bool(result.get("unnat_bharat_scheme_adopted")),
            "minority_managed_institution": to_bool(result.get("minority_managed_institution")),
            "minority_type": (result.get("minority_type") or "").strip(),
            "has_ncc": to_bool(result.get("has_ncc")),
            "has_nss": to_bool(result.get("has_nss")),
            "is_specialized_university": to_bool(result.get("is_specialized_university")),
            "specialization_type": (result.get("specialization_type") or "").strip(),
            "other_specialization": (result.get("other_specialization") or "").strip(),
        },
    }
    return parsed


def parse_officers_and_address(page_text: str) -> Dict[str, Dict[str, Optional[str]]]:
    lines = [ln.rstrip() for ln in page_text.splitlines()]

    vc_map = {
        "name": [
            "Name of Vice chancellor/Director/Head/Principal",
            "Name of Vice chancellor Director Head Principal",
        ],
        "designation": ["Designation"],
        "mobile": ["Mobile No", "Mobile Number"],
        "email": ["Email"],
        "telephone": ["Telephone No", "Telephone No with STD Code", "Telephone with STD Code"],
    }

    vc: Dict[str, Optional[str]] = {}
    cursor = 0
    for key, candidates in vc_map.items():
        vc[key], pos = find_value_for_label(lines, candidates, start_at=cursor)
        if pos != -1:
            cursor = pos

    nodal_map = {
        "name": ["Name of Nodal Officer for AISHE"],
        "designation": ["Designation"],
        "mobile": ["Mobile No", "Mobile Number"],
        "email": ["Email"],
        "telephone": ["Telephone No", "Telephone No with STD Code", "Telephone with STD Code"],
    }

    nodal: Dict[str, Optional[str]] = {}
    # Search after where VC likely ended
    for key, candidates in nodal_map.items():
        nodal[key], pos = find_value_for_label(lines, candidates, start_at=cursor)
        if pos != -1:
            cursor = pos

    address_map = {
        "location": ["Location of the University"],
        "address_line1": ["Address Line1", "Address Line 1"],
        "address_line2": ["Address Line2", "Address Line 2"],
        "city": ["City"],
        "state": ["State"],
        "district": ["District"],
        "block_or_urban_local_body": ["Block / Urban Local Body", "Block Urban Local Body"],
        "pin_code": ["Pin Code", "Pincode"],
        "longitude": ["Longitude(in degree)", "Longitude (in degree)", "Longitude"],
        "latitude": ["Latitude(in degree)", "Latitude (in degree)", "Latitude"],
        "total_area_acre": ["Total Area (in acre)", "Total Area in acre"],
        "total_constructed_area_sqm": ["Total Constructed Area (in sq m)", "Total Constructed Area"],
        "website": ["Website"],
    }

    address: Dict[str, Optional[str]] = {}
    for key, candidates in address_map.items():
        address[key], pos = find_value_for_label(lines, candidates, start_at=cursor)
        if pos != -1:
            cursor = pos

    # Post-process some fields
    if address.get("pin_code"):
        m = re.search(r"\b(\d{6})\b", address["pin_code"])  # Indian PIN
        address["pin_code"] = m.group(1) if m else address["pin_code"]

    for fld in ("latitude", "longitude"):
        val = address.get(fld)
        if val:
            m = re.search(r"-?\d{1,3}\.\d+", val)
            address[fld] = m.group(0) if m else val

    return {"vice_chancellor": vc, "nodal_officer": nodal, "address": address}


def extract_aishe(pdf_path: str) -> Dict[str, Dict]:
    doc = fitz.open(pdf_path)
    if doc.page_count < 1:
        raise ValueError("PDF has no pages")

    basic_text = ensure_text(doc, 0)
    officers_text = ensure_text(doc, 1) if doc.page_count >= 2 else ""

    data: Dict[str, Dict] = {
        "basic_details": parse_basic_details(basic_text),
    }

    if officers_text:
        parsed = parse_officers_and_address(officers_text)
        data.update(parsed)

    return data


def main():
    parser = argparse.ArgumentParser(description="Extract AISHE Basic Details, Officers, and Address from first two pages of a PDF into JSON.")
    parser.add_argument("--pdf", required=True, help="Absolute path to the input PDF")
    parser.add_argument("--out", default="", help="Output JSON file path. If omitted, prints to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    result = extract_aishe(args.pdf)
    text = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()