#!/usr/bin/env python3
"""
Convert all PDF files in the current directory into PNG images.

Outputs are written next to each PDF using the pattern:
  <pdf_stem>_<page_number_padded>.png

Default settings:
- DPI: 300
- Overwrite existing PNGs: False (skips existing files)

Backends:
- Prefer PyMuPDF (fitz) if available: pip install pymupdf
- Fallback to pdf2image if available: pip install pdf2image
  (requires Poppler, e.g., macOS: brew install poppler)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


def is_pymupdf_available() -> bool:
    try:
        import fitz  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def is_pdf2image_available() -> bool:
    try:
        from pdf2image import convert_from_path  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def list_pdfs_in_current_directory() -> List[Path]:
    current_dir = Path.cwd()
    return [p for p in current_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]


def convert_with_pymupdf(pdf_path: Path, dpi: int, overwrite: bool) -> None:
    import fitz  # type: ignore

    try:
        document = fitz.open(pdf_path.as_posix())
    except Exception as error:  # pragma: no cover - environment-specific
        print(f"[ERROR] Failed to open '{pdf_path.name}' with PyMuPDF: {error}")
        return

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    try:
        total_pages = len(document)
        for page_index in range(total_pages):
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_name = f"{pdf_path.stem}_{page_index + 1:04d}.png"
            output_path = pdf_path.with_name(output_name)

            if output_path.exists() and not overwrite:
                print(f"[SKIP] {output_path.name} already exists")
                continue

            pixmap.save(output_path.as_posix())
            print(f"[OK]   {pdf_path.name} -> {output_path.name}")
    finally:
        document.close()


def convert_with_pdf2image(pdf_path: Path, dpi: int, overwrite: bool) -> None:
    from pdf2image import convert_from_path  # type: ignore

    try:
        images = convert_from_path(pdf_path.as_posix(), dpi=dpi)
    except Exception as error:  # pragma: no cover - environment-specific
        print(f"[ERROR] Failed to convert '{pdf_path.name}' with pdf2image: {error}")
        return

    for index, image in enumerate(images, start=1):
        output_name = f"{pdf_path.stem}_{index:04d}.png"
        output_path = pdf_path.with_name(output_name)

        if output_path.exists() and not overwrite:
            print(f"[SKIP] {output_path.name} already exists")
            continue

        image.save(output_path.as_posix(), "PNG")
        print(f"[OK]   {pdf_path.name} -> {output_path.name}")


def convert_all_pdfs(pdf_paths: Iterable[Path], dpi: int, overwrite: bool) -> None:
    backend = None
    if is_pymupdf_available():
        backend = "pymupdf"
    elif is_pdf2image_available():
        backend = "pdf2image"

    if backend is None:
        print(
            "[ERROR] No PDF rendering backend found. Install one of the following:\n"
            "  - pip install pymupdf\n"
            "  - pip install pdf2image  (requires Poppler; macOS: 'brew install poppler')"
        )
        sys.exit(1)

    print(f"[INFO] Using backend: {backend}")
    print(f"[INFO] DPI: {dpi} | Overwrite: {overwrite}")

    for pdf_path in sorted(pdf_paths, key=lambda p: p.name.lower()):
        if backend == "pymupdf":
            convert_with_pymupdf(pdf_path, dpi=dpi, overwrite=overwrite)
        else:
            convert_with_pdf2image(pdf_path, dpi=dpi, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in the current directory into PNGs."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution (dots per inch). Default: 300",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files instead of skipping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_files = list_pdfs_in_current_directory()

    if not pdf_files:
        print("[INFO] No PDF files found in the current directory.")
        return

    convert_all_pdfs(pdf_files, dpi=args.dpi, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


