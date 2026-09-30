"""Bounded subprocess entrypoint for untrusted downloaded PDFs."""

import json
import sys

from pypdf import PdfReader

if __name__ == "__main__":
    reader = PdfReader(sys.argv[1])
    print(
        json.dumps({
            str(i + 1): (page.extract_text() or "")[:30_000]
            for i, page in enumerate(reader.pages[:80])
        })
    )
