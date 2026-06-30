import fitz  # PyMuPDF library


def extract_raw_text_from_pdf(pdf_path: str) -> str:
    print(f"[PDF READER] Opening file: '{pdf_path}'...")

    try:
        # Open the PDF document asset
        doc = fitz.open(pdf_path)
        print(f"[PDF READER] Document loaded successfully. Total pages: {len(doc)}")

        full_text_accumulator = []

        # Iterate over each page sequentially
        for page_num, page in enumerate(doc):  # type: ignore
            # Extract clean, layout-aware text from the page
            page_text = page.get_text("text")

            print(
                f" -> Extracted page {page_num + 1}/{len(doc)} ({len(page_text)} characters)"
            )

            # Store the text block
            full_text_accumulator.append(page_text)

        # Combine all extracted pages into a single raw text string
        complete_raw_text = "\n--- PAGE BREAK ---\n".join(full_text_accumulator)
        return complete_raw_text

    except Exception as e:
        print(f"[ERROR] Failed to read the PDF asset file: {e}")
        return ""


if __name__ == "__main__":
    # Define the target path to your company policy document file
    target_pdf = "company_policy.pdf"

    # Run extraction process
    raw_text_output = extract_raw_text_from_pdf(target_pdf)

    if raw_text_output:
        print("\n================== RAW TEXT PREVIEW ==================")
        # Print the first 1000 characters to verify content quality in your console
        print(raw_text_output[:1000])
        print("\n======================================================")
