import os
import argparse
from pathlib import Path
import PyPDF2

def extract_page_text(pdf_reader, page_num):
    """
    Extract text from a specific page in a PDF

    Args:
        pdf_reader: PyPDF2.PdfReader object
        page_num: Page number (0-based index)

    Returns:
        String containing the extracted text
    """
    try:
        page = pdf_reader.pages[page_num]
        return page.extract_text()
    except Exception as e:
        print(f"Error extracting text from page {page_num+1}: {str(e)}")
        return ""

def process_pdf_folder(input_folder, output_folder=None, chunk_by_page=True):
    """
    Process all PDF files in a folder and convert them to text files,
    chunking by page if specified

    Args:
        input_folder: Path to folder containing PDF files
        output_folder: Path to folder where text files will be saved (optional)
                      If not provided, will create a 'text_output' folder in the same directory
        chunk_by_page: If True, create a separate text file for each page

    Returns:
        Number of successfully processed files/pages
    """
    # Create input folder path object
    input_path = Path(input_folder)

    # Set default output path if not provided
    if output_folder is None:
        output_path = input_path.parent / "text_output"
    else:
        output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Find all PDF files
    pdf_files = list(input_path.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return 0

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF file
    success_count = 0
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")

        try:
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)

                # Get the base name of the PDF file without extension
                base_name = pdf_file.stem

                # Create subfolder for PDF if chunking by page
                if chunk_by_page:
                    pdf_output_dir = output_path / base_name
                    os.makedirs(pdf_output_dir, exist_ok=True)

                    # Process each page separately
                    for page_num in range(total_pages):
                        page_text = extract_page_text(reader, page_num)

                        if page_text:
                            # Create output filename: base_name_page_001.txt
                            page_file = pdf_output_dir / f"{base_name}_page_{page_num+1:03d}.txt"

                            with open(page_file, 'w', encoding='utf-8') as f:
                                f.write(page_text)

                            success_count += 1
                            print(f"Saved page {page_num+1}/{total_pages}: {page_file}")
                        else:
                            print(f"Failed to extract text from page {page_num+1}/{total_pages}")
                else:
                    # Process entire PDF as one file
                    # Get relative path within input folder
                    rel_path = pdf_file.relative_to(input_path)

                    # Create corresponding output folder structure
                    output_file = output_path / rel_path.with_suffix('.txt')
                    os.makedirs(output_file.parent, exist_ok=True)

                    # Combine text from all pages
                    full_text = ""
                    for page_num in range(total_pages):
                        page_text = extract_page_text(reader, page_num)
                        full_text += page_text + "\n\n"

                    # Write to output file
                    if full_text.strip():
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(full_text)
                        success_count += 1
                        print(f"Saved: {output_file}")
                    else:
                        print(f"Failed to extract any text from {pdf_file}")

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    return success_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PDF files to text")
    parser.add_argument("input_folder", help="Folder containing PDF files")
    parser.add_argument("-o", "--output_folder", help="Folder to save text files (optional)")
    parser.add_argument("-c", "--chunk", action="store_true",
                        help="Chunk by page (create separate text file for each page)")

    args = parser.parse_args()

    # Process the folder
    processed_count = process_pdf_folder(args.input_folder, args.output_folder, args.chunk)

    if args.chunk:
        print(f"\nSuccessfully processed {processed_count} pages from PDF files")
    else:
        print(f"\nSuccessfully processed {processed_count} PDF files")

if __name__ == "__main__":
    main()