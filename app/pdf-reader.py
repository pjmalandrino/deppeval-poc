import os
import argparse
from pathlib import Path
import PyPDF2

def convert_pdf_to_text(pdf_path):
    """
    Convert a single PDF file to text

    Args:
        pdf_path: Path to the PDF file

    Returns:
        String containing the extracted text
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"

            return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def process_pdf_folder(input_folder, output_folder=None):
    """
    Process all PDF files in a folder and convert them to text files

    Args:
        input_folder: Path to folder containing PDF files
        output_folder: Path to folder where text files will be saved (optional)
                       If not provided, will create a 'text_output' folder in the same directory

    Returns:
        Number of successfully processed files
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
        # Get relative path within input folder
        rel_path = pdf_file.relative_to(input_path)

        # Create corresponding output folder structure
        output_file = output_path / rel_path.with_suffix('.txt')
        os.makedirs(output_file.parent, exist_ok=True)

        print(f"Processing: {pdf_file}")
        text = convert_pdf_to_text(pdf_file)

        if text:
            # Write text to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            success_count += 1
            print(f"Saved: {output_file}")
        else:
            print(f"Failed to extract text from {pdf_file}")

    return success_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PDF files to text")
    parser.add_argument("input_folder", help="Folder containing PDF files")
    parser.add_argument("-o", "--output_folder", help="Folder to save text files (optional)")

    args = parser.parse_args()

    # Process the folder
    processed_count = process_pdf_folder(args.input_folder, args.output_folder)

    print(f"\nSuccessfully processed {processed_count} PDF files")

if __name__ == "__main__":
    main()