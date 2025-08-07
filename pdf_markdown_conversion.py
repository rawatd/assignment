import os
from docling.document_converter import DocumentConverter

def convert_pdfs_to_markdown(input_folder="data", output_folder="output_markdown_new"):
    """
    Converts all PDF files in an input folder to Markdown format
    and saves them in an output folder.

    Args:
        input_folder (str): The name of the folder containing the PDF files.
        output_folder (str): The name of the folder where Markdown files will be saved.
    """
    print(f"Checking for output folder: '{output_folder}'...")
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' ensured.")

    # Initialize the DocumentConverter
    converter = DocumentConverter()
    print("Docling DocumentConverter initialized.")
    
    print(f"\nScanning for PDF files in '{input_folder}'...")
    
    # Dynamically find all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("🔍 No PDF files found in the specified folder. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    print("\nStarting PDF to Markdown conversion...")

    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file_name)
        
        try:
            print(f"\n➡️ Converting '{pdf_file_name}' to Markdown...")
            # Convert the PDF document using Docling
            result = converter.convert(pdf_path)

            # Export the document content to Markdown format
            markdown_content = result.document.export_to_markdown()

            # Define the output Markdown file name (e.g., "filename.pdf" -> "filename.md")
            markdown_file_name = os.path.splitext(pdf_file_name)[0] + ".md"
            markdown_output_path = os.path.join(output_folder, markdown_file_name)

            # Write the Markdown content to a file in the output folder
            with open(markdown_output_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
            
            print(f"✅ Successfully converted '{pdf_file_name}' to '{markdown_file_name}'")

        except Exception as e:
            print(f"❌ Error converting '{pdf_file_name}': {e}")

    print("\n🎉 PDF to Markdown conversion complete!")
    print(f"All converted Markdown files are saved in the '{output_folder}' folder.")

# --- Execution of the function ---
if __name__ == "__main__":
    convert_pdfs_to_markdown()