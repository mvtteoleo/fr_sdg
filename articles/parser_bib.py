import sys
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

def extract_doi_from_pdf(pdf_path):
    try:
        # Extract text from only the first 2 pages using pdftotext
        result = subprocess.run(
            ['pdftotext', '-f', '1', '-l', '2', str(pdf_path), '-'], 
            capture_output=True, text=True, check=True
        )
        text = result.stdout
        
        # Regex to find a standard DOI (e.g., 10.1017/jfm.2025...)
        match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)', text)
        if match:
            # Clean up any trailing punctuation that might get caught
            return match.group(1).rstrip('.:,;')
    except FileNotFoundError:
        print("🚨 Error: 'pdftotext' is not installed. Run: nix shell nixpkgs#poppler_utils")
        sys.exit(1)
    except Exception as e:
        pass
    
    return None

def fetch_bibtex_from_doi(doi):
    # Crossref API returns perfect BibTeX for DOIs
    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/x-bibtex'})
        with urllib.request.urlopen(req) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError:
        return None

def build_bibliography(directory_path, output_filename="master_bibliography.bib"):
    target_dir = Path(directory_path)
    
    if not target_dir.is_dir():
        print(f"❌ Error: '{directory_path}' is not a valid directory.")
        return

    master_bib_path = Path(output_filename)
    if master_bib_path.exists():
        master_bib_path.unlink()

    pdf_files = list(target_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDFs found.")
        return

    print(f"🔍 Found {len(pdf_files)} PDFs. Hunting for DOIs...\n")

    success_count = 0

    with open(master_bib_path, 'a', encoding='utf-8') as master_file:
        for pdf in pdf_files:
            print(f"⚙️ Scanning: {pdf.name}...")
            
            doi = extract_doi_from_pdf(pdf)
            if not doi:
                print("   ❌ No DOI found on the first pages. You'll need to cite this manually.")
                master_file.write(f"%"*50 + "\n")
                master_file.write(f"%% {pdf.name} Misses the DOI!! TODO cite manually\n")
                master_file.write(f"%"*50 + "\n")
                continue
                
            print(f"   🔎 Found DOI: {doi} -> Fetching BibTeX...")
            bibtex = fetch_bibtex_from_doi(doi)
            
            if bibtex:
                master_file.write(f"%% Extracted from: {pdf.name}\n")
                master_file.write(bibtex.strip() + "\n\n")
                print("   ✅ Success!")
                success_count += 1
            else:
                master_file.write(f"%"*50 + "\n")
                master_file.write(f"%% {pdf.name} Misses the BibTex from Crossref!! TODO cite manually\n")
                master_file.write(f"%"*50 + "\n")
                print("   ❌ Failed to download BibTeX from Crossref.")

    print(f"\n🎉 Done! Successfully generated {success_count}/{len(pdf_files)} citations into '{output_filename}'.")

if __name__ == "__main__":
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    build_bibliography(target_directory)

