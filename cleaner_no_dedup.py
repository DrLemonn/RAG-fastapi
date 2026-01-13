import re
import os
from bs4 import BeautifulSoup

def clean_fastapi_markdown(text, root_dir="."):
    """
    Chunk-friendly cleaning strategy (no code deduplication):
    1. Regex: Handle file includes (macro replacement) - ALWAYS include code
    2. Regex: Handle special Termy format (HTML -> Markdown code blocks)
    3. BeautifulSoup: Structured HTML cleaning (remove tags, extract images/links)
    4. Regex: Text noise cleaning (remove extra symbols, blank lines)

    Key difference from cleaner.py:
    - Does NOT track seen_files - every {* ... *} tag gets replaced with actual code
    - This ensures each chunk is self-contained for chunk-based retrieval
    """

    # --- 2. Handle <div class="termy"> (keep original logic) ---
    # Must be done before BS4 cleaning to extract content into console code blocks
    def clean_termy(match):
        content = match.group(1)
        clean_content = re.sub(r'<[^>]+>', '', content)
        return f"{clean_content.strip()}"

    text = re.sub(r'<div class="termy">\s*(.*?)\s*</div>', clean_termy, text, flags=re.DOTALL)

    # --- 3. [Key modification] Use BeautifulSoup for structured cleaning ---
    # Process images, links, and other HTML tags
    soup = BeautifulSoup(text, "html.parser")

    # 3.1 Remove useless tags (including content)
    for tag in soup.find_all(['script', 'style', 'svg']):
        tag.decompose()

    # 3.2 Handle images (img)
    # Logic: If has alt, keep as [Image: alt]; otherwise delete
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '').strip()
        if alt_text:
            # Replace with text placeholder
            img.replace_with(f" [Image: {alt_text}] ")
        else:
            # No alt attribute, treat as decorative, delete
            img.decompose()

    # 3.3 Handle hyperlinks (a)
    # Logic:
    # A. Get text inside link (note: if img was replaced with text above, we can get it here)
    # B. If text contains "[Image:" or link contains "badge", treat as decorative, delete whole line
    # C. Otherwise, unwrap <a> tag, keep only text
    for a in soup.find_all('a'):
        # Get text inside tag (including [Image: ...] we replaced)
        inner_text = a.get_text(strip=True)

        # Check if it's a badge link or image wrapper
        is_badge_link = "badge" in str(a) or "shield" in str(a)
        is_image_wrapper = "[Image:" in inner_text

        if is_badge_link or is_image_wrapper:
            a.decompose()  # Delete entire <a> node
        else:
            a.replace_with(inner_text)  # Keep text only, remove link wrapper

    # ================= [New step] Handle paragraphs and formatting tags =================

    # Define tags to "unwrap" (remove tag but keep content)
    # p: paragraph, div: block, span: inline, em/i: italic, strong/b: bold, center: centered
    formatting_tags = ['p', 'div', 'span', 'em', 'i', 'strong', 'b', 'u', 'center', 'font', 'small', 'attr']

    for tag_name in formatting_tags:
        for tag in soup.find_all(tag_name):
            # 1. Check if tag is "empty" (only spaces or newlines)
            # Example: <p>  </p> should be deleted, not leaving blank lines
            if not tag.get_text(strip=True):
                tag.decompose()
            else:
                # 2. If has content, unwrap tag, keep content
                # Example: <p align="center">Text</p>  ==>  Text
                tag.unwrap()

    # Convert processed DOM tree back to string
    text = str(soup)

    # --- 4. Regex cleanup (Text Noise Cleaning) ---

    # 4.1 Remove Markdown separators --- (usually noise for RAG)
    text = re.sub(r'-{3,}', '', text)

    # 4.2 Clean remaining HTML entities (optional, BS4 conversion may produce &gt;)
    # text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")

    # Remove markdown bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # 4.3 Clean excessive blank lines (very important, merge 2+ newlines to 2)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # --- 1. Handle file includes {* ... *} (NO DEDUPLICATION) ---
    # Key difference: removed seen_files tracking
    # Every {* ... *} tag is replaced with actual code content

    def load_file_content(match):
        tag_content = match.group(0)
        content_inner = tag_content.replace('{*', '').replace('*}', '').strip()
        parts = content_inner.split()
        if not parts: return ""

        raw_path = parts[0]

        if "docs_src" in raw_path:
            start_index = raw_path.find("docs_src")
            rel_path = raw_path[start_index:]
        else:
            rel_path = raw_path.lstrip('./').replace('../', '')

        full_path_raw = os.path.join(root_dir, rel_path)
        dir_name = os.path.dirname(full_path_raw)
        base_name = os.path.basename(full_path_raw)
        file_name_no_ext, ext = os.path.splitext(base_name)

        candidate_filenames = [
            f"{file_name_no_ext}_an_py39{ext}",
            f"{file_name_no_ext}_py39{ext}",
            f"{file_name_no_ext}{ext}"
        ]

        target_content = None

        for candidate in candidate_filenames:
            check_path = os.path.join(dir_name, candidate)
            if os.path.exists(check_path):
                # NO DEDUPLICATION - always load the file
                try:
                    with open(check_path, 'r', encoding='utf-8') as f:
                        target_content = f.read()
                    break
                except Exception as e:
                    return f"\n[Error reading file {candidate}: {e}]\n"

        if target_content is not None:
            lang_tag = ext.lstrip('.')
            return f"\n```{lang_tag}\n{target_content}\n```\n"
        else:
            return f"\n[Warning: Code file not found. Looked for: {', '.join(candidate_filenames)} at {dir_name}]\n"

    pattern_include = r'\{\*.*?\*\}'
    text = re.sub(pattern_include, load_file_content, text)

    # 4.4 Remove leading/trailing whitespace
    return text.strip()


if __name__ == "__main__":
    # --- Configuration ---
    current_root_dir = os.path.dirname(os.path.abspath(__file__))
    input_md_file = "rag_test_doc.md"

    # --- Test with sample data ---
    fake_md_content = """
    # FastAPI [Image: FastAPI]

    FastAPI framework, high performance, easy to learn.

    <a href="https://github.com/fastapi/fastapi/actions?query=workflow%3ATest+event%3Apush+branch%3Amaster" target="_blank">
    <img src="https://github.com/fastapi/fastapi/workflows/Test/badge.svg" alt="Test">
    </a>
    <a href="https://pypi.org/project/fastapi" target="_blank">
        <img src="https://img.shields.io/pypi/v/fastapi?color=%2334D058&label=pypi%20package" alt="Package version">
    </a>

    Here is a code example:
    {* ../../docs_src/main.py *}

    Same code appears again (should be included, not replaced with reference):
    {* ../../docs_src/main.py *}

    <div class="termy">
    ```console
    $ uvicorn main:app --reload
    </div> """

    target_content = ""

    if os.path.exists(input_md_file):
        print(f"Reading file: {input_md_file} ...")
        with open(input_md_file, "r", encoding="utf-8") as f:
            target_content = f.read()
    else:
        print(f"⚠️ File '{input_md_file}' not found, using test string...")
        target_content = fake_md_content

    # Execute
    processed_result = clean_fastapi_markdown(target_content, root_dir=current_root_dir)

    print("\n" + "="*20 + " Processing Result Preview " + "="*20)
    print(processed_result[:1000])
    print("="*50)

    output_filename = "test_clean_result_no_dedup.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(processed_result)

    print(f"\n✅ Done! Full result saved to: {os.path.join(current_root_dir, output_filename)}")
