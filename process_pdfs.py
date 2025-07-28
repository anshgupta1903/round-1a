import fitz  # PyMuPDF
import json
import os
import re
import time
from collections import Counter
import pandas as pd
import joblib

def detect_headers_and_footers(doc, line_threshold=0.4):
    """
    Detects repeating text that is likely a header or footer based on
    frequency and position on the page. This is a critical first step.
    """
    page_count = len(doc)
    if page_count < 4: return set()

    text_counts = Counter()
    # Scan the middle half of the document to avoid title and reference pages
    start_page = page_count // 4
    end_page = page_count - start_page

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        for b in blocks:
            # Check a wider vertical area: top 20% and bottom 15% of the page
            if b[1] < page_height * 0.20 or b[3] > page_height * 0.85:
                line_text = b[4].strip().replace('\n', ' ')
                if 5 < len(line_text) < 100 and not line_text.endswith('.'):
                    text_counts[line_text] += 1
    
    ignore_set = set()
    # Lower the threshold to catch text that appears on 40% of scanned pages
    min_occurrences = (end_page - start_page) * line_threshold
    for text, count in text_counts.items():
        if count >= min_occurrences:
            ignore_set.add(text)
            
    print(f"INFO: Detected {len(ignore_set)} repeating lines to ignore as headers/footers.")
    return ignore_set

def is_line_in_table(line_bbox, page_table_areas):
    """Checks if a line's bounding box is inside any of a page's table areas."""
    if not page_table_areas:
        return False
    
    l_x0, l_y0, l_x1, l_y1 = line_bbox
    for t_bbox in page_table_areas:
        t_x0, t_y0, t_x1, t_y1 = t_bbox
        # Check for containment. A line is in a table if its bbox is inside the table's bbox.
        if l_x0 >= t_x0 and l_y0 >= t_y0 and l_x1 <= t_x1 and l_y1 <= t_y1:
            return True
    return False

def get_dominant_style(line):
    """
    Determines the most common (dominant) style in a line of text.
    This is more robust than just checking the first span.
    """
    if not line["spans"]:
        return (10, False) # Default style

    style_counts = Counter()
    for span in line["spans"]:
        # More robust check for bold fonts
        is_bold = bool(re.search(r'bold|black|heavy', span["font"], re.IGNORECASE))
        style = (round(span["size"]), is_bold)
        # We weigh the style by the length of the text in the span
        style_counts[style] += len(span["text"].strip())
    
    # Return the most common style
    return style_counts.most_common(1)[0][0]

def is_mostly_uppercase(s):
    """
    Checks if a string is predominantly uppercase. More robust than isupper().
    """
    letters = [char for char in s if char.isalpha()]
    if not letters:
        return False
    uppercase_letters = [char for char in letters if char.isupper()]
    return (len(uppercase_letters) / len(letters)) > 0.8

def get_page_layout(page, threshold=0.3):
    """
    Analyzes the layout of a page to determine if it is single or multi-column.
    Returns the number of detected columns (1 or 2).
    """
    page_width = page.rect.width
    midpoint = page_width / 2
    
    blocks = page.get_text("blocks")
    if not blocks:
        return 1 # Default to 1 column if no text

    left_blocks = 0
    right_blocks = 0
    
    for b in blocks:
        if b[2] < midpoint: # Block ends before midpoint
            left_blocks += 1
        elif b[0] > midpoint: # Block starts after midpoint
            right_blocks += 1

    total_sided_blocks = left_blocks + right_blocks
    if total_sided_blocks == 0:
        return 1

    # Heuristic: If there are a significant number of blocks on both sides, it's a 2-column layout.
    if (left_blocks > 0 and right_blocks > 0):
        if (left_blocks / total_sided_blocks > threshold) and (right_blocks / total_sided_blocks > threshold):
            return 2
    
    return 1

def process_pdf(pdf_path, ml_output_path=None):
    """
    Processes a PDF using a hybrid ML and rule-based filtering approach.
    """
    doc = fitz.open(pdf_path)
    ignored_texts = detect_headers_and_footers(doc)
    
    table_areas = {}
    for page_num, page in enumerate(doc):
        tables = page.find_tables()
        if tables.tables:
            table_areas[page_num] = [t.bbox for t in tables]
    
    if table_areas:
        print(f"INFO: Detected tables on pages: {list(table_areas.keys())}")
        
    all_lines = []
    style_counts = Counter()
    page_heights = {}
    
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_heights[page_num] = page.rect.height
        page_table_bboxes = table_areas.get(page_num, [])
        
        num_columns = get_page_layout(page)
        page_midpoint = page_width / 2

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if not line["spans"]: continue
                    if is_line_in_table(line["bbox"], page_table_bboxes): continue

                    line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if not line_text or line_text in ignored_texts: continue

                    # Skip common date lines
                    text_lower = line_text.lower()
                    if re.search(r'\b\d{4}\b', text_lower):
                        if any(month in text_lower for month in [
                            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                            'aug', 'sep', 'oct', 'nov', 'dec']):
                            if len(line_text.split()) <= 4:
                                continue
                    if re.match(r'^Page \d+\s*of\s*\d+$', line_text, re.I): continue

                    style = get_dominant_style(line)
                    style_counts[style] += 1

                    # Detect column index
                    line_center_x = (line["bbox"][0] + line["bbox"][2]) / 2
                    column_index = 0
                    if num_columns == 2 and line_center_x > page_midpoint:
                        column_index = 1

                    all_lines.append({
                        "page": page_num,
                        "text": line_text,
                        "style": style,
                        "is_bold": style[1],
                        "size": style[0],
                        "x0": line["bbox"][0],
                        "y0": line["bbox"][1],
                        "x1": line["bbox"][2],
                        "y1": line["bbox"][3],
                        "column": column_index,
                        "id": f"{page_num}-{line['bbox'][1]}",
                    })

    if not all_lines:
        return {"title": "", "outline": []}
    
    # --- TITLE IDENTIFICATION ---
    page_0_height = page_heights.get(0, 1000)
    page_0_lines_top_half = sorted(
        [line for line in all_lines if line["page"] == 0 and line["y0"] < page_0_height / 2],
        key=lambda x: (x["column"], x["y0"])
    )

    doc_title = ""
    title_line_ids = set()
    if page_0_lines_top_half:
        try:
            sizes = [line["size"] for line in page_0_lines_top_half]
            max_size = max(sizes)
            min_size = min(sizes)

            if max_size - min_size < 1:
                # fallback: bold & centered
                page_width = doc[0].rect.width
                center_x = page_width / 2
                centered_bold_lines = [
                    line for line in page_0_lines_top_half
                    if line["is_bold"] and abs((line["x0"] + line["x1"]) / 2 - center_x) < page_width * 0.1
                ]
                if centered_bold_lines:
                    title_lines = [centered_bold_lines[0]["text"]]
                    title_line_ids.add(centered_bold_lines[0]["id"])
                    if len(centered_bold_lines) > 1:
                        title_lines.append(centered_bold_lines[1]["text"])
                        title_line_ids.add(centered_bold_lines[1]["id"])
                    doc_title = " ".join(title_lines)
            else:
                first_title_line = next(line for line in page_0_lines_top_half if line["size"] == max_size)
                start_index = page_0_lines_top_half.index(first_title_line)
                title_lines = []
                last_line = None
                for i in range(start_index, len(page_0_lines_top_half)):
                    current_line = page_0_lines_top_half[i]
                    if len(title_lines) >= 2: break
                    if last_line:
                        if abs(current_line["y0"] - last_line["y0"]) > last_line["size"] * 2.5: break
                        if current_line["size"] < first_title_line["size"] * 0.7: break
                    title_lines.append(current_line["text"])
                    title_line_ids.add(current_line["id"])
                    last_line = current_line
                doc_title = " ".join(title_lines)
        except Exception:
            doc_title = ""

    if title_line_ids:
        print(f"INFO: Identified title: '{doc_title}'. Excluding {len(title_line_ids)} lines from heading analysis.")
        all_lines = [line for line in all_lines if line['id'] not in title_line_ids]

    # --- HEADING DETECTION ---
    non_bold_styles = [s for s, c in style_counts.items() if not s[1]]
    body_style = (10, False)
    if non_bold_styles:
        body_style = Counter({s: style_counts[s] for s in non_bold_styles}).most_common(1)[0][0]
    elif style_counts:
        body_style = style_counts.most_common(1)[0][0]
    print(f"INFO: Deduced body text style: {body_style} (size, is_bold)")

    # Detect if page 0 has real paragraphs
    page_0_has_paragraphs = any(
        line['page'] == 0 and line['style'] == body_style
        and len(line['text'].split()) >= 30
        and not is_mostly_uppercase(line['text'])
        and len(line['text']) > 30
        for line in all_lines
    )
    if not page_0_has_paragraphs:
        print("INFO: Page 0 has no paragraph text. It will be ignored for headings.")

    # Filter initial heading candidates
    initial_candidates = []
    for line in all_lines:
        if not page_0_has_paragraphs and line['page'] == 0:
            continue
        is_distinct = (line['size'] > body_style[0]) or (line['is_bold'] and not body_style[1])
        if not is_distinct: continue
        if not (3 < len(line['text']) < 250): continue
        if len(line['text'].split()) > 25: continue
        if re.fullmatch(r"[\d\W_]+", line['text']): continue
        if line['text'].endswith(('.', ',', ';')) and len(line['text'].split()) > 15: continue
        initial_candidates.append(line)

    if not initial_candidates:
        return {"title": doc_title, "outline": []}

    # Assign heading levels (limit to H1–H3)
    heading_styles = sorted(set(c['style'] for c in initial_candidates), key=lambda s: (-s[0], -s[1]))
    style_to_level = {style: f"H{i+1}" for i, style in enumerate(heading_styles[:3])}
    print("INFO: Detected heading style hierarchy:")
    for style, level in style_to_level.items():
        print(f"  - {level}: {style}")

    refined_headings = []
    for cand in initial_candidates:
        cand['level'] = style_to_level.get(cand['style'])
        if cand['level']:
            refined_headings.append(cand)

    # Remove page 0 headings if no paragraphs there
    if not page_0_has_paragraphs:
        refined_headings = [h for h in refined_headings if h['page'] != 0]

    # --- Final merging & cleaning ---
    sorted_headings = sorted(refined_headings, key=lambda x: (x['page'], x['column'], x['y0']))
    outline = []
    i = 0
    while i < len(sorted_headings):
        current = sorted_headings[i]
        j = i+1
        while j < len(sorted_headings):
            next_line = sorted_headings[j]
            if (next_line['page'] == current['page'] and next_line['column'] == current['column']
                and next_line['style'] == current['style']
                and abs(next_line['y0'] - sorted_headings[j-1]['y1']) < current['size'] * 0.5):
                current['text'] += " " + next_line['text']
                current['y1'] = next_line['y1']
                j += 1
            else:
                break
        outline.append({"level": current['level'], "text": current['text'].strip(), "page": current['page']})
        i = j

    return {"title": doc_title, "outline": outline}



def process_all_pdfs(input_dir, output_dir):
    """
    Processes all PDF files in a given directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"--- Processing {filename} ---")
            start_time = time.time()
            
            base_filename = os.path.splitext(filename)[0]
            json_output_path = os.path.join(output_dir, base_filename + ".json")
            
            output_data = process_pdf(pdf_path)
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            end_time = time.time()
            print(f"--- Finished {filename} in {end_time - start_time:.2f} seconds. ---")




if __name__ == "__main__":
    print("Starting PDF processing...")
    INPUT_DIR = "./input"
    OUTPUT_DIR = "./output"
    
    # if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
    # if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    # if not os.path.exists(ML_OUTPUT_DIR): os.makedirs(ML_OUTPUT_DIR)
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR)


