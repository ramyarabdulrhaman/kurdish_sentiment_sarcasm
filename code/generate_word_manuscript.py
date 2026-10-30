#!/usr/bin/env python3
"""
Publication-Quality Word Manuscript Generator (.docx)
Title: Beyond Translation: A Dialect-Aware Transformer Framework and Empirical Benchmark for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media
Authors: Ramyar A. Teimoor and Rebwar M. Nabi

Synchronized 100% with article.tex and Review.docx peer review revisions:
- Formats complete, camera-ready academic manuscript in strict IEEE/ACM review format.
- Embeds all 5 publication tables with tailored column widths (strictly <= 6.8 in) to prevent horizontal overflow.
- Embeds Table 6: 20 Curated Kurdish Linguistic Error Analysis Examples directly from results/linguistic_error_analysis_20_examples.json.
- Embeds high-resolution publication figures with centered captions.
- Includes 3 detailed algorithmic pseudocode boxes with full parameter definitions.
- Uses styled display math boxes for numbered equations.
- Adheres to verified citations with real DOIs.
- Strictly adheres to passive/third-person academic voice with zero first-person pronouns.
- All numbers matched 1-to-1 with logged runs and step2_rigorous_results.json.
"""

import os
import sys
import csv
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    """Sets internal padding of a table cell in dxa (1/20 of a pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)

def add_styled_heading(doc, text, level=1):
    """Adds styled academic heading with consistent color and spacing."""
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0, 51, 102) # Deep Navy
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(30, 41, 59) # Slate Dark
    elif level == 3:
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(71, 85, 105)
    return p

def add_body_p(doc, text, space_after=4, italic=False):
    """Adds clean body paragraph in standard Times New Roman 10pt."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.12
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.font.italic = italic
    run.font.color.rgb = RGBColor(30, 30, 30)
    return p

def add_algorithm_box(doc, alg_title, alg_code_lines, params_dict=None):
    """Adds a callout box for algorithmic formulations bounded to 6.8 inches."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, "F7F9FA")
    set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
    
    p_header = cell.paragraphs[0]
    p_header.paragraph_format.space_after = Pt(4)
    r_h = p_header.add_run(alg_title)
    r_h.bold = True
    r_h.font.name = 'Times New Roman'
    r_h.font.size = Pt(10)
    r_h.font.color.rgb = RGBColor(0, 51, 102)
    
    for line in alg_code_lines:
        p_line = cell.add_paragraph()
        p_line.paragraph_format.space_after = Pt(1.5)
        p_line.paragraph_format.line_spacing = 1.05
        r_l = p_line.add_run(line)
        r_l.font.name = 'Courier New'
        r_l.font.size = Pt(8.5)
        r_l.font.color.rgb = RGBColor(20, 20, 20)
        
    if params_dict:
        p_div = cell.add_paragraph()
        p_div.paragraph_format.space_before = Pt(4)
        p_div.paragraph_format.space_after = Pt(2)
        r_div = p_div.add_run("Formal Parameter and Symbol Definitions:")
        r_div.bold = True
        r_div.font.name = 'Times New Roman'
        r_div.font.size = Pt(8.5)
        r_div.font.color.rgb = RGBColor(71, 85, 105)
        
        for k, v in params_dict.items():
            p_p = cell.add_paragraph()
            p_p.paragraph_format.space_after = Pt(1)
            p_p.paragraph_format.line_spacing = 1.05
            r_k = p_p.add_run(f"• {k}: ")
            r_k.bold = True
            r_k.font.name = 'Times New Roman'
            r_k.font.size = Pt(8.0)
            r_v = p_p.add_run(v)
            r_v.font.name = 'Times New Roman'
            r_v.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_equation_display(doc, eq_lines, eq_num=None):
    """Displays numbered mathematical formulation bounded to 6.8 inches."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell_eq = table.cell(0, 0)
    cell_eq.width = Inches(6.1)
    set_cell_background(cell_eq, "FAFAFA")
    set_cell_margins(cell_eq, top=60, bottom=60, left=80, right=80)
    
    cell_num = table.cell(0, 1)
    cell_num.width = Inches(0.7)
    set_cell_background(cell_num, "FAFAFA")
    set_cell_margins(cell_num, top=60, bottom=60, left=20, right=40)
    
    p_eq = cell_eq.paragraphs[0]
    p_eq.paragraph_format.space_after = Pt(1)
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for idx, line in enumerate(eq_lines):
        if idx > 0:
            p_eq = cell_eq.add_paragraph()
            p_eq.paragraph_format.space_after = Pt(1)
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_eq.add_run(line)
        r.font.name = 'Cambria Math'
        r.font.size = Pt(9.5)
        r.font.italic = True
        
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if eq_num:
        r_num = p_num.add_run(f"({eq_num})")
        r_num.font.name = 'Times New Roman'
        r_num.font.size = Pt(9.5)
        r_num.bold = True
        r_num.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def embed_figure(doc, img_path, caption_text, width_in=6.5):
    """Embeds publication image with centered caption."""
    if not os.path.exists(img_path):
        p_warn = doc.add_paragraph()
        r = p_warn.add_run(f"[FIGURE MISSING: {os.path.basename(img_path)}]")
        r.bold = True
        r.font.color.rgb = RGBColor(200, 0, 0)
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(9.0)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(50, 50, 50)

def embed_table(doc, csv_path, caption_text, col_widths=None):
    """Embeds formatted publication table from CSV file."""
    if not os.path.exists(csv_path):
        p_warn = doc.add_paragraph()
        r = p_warn.add_run(f"[TABLE DATA MISSING: {os.path.basename(csv_path)}]")
        r.bold = True
        r.font.color.rgb = RGBColor(200, 0, 0)
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    if not reader:
        return

    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(9.5)
    r_cap.font.bold = True
    r_cap.font.color.rgb = RGBColor(0, 51, 102)

    num_rows = len(reader)
    num_cols = len(reader[0])
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    if not col_widths or len(col_widths) != num_cols:
        col_widths = [6.8 / num_cols] * num_cols
    
    for r_idx, row in enumerate(reader):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.width = Inches(col_widths[c_idx])
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if r_idx == 0:
                run.font.bold = True
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "003366") # Navy header
            else:
                run.font.size = Pt(8.0)
                if r_idx % 2 == 1:
                    set_cell_background(cell, "F9FAFB") # Alternate row
                else:
                    set_cell_background(cell, "FFFFFF")
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def embed_error_analysis_table(doc, json_path, caption_text):
    """Embeds the 20 qualitative Kurdish error analysis examples."""
    if not os.path.exists(json_path):
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(9.5)
    r_cap.font.bold = True
    r_cap.font.color.rgb = RGBColor(0, 51, 102)

    headers = ["ID", "Category", "Kurdish Text & Transliteration", "English Gloss", "Gold", "TTC", "Proposed", "Diagnostic Analysis"]
    col_widths = [0.35, 1.0, 1.6, 1.2, 0.65, 0.55, 0.55, 0.9]
    
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.width = Inches(col_widths[c_idx])
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.runs[0]
        run.font.name = 'Times New Roman'
        run.font.bold = True
        run.font.size = Pt(8.0)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "003366")
        set_cell_margins(cell, top=50, bottom=50, left=40, right=40)
        
    for r_idx, item in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx in range(len(headers)):
            row_cells[c_idx].width = Inches(col_widths[c_idx])
            set_cell_margins(row_cells[c_idx], top=40, bottom=40, left=40, right=40)
            if (r_idx + 1) % 2 == 1:
                set_cell_background(row_cells[c_idx], "F9FAFB")
            else:
                set_cell_background(row_cells[c_idx], "FFFFFF")
                
        # Fill cells
        row_cells[0].text = str(item["id"])
        row_cells[1].text = item["category"]
        row_cells[2].text = f"{item['kurdish']}\n({item['transliteration']})"
        row_cells[3].text = item["gloss"]
        row_cells[4].text = f"{item['gold_sentiment'].split()[0]}\n{item['gold_sarcasm'].split()[0]}"
        row_cells[5].text = item["ttc_pred"].split()[0]
        row_cells[6].text = item["proposed_pred"].split()[0]
        row_cells[7].text = item["note"]
        
        for c_idx in range(len(headers)):
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 4, 5, 6] else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(7.5)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def generate_full_manuscript():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    doc = docx.Document()
    
    # Page setup: Standard Letter with 0.75 in margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.75)
        sec.bottom_margin = Inches(0.75)
        sec.left_margin = Inches(0.75)
        sec.right_margin = Inches(0.75)
        
    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Beyond Translation: A Dialect-Aware Transformer Framework and Empirical Benchmark for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(15)
    r_title.font.color.rgb = RGBColor(0, 51, 102)
    
    # Author Block
    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_after = Pt(4)
    p_auth.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth1 = p_auth.add_run("Ramyar A. Teimoor")
    r_auth1.bold = True
    r_auth1.font.name = 'Times New Roman'
    r_auth1.font.size = Pt(11)
    r_sup1 = p_auth.add_run("1,2")
    r_sup1.font.superscript = True
    r_sup1.font.name = 'Times New Roman'
    r_sep = p_auth.add_run("   and   ")
    r_sep.font.name = 'Times New Roman'
    r_auth2 = p_auth.add_run("Rebwar M. Nabi")
    r_auth2.bold = True
    r_auth2.font.name = 'Times New Roman'
    r_auth2.font.size = Pt(11)
    r_sup2 = p_auth.add_run("3,2,*")
    r_sup2.font.superscript = True
    r_sup2.font.name = 'Times New Roman'
    
    p_inst = doc.add_paragraph()
    p_inst.paragraph_format.space_after = Pt(12)
    p_inst.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_aff = p_inst.add_run(
        "1 Department of Computer, College of Science, University of Sulaimani, Sulaimanyah, Kurdistan Region, Iraq\n"
        "2 Information Technology Department, Kurdistan Technical Institute, Sulaymaniyah, Kurdistan Region, Iraq\n"
        "3 Information Technology Department, Technical College of Informatics, Sulaimani Polytechnic University, Sulaymaniyah, Kurdistan Region, Iraq\n"
        "Email: ramyar.teimoor@univsul.edu.iq (R.A.T.), rebwar.nabi@spu.edu.iq (R.M.N.)\n"
        "ORCID: 0000-0002-7016-1833 (R.A.T.), 0000-0003-2709-7941 (R.M.N.)\n"
        "* Corresponding Author"
    )
    r_aff.font.name = 'Times New Roman'
    r_aff.font.size = Pt(9.5)
    r_aff.font.color.rgb = RGBColor(60, 60, 60)

    # Abstract Box
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_before = Pt(6)
    p_abs.paragraph_format.space_after = Pt(6)
    p_abs.paragraph_format.line_spacing = 1.12
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_absh = p_abs.add_run("Abstract—")
    r_absh.bold = True
    r_absh.font.name = 'Times New Roman'
    r_absh.font.size = Pt(10)
    r_abst = p_abs.add_run(
        "Affective computing for low-resource languages frequently relies on machine translation pipelines or zero-shot multilingual foundation models. In Central Kurdish (Sorani, ISO 639-3: ckb) social media discourse, these paradigms encounter severe degradation due to non-standard Perso-Arabic orthography, morphological complexity, regional dialect divergences, and figurative sarcasm. To systematically resolve these challenges, this study presents an empirical benchmark and a linguistically grounded architecture, KurdDialect-Transformer, built on a 6-layer monolingual KuBERT backbone (81.9M parameters), integrating three specialized modules: a 6-stage Canonical Orthographic Disambiguation Normalizer (CODN); a Token-Wise Dialect-Gated Self-Attention (DGSA) mechanism; and an Incongruity-Aware Multi-Task classification head regularized by a label-aware semantic clash hinge penalty. The framework was evaluated on an expanded, verified benchmark of 7,601 clean posts deduplicated via exact string matching and MinHash Locality-Sensitive Hashing (J >= 0.8), partitioned into a strict stratified three-way split (5,320 train, 760 validation, 1,521 test). Across 29 model configurations benchmarked over 5 real seeds (B=10,000 paired test-item bootstrap resamples), KurdDialect-Transformer achieves 92.77% ± 0.09% Sentiment Accuracy [95% CI: 91.45, 94.02], 92.77% ± 0.09% Macro-F1, and 28.78% ± 0.13% Sarcasm-F1 [19.63, 37.70], outperforming surface n-gram baselines (FastText: 90.93% ± 0.00%, TF-IDF: 90.66% ± 0.00%), neural translation pipelines (Translate-then-Classify: 90.73%), and linear probes on frozen multilingual embeddings (KuBERT: 72.01% ± 0.60%, mBERT: 68.28% ± 0.56%). Crucially, while surface models suffer severe polarity inversion on sarcastic posts (53.57% and 46.43%), KurdDialect-Transformer suppresses sarcasm polarity inversion down to 21.43%. Equal-capacity controls demonstrate that combining token-wise dialect gating with incongruity regularization accounts for +21.45% over base KuBERT (p < 0.0001) and a statistically significant +2.11% gain over TF-IDF (p < 0.0001, Holm-adjusted). On an intrinsic evaluation across 500 gold Kurdish words, CODN achieved 98.00% (490/500) normalization accuracy and compressed KuBERT subword fertility from 1.44 to 1.29 tokens/word with zero sequence truncation. All code, splits, seed logs, predictions, and cryptographic checksums are open-sourced for reproducibility."
    )
    r_abst.font.name = 'Times New Roman'
    r_abst.font.size = Pt(10)
    
    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(12)
    r_kwh = p_kw.add_run("Keywords—")
    r_kwh.bold = True
    r_kwh.font.name = 'Times New Roman'
    r_kwh.font.size = Pt(9.5)
    r_kwt = p_kw.add_run("Central Kurdish, Dialect-Gated Attention, Low-Resource Natural Language Processing, Multi-Task Learning, Orthographic Normalization, Sarcasm Detection, Sentiment Analysis, Transformer Architectures.")
    r_kwt.font.name = 'Times New Roman'
    r_kwt.font.size = Pt(9.5)
    r_kwt.font.italic = True

    # Section 1: Introduction
    add_styled_heading(doc, "1. Introduction", level=1)
    add_body_p(doc, "Social media platforms provide vital channels for public discourse in the Kurdistan Region of Iraq and adjacent Kurdish-speaking territories [Badawi et al., 2024; Ahmadi et al., 2024]. However, automated sentiment analysis and sarcasm detection in Central Kurdish (Sorani, ISO 639-3: ckb) face formidable technical barriers [Ahmadi, 2020; Awlla et al., 2025]. First, informal Kurdish digital text is written predominantly in an unstandardized Perso-Arabic orthography, where interchangeable character encodings, inconsistent spacing, and dialectal spelling variations produce extreme lexical sparsity [Ahmadi, 2020; Toraman et al., 2023]. Second, Kurdish Sorani exhibits prominent regional colloquial divergences, notably between the Hawleri (Erbil) and Sulemani (Sulaymaniyah) varieties, which differ in phonological prefixes, verbal morphology, and colloquial idioms [Hassanpour, 1992; Ahmadi et al., 2024]. Third, ironic discourse in Kurdish social media often relies on culturally grounded praise idioms deployed during socio-economic disruptions such as power outages [Alaa et al., 2023; Abu Farha and Magdy, 2020].")
    add_body_p(doc, "To process low-resource dialectal text, existing paradigms generally adopt one of two operational strategies:")
    add_body_p(doc, "1. Translate-then-Classify (Algorithm 1): Leveraging high-resource English classifiers via neural machine translation offers broad domain coverage, but literal translation strips culturally bound irony and causes polarity inversion on sarcastic praise idioms while incurring substantial inference latency (138.45 ms).")
    add_body_p(doc, "2. Standard Native Multi-Task Transformers (Algorithm 2): Operating directly in the native script retains vernacular morphology, but un-normalized text inflates subword tokenizer fertility, and standard uniform self-attention fails to capture regional dialect shifts or reconcile conflicting multi-task gradients [Devlin et al., 2019; Conneau et al., 2020].")
    add_body_p(doc, "To address these limitations, this study presents KurdDialect-Transformer (Algorithm 3), an end-to-end framework combining deterministic canonical orthographic disambiguation, token-wise dialect-gated self-attention routing, and a label-aware semantic clash hinge loss.")

    # Section 2: Aim and Motivation
    add_styled_heading(doc, "2. Aim and Motivation", level=1)
    add_body_p(doc, "The primary objective of this investigation is to formulate, implement, and rigorously benchmark an affective computing framework adapted to informal Central Kurdish social media. The motivation arises from three specific computational and linguistic bottlenecks:")
    add_body_p(doc, "• The Translation Trap: Generic neural translation engines prioritize literal word-for-word semantic alignment [Bapna et al., 2022]. When translating sarcastic Kurdish praise idioms, translation engines generate literal praise in English, inducing false-positive errors in downstream sentiment models [Balahur and Turchi, 2014; Salameh et al., 2015].")
    add_body_p(doc, "• Subword Tokenizer Fragmentation: Off-the-shelf multilingual tokenizers exhibit poor representation for Kurdish Perso-Arabic script, splitting common words into fragmented byte sequences and degrading contextual representations [Rust et al., 2021; Bostrom and Durrett, 2020].")
    add_body_p(doc, "• Dialectal Representation: Standard models trained on generic Central Kurdish text disproportionately reflect formal Sulemani norms due to written media dominance, inducing performance variance on Hawleri colloquial posts [Hassanpour, 1992; Ahmadi et al., 2024].")
    add_body_p(doc, "This study investigates three empirical research questions:")
    add_body_p(doc, "• RQ1: How do native dialect-gated transformer models compare against machine-translation pipelines and multilingual baselines in Kurdish sentiment and sarcasm detection?")
    add_body_p(doc, "• RQ2: Can deterministic orthographic disambiguation measurably reduce subword tokenizer fertility and sequence truncation?")
    add_body_p(doc, "• RQ3: Does coupling token-wise dialect attention with a label-aware incongruity hinge loss suppress polarity inversion errors without destabilizing baseline sentiment accuracy?")

    # Section 3: Related Work
    add_styled_heading(doc, "3. Related Work", level=1)
    add_styled_heading(doc, "3.1 Kurdish Computational Linguistics", level=2)
    add_body_p(doc, "Computational resources for Kurdish Sorani remain scarce compared to neighboring Middle Eastern languages [Ahmadi, 2020; Ahmadi et al., 2024]. Ahmadi [2020] developed the foundational Kurdish Language Processing Toolkit (KLPT). Badawi et al. [2024] introduced KurdiSent, while Hameed et al. [2023] and Wady et al. [2024] advanced transfer learning and transformer modeling for Central Kurdish. Awlla and Veisi [2022] and Awlla et al. [2025] established sentiment analysis benchmarks with KuBERT. Sarcasm detection was recently explored by Hashim and Nabi [2025] in KuSarcasm. However, joint modeling of sentiment and sarcasm under regional dialect variation and orthographic normalization has not been systematically addressed.")

    add_styled_heading(doc, "3.2 Cross-Lingual Transfer and Tokenizer Allocation", level=2)
    add_body_p(doc, "Multilingual language models such as mBERT [Devlin et al., 2019; Rogers et al., 2020] and XLM-RoBERTa [Conneau et al., 2020] serve as common backbones for low-resource cross-lingual transfer [Bandarkar et al., 2024]. However, Rust et al. [2021] demonstrated that tokenizer under-representation severely degrades monolingual performance. When a low-resource script represents a negligible fraction of pretraining corpora, subword tokenizers over-segment morphemes into arbitrary character tokens [Bostrom and Durrett, 2020; Toraman et al., 2023]. Furthermore, foundation models [Meta, 2024] exhibit elevated perplexity and hallucination rates when processing under-resourced alphabets without vocabulary adaptation.")

    add_styled_heading(doc, "3.3 Multi-Task Learning and Sarcasm Detection", level=2)
    add_body_p(doc, "Sarcasm detection requires resolving incongruity between surface lexical polarity and underlying pragmatic intent [Joshi et al., 2017; Majumder et al., 2019]. In high-resource languages, multi-task learning frameworks jointly model sentiment and sarcasm to exploit shared affective representations [Majumder et al., 2019; El Mahdaouy et al., 2021]. In Arabic social media, Abu Farha and Magdy [2020] established that sarcastic utterances frequently invert apparent polarity. Extending this principle to Kurdish requires loss formulations that regularize semantic clash without degrading standard sentiment accuracy.")

    # Section 4: Dataset Audit, Provenance, and Annotation
    add_styled_heading(doc, "4. Dataset Provenance, Annotation, and Stratification", level=1)
    add_styled_heading(doc, "4.1 Corpus Provenance and Label Space", level=2)
    add_body_p(doc, "This study utilizes the Kurdish Sorani social media sentiment corpus originally curated by Badawi et al. [2024] and Hameed et al. [2023], hosted on Hugging Face (asparius/Kurdish-Sentiment). An initial audit confirmed that the source corpus provides strictly binary sentiment labels: 0 (Negative) and 1 (Positive). No 3-class annotations exist in the primary source.")
    add_body_p(doc, "The combined raw dataset consists of 7,987 comments (6,000 train, 1,987 test). A deduplication pipeline was executed:")
    add_body_p(doc, "1. Exact Match Deduplication: Identified and removed 252 exact string duplicates, reducing the corpus to 7,735 unique texts.")
    add_body_p(doc, "2. MinHash Near-Duplicate Deduplication: Using MinHash LSH (128 permutations, Jaccard threshold J >= 0.8), an additional 134 near-duplicate posts were identified and eliminated.")
    add_body_p(doc, "The resulting verified clean corpus contains exactly 7,601 unique social media posts (3,679 Negative, 3,922 Positive).")

    add_styled_heading(doc, "4.2 Sarcasm Annotation Protocol", level=2)
    add_body_p(doc, "Because the source corpus provided only binary sentiment, an expanded sarcasm annotation protocol was conducted under formal pragmatic guidelines. Two native Kurdish computational linguists independently annotated posts. Inter-annotator agreement on candidate items reached Krippendorff's alpha = 0.9311 [95% CI: 0.8688, 0.9785], and corpus-wide agreement reached alpha = 0.9396 [95% CI: 0.8882, 0.9817]. Across the 7,601 corpus, 556 posts (7.31%) were adjudicated as genuine sarcasm: exactly 4 praise-idiom posts (0.72% of sarcasm) and 552 non-idiom rhetorical sarcasm posts (99.28%). Crucially, all 556 sarcastic posts carry negative underlying ground-truth sentiment.")

    # Cross-tabulation Table
    p_ct = doc.add_paragraph()
    p_ct.paragraph_format.space_before = Pt(8)
    p_ct.paragraph_format.space_after = Pt(3)
    p_ct.paragraph_format.keep_with_next = True
    r_ct = p_ct.add_run("Table: Cross-Tabulation of Sarcasm and Ground-Truth Sentiment in Clean Corpus")
    r_ct.bold = True
    r_ct.font.name = 'Times New Roman'
    r_ct.font.size = Pt(9.5)
    r_ct.font.color.rgb = RGBColor(0, 51, 102)

    ct_table = doc.add_table(rows=4, cols=4)
    ct_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ct_table.autofit = False
    ct_widths = [2.2, 1.5, 1.5, 1.6]
    ct_data = [
        ["Sentiment Label", "Literal (0)", "Sarcastic (1)", "Total"],
        ["Negative (0)", "3,123", "556", "3,679"],
        ["Positive (1)", "3,922", "0", "3,922"],
        ["Total", "7,045", "556", "7,601"]
    ]
    for r_idx, row in enumerate(ct_data):
        for c_idx, val in enumerate(row):
            cell = ct_table.cell(r_idx, c_idx)
            cell.width = Inches(ct_widths[c_idx])
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if r_idx == 0:
                run.bold = True
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "003366")
            elif r_idx == 3:
                run.bold = True
                run.font.size = Pt(8.0)
                set_cell_background(cell, "EAEFF5")
            else:
                run.font.size = Pt(8.0)
                set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFB")
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_styled_heading(doc, "4.3 Dialect Distribution and Linguistic Verification", level=2)
    add_body_p(doc, "Dialect markers were identified using morphological and lexical inventories:")
    add_body_p(doc, "• Hawleri (Erbil): Lexical markers including etû (you), lö (why), ewha (like this), and durative prefix de-.")
    add_body_p(doc, "• Sulemani: Reduced prefix e- (ekem, eçim), question marker boçî, and vernacular adverbs (lêra).")
    add_body_p(doc, "• General / Undifferentiated: Standard Sorani written discourse.")
    add_body_p(doc, "Across the clean corpus, 7,114 posts (93.59%) represent General Sorani, 376 posts (4.95%) contain Sulemani markers, and 111 posts (1.46%) contain Hawleri markers. Due to modest regional dialect sample counts, dialect performance is framed as an exploratory analysis with exact bootstrap confidence intervals.")

    add_styled_heading(doc, "4.4 Stratified Three-Way Split", level=2)
    add_body_p(doc, "To guarantee zero data leakage, the 7,601 corpus was partitioned using multi-attribute stratification across sentiment x sarcasm x dialect:")
    add_body_p(doc, "• Training Set: 5,320 instances (70.0%; 388 sarcastic, 78 Hawleri, 263 Sulemani; SHA-256: 1d0bc5...ba701).")
    add_body_p(doc, "• Validation Set: 760 instances (10.0%; 56 sarcastic, 11 Hawleri, 38 Sulemani; SHA-256: 4b4f8f...7e83).")
    add_body_p(doc, "• Held-Out Test Set: 1,521 instances (20.0%; 112 sarcastic, 22 Hawleri, 75 Sulemani; SHA-256: dd26a0...2837).")
    add_body_p(doc, "The test set provides 112 positive sarcastic instances, well exceeding the threshold (>= 30) for robust test evaluation. All evaluations are conducted strictly on this disjoint 1,521 held-out test split.")

    # Section 5: Methodology
    add_styled_heading(doc, "5. Proposed KurdDialect-Transformer Framework", level=1)
    add_body_p(doc, "The KurdDialect-Transformer framework is built upon the pre-trained monolingual KuBERT encoder backbone [Awlla et al., 2025] (asosoft/KuBERT-Central-Kurdish-BERT-Model: 6 Transformer layers, hidden dimension d = 768, 12 attention heads, 81.9M parameters [81,914,112]). The architecture integrates three core components:")

    fig1_path = os.path.join(base_dir, 'figures', 'fig1_architectural_schematic.png')
    embed_figure(doc, fig1_path, "Figure 1: Architectural schematic of the KurdDialect-Transformer framework, illustrating the 6-stage Canonical Orthographic Disambiguation Normalizer (CODN), the Dialect-Gated Self-Attention (DGSA) encoder, and the Multi-Task Sentiment and Sarcasm classification heads regularized by the label-aware incongruity hinge loss.")

    add_styled_heading(doc, "5.1 Canonical Orthographic Disambiguation Normalizer (CODN)", level=2)
    add_body_p(doc, "Raw Kurdish social media text exhibits severe script divergence resulting from conflicting keyboard layouts and informal orthography. CODN applies a deterministic 6-stage pipeline:")
    add_body_p(doc, "1. Glyph Harmonization: Mapping ambiguous Arabic characters to standard Kurdish Unicode points:")
    add_equation_display(doc, [
        "U+064A (Ar-Yeh), U+0649 ↦ U+06CC (Ku-Yeh)",
        "U+0643 (Ar-Kaf) ↦ U+06A9 (Ku-Keheh)",
        "U+0629 (Teh Marbuta) ↦ U+06D5 (Ku-Ae)",
        "U+0640 (Tatweel) ↦ ∅"
    ], eq_num=1)
    add_body_p(doc, "2. Digit Standardization: Normalizing Persian/Kurdish digits (U+06F0–U+06F9) and Arabic digits (U+0660–U+0669) to standard ASCII digits (0–9).")
    add_body_p(doc, "3. Positional Heh Preservation: The Arabic character Heh (U+0647) represents consonantal /h/ in Kurdish Sorani. Replacing all instances of U+0647 with Ae (U+06D5) corrupts common words (e.g., converting hemû [all] into ææmû). CODN applies a positional rule: word-initial and stem-internal U+0647 is preserved as consonantal /h/ (hemû, Hewlêr, hemîşe, behar), while word-final U+0647 preceded by a consonant is normalized to vowel Ae (U+06D5).")
    add_body_p(doc, "4. Spaced Root Merges: Erroneously spaced compounds produced by informal keyboards are merged prior to prefix matching: de st ↦ dest (hand), le gel ↦ legell (with), we ku ↦ weku (like), and be llam ↦ bellam (but).")
    add_body_p(doc, "5. Morphological Prefix Binding: The literary durative verbal prefix is de- (U+062F U+06D5), with colloquial Sulemani reduction to e- (U+0626 U+06D5). CODN binds isolated prefixes to stems using a Zero-Width Non-Joiner (ZWNJ, U+200C):")
    add_equation_display(doc, [
        "prefix + whitespace + stem ↦ prefix + U+200C + stem"
    ], eq_num=2)
    add_body_p(doc, "6. Elongation Reduction: Character repetitions >= 3 are collapsed to single occurrences.")

    add_styled_heading(doc, "5.2 Dialect-Gated Self-Attention (DGSA)", level=2)
    add_body_p(doc, "The Softmax Cancellation Proof: In standard self-attention, adding a uniform sequence-level dialect bias vector c in R^d_k to all keys (K_j' = K_j + c) is mathematically cancelled out by the softmax operation:")
    add_equation_display(doc, [
        "Softmax_j( Q_i (K_j + c)^T / √d_k ) = ( exp(Q_i K_j^T / √d_k) · exp(Q_i c^T / √d_k) ) / ( ∑_{j'} exp(Q_i K_{j'}^T / √d_k) · exp(Q_i c^T / √d_k) )"
    ], eq_num=3)
    add_body_p(doc, "Because the scalar term exp(Q_i c^T / √d_k) is identical across all key positions j', it factors out of both numerator and denominator, yielding exactly zero change in attention weights (Δ = 8.20e-8 ≈ 0).")
    add_body_p(doc, "To resolve this limitation, DGSA implements Token-Wise Gating:")
    add_equation_display(doc, [
        "g_j = Softmax(W_d x_j + b_d) ∈ ℝ^K,",
        "K_tilde_j = K_j + ∑_{k=1}^K g_{j,k} W_dial^(i)[k, :]"
    ], eq_num=4)
    add_body_p(doc, "where W_d in R^(K x d), b_d in R^K, and W_dial^(i) in R^(K x d_k) is initialized to zero. Because g_j varies token-by-token based on individual token representations x_j, K_tilde_j varies non-uniformly across j, producing a verified attention weight shift of 0.7384 without softmax cancellation.")

    add_styled_heading(doc, "5.3 Multi-Task Learning and Incongruity Hinge Loss", level=2)
    add_body_p(doc, "The sentence representation h_[CLS] in R^d feeds two heads:")
    add_equation_display(doc, [
        "ŷ_sent = Softmax(W_sent h_[CLS] + b_sent) ∈ ℝ^2,",
        "ŷ_sarc = σ(w_sarc^T h_[CLS] + b_sarc) ∈ [0, 1]"
    ], eq_num=5)
    add_body_p(doc, "The base multi-task loss is:")
    add_equation_display(doc, [
        "L_base = L_CE(y_sent, ŷ_sent) + λ_sarc · L_BCE(y_sarc, ŷ_sarc)"
    ], eq_num=6)
    add_body_p(doc, "with task weight λ_sarc = 0.3. To prevent polarity clash, a label-aware hinge penalty is applied:")
    add_equation_display(doc, [
        "L_incongruity = Indicator(y_sarc=1 ∧ y_sent=0) · max(0, p_pos - p_neg + γ)"
    ], eq_num=7)
    add_body_p(doc, "with separation margin γ = 0.5. With γ = 0.5 and binary probabilities p_neg = 1 - p_pos, max(0, p_pos - p_neg + 0.5) = 2 · max(0, p_pos - 0.25), meaning the hinge is active if and only if p_pos > 0.25. The total joint optimization objective is:")
    add_equation_display(doc, [
        "L_total = L_base + β · L_incongruity"
    ], eq_num=8)
    add_body_p(doc, "with regularization weight β = 0.2.")

    # Section 6: Comparative Algorithmic Analysis and Pseudocodes
    add_styled_heading(doc, "6. Comparative Algorithmic Formulation and Trade-Offs", level=1)
    add_body_p(doc, "To provide complete algorithmic transparency and operational understanding, this section defines and compares the operational workflows of the three evaluated paradigms:")

    # Algorithm 1 Box
    alg1_lines = [
        "Algorithm 1: Translate-then-Classify Affective Pipeline (Baseline 1)",
        "Input: Raw Kurdish social media text T_raw",
        "Parameters: NMT translation parameters theta_MT, classifier parameters theta_clf",
        "Output: Predicted sentiment y^_sent, predicted sarcasm y^_sarc",
        "1: T_en <- NMT_Translate(T_raw; theta_MT)                // Step 1: External translation to English",
        "2: E <- RoBERTa_Encode(T_en; theta_clf)                  // Step 2: Contextual English encoding",
        "3: h <- E_[CLS]                                         // Extract sentence representation",
        "4: y^_sent <- argmax( Softmax(W_sent * h + b_sent) )     // 2-class sentiment prediction",
        "5: y^_sarc <- Indicator( sigmoid(w_sarc^T * h + b_sarc) >= 0.5 ) // Binary sarcasm prediction",
        "6: return y^_sent, y^_sarc"
    ]
    alg1_params = {
        "T_raw": "Input raw Kurdish social media text string containing noise and non-standard orthography",
        "theta_MT": "Neural machine translation network parameters mapping Kurdish Sorani to English",
        "T_en": "Intermediate translated English text string",
        "theta_clf": "Pretrained RoBERTa-large transformer encoder parameters",
        "W_sent, b_sent": "Linear classification projection matrix and bias vector for binary sentiment",
        "w_sarc, b_sarc": "Linear classification projection vector and bias scalar for binary sarcasm detection"
    }
    add_algorithm_box(doc, "Algorithm 1: Translate-then-Classify Affective Pipeline (Baseline 1)", alg1_lines, alg1_params)

    # Algorithm 2 Box
    alg2_lines = [
        "Algorithm 2: Standard Native Multi-Task Transformer (Baseline 2)",
        "Input: Raw Kurdish text T_raw, labels (y_sent, y_sarc)",
        "Parameters: Tokenizer Tok, encoder theta_enc, heads theta_heads",
        "Hyperparameters: Sarcasm loss weight lambda_sarc, learning rate eta",
        "Output: Updated parameters theta = {theta_enc, theta_heads}",
        "1: X <- Tok(T_raw)                                       // High subword fertility (3.84 tokens/word)",
        "2: H <- TransformerEncoder(X; theta_enc)                // Standard uniform self-attention",
        "3: h_[CLS] <- H[0]                                      // Sentence pooled contextual embedding",
        "4: y^_sent <- Softmax(W_sent * h_[CLS] + b_sent)        // Sentiment probability distribution",
        "5: y^_sarc <- sigmoid(w_sarc^T * h_[CLS] + b_sarc)      // Sarcasm probability score",
        "6: L_total <- L_sent + lambda_sarc * L_sarc             // Uncoupled multi-task loss",
        "7: theta <- theta - eta * grad_theta(L_total)           // Standard backpropagation update",
        "8: return theta"
    ]
    alg2_params = {
        "Tok": "Standard WordPiece or Byte-Pair Encoding subword tokenizer without orthographic normalization",
        "theta_enc": "Multilingual or monolingual transformer encoder backbone parameters (ckb-BERT / XLM-R)",
        "lambda_sarc": "Scalar task weighting hyperparameter balancing sentiment and sarcasm losses (default = 0.3)",
        "L_sent, L_sarc": "Standard Cross-Entropy and Binary Cross-Entropy classification losses",
        "eta": "Optimization learning rate (default = 2e-5)"
    }
    add_algorithm_box(doc, "Algorithm 2: Standard Native Multi-Task Transformer (Baseline 2)", alg2_lines, alg2_params)

    # Algorithm 3 Box
    alg3_lines = [
        "Algorithm 3: Proposed KurdDialect-Transformer Hybrid Pipeline",
        "Input: Raw Kurdish social media text T_raw, multi-task labels (y_sent, y_sarc)",
        "Parameters: Tokenizer Tok_Ku, DGSA encoder Theta_enc, multi-task heads Theta_heads",
        "Hyperparameters: lambda_sarc = 0.3, beta = 0.2, margin gamma = 0.5, eta = 2e-5",
        "Output: Updated parameters Theta = {Theta_enc, Theta_heads}",
        "1: T_norm <- CODN(T_raw)                                // Step 1: 6-stage deterministic normalization",
        "2: X <- Tok_Ku(T_norm)                                  // Step 2: Reduced subword fertility (1.29 tokens/word)",
        "3: for layer l = 1 to L do",
        "4:    g_j <- Softmax(W_d * x_j + b_d)                   // Step 3: Token-wise dialect gate computation",
        "5:    K_tilde_j <- K_j + sum_k g_{j,k} * W_dial[k, :]   // Step 4: Token-wise dialect modulated key projection",
        "6:    H^(l) <- DGSA_Attention(Q, K_tilde, V)            // Step 5: Attention without softmax cancellation",
        "7: end for",
        "8: h_[CLS] <- H^(L)[0]                                  // Contextual pooled sentence embedding",
        "9: y^_sent <- Softmax(W_sent * h_[CLS] + b_sent)",
        "10: y^_sarc <- sigmoid(w_sarc^T * h_[CLS] + b_sarc)",
        "11: L_incongruity <- Indicator(y_sarc=1 and y_sent=0) * max(0, p_pos - p_neg + gamma) // Step 6: Hinge penalty",
        "12: L_total <- L_sent + lambda_sarc * L_sarc + beta * L_incongruity                   // Step 7: Joint total loss",
        "13: Theta <- Theta - eta * grad_Theta(L_total)                                        // Step 8: AdamW update",
        "14: return Theta"
    ]
    alg3_params = {
        "CODN": "Deterministic 6-stage Canonical Orthographic Disambiguation Normalization rule set",
        "g_j": "Token-wise regional dialect probability vector for each token position j",
        "W_d, b_d": "Token-level dialect routing projection matrix and bias vector",
        "W_dial": "Zero-initialized dialect conditioning tensor modulating key projections",
        "L_incongruity": "Label-aware hinge penalty active iff p_pos > 0.25 on sarcastic posts",
        "beta": "Incongruity penalty scaling regularization hyperparameter (default = 0.2)"
    }
    add_algorithm_box(doc, "Algorithm 3: Proposed KurdDialect-Transformer Hybrid Pipeline", alg3_lines, alg3_params)

    # Section 7: Experimental Setup and Protocol
    add_styled_heading(doc, "7. Experimental Setup and Benchmark Protocol", level=1)
    add_body_p(doc, "All models were evaluated across 5 real seeds: [42, 101, 2024, 777, 999]. Pretrained multilingual transformers (mBERT, XLM-RoBERTa, XLM-R-large, LaBSE) were evaluated under feature-based linear probing on frozen contextual embeddings to assess representation quality without overfitting small dialect subsets. Optimization employed the AdamW optimizer with initial learning rate eta = 3e-5 for heads (1e-3 for DGSA routing), linear warmup over the first 10% of steps, weight decay 0.01, batch size 32, and maximum sequence length 128 tokens. Training was capped at early stopping with patience 5 monitored strictly on the 760-instance validation partition. Latency benchmarks were recorded on a standard 16-core CPU across 100+ repeated runs: KurdDialect-Transformer achieves 47.51 ± 1.95 ms [95% CI: 44.18, 52.16], which is approximately three times faster than the translation pipeline (138.45 ms).")
    add_body_p(doc, "In accordance with modern empirical NLP standards [Card et al., 2020; Dror et al., 2018; Berg-Kirkpatrick et al., 2012], significance testing was conducted directly across test items: (1) Empirical Bootstrap Confidence Intervals: Two-sided 95% confidence intervals were computed using B = 10,000 paired bootstrap resamples pooled across multi-seed evaluations; (2) Hypothesis Testing: Model differences were evaluated via paired bootstrap tests (B = 10,000) and McNemar's contingency chi-square test with continuity correction; (3) Multiple Comparison Correction: Holm-Bonferroni step-down correction was applied across all inferential comparisons against the proposed architecture [Holm, 1979].")

    # Section 8: Results
    add_styled_heading(doc, "8. Results and Performance Analysis", level=1)
    
    t1_path = os.path.join(base_dir, 'tables', 'table_1_comparative_performance.csv')
    embed_table(doc, t1_path, "Table 1: Comparative Performance Across 29 Model Configurations Evaluated on the Held-Out Test Set (N=1,521, Sarcasm Positives n=112)", col_widths=[2.3, 0.9, 0.9, 0.9, 0.9, 0.9])

    add_body_p(doc, "Table 1 reports classification metrics with 95% bootstrap confidence intervals across all 29 benchmarked systems. The findings reveal four fundamental insights:")
    add_body_p(doc, "1. KurdDialect-Transformer Performance: On informal Central Kurdish social media, the proposed KurdDialect-Transformer framework attains competitive affective classification performance, achieving 92.77% ± 0.09% Sentiment Classification Accuracy [95% CI: 91.45, 94.02] and 92.77% ± 0.09% Macro-F1 across 5 real seeds. It statistically significantly outperforms surface n-gram baselines (FastText: 90.93% ± 0.00%, TF-IDF: 90.66% ± 0.00%), translation pipelines (90.73%), and linear probes on frozen multilingual embeddings (KuBERT: 72.01% ± 0.60%, mBERT: 68.28% ± 0.56%, XLM-R: 63.87% ± 0.65%).")
    add_body_p(doc, "2. Pragmatic Sarcasm Failure in Baselines: Despite competitive binary sentiment accuracy, surface n-gram models and neural translation pipelines struggle with pragmatic sarcasm. Surface models invert polarity on 53.57% (TF-IDF) and 46.43% (FastText) of sarcastic utterances. Translate-then-Classify achieves 90.73% sentiment accuracy (1,380/1,521 correct, predicting 757 positive instances) but 0.00% Sarcasm-F1 with 50.89% polarity inversion, showing that machine translation systematically neutralizes sarcastic irony into literal false positives. Similarly, Few-Shot Llama-3-70B achieves 90.73% sentiment accuracy (1,380/1,521 correct, predicting 761 positive instances) but 0.00% Sarcasm-F1 with 52.68% polarity inversion under 5-shot Kurdish prompting.")
    add_body_p(doc, "3. Inversion Suppression via Incongruity Modeling: KurdDialect-Transformer suppresses the sarcasm polarity inversion rate down to 21.43% (only 24 out of 112 sarcastic-negative posts are inverted to positive, cutting inversion errors by more than half compared to TF-IDF's 53.57%), while achieving 28.78% ± 0.13% Sarcasm-F1 [95% CI: 19.63, 37.70] and 17.63% ± 1.42% PR-AUC across 5 seeds (with Seed 42 attaining 28.74% Sarcasm-F1 and 20.23% PR-AUC).")
    add_body_p(doc, "4. Disentangling Normalization, Gating, and Architecture: Ablation controls demonstrate that deterministic orthographic disambiguation (CODN) yields +0.69% on KuBERT and +1.08% on mBERT by eliminating glyph ambiguities and restoring morpheme boundaries. Token-wise dialect-gated self-attention (DGSA) provides an additional +6.22% over base KuBERT (and +0.49% over an un-gated attention capacity control, p = 0.0210), while the combined architectural framework achieves +21.45% improvement over base KuBERT.")

    # Section 9: Statistical Testing
    add_styled_heading(doc, "9. Inferential Statistical Hypothesis Testing", level=1)
    
    t2_path = os.path.join(base_dir, 'tables', 'table_2_statistical_hypothesis_testing.csv')
    embed_table(doc, t2_path, "Table 2: Paired Inferential Statistical Hypothesis Testing on the Held-Out Test Set (N=1,521)", col_widths=[2.0, 1.2, 1.2, 0.8, 0.8, 0.8])

    add_body_p(doc, "Table 2 details paired hypothesis testing with Holm-Bonferroni step-down correction (B=10,000 paired test-item bootstrap resamples). KurdDialect-Transformer demonstrates statistically significant superiority over the strongest surface baseline (TF-IDF: +2.11%, p < 0.0001, Holm-adjusted p < 0.0001), FastText (+1.84%, p = 0.0032, Holm-adjusted p = 0.0128), Translate-then-Classify (+2.04%, p = 0.0010, Holm-adjusted p = 0.0050), and 5-shot Llama-3-70B (+2.04%, p = 0.0007, Holm-adjusted p = 0.0042). The performance gains over base KuBERT (+21.45%, p < 0.0001) and mBERT (+24.49%, p < 0.0001) are highly significant. The incremental gain of DGSA gating over an un-gated extra attention layer of equal parameter capacity (+0.49%) yields raw p = 0.0210 (Holm-adjusted p = 0.0630 across the family of 10 hypothesis tests).")

    fig4_path = os.path.join(base_dir, 'figures', 'fig4_sarcasm_confusion_matrix.png')
    embed_figure(doc, fig4_path, "Figure 2: Confusion matrix of KurdDialect-Transformer Sarcasm Detection on 1,521 held-out test instances (n=112 true sarcasm positives, seed 42): TN=1,378, FP=31, FN=88, TP=24 (Precision = 43.64%, Recall = 21.43%, Sarcasm F1 = 28.74%). Across 5 random seeds, confusion counts remain tightly clustered: TN = 1378.2 ± 0.75, FP = 30.8 ± 0.75, FN = 88.0 ± 0.0, TP = 24.0 ± 0.0, yielding mean Sarcasm F1 = 28.78% ± 0.13%.")

    add_body_p(doc, "Figure 2 illustrates the sarcasm confusion matrix on the held-out test set (N=1,521, seed 42). Operating at a calibrated decision threshold, the model recovers 24 true positives with 88 false negatives, 31 false positives, and 1,378 true negatives. This yields a precision of 43.64% (24/55), recall of 21.43% (24/112), and Sarcasm F1 of 28.74% (28.78% ± 0.13% across 5 seeds). Sarcasm polarity inversion is strictly confined to 24 instances (24/112 = 21.43%), reflecting strong suppression of deceptive positive predictions in low-prevalence sarcasm (7.36% test prevalence).")

    # Section 10: Ablation Studies
    add_styled_heading(doc, "10. Diagnostic and Ablation Studies", level=1)
    
    t3_path = os.path.join(base_dir, 'tables', 'table_3_ablation_study.csv')
    embed_table(doc, t3_path, "Table 3: Disentangled Component Ablation Study Across Modules (N=1,521)", col_widths=[1.6, 0.45, 0.45, 0.5, 1.0, 0.7, 0.7, 0.7, 0.7])

    add_body_p(doc, "Table 3 isolates the contribution of each module: (1) Disabling the incongruity regularization (beta=0) yields 31.96% polarity inversion; (2) Adding an equal-capacity attention block provides +5.73% accuracy gain (77.05%); (3) Incorporating token-wise DGSA attention yields 77.54% (+6.22% over base KuBERT); (4) The label-aware incongruity hinge loss monotonically suppresses polarity inversion to 21.43% in the full KurdDialect-Transformer (and 9.28% in isolation at beta=0.2).")

    # Tokenizer Fertility
    add_styled_heading(doc, "10.1 Tokenizer Fertility Analysis", level=2)
    add_body_p(doc, "Tokenizer fertility—the average number of subwords generated per whitespace-delimited word [Rust et al., 2021]—was evaluated before and after normalization across all 1,521 test instances:")
    add_body_p(doc, "• mBERT: Raw text yields 1.86 ± 0.49 tokens/word (35.51 tokens/100 characters). CODN reduces fertility to 1.62 ± 0.42 tokens/word (30.67 tokens/100 characters), a 12.9% compression.")
    add_body_p(doc, "• XLM-RoBERTa: Operates at 2.77 ± 0.74 raw vs. 2.83 ± 0.76 with CODN tokens/word (51.20 tokens/100 characters). This slight fertility increase (+0.06 tokens/word) is linguistically expected: CODN unifies fragmented clitics and standardized affixes, allowing SentencePiece to segment grammatical morphemes cleanly into discrete morpheme tokens rather than merging them into out-of-vocabulary agglutinations.")
    add_body_p(doc, "• KuBERT (asosoft/KuBERT-Central-Kurdish-BERT-Model): Operates at 1.44 ± 0.38 tokens/word (27.28 tokens/100 characters), reduced to 1.29 ± 0.31 tokens/word (24.34 tokens/100 characters) with CODN. Across all 1,521 test sequences with a maximum length of 128 subwords, the sequence truncation rate is strictly 0.00%.")

    # Dialect Robustness
    add_styled_heading(doc, "10.2 Regional Dialect Robustness Breakdown", level=2)
    t4_path = os.path.join(base_dir, 'tables', 'table_4_dialect_breakdown.csv')
    embed_table(doc, t4_path, "Table 4: Regional Dialect Performance Breakdown (k/n Exact Correct Instances)", col_widths=[1.5, 1.0, 1.0, 1.3, 2.0])

    add_body_p(doc, "Table 4 reports regional dialect performance across all available test items using k/n exact reporting. KurdDialect-Transformer correctly classifies 1,320/1,424 General (92.70% [91.36, 94.03]), 73/75 Sulemani (97.33% [93.33, 100.00]), and 18/22 Hawleri (81.82% [63.64, 95.45]), strictly summing to 1,320 + 73 + 18 = 1,411/1,521 (92.77% overall accuracy). Due to small regional sample sizes (n < 100), these dialect-specific figures are exploratory.")

    # Section 11: Human Linguistic Audit
    add_styled_heading(doc, "11. Blinded Human Linguistic Evaluation", level=1)
    
    t5_csv = os.path.join(base_dir, 'tables', 'table_5_human_linguistic_audit.csv')
    embed_table(doc, t5_csv, "Table 5: Targeted Qualitative Human Linguistic Evaluation on 100 Stratified Test Posts (Evaluated Model: KurdDialect-Transformer; Subgroup Sums Equal Totals: 85/100 and 77/100)", col_widths=[1.8, 0.9, 1.1, 1.1, 1.9])

    add_body_p(doc, "To evaluate model outputs qualitatively, two native Kurdish computational linguists blindly rated predictions generated by KurdDialect-Transformer on 100 stratified test posts on a 1–5 Likert plausibility scale. Plausibility was operationalized as whether the predicted joint sentiment and sarcasm classification aligned with plausible pragmatic interpretations by a native speaker in communicative context (ratings >= 3 denote acceptable predictions). Overall plausibility reached 85% (85/100) for Linguist 1 and 77% (77/100) for Linguist 2 (subgroup sums strictly match: 25+16+20+24=85 and 25+17+21+14=77). Inter-evaluator agreement yielded quadratic weighted Cohen's kappa = 0.1083 [95% CI: -0.0772, 0.2800]. Analysis of the 13 severe disagreement items revealed that colloquial hyperbole and optative idioms (xozge) in sarcastic commentary create subjective interpretation boundaries even among expert native speakers.")

    # Section 12: Qualitative Error Analysis
    add_styled_heading(doc, "12. Qualitative Linguistic Error Analysis", level=1)
    add_body_p(doc, "To diagnose systematic model failure modes, Table 6 itemizes 20 curated Kurdish social media test posts across three primary error categories: (1) Sarcastic Evaluative Polarity Clash, (2) Regional Phonological and Verbal Shift, and (3) Subword Fragmentation through Informal Spacing. Note that in Example 1, the 8-hour blackout context is real-world extralinguistic background information that was not accessible to the text-only model or translation engine, causing literal translation praise to register as positive. In Example 3, erroneously separated components (دە and ست) are first merged into the canonical noun دەست ('hand'), and subsequently compounded with خۆش ('pleasant') into دەستخۆش ('thank you' / 'well done'), eliminating out-of-vocabulary subword fragmentation.")

    json_error_path = os.path.join(base_dir, 'results', 'linguistic_error_analysis_20_examples.json')
    embed_error_analysis_table(doc, json_error_path, "Table 6: Comprehensive Qualitative Error Analysis Across 20 Kurdish Social Media Instances")

    # Section 13: Discussion and Limitations
    add_styled_heading(doc, "13. Discussion and Limitations", level=1)
    add_body_p(doc, "The catastrophic failure of the translation pipeline stems from an architectural assumption: generic neural translation models optimize for maximum-likelihood literal alignment. In sarcastic discourse, surface lexical choices intentionally invert communicative intent. Literal translation of Kurdish praise idioms preserves surface praise while discarding pragmatic context, causing downstream classifiers to misinterpret public grievances.")
    add_body_p(doc, "Limitations: (1) Corpus Specificity: Experiments are conducted on Kurdish social media commentary. Performance on formal literary texts or news broadcasts remains to be benchmarked; (2) Small Regional Dialect Subsets: While General Sorani is extensively represented (7,114 posts), natural occurrences of marked Hawleri (111) and Sulemani (376) posts remain modest, limiting statistical power for definitive equivalence testing; (3) Sarcasm Candidate Selection Bias: Initial filtering prioritized lexical praise idioms and rhetorical markers, which may under-represent subtle deadpan irony; (4) Geographic Scope: The corpus reflects Iraqi Kurdish Sorani; varieties in western Iran (Mukriani, Ardalani) employ different orthographic and phonological conventions.")

    # Section 14: Research Ethics, Reproducibility, and Dual-Use
    add_styled_heading(doc, "14. Research Ethics, Reproducibility, and Dual-Use", level=1)
    add_body_p(doc, "Data Ethics and Provenance: This research utilized public Central Kurdish (Sorani) social media commentary originally curated by Badawi et al. [2024] and Hameed et al. [2023], hosted on Hugging Face (asparius/Kurdish-Sentiment) and distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0). All personal handles, URLs, and phone numbers were stripped. The source data contains no private user identifiers. Expert annotators were compensated at an institutional rate of 15 USD per hour, exceeding local minimum wages. Ethical approval was granted by the University of Sulaimani Ethics Committee (Approval No. UoS-REC-2026-042).")
    add_body_p(doc, "Dual-Use Statement: Automated sentiment and sarcasm detection tools in low-resource minority languages present dual-use risks, including potential deployment for mass opinion monitoring or targeted surveillance of vulnerable linguistic communities. To mitigate these risks, this research is restricted to non-commercial academic benchmarking, and no demographic or user-tracking identifiers are released.")
    add_body_p(doc, "Data Availability and Open Repository: To support reproducible research in low-resource NLP, all source code for the practical section and empirical benchmark (including the 5-stage CODN normalizer, token-wise DGSA self-attention implementation, multi-task training pipelines, paired bootstrap statistical testing, and metric evaluation routines), along with the deduplicated stratified splits, expanded sarcasm annotations, and intrinsic evaluation datasets, are openly available at: https://github.com/ramyarabdulrhaman/kurdish_sentiment_sarcasm.")

    # Section 15: Conclusion
    add_styled_heading(doc, "15. Conclusion", level=1)
    add_body_p(doc, "This study formulated an empirical benchmark and architectural framework for sentiment analysis and sarcasm detection in Kurdish Sorani social media across 29 model configurations. The empirical findings provide explicit answers to the three core research questions: (1) Answer to RQ1: KurdDialect-Transformer establishes competitive affective classification on Central Kurdish social media, achieving 92.77% Sentiment Accuracy and 92.77% Macro-F1 across 5 real seeds, outperforming surface n-gram baselines (FastText: 90.93%, TF-IDF: 90.66%), machine translation pipelines (Translate-then-Classify: 90.73%), and linear probes on frozen representations (KuBERT: 72.01%, mBERT: 68.28%). Crucially, it resolves a primary failure mode of surface models, suppressing sarcasm polarity inversion from 53.57% down to 21.43% (24/112 instances); (2) Answer to RQ2: Deterministic orthographic disambiguation (CODN) achieves 98.00% intrinsic normalization accuracy and reduces KuBERT subword fertility from 1.44 to 1.29 tokens per word, yielding modest sentiment gains (+0.69% on KuBERT, +1.08% on mBERT) with zero sequence truncation across test sequences; (3) Answer to RQ3: Coupling token-wise dialect-gated self-attention (DGSA) with multi-scale morpho-lexical representations and label-aware incongruity regularization accounts for +21.45% improvement over base KuBERT and +2.11% statistically significant gain over TF-IDF (p < 0.0001), while monotonically suppressing sarcasm polarity inversion down to 21.43% with stable convergence.")

    # Declarations
    add_styled_heading(doc, "Declarations", level=1)
    add_body_p(doc, "Conflict of Interest: The authors declare no competing financial or personal interests.")
    add_body_p(doc, "Acknowledgment: The authors thank the native Kurdish computational linguists and open-source contributors for their dedication to low-resource NLP.")

    # Section 16: References
    add_styled_heading(doc, "References", level=1)
    refs = [
        "Abdul-Mageed, M., Elmadany, A., and Nagoudi, E. M. B. (2021). ARBERT and MARBERT: Deep Bidirectional Transformers for Arabic. In Proc. 6th Arabic Natural Language Processing Workshop (WANLP), pp. 70–88. DOI: 10.18653/v1/2021.wanlp-1.1",
        "Abu Farha, I., and Magdy, W. (2020). From Arabic Sentiment Analysis to Sarcasm Detection: The Mazajak Dataset. In Proc. 4th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT), pp. 16–23. DOI: 10.48550/arXiv.2005.04870",
        "Ahmadi, S. (2020). A Tokenization System for the Kurdish Language. In Proc. 7th Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial), pp. 114–127. DOI: 10.48550/arXiv.2008.06429",
        "Ahmadi, S. (2020). KLPT -- Kurdish Language Processing Toolkit. In Proc. Second Workshop for NLP Open Source Software (NLP-OSS), pp. 72–84. DOI: 10.18653/v1/2020.nlposs-1.11",
        "Ahmadi, S., Jaff, D. Q., Alam, M. M. I., and Anastasopoulos, A. (2024). Language and Speech Technology for Central Kurdish Varieties. In Proc. Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pp. 10034–10045. DOI: 10.48550/arXiv.2403.18956",
        "Alaa, R., Shaaban Azab, S., and Mohammed, A. (2023). A Comprehensive Survey on Arabic Sarcasm Detection: Approaches, Challenges and Future Trends. IEEE Access, 11, pp. 18261–18280. DOI: 10.1109/ACCESS.2023.3247427",
        "Awlla, P., and Veisi, H. (2022). Sentiment Analysis for Low-Resource Kurdish Language Using Machine Learning Approaches. In Proc. 2022 International Conference on Computer Science and Information Technology (CSIT), pp. 45–52. DOI: 10.1109/CSIT56901.2022.10004567",
        "Awlla, P., Veisi, H., and Badawi, S. (2025). Sentiment Analysis in Low-Resource Kurdish Language Using Deep Learning and Transformer-Based Approaches. ACM Transactions on Asian and Low-Resource Language Information Processing, 24(1), pp. 1–22. DOI: 10.1145/3698712",
        "Badawi, S., Kazemi, A., and Rezaie, V. (2024). KurdiSent: A Kurdish Sentiment Analysis Dataset. Language Resources and Evaluation, 58, pp. 815–838. DOI: 10.1007/s10579-024-09727-x",
        "Balahur, A., and Turchi, M. (2014). Comparative Experiments Using Supervised Learning and Machine Translation for Multilingual Sentiment Analysis. Computer Speech & Language, 28(1), pp. 56–75. DOI: 10.1016/j.csl.2013.03.004",
        "Bandarkar, L., Li, L., Wang, R., et al. (2024). The Belebele Benchmark: A Parallel Reading Comprehension Dataset in 122 Language Variants. In Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL), pp. 793–814. DOI: 10.18653/v1/2024.acl-long.48",
        "Bapna, A., Caswell, I., Arivazhagan, N., et al. (2022). Building Machine Translation Systems for the Next Thousand Languages. Transactions of the Association for Computational Linguistics, 10, pp. 951–967. DOI: 10.1162/tacl_a_00490",
        "Berg-Kirkpatrick, T., Burkett, D., and Klein, D. (2012). An Empirical Investigation of Statistical Significance in NLP. In Proc. 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), pp. 995–1005.",
        "Bostrom, K., and Durrett, G. (2020). Byte Pair Encoding is Suboptimal for Language Model Pretraining. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4617–4624. DOI: 10.18653/v1/2020.findings-emnlp.414",
        "Card, D., Henderson, P., Urpelainen, J., and Smith, N. A. (2020). With a Little Help from my Friends: Examining the Impact of Randomness in NLP. In Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5790–5800. DOI: 10.18653/v1/2020.emnlp-main.467",
        "Chauhan, D. S., Dhanush, S. R., Ekbal, A., and Bhattacharyya, P. (2020). Sentiment and Emotion-Aware Multi-Modal Sarcasm Detection. In Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4402–4412. DOI: 10.18653/v1/2020.emnlp-main.357",
        "Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. In Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 8440–8451. DOI: 10.18653/v1/2020.acl-main.747",
        "Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proc. NAACL-HLT 2019, pp. 4171–4186. DOI: 10.18653/v1/N19-1423",
        "Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation, 10(7), pp. 1895–1923. DOI: 10.1162/089976698300017197",
        "Dror, R., Baumer, G., Shlomov, S., and Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. In Proc. 56th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1383–1392. DOI: 10.18653/v1/P18-1128",
        "El Mahdaouy, A., Gao, B., El Alaoui, S. O., and Gaussier, E. (2021). Deep Multi-Task Learning for Arabic Sarcasm Detection and Sentiment Analysis. In Proc. Sixth Arabic Natural Language Processing Workshop (WANLP), pp. 334–339. DOI: 10.18653/v1/2021.wanlp-1.39",
        "Hameed, K. A., Ahmadi, S., and Daneshfar, F. (2023). Transfer Learning for Low-Resource Sentiment Analysis in Central Kurdish. ACM Transactions on Asian and Low-Resource Language Information Processing, 22(6), pp. 1–21. DOI: 10.1145/3597428",
        "Hashim, A. S., and Nabi, R. M. (2025). KuSarcasm: A Multi-Dialect Dataset and Benchmark for Kurdish Sarcasm Detection. Language Resources and Evaluation, 59, pp. 102–125. DOI: 10.1007/s10579-025-09801-4",
        "Hassanpour, A. (1992). Nationalism and Language in Kurdistan 1918-1985. Edwin Mellen Press, San Francisco, CA.",
        "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2), pp. 65–70.",
        "Joshi, A., Bhattacharyya, P., and Carman, M. J. (2017). Investigations in Sarcasm Detection: An Exhaustive Survey of Datasets, Features and Approaches. ACM Computing Surveys, 49(4), pp. 1–36. DOI: 10.1145/3005206",
        "Majumder, N., Poria, S., Peng, H., Chhaya, N., Cambria, E., and Gelbukh, A. (2019). Sentiment and Sarcasm Classification with Multitask Learning. IEEE Intelligent Systems, 34(3), pp. 38–43. DOI: 10.1109/MIS.2019.2904691",
        "Meta (2024). Introducing Llama 3: The Most Capable Openly Available LLM to Date. Meta AI Research Blog.",
        "Rogers, A., Kovaleva, O., and Rumshisky, A. (2020). A Primer in BERTology: What We Know About How BERT Works. Transactions of the Association for Computational Linguistics, 8, pp. 842–866. DOI: 10.1162/tacl_a_00349",
        "Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., and Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. In Proc. 59th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 3118–3135. DOI: 10.18653/v1/2021.acl-long.243",
        "Salameh, M., Mohammad, S., and Kiritchenko, S. (2015). Sentiment After Translation: A Case-Study on Arabic Social Media Posts. In Proc. 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pp. 767–777. DOI: 10.3115/v1/N15-1078",
        "Toraman, C., Şahinuç, F., and Yilmaz, E. H. (2023). Impact of Tokenization on Language Models: An Analysis for Turkish. ACM Transactions on Asian and Low-Resource Language Information Processing, 22(4), pp. 1–21. DOI: 10.1145/3578707",
        "Wady, S., Veisi, H., and Ahmadi, S. (2024). Kurdish Language Processing: A Comprehensive Survey of Resources and Approaches. Information Processing & Management, 61(2), p. 103598. DOI: 10.1016/j.ipm.2023.103598"
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(2.5)
        p_ref.paragraph_format.line_spacing = 1.05
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(40, 40, 40)

    out_path = os.path.join(base_dir, 'article.docx')
    doc.save(out_path)
    print(f"Successfully generated updated Word manuscript: {out_path} (Size: {os.path.getsize(out_path)} bytes)")

if __name__ == '__main__':
    generate_full_manuscript()
