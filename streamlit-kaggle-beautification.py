import streamlit as st
import nbformat as nbf
from nbformat.v4 import new_markdown_cell
from nbformat import NotebookNode
from models import model, clip_index, clip_model, clip_processor
import json
import re
import sys
import os
import pickle
import torch
import tempfile
import time
from dotenv import load_dotenv

# =========================
# STREAMLIT UI
# =========================
st.title("Kaggle Notebook Beautifier & Cover Generator")

uploaded_file = st.file_uploader(
    "Upload your Kaggle notebook (.ipynb)",
    type=["ipynb"]
)

if uploaded_file is None:
    st.stop()

# =========================
# ENV & ENCODING FOR API USAGE AND CONSISTENT ENCODING
# =========================
load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# =========================
# LOADING PRETRAINED CLIP PATHS
# =========================
with open("image_paths_updated.pkl", "rb") as f:  
    image_paths = pickle.load(f)

# =========================
# NOTEBOOK CELLS EXTRACTION
# =========================
def extract_notebook_content(file_path):
    nb = nbf.read(file_path, as_version=4)
    all_text = ""

    for i, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        content = cell["source"]
        all_text += f"\n\n# Cell {i} ({cell_type})\n{content}\n"

    return all_text

# =========================
# BEAUTIFICATION PROMPT FOR UPGRADING THE PRESENTATION OF NOTEBOOK
# =========================
def get_beautified_notebook(notebook_text):
    prompt = f"""
    You are an advanced AI assistant specializing in transforming raw Notebook files into polished, professional notebooks.

    Your tasks:
    1. **Organization**
    - Reorder and structure cells logically (intro → data loading → processing → analysis → visualization → results).
    - Ensure smooth narrative flow between code and markdown.

    2. **Markdown Improvements**
    - Write Markdown in a professional, natural, and tutorial-friendly style.
    - Add clear headings, subheadings, and step-by-step explanations.
    - Include short introductions before code cells and meaningful conclusions after results.
    - Use lists, bold, and italics where necessary for clarity.

    3. **Code Cells**
    - All code must be wrapped as `"cell_type": "code"`.
    - Keep code intact but organize placement logically.
    - Remove redundant or duplicate cells if any.
    - Use clear, short and direct comments in the Code Cells.

    4. **Markdown Cells**
    - All explanations must be `"cell_type": "markdown"`.
    - Ensure the tone is explanatory, engaging, and reader-friendly.
    - Use clear, direct and minimal markdowns with accurate formatting.

    5. **Output Format**
    - Final output should be a single JSON dict of the improved notebook.
    - Strictly preserve the Jupyter Notebook JSON schema (cells, metadata, etc.).

    6. **Flow and Smoothness**
    - Make the flow sound smooth for the other users to read it.
    - No use of words like tutorial, guide, etc. It will be framed as a Notebook of the analysis work of a Kaggle user to be presented for other users to read it.
    
    Notebook content to beautify:
    {notebook_text}
    """
    beautified_response = model.invoke(prompt)
    return beautified_response

# =========================
# VISUAL PROMPT GENERATION
# =========================
def generate_visual_prompt(notebook_text):
    visual_prompt_request = f"""
Create a concise, vivid visual description suitable for generating a notebook cover image using only simple, clear keywords extracted directly from the notebook content without summarizing or abstracting.
Constraints:
1–2 sentences only
No markdown
No explanations
No code references
Use straightforward keywords from the notebook content to enable direct image retrieval from a FAISS index

Notebook content:
{notebook_text[:2000]}
"""
    response = model.invoke(visual_prompt_request)
    return response.content.strip()

# =========================
# NOTEBOOK RECONSTRUCTION
# =========================
def text_to_notebook(beautified_text):
    beautified_text = re.sub(r"```json|```", "", beautified_text).strip()
    parsed = json.loads(beautified_text)

    cells = parsed["cells"] if isinstance(parsed, dict) else parsed

    for cell in cells:
        if isinstance(cell, NotebookNode):
            cell.metadata.pop("trusted", None)
        elif isinstance(cell, dict) and "metadata" in cell:
            cell["metadata"].pop("trusted", None)

    for cell in cells:
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return nbf.from_dict(notebook_dict)

# =========================
# RETRY LOOP FOR EACH FUNCTION
# =========================
def retry_until_third_attempt(func, data, retries=3, delay=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return func(data)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay)

    raise last_error

            
# TEMP DIRECTORY (PER USER)
with tempfile.TemporaryDirectory() as tmpdir:

    input_path = os.path.join(tmpdir, uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    OUTPUT_NOTEBOOK = os.path.join(
        tmpdir,
        uploaded_file.name.replace(".ipynb", "_beautified_updated.ipynb")
    )

    notebook_text = retry_until_third_attempt(extract_notebook_content, input_path)

    with st.spinner("Beautifying notebook..."):
        # BEAUTIFICATION AND VISUAL PROMPT GENERATION
        beautified_response = retry_until_third_attempt(get_beautified_notebook, notebook_text)
        beautified_text = beautified_response.content

    visual_prompt = retry_until_third_attempt(generate_visual_prompt, notebook_text)

    # IMAGE SEARCH (FAISS)
    inputs = clip_processor(text=[visual_prompt], return_tensors="pt")

    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)

    text_embedding_np = text_embedding.cpu().numpy().astype("float32")
    distances, indices = clip_index.search(text_embedding_np, k=1)

    image_path = image_paths[indices[0][0]]
    full_image_path = os.path.join("train_data", image_path)

    # FINAL NOTEBOOK EXTRACTION
    beautified_nb = retry_until_third_attempt(text_to_notebook, beautified_text)

    # COVER PREVIEW (IMAGE OR FALLBACK)
    st.subheader("Cover Preview")

    if os.path.exists(full_image_path):
        st.image(full_image_path, caption="Selected cover image", width='stretch')
    else:
        st.warning("Image not found. Using text-based cover preview instead.")
        st.markdown(
            f"""
            **Notebook Cover Preview (Text-Based)**  
            {visual_prompt}
            """
        )

        st.download_button(
            label="Download cover description",
            data=visual_prompt,
            file_name="cover_description.txt",
            mime="text/plain",
            key=f"cover_desc_{input_path}"
        )

    # INSERT COVER IMAGE (ONLY IF EXISTS)
    if os.path.exists(full_image_path):
        cover_cell = new_markdown_cell(f"![Notebook Cover]({full_image_path})")
        beautified_nb.cells.insert(1, cover_cell)

    # NOTEBOOK PREVIEW
    st.subheader("Notebook Preview")

    with st.expander("Preview first few cells"):
        for cell in beautified_nb.cells[:10]:
            if cell.cell_type == "markdown":
                st.markdown(cell.source)
            else:
                st.code(cell.source, language="python")

    # WRITE OUTPUT
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        nbf.write(beautified_nb, f)

    # DOWNLOAD BUTTON
    with open(OUTPUT_NOTEBOOK, "rb") as f:
        st.download_button(
            label="Download Beautified Notebook",
            data=f,
            file_name=os.path.basename(OUTPUT_NOTEBOOK),
            mime="application/x-ipynb+json",
            key=f"notebook_dl_{input_path}"
        )


    st.success("Notebook beautified successfully!")

