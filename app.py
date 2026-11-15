"""Minimal web UI to upload/index PDFs and search the corpus in Typesense.

Run with:  python app.py   (then open http://localhost:5000)
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from markupsafe import Markup, escape
from werkzeug.utils import secure_filename

from index_corpus import DEFAULT_COLLECTION, DEFAULT_CORPUS, DEFAULT_MODEL, ensure_collection, index_pdf, make_client

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", DEFAULT_CORPUS)).resolve()
COLLECTION = os.getenv("TYPESENSE_COLLECTION", DEFAULT_COLLECTION)
MODEL = os.getenv("TYPESENSE_MODEL", DEFAULT_MODEL)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SEARCH_MODES = {
    "semantic": "embedding",
    "hybrid": "text,embedding",
    "keyword": "text",
}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(16))
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

client = make_client(
    os.getenv("TYPESENSE_HOST", "localhost"),
    int(os.getenv("TYPESENSE_PORT", "8108")),
    os.getenv("TYPESENSE_PROTOCOL", "http"),
    os.getenv("TYPESENSE_API_KEY", "dev-secret-key"),
)


def indexed_files() -> list[dict]:
    """Return indexed PDFs with their chunk counts, using a facet on file_name."""
    res = client.collections[COLLECTION].documents.search({
        "q": "*",
        "facet_by": "file_name",
        "max_facet_values": 1000,
        "per_page": 0,
    })
    counts = res["facet_counts"][0]["counts"] if res.get("facet_counts") else []
    return sorted(({"name": c["value"], "chunks": c["count"]} for c in counts), key=lambda f: f["name"].lower())


def search(query: str, mode: str, per_page: int) -> list[dict]:
    params = {
        "q": query,
        "query_by": SEARCH_MODES[mode],
        "exclude_fields": "embedding",
        "per_page": per_page,
        "highlight_full_fields": "text",
        "highlight_start_tag": "<mark>",
        "highlight_end_tag": "</mark>",
    }
    if mode != "keyword":
        # Hide chunks that are semantically far from the query.
        params["vector_query"] = "embedding:([], distance_threshold: 0.8)"
    res = client.collections[COLLECTION].documents.search(params)

    results = []
    for hit in res["hits"]:
        doc = hit["document"]
        highlight = next((h for h in hit.get("highlights", []) if h.get("field") == "text"), None)
        html = None
        if highlight and highlight.get("value"):
            # Escape the PDF text, then restore only the highlight tags.
            html = Markup(str(escape(highlight["value"]))
                          .replace("&lt;mark&gt;", "<mark>").replace("&lt;/mark&gt;", "</mark>"))
        results.append({
            **doc,
            "html": html,
            "vector_distance": hit.get("vector_distance"),
            "text_match": hit.get("text_match"),
        })
    return results


@app.get("/")
def home():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "semantic")
    if mode not in SEARCH_MODES:
        mode = "semantic"

    results, error = [], None
    try:
        files = indexed_files()
        if query:
            results = search(query, mode, per_page=10)
    except Exception as exc:
        files = []
        error = f"Typesense error: {exc}"
    return render_template("index.html", query=query, mode=mode, modes=SEARCH_MODES,
                           results=results, files=files, error=error)


@app.post("/upload")
def upload():
    file = request.files.get("pdf")
    if not file or not file.filename:
        flash("Select a PDF to upload.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        flash("Only .pdf files are accepted.", "error")
        return redirect(url_for("home"))
    if file.stream.read(5) != b"%PDF-":
        flash(f"{filename} is not a valid PDF.", "error")
        return redirect(url_for("home"))
    file.stream.seek(0)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    target = CORPUS_DIR / filename
    replaced = target.exists()
    backup = target.with_name(target.name + ".bak")
    if replaced:
        os.replace(target, backup)
    file.save(target)

    try:
        # Indexed from its final path so doc_id/file_path match the CLI indexer.
        indexed, total = index_pdf(client, COLLECTION, target, CORPUS_DIR, CHUNK_SIZE, OVERLAP)
    except Exception as exc:
        target.unlink(missing_ok=True)
        if replaced:
            os.replace(backup, target)
        flash(f"Could not index {filename}: {exc}", "error")
        return redirect(url_for("home"))
    backup.unlink(missing_ok=True)

    action = "Re-indexed" if replaced else "Indexed"
    level = "success" if indexed == total else "error"
    flash(f"{action} {filename}: {indexed}/{total} chunks.", level)
    return redirect(url_for("home"))


@app.get("/pdf/<path:file_path>")
def pdf(file_path: str):
    if not file_path.lower().endswith(".pdf"):
        abort(404)
    return send_from_directory(CORPUS_DIR, file_path, mimetype="application/pdf")


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB).", "error")
    return redirect(url_for("home"))


if __name__ == "__main__":
    ensure_collection(client, COLLECTION, MODEL, recreate=False)
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
