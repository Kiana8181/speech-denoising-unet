"""Execute the project notebook and export its results.

Steps
1. Execute ``notebooks/<name>.ipynb`` in a fresh Jupyter kernel (skip with ``--no-execute``).
2. Export the executed notebook to ``site/index.html`` (outputs and plots embedded).
3. Extract the figures of cells tagged with ``metadata.report`` into ``outputs/figures/``.
4. Build a PDF results report from those figures, tables and text outputs.

Usage (from the repository root)::

    python scripts/run_pipeline.py              # execute + export
    python scripts/run_pipeline.py --inplace    # also write outputs back into the notebook
    python scripts/run_pipeline.py --no-execute # export the stored outputs (used by CI)
"""
import argparse
import base64
import io
import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from nbconvert import HTMLExporter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "speech_denoising_unet.ipynb"


def display_path(path):
    """Path relative to the repository root when possible (for log messages)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def execute(nb, notebook_path, timeout):
    from nbclient import NotebookClient  # only needed when executing
    print(f"Executing {notebook_path.name} ...")
    start = time.time()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(notebook_path.parent)}})
    client.execute()
    print(f"Finished in {time.time() - start:.1f} s")
    return nb


def export_html(nb, site_dir):
    site_dir.mkdir(parents=True, exist_ok=True)
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    out = site_dir / "index.html"
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {display_path(out)}")


def collect_report_items(nb):
    """Return figures, tables and text blocks from cells carrying ``metadata.report``."""
    items = []
    for cell in nb.cells:
        spec = cell.get("metadata", {}).get("report")
        if cell.cell_type != "code" or not spec:
            continue
        images = [o["data"]["image/png"] for o in cell.outputs if "data" in o and "image/png" in o["data"]]
        names = spec.get("figures", [])
        captions = spec.get("captions", [spec.get("caption", "")] * len(names))
        for name, caption, png in zip(names, captions, images):
            items.append({"kind": "figure", "name": name, "png": base64.b64decode(png), "caption": caption})
        if spec.get("tables") or spec.get("text"):
            text = []
            for o in cell.outputs:
                if o.get("output_type") == "stream" and o.get("name") == "stdout":  # skip progress bars
                    text.append(o["text"])
                elif "data" in o and "text/plain" in o["data"]:
                    text.append(o["data"]["text/plain"])
            items.append({"kind": "text", "name": (spec.get("tables") or [spec.get("text")])[0],
                          "text": "\n".join(text).rstrip(), "caption": spec.get("caption", "")})
    return items


def save_figures(items, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item["kind"] == "figure":
            (fig_dir / f"{item['name']}.png").write_bytes(item["png"])
    print(f"Wrote {sum(i['kind'] == 'figure' for i in items)} figures to {display_path(fig_dir)}")


def build_pdf(nb, items, pdf_path):
    meta = nb.metadata.get("report", {})
    a4 = (8.27, 11.69)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=a4)
        fig.text(0.08, 0.90, meta.get("title", "Results report"), fontsize=18, weight="bold", wrap=True)
        y = 0.85
        for para in meta.get("summary", []):
            wrapped = textwrap.fill(para, 90)
            fig.text(0.08, y, wrapped, fontsize=10, va="top")
            y -= 0.02 * (wrapped.count("\n") + 1) + 0.02
        fig.text(0.08, y - 0.01, "Execution: " + meta.get("execution_note", ""), fontsize=9, style="italic",
                 va="top", wrap=True)
        y -= 0.06
        for item in items:
            if item["kind"] != "text":
                continue
            fig.text(0.08, y, item["caption"], fontsize=10, weight="bold", va="top")
            y -= 0.025
            fig.text(0.08, y, item["text"], fontsize=7.5, family="monospace", va="top")
            y -= 0.014 * (item["text"].count("\n") + 1) + 0.04
        pdf.savefig(fig)
        plt.close(fig)

        # Figures flow down the page; each is scaled by its aspect ratio with the caption directly below.
        figures = [i for i in items if i["kind"] == "figure"]
        page_ratio = a4[0] / a4[1]
        fig, y = None, 0.0
        for number, item in enumerate(figures, start=1):
            img = mpimg.imread(io.BytesIO(item["png"]), format="png")
            width = 0.86
            height = width * page_ratio * img.shape[0] / img.shape[1]
            if height > 0.62:
                width *= 0.62 / height
                height = 0.62
            if fig is None or y - height - 0.05 < 0.03:
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
                fig, y = plt.figure(figsize=a4), 0.95
            ax = fig.add_axes([(1 - width) / 2, y - height, width, height])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, y - height - 0.008, f"Figure {number}. {item['caption']}",
                     ha="center", va="top", fontsize=9, wrap=True)
            y -= height + 0.06
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {display_path(pdf_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true", help="export the notebook without running it")
    parser.add_argument("--inplace", action="store_true", help="save the executed notebook over the source")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    nb = nbformat.read(notebook_path, as_version=4)
    if not args.no_execute:
        nb = execute(nb, notebook_path, args.timeout)
        target = notebook_path if args.inplace else ROOT / "build" / notebook_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, target)
        print(f"Saved executed notebook to {display_path(target)}")

    export_html(nb, args.site_dir)
    items = collect_report_items(nb)
    save_figures(items, args.outputs_dir / "figures")
    pdf_name = nb.metadata.get("report", {}).get("pdf_name", "results_report.pdf")
    build_pdf(nb, items, args.outputs_dir / pdf_name)


if __name__ == "__main__":
    main()
