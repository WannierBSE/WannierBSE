"""Plot the exciton energy spectrum produced by the Python WBSE solver.

The utility reads the real-valued exciton energies in
``Exciton_data/Ex.txt``. The file is written by ``WBSE.py`` in electron-volts,
with one energy per row and a comment header. The resulting figure shows the
energy levels using the package's publication-oriented styling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Load exciton energies, plot their levels, and save the figure."""

    config = {
        "dataPath": "Exciton_data",
        "dataFile": "Ex.txt",
        "pngOutput": "Ex_plot.png",
        # "pdfOutput": "Ex_plot.pdf",
        "lineWidth": 2.0,
        "lineColor": [0.1, 0.25, 0.60],
        "fontSize": 18,
        "fontName": "Helvetica",
        "aspectRatio": 2.8,
        "dpi": "-r300",
        "rightMargin": 0.30,
    }

    plt.close("all")
    print("\n>> Initializing Exciton Plotting Suite...")

    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / config["dataPath"] / config["dataFile"]

    if not file_path.is_file():
        raise FileNotFoundError(f"Error: Target file [{file_path}] not found.")

    print(f">> Loading dataset: {file_path}")
    try:
        # ndmin=1 keeps a one-energy file compatible with the plotting loop.
        ex = np.loadtxt(file_path, comments="#", ndmin=1)
    except ValueError as exc:
        raise ValueError(
            f"Exciton energy file is empty or invalid: {file_path}"
        ) from exc
    ex = np.asarray(ex, dtype=float).reshape(-1)
    if ex.size == 0:
        raise ValueError(f"Exciton energy file is empty: {file_path}")

    fig, ax = plt.subplots(
        figsize=(4.0, 6.0),
        facecolor="white",
        num="Exciton Energy Spectrum",
    )
    fig.patch.set_facecolor("white")

    for energy in ex:
        ax.plot(
            [0.0, 1.0],
            [energy, energy],
            color=config["lineColor"],
            linewidth=config["lineWidth"],
        )

    ax.set_box_aspect(config["aspectRatio"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel(
        r"$\mathrm{Exciton\ energy}\ E^{X} \ \mathrm{(eV)}$",
        fontsize=config["fontSize"] + 2,
    )

    y_limits = ax.get_ylim()
    ylim_padding = (y_limits[1] - y_limits[0]) * 0.05
    ax.set_ylim(y_limits[0] - ylim_padding, np.max(ex))

    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        direction="in",
        length=4.0,
        width=1.5,
        labelsize=config["fontSize"],
        which="major",
    )
    ax.minorticks_off()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(config["fontName"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    output_relative_path = Path(config["dataPath"]) / "Figures"
    output_dir = script_dir / output_relative_path
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / config["pngOutput"]
    pdf_path = None
    if "pdfOutput" in config and config["pdfOutput"]:
        pdf_path = output_dir / config["pdfOutput"]

    fig.subplots_adjust(
        left=0.20,
        right=max(0.20, 1.0 - config["rightMargin"] / fig.get_figwidth()),
        bottom=0.08,
        top=0.98,
    )
    fig.set_size_inches(
        fig.get_figwidth() + config["rightMargin"],
        fig.get_figheight() + 0.01,
    )

    print(">> Exporting publication-quality files...")
    dpi_setting = str(config["dpi"]).lower().replace("-r", "")
    fig.savefig(png_path, dpi=int(dpi_setting), facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", facecolor="white")

    print(f"[Info] The exciton energy spectrum was generated at {output_relative_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
