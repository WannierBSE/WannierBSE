# =========================================================================
# PLOT_EXCITON_LEVELS.PY
# =========================================================================
# Summary:   Loads and visualizes exciton energy spectra from Ex.txt data.
#
# Features:
#   - Aspect Ratio Control for publication-quality visuals.
#   - Automated multi-format export (PDF/PNG).
# =========================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Load exciton energies, plot their levels, and save the figure."""

    # %% ---------------------------------------------------------------------
    #  1. GLOBAL USER CONFIGURATION (EDIT HERE)
    # ------------------------------------------------------------------------

    # -- Bandgap Energy --
    Eg = 0.0  # In eV. If Eg=0, plot the exciton energy; otherwise, plot Eb=Ex-Eg.

    # -- Energy Display Unit --
    energyUnit = "eV"  # Options: 'eV' or 'meV'

    # -- File & Path Settings --
    config = {
        "dataPath": "Exciton_data",
        "dataFile": "Ex.txt",
        "pngOutput": "Ex_plot.png",
        # "pdfOutput": "Ex_plot.pdf",

        # -- Visual Styling --
        "lineWidth": 2.0,
        "lineColor": [0.1, 0.25, 0.60],
        "fontSize": 18,
        "fontName": "Nimbus Sans",
        "aspectRatio": 2.8,

        # -- Export Parameters --
        "dpi": "-r300",
        "rightMargin": 0.30,
    }

    plt.close("all")
    print("\n>> Initializing Exciton Plotting Suite...")

    # %% ---------------------------------------------------------------------
    #  2. DATA INITIALIZATION & LOADING
    # ------------------------------------------------------------------------

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

    # Calculate the physical energy quantity first, then convert for display.
    if Eg == 0:
        plot_energy = ex
    else:
        plot_energy = ex - Eg

    normalized_energy_unit = energyUnit.lower()
    if normalized_energy_unit == "ev":
        energy_scale = 1.0
        energy_unit_label = "eV"
    elif normalized_energy_unit == "mev":
        energy_scale = 1000.0
        energy_unit_label = "meV"
    else:
        raise ValueError(
            f"Invalid energy unit [{energyUnit}]. Use 'eV' or 'meV'."
        )

    plot_energy = energy_scale * plot_energy

    if Eg == 0:
        y_axis_label = (
            rf"$E^{{X}}\ "
            rf"\mathrm{{({energy_unit_label})}}$"
        )
    else:
        y_axis_label = (
            rf"$E^{{X}}-E_{{g}}\ "
            rf"\mathrm{{({energy_unit_label})}}$"
        )

    # %% ---------------------------------------------------------------------
    #  3. CORE PLOTTING ENGINE
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(4.0, 6.0),
        facecolor="white",
        num="Exciton Energy Spectrum",
    )
    fig.patch.set_facecolor("white")

    for energy in plot_energy:
        ax.plot(
            [0.0, 1.0],
            [energy, energy],
            color=config["lineColor"],
            linewidth=config["lineWidth"],
        )

    ax.set_box_aspect(config["aspectRatio"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel(y_axis_label, fontsize=config["fontSize"] + 2)

    y_limits = ax.get_ylim()
    ylim_padding = (y_limits[1] - y_limits[0]) * 0.05
    ax.set_ylim(y_limits[0] - ylim_padding, np.max(plot_energy))

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

    # %% ---------------------------------------------------------------------
    #  4. EXPORT & FILESYSTEM MANAGEMENT
    # ------------------------------------------------------------------------

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

# =========================================================================
# END OF SCRIPT
# =========================================================================
