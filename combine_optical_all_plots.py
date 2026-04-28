import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    raise ImportError(
        "This script requires scipy for Savitzky-Golay smoothing.\n"
        "Install it with: pip install scipy"
    )


# =========================================================
# Helper: chemical formula with subscripts
# =========================================================
def to_subscript(formula):
    """
    Converts digits in formula names to subscripts for matplotlib legends.
    Example: MoSeSiN2 -> MoSeSiN$_2$
    """
    return ''.join([f"$_{ch}$" if ch.isdigit() else ch for ch in formula])


# =========================================================
# User options
# =========================================================
SMOOTH = True
WINDOW = 15
POLYORDER = 3

SHOW_RAW = False
RAW_ALPHA = 0.18

# Three different materials
# All will be plotted only for out-of-plane direction: E/c
#folders = ["CrSeSiN2", "CrSSiN2", "CrTeSiN2"]

#folders = ["MoSeSiN2", "MoSSiN2", "MoTeSiN2"]

folders = ["WSeSiN2", "WSSiN2", "WTeSiN2"]

PLOT_OUT_OF_PLANE = True

ENERGY_MIN = 0
ENERGY_MAX = 8

ABS_SCALE = 1e5
ABS_LABEL = r"$\alpha(\omega)$ ($10^5$ cm$^{-1}$)"

OUTPUT_NAME = "optical_plots_Ec_bold.png"

# =========================================================
# Spectrum background options
# =========================================================
SHOW_SPECTRUM_BACKGROUND = True

VISIBLE_MIN = 2.55   # ~750 nm
VISIBLE_MAX = 7.20   # ~400 nm

DULL_COLOR = "#eeeeee"
DULL_ALPHA = 0.15
VISIBLE_ALPHA = 0.85

# =========================================================
# Strong colors for three materials
# =========================================================
colors = [
    "#0072B2",   # MoSeSiN2: blue
    "#D55E00",   # MoSSiN2: orange-red
    "#009E73",   # MoTeSiN2: green
]

line_styles = ['-', '-', '-']


# =========================================================
# Smoothing helpers
# =========================================================
def get_valid_window(n, preferred_window, polyorder):
    if n < 3:
        return None

    max_window = n if n % 2 == 1 else n - 1

    min_window = polyorder + 1
    if min_window % 2 == 0:
        min_window += 1

    if max_window < min_window:
        return None

    window = min(preferred_window, max_window)

    if window % 2 == 0:
        window -= 1

    if window < min_window:
        window = min_window

    if window > max_window:
        return None

    return window


def smooth_curve(y, preferred_window=15, polyorder=3):
    y = np.asarray(y, dtype=float)

    if len(y) < 3:
        return y.copy()

    y_work = y.copy()
    finite_mask = np.isfinite(y_work)

    if not finite_mask.any():
        return y.copy()

    if not finite_mask.all():
        idx = np.arange(len(y_work))
        y_work[~finite_mask] = np.interp(
            idx[~finite_mask],
            idx[finite_mask],
            y_work[finite_mask]
        )

    window = get_valid_window(len(y_work), preferred_window, polyorder)

    if window is None:
        return y_work

    return savgol_filter(
        y_work,
        window_length=window,
        polyorder=polyorder,
        mode='interp'
    )


def prepare_xy(x, y, smooth=False, window=15, polyorder=3,
               clip_min=None, clip_max=None):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if smooth:
        y = smooth_curve(y, preferred_window=window, polyorder=polyorder)

    if clip_min is not None:
        y = np.maximum(y, clip_min)

    if clip_max is not None:
        y = np.minimum(y, clip_max)

    return x, y


def plot_series(ax, x, y, color, linestyle, label,
                smooth=False, window=15, polyorder=3,
                show_raw=False, raw_alpha=0.2,
                clip_min=None, clip_max=None):

    x_raw, y_raw = prepare_xy(x, y, smooth=False)

    if show_raw and smooth:
        ax.plot(
            x_raw,
            y_raw,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=raw_alpha
        )

    x_plot, y_plot = prepare_xy(
        x,
        y,
        smooth=smooth,
        window=window,
        polyorder=polyorder,
        clip_min=clip_min,
        clip_max=clip_max
    )

    ax.plot(
        x_plot,
        y_plot,
        color=color,
        linestyle=linestyle,
        linewidth=3.0,
        label=label
    )

# =========================================================
# Helper: visible + non-visible spectrum background
# =========================================================
def add_spectrum_background(
    ax,
    visible_min=1.65,
    visible_max=3.10,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    dull_color="#eeeeee",
    dull_alpha=0.85,
    visible_alpha=0.55
):
    """
    Add a dull/light background for the full optical range and
    a brighter rainbow background only for the visible region.

    visible_min, visible_max : visible-light energy range in eV
    x_min, x_max             : full x-range to shade
    y_min, y_max             : full y-range to shade
    """

    # Get axis limits if not given
    if x_min is None or x_max is None:
        current_xlim = ax.get_xlim()
        x_min = current_xlim[0]
        x_max = current_xlim[1]

    if y_min is None or y_max is None:
        current_ylim = ax.get_ylim()
        y_min = current_ylim[0]
        y_max = current_ylim[1]

    # -----------------------------------------------------
    # 1) Dull/light background over the whole panel
    # -----------------------------------------------------
    ax.axvspan(
        x_min,
        x_max,
        ymin=0,
        ymax=1,
        facecolor=dull_color,
        alpha=dull_alpha,
        zorder=0
    )

    # -----------------------------------------------------
    # 2) Bright visible rainbow only in visible range
    # -----------------------------------------------------
    colors_visible = [
        "red",
        "orange",
        "yellow",
        "limegreen",
        "cyan",
        "blue",
        "violet"
    ]

    cmap_visible = LinearSegmentedColormap.from_list(
        "visible_spectrum",
        colors_visible
    )

    gradient = np.linspace(0, 1, 512).reshape(1, -1)

    ax.imshow(
        gradient,
        extent=[visible_min, visible_max, y_min, y_max],
        aspect="auto",
        cmap=cmap_visible,
        alpha=visible_alpha,
        origin="lower",
        zorder=1
    )
    
    
    
# =========================================================
# Styling helper for every subplot
# =========================================================
def style_axis(ax, panel_label, xlim=(0, 5), ylim=None):
    ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lines = ax.get_lines()
        y_all = []

        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            mask = (
                np.isfinite(x_data) &
                np.isfinite(y_data) &
                (x_data >= xlim[0]) &
                (x_data <= xlim[1])
            )

            if mask.any():
                y_all.extend(y_data[mask])

        if len(y_all) > 0:
            y_all = np.asarray(y_all)
            ymin = np.nanmin(y_all)
            ymax = np.nanmax(y_all)

            if np.isclose(ymin, ymax):
                margin = 0.1 * abs(ymax) if ymax != 0 else 0.1
            else:
                margin = 0.08 * (ymax - ymin)

            ax.set_ylim(ymin - margin, ymax + margin)

    # Bold subplot boundary
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
        spine.set_color("black")

    # Bold ticks
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        width=2.2,
        length=7,
        labelsize=12,
        top=True,
        right=True
    )

    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        width=1.5,
        length=4,
        top=True,
        right=True
    )

    ax.minorticks_on()

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # Panel label: (a), (b), ...
    ax.text(
        0.04,
        0.90,
        panel_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        color='black'
    )

    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.28)

    legend = ax.legend(
        fontsize=11,
        frameon=True,
        loc='best',
        edgecolor='black'
    )

    if legend is not None:
        legend.get_frame().set_linewidth(1.4)
        legend.get_frame().set_alpha(0.95)

        for text in legend.get_texts():
            text.set_fontweight('bold')


# =========================================================
# Read data
# =========================================================
data_list = {}

required_columns = [
    "energy_eV",
    "n_out",
    "k_out",
    "alpha_out_cm^-1",
    "R_out",
    "eps1_out",
    "eps2_out"
]

for folder in folders:
    csv_file = os.path.join(folder, "output.csv")

    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_file}: {missing}")

        data_list[folder] = data
    else:
        print(f"CSV not found: {csv_file}")

if not data_list:
    raise FileNotFoundError(
        "No valid output.csv files were found in the listed folders."
    )


# =========================================================
# Create figure: 3 rows x 2 columns
# =========================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"

fig, axs = plt.subplots(
    3,
    2,
    figsize=(14.5, 15.5),
    constrained_layout=False
)

axs = np.asarray(axs)


# =========================================================
# Plot 1: Refractive index n, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[0, 0],
        data["energy_eV"],
        data["n_out"],
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA
    )

axs[0, 0].set_title("Refractive Index", fontsize=15, fontweight='bold')
axs[0, 0].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[0, 0].set_ylabel(r"$n(\omega)$", fontsize=14, fontweight='bold')
style_axis(axs[0, 0], "(a)", xlim=(ENERGY_MIN, ENERGY_MAX)) #, ylim=(1.6, 2.8))


# =========================================================
# Plot 2: Extinction coefficient k, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[0, 1],
        data["energy_eV"],
        data["k_out"],
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA,
        clip_min=0.0
    )

axs[0, 1].set_title("Extinction Coefficient", fontsize=15, fontweight='bold')
axs[0, 1].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[0, 1].set_ylabel(r"$k(\omega)$", fontsize=14, fontweight='bold')
style_axis(axs[0, 1], "(b)", xlim=(ENERGY_MIN, ENERGY_MAX))  #, ylim=(0, 1.2))


# =========================================================
# Plot 3: Absorption coefficient, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[1, 0],
        data["energy_eV"],
        data["alpha_out_cm^-1"] / ABS_SCALE,
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA,
        clip_min=0.0
    )

axs[1, 0].set_title("Absorption Coefficient", fontsize=15, fontweight='bold')
axs[1, 0].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[1, 0].set_ylabel(ABS_LABEL, fontsize=14, fontweight='bold')

# First style axis so limits are available
style_axis(axs[1, 0], "(c)", xlim=(ENERGY_MIN, ENERGY_MAX))

# Add dull + bright spectrum background
if SHOW_SPECTRUM_BACKGROUND:
    x0, x1 = axs[1, 0].get_xlim()
    y0, y1 = axs[1, 0].get_ylim()

    add_spectrum_background(
        axs[1, 0],
        visible_min=VISIBLE_MIN,
        visible_max=VISIBLE_MAX,
        x_min=x0,
        x_max=x1,
        y_min=y0,
        y_max=y1,
        dull_color=DULL_COLOR,
        dull_alpha=DULL_ALPHA,
        visible_alpha=VISIBLE_ALPHA
    )

# Bring plotted curves above background
for line in axs[1, 0].get_lines():
    line.set_zorder(3)

# Keep grid above background, below curves
axs[1, 0].grid(True, linestyle='--', linewidth=0.7, alpha=0.28, zorder=2)

# Rebuild legend so it stays above everything
legend = axs[1, 0].legend(
    fontsize=11,
    frameon=True,
    loc='best',
    edgecolor='black'
)

if legend is not None:
    legend.set_zorder(5)
    legend.get_frame().set_linewidth(1.4)
    legend.get_frame().set_alpha(0.95)

    for text in legend.get_texts():
        text.set_fontweight('bold')

# =========================================================
# Plot 4: Optical reflectivity, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[1, 1],
        data["energy_eV"],
        data["R_out"],
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA,
        clip_min=0.0,
        clip_max=1.0
    )

axs[1, 1].set_title("Optical Reflectivity", fontsize=15, fontweight='bold')
axs[1, 1].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[1, 1].set_ylabel(r"$R(\omega)$", fontsize=14, fontweight='bold')
style_axis(axs[1, 1], "(d)", xlim=(ENERGY_MIN, ENERGY_MAX)) #, ylim=(0, 0.25))


# =========================================================
# Plot 5: Real dielectric function epsilon_1, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[2, 0],
        data["energy_eV"],
        data["eps1_out"],
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA
    )

axs[2, 0].set_title("Real Part of Dielectric Function", fontsize=15, fontweight='bold')
axs[2, 0].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[2, 0].set_ylabel(r"$\varepsilon_1(\omega)$", fontsize=14, fontweight='bold')
style_axis(axs[2, 0], "(e)", xlim=(ENERGY_MIN, ENERGY_MAX)) #, ylim=(2.4, 6.6))


# =========================================================
# Plot 6: Imaginary dielectric function epsilon_2, out-of-plane E/c
# =========================================================
for i, (folder, data) in enumerate(data_list.items()):
    color = colors[i % len(colors)]
    label_name = to_subscript(folder)
    legend_label = rf"{label_name}- $E/c$"

    plot_series(
        axs[2, 1],
        data["energy_eV"],
        data["eps2_out"],
        color=color,
        linestyle=line_styles[i % len(line_styles)],
        label=legend_label,
        smooth=SMOOTH,
        window=WINDOW,
        polyorder=POLYORDER,
        show_raw=SHOW_RAW,
        raw_alpha=RAW_ALPHA
    )

axs[2, 1].set_title("Imaginary Part of Dielectric Function", fontsize=15, fontweight='bold')
axs[2, 1].set_xlabel("Energy (eV)", fontsize=14, fontweight='bold')
axs[2, 1].set_ylabel(r"$\varepsilon_2(\omega)$", fontsize=14, fontweight='bold')
style_axis(axs[2, 1], "(f)", xlim=(ENERGY_MIN, ENERGY_MAX))


# =========================================================
# Final layout and save
# =========================================================
plt.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.07,
    top=0.94,
    wspace=0.25,
    hspace=0.35
)

plt.savefig(OUTPUT_NAME, dpi=700, bbox_inches="tight")
plt.savefig("optical_plots_Ec_bold.pdf", dpi=700, bbox_inches="tight")

plt.show()