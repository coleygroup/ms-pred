""" plot_utils.py

Set plot utils

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from matplotlib.path import Path
import matplotlib.patches as patches
import re
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import gridspec


legend_params = dict(frameon=False, facecolor="none", fancybox=False)
method_colors = {
    "CFM-ID": "#D7D7D7",
    "MassFormer": "#E9B382",
    "GrAFF-MS": "#FFD593",
    "FraGNNet": "#7C9D97",
    "ICEBERG (Goldman'24)": "#A7B7C3",
    "ICEBERG (Ours)": "#7B94CC",
    "MetFrag": "#818181",
}

# List all marker symbols in list in commnet
# https://matplotlib.org/3.1.1/api/markers_api.html
# markers: [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s",]

method_markers = {
    "CFM-ID": ".",
    "MassFormer": "<",
    "Graff-MS": "v",
    "FraGNNet": ">",
    "ICEBERG (Goldman'24)": "x",
    "ICEBERG (Ours)": "o",
}

plt_dataset_names = {"canopus_train_public": "NPLIB1", "nist20": "NIST20"}


def export_mol(
    mol,
    name,
    width=100,
    height=100,
):
    """Save structure as PDF"""
    from rdkit.Chem.Draw import rdMolDraw2D
    import cairosvg
    import io

    drawer = rdMolDraw2D.MolDraw2DSVG(
        width,
        height,
    )
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    cairosvg.svg2pdf(bytestring=drawer.GetDrawingText().encode(), write_to=str(name))


def export_mol_highlight(
    mol,
    name,
    hatoms,
    hbonds,
    width=100,
    height=100,
    color=(0.925, 0.688, 0.355),
):
    """Save structure as PDF"""
    from rdkit.Chem.Draw import rdMolDraw2D
    import cairosvg
    import io

    d = rdMolDraw2D.MolDraw2DSVG(
        width,
        height,
    )
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=hatoms,
        highlightBonds=hbonds,
        highlightBondColors={i: color for i in hbonds},
        highlightAtomColors={i: color for i in hatoms},
    )
    d.FinishDrawing()
    cairosvg.svg2pdf(bytestring=d.GetDrawingText().encode(), write_to=str(name))


def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = 'Arial'
    # If Arial did not show up:
    # 1) install Microsoft fonts: sudo apt-get install ttf-mscorefonts-installer
    # 2) clear local cache: fc-cache -f -v && rm -r .cache/matplotlib/*
    # 3) restart Python environment
    sns.set(context="paper", style="ticks")
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5
    mpl.rcParams["xtick.minor.size"] = 2.0
    mpl.rcParams["ytick.minor.size"] = 2.0

    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45
    mpl.rcParams["xtick.minor.width"] = 0.45
    mpl.rcParams["ytick.minor.width"] = 0.45
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.pad'] = 1
    mpl.rcParams['ytick.major.pad'] = 1
    mpl.rcParams['xtick.minor.pad'] = 1
    mpl.rcParams['ytick.minor.pad'] = 1

    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 6
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 6
    mpl.rcParams["figure.titlesize"] = 6
    mpl.rcParams["figure.titlesize"] = 6
    mpl.rcParams["legend.fontsize"] = 5
    mpl.rcParams["legend.title_fontsize"] = 6
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5


def set_size(w, h, ax=None):
    """w, h: width, height in inches

    Resize the axis to have exactly these dimensions

    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_compare_ms(spec1, spec2, spec1_name='', spec2_name='', ce_label='', dpi=300, ppm=20, ax=None, largest_mz=None,
                    matched_color='#7C9D97', spec1_color='#58595B', spec2_color='#58595B'):
    """
    spec1 and spec2 are 2d arrays [(mz1, inten1), (mz2, inten2), ...]
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 4), dpi=dpi)
        ax = plt.gca()
    else:
        plt.sca(ax)
    if largest_mz is None:
        largest_mz = 0
    for idx, spec in enumerate((spec1, spec2)):
        spec = np.array(spec).astype(np.float64)
        spec[:, 1] = spec[:, 1] / spec[:, 1].max()
        spec = spec[spec[:, 1] > 0.01]  # remove low intensity peaks
        largest_mz = max(largest_mz, spec[:, 0].max())
        intensity_arr = spec[:, 1] if idx == 0 else -spec[:, 1]
        for mz, inten in zip(spec[:, 0], intensity_arr):
            mz_in_spec1 = np.min(np.abs(mz - spec1[:, 0])) / mz < 1e-6 * ppm
            mz_in_spec2 = np.min(np.abs(mz - spec2[:, 0])) / mz < 1e-6 * ppm
            if mz_in_spec1 and mz_in_spec2:
                color = matched_color
            elif inten > 0:
                color = spec1_color
            else:
                color = spec2_color
            markerline, stemlines, baseline = plt.stem(mz, inten, color, markerfmt=" ", basefmt=" ")
            plt.setp(stemlines, 'linewidth', 0.25)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.4)
    plt.text(-0.07, 0.6, spec1_name, rotation=90, rotation_mode='anchor',
             verticalalignment='bottom', horizontalalignment='center', transform=ax.get_yaxis_transform())
    plt.text(-0.07, -0.6, spec2_name, rotation=90, rotation_mode='anchor',
             verticalalignment='bottom', horizontalalignment='center', transform=ax.get_yaxis_transform())
    if ce_label:
        plt.text(0.02, 0.02, ce_label,
                 verticalalignment='bottom', horizontalalignment='left', transform=ax.get_yaxis_transform())

    plt.xlabel('m/z', rotation=0)
    ax.xaxis.set_label_coords(0.5, -0.2)

    ax.set_yticks(ticks=[-1, 0, 1])
    ax.set_yticklabels(['1', '0', '1'])

    ax.set_xlim(0, largest_mz * 1.05)
    ax.set_ylim(-1.1, 1.1)

    # hide top and right rules
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_compare_ref_ms_with_structures(
        spec1, spec2, spec3=None, spec4=None,
        spec1_name='Target', spec2_name='Reference',
        spec3_name='Predicted', spec4_name='Additional',
        spec1_smiles=None, spec2_smiles=None,
        spec3_smiles=None, spec4_smiles=None,
        spec1_ce_label='', spec2_ce_label='', dpi=300, ppm=20, 
        ax=None, save_path=None):
    """
    Plot up to 4 MS spectra with optional molecule structures in a 2x2 layout.
    """

    spectra = [spec1, spec2]
    names = [f'{spec1_name} ({spec1_ce_label} eV)', f'{spec2_name} ({spec2_ce_label} eV)']
    smiles = [spec1_smiles, spec2_smiles]

    if spec3 is not None:
        spectra.append(spec3)
        names.append(spec3_name)
        smiles.append(spec3_smiles)

    if spec4 is not None:
        spectra.append(spec4)
        names.append(spec4_name)
        smiles.append(spec4_smiles)

    n_specs = len(spectra)
    n_plots = min(n_specs, 4)  # support up to 4

    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # Compute global largest m/z
    largest_mz = np.max(np.concatenate([s[:, 0] for s in spectra]))

    for idx in range(n_plots):
        row, col = divmod(idx, 2)
        ax_spec = fig.add_subplot(gs[row, col])

        ax_spec.set_title(names[idx], y=0.9, fontsize=16)
        plot_ms(spectra[idx], ax=ax_spec, linewidth=1.0, largest_mz=largest_mz)

        if idx < n_plots - 2:
            ax_spec.set_xlabel("")
            ax_spec.set_xticklabels([])

        if smiles[idx]:
            mol = Chem.MolFromSmiles(smiles[idx])
            img = Draw.MolToImage(mol, size=(300, 300))
            # Use inset axes for structure
            inset_ax = ax_spec.inset_axes([0.7, 0.55, 0.25, 0.4])
            inset_ax.imshow(img)
            inset_ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_ms(spec, spec_name='', ce_label='', dpi=300, ax=None, largest_mz=None, linewidth=0.25):
    """
    spec is a 2d array [(mz1, inten1), (mz2, inten2), ...]
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 2), dpi=dpi)
        ax = plt.gca()
    else:
        plt.sca(ax)
    spec = np.array(spec).astype(np.float64)
    spec[:, 1] = spec[:, 1] / spec[:, 1].max()
    spec = spec[spec[:, 1] > 0.01]
    if largest_mz is None:
        largest_mz = 0
        largest_mz = max(largest_mz, spec[:, 0].max())
    intensity_arr = spec[:, 1]
    for mz, inten in zip(spec[:, 0], intensity_arr):
        markerline, stemlines, baseline = plt.stem(mz, inten, '#58595B', markerfmt=" ", basefmt=" ")
        plt.setp(stemlines, 'linewidth', linewidth)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.4)
    plt.ylabel(spec_name)
    ax.yaxis.set_label_coords(-0.05, 0.5)

    if ce_label:
        plt.text(0.02, 0.02, ce_label,
                 verticalalignment='bottom', horizontalalignment='left', transform=ax.get_yaxis_transform())

    plt.xlabel('m/z', rotation=0)
    ax.xaxis.set_label_coords(0.5, -0.25)

    ax.set_yticks(ticks=[0, 1])
    ax.set_yticklabels(['0', '1'])

    ax.set_xlim(0, largest_mz * 1.05)
    ax.set_ylim(0, 1.1)

    # hide top and right rules
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def _parse_svg_path_data(d, offset=(0, 0), zoom=(1, 1)):
    """Function to parse SVG path data into Matplotlib Path"""
    vertices = []
    codes = []
    parts = re.split(',| ', d)
    parts = [p for p in parts if len(p) > 0]
    i = 0
    while i < len(parts):
        cmd = parts[i]
        if cmd == 'M':
            x, y = float(parts[i + 1]), float(parts[i + 2])
            vertices.append((x, y))
            codes.append(Path.MOVETO)
            i += 3
        elif cmd == 'L':
            x, y = float(parts[i + 1]), float(parts[i + 2])
            vertices.append((x, y))
            codes.append(Path.LINETO)
            i += 3
        elif cmd == 'Q':
            x1, y1 = float(parts[i + 1]), float(parts[i + 2])
            x2, y2 = float(parts[i + 3]), float(parts[i + 4])
            vertices.append((x1, y1))
            vertices.append((x2, y2))
            codes.append(Path.CURVE3)
            codes.append(Path.CURVE3)
            i += 5
        elif cmd == 'Z':
            codes.append(Path.CLOSEPOLY)
            vertices.append(vertices[0])  # Close the path
            i += 1
        else:
            i += 1  # Handle other cases if needed
    vertices = np.array(vertices)
    vertices[:, 0] *= zoom[0]
    vertices[:, 1] *= -zoom[1]
    vertices[:, 0] += offset[0]
    vertices[:, 1] += offset[1]
    return Path(vertices, codes)


def _parse_style(style_string):
    """Function to convert SVG style string to a dictionary"""
    styles = style_string.split(';')
    style_dict = {}
    for style in styles:
        if ':' in style:
            key, value = style.split(':')
            style_dict[key.strip()] = value.strip()
    return style_dict


def plot_mol_as_vector(mol, hatoms=None, hbonds=None, atomcmap=None, atomscores=None, ax=None, offset=(0, 0), zoom=1):
    # Draw plot as SVG
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG

    svg_size = (300, 300)
    d2d = rdMolDraw2D.MolDraw2DSVG(*svg_size)
    if atomcmap is not None and atomscores is not None:
        cmap = plt.get_cmap(atomcmap)
        cnorm = mpl.colors.Normalize(vmin=min(atomscores), vmax=max(atomscores))
        colors = {i: cmap(cnorm(s)) for i, s in enumerate(atomscores)}
        d2d.DrawMolecule(mol, highlightAtoms=hatoms, highlightBonds=hbonds, highlightAtomColors=colors)
        plt.colorbar(plt.cm.ScalarMappable(cnorm, cmap))
    else:
        d2d.DrawMolecule(mol, highlightAtoms=hatoms, highlightBonds=hbonds)
    d2d.FinishDrawing()
    a = SVG(d2d.GetDrawingText())

    # Example SVG string
    svg_string = a.data

    # Parse the SVG string
    root = ET.fromstring(svg_string)

    # Create a Matplotlib figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # Handle zoom coordinates
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_to_y_ratio = (xmax - xmin) / (ymax - ymin) * bbox.height / bbox.width
    zoom = (zoom * x_to_y_ratio, zoom)

    # move offset to plot center
    offset = (offset[0] - svg_size[0] * zoom[0] / 2, offset[1] + svg_size[1] * zoom[1] / 2)

    # Loop through each path element in the SVG
    for elem in root:
        if elem.tag.endswith('path'):
            path_data = elem.attrib['d']
            style_string = elem.attrib.get('style', '')

            # Convert the path data to a Matplotlib Path
            mpl_path = _parse_svg_path_data(path_data, offset=offset, zoom=zoom)

            # Parse the style attributes
            style = _parse_style(style_string)

            # Extract specific style attributes
            stroke_color = style.get('stroke', '#000000')  # Default to black
            linewidth = float(style.get('stroke-width', '0').strip('px')) * zoom[0]
            fill_color = style.get('fill', 'none')
            if fill_color == 'none':
                fill_color = elem.attrib.get('fill', 'none')

            name = elem.attrib.get('class', 'none')
            if name.startswith('bond'):
                linewidth = max(linewidth, 0.1)
            elif name.startswith('atom'):
                pass
            else:
                linewidth = max(linewidth, 0.15)  # highlights

            # Create a PathPatch with the extracted style
            patch = patches.PathPatch(mpl_path, facecolor=fill_color, edgecolor=stroke_color, lw=linewidth)

            # Add the patch to the Matplotlib axes
            ax.add_patch(patch)

        elif elem.tag.endswith('ellipse'):
            # Extract ellipse attributes
            cx = float(elem.attrib['cx'])
            cy = float(elem.attrib['cy'])
            rx = float(elem.attrib['rx'])
            ry = float(elem.attrib['ry'])
            style_string = elem.attrib.get('style', '')

            # Parse the style attributes
            style = _parse_style(style_string)

            # Extract specific style attributes
            stroke_color = style.get('stroke', '#000000')  # Default to black
            linewidth = float(style.get('stroke-width', '1.0').strip('px')) * zoom[0]
            fill_color = style.get('fill', 'none')

            # Create an Ellipse patch
            ellipse = patches.Ellipse((cx * zoom[0] + offset[0], -cy * zoom[1] + offset[1]), width=2*rx*zoom[0], height=2*ry*zoom[1], facecolor=fill_color, edgecolor=stroke_color, lw=linewidth)

            # Add the ellipse to the Matplotlib axes
            ax.add_patch(ellipse)
