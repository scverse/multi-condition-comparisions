import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import adjustText
import matplotlib.patheffects as PathEffects
import os
import anndata as ad
from typing import Union, List, Tuple, Optional, Dict


def volcano(
    data: Union[pd.DataFrame, ad.AnnData],
    log2fc_col: str = "log2FoldChange",
    pvalue_col: str = "padj",
    symbol_col: str = "symbol",
    pval_thresh: float = 0.05,
    log2fc_thresh: float = 0.75,
    to_label: Union[int, List[str]] = 5,
    s_curve: Optional[bool] = False,
    colors: List[str] = ["gray", "#D62728", "#1F77B4"],
    varm_key: Optional[str] = None,
    color_dict: Optional[Dict[str, List[str]]] = None,
    shape_dict: Optional[Dict[str, List[str]]] = None,
    size_col: Optional[str] = None,
    fontsize: int = 10,
    top_right_frame: bool = False,
    figsize: Tuple[int, int] = (5, 5),
    legend_pos: Tuple[float, float] = (1.6, 1),
    point_sizes: Tuple[int, int] = (15, 150),
    save: Optional[Union[bool, str]] = None,
    shapes: Optional[List[str]] = None,
    shape_order: Optional[List[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    **kwargs: int,
) -> None:
    """
    Create a volcano plot from a pandas dataframe

    Parameters:
    - data: pandas.DataFrame or Anndata
    - log2fc_col: string, default: 'log2FoldChange'
        Column name of log2 Fold-Change values
    - pvalue_col: string, default: 'padj'
        Column name of the p values to be converted to -log10 P values
    - symbol_col: string, default: 'symbol'
        Column name of gene IDs to use
    - size_col: string, default: None
        Numeric column to size points by.
    - point_sizes: tuple, default: (15, 150)
        Lower and upper bounds of point sizes. If size_col is not None.
    - pval_thresh: numeric, default: 0.05
        Threshold pvalue for points to be significant. Also controls horizontal line.
    - log2fc_thresh: numeric, default: 0.75
        Threshold for the absolute value of the log2 fold change to be considered significant. Also controls vertical lines
    - to_label: int or list, default: 5
        If an int is passed, that number of top down and up genes will be labeled.
        If a list of gene Ids is passed, only those will be labeled.
    - color_dict: dictionary, default: None
        Dictionary to color dots by. Up to 11 categories with default colors.
        Pass list of genes and the category to group them by. {category : ['gene1', gene2]}
        Default colors are: ['dimgrey', 'lightgrey', 'tab:blue', 'tab:orange',
        'tab:green', 'tab:red', 'tab:purple','tab:brown', 'tab:pink',
        'tab:olive', 'tab:cyan']
    - shape_dict: dictionary, default: None
        Dictionary to shape dots by. Up to 6 categories. Pass list of genes as values
        and category as key. {category : ['gene1', gene2], category2 : ['gene3']}
    - fontsize: int, default: 10
        Size of labels
    - colors: list, default: ['gray', '#D62728', '#1F77B4']
        Colors for [non-DE, up, down] genes. Only applies if color_dict is None.
    - top_right_frame: Boolean, default: False
        Show the top and right frame. True/False
    - figsize: tuple, default: (5, 5)
        Size of figure. (x, y)
    - save: boolean | string, default: None
        If true saves default file name. Pass string as path to output file. Will
        add a .svg/.png to string. Saves as both png and svg.
    - shapes: list, default: None
        Pass matplotlib marker ids to change default shapes/order
        Default shapes order is: ['o', '^', 's', 'X', '*', 'd']
    - shape_order: list, default: None
        If you want to change the order of categories for your shapes. Pass
        a list of your categories.
    - x_label: string, default: None
        Label for x axis. Default: column name of log2 Fold-Change"
    - y_label: string, default: None
        Label for y axis. Default: column name of -log10 P value"
    - **kwargs:
        Additional keyword arguments to pass to seaborn.scatterplot
    """

    # add type annotations
    def pval_reciprocal(lfc: float) -> float:
        """
        Function for relating -log10(pvalue) and logfoldchange in a reciprocal.
        Used for plotting the S-curve
        """
        return pval_thresh / (lfc - log2fc_thresh)

    def map_shape(symbol: str) -> str:
        if shape_dict is not None:
            for k in shape_dict.keys():
                if shape_dict[k] is not None and symbol in shape_dict[k]:
                    return k
        return "other"

    # TODO join the two mapping functions
    def map_genes_categories(
        row: pd.Series,
        log2fc_col: str,
        nlog10_col: str,
        log2fc_thresh: float,
        pval_thresh: float = None,
        s_curve: bool = False,
    ) -> str:
        """
        Map genes to categories based on log2fc and pvalue.
        These categories are used for coloring the dots.
        Used when no color_dict is passed, sets up/down/nonsignificant.
        """
        log2fc = row[log2fc_col]
        nlog10 = row[nlog10_col]

        if s_curve:
            # S-curve condition for Up or Down categorization
            reciprocal_thresh = pval_reciprocal(abs(log2fc))
            if log2fc > log2fc_thresh and nlog10 > reciprocal_thresh:
                return "Up"
            elif log2fc < -log2fc_thresh and nlog10 > reciprocal_thresh:
                return "Down"
            else:
                return "not DE"
        else:
            # Standard condition for Up or Down categorization
            if log2fc > log2fc_thresh and nlog10 > pval_thresh:
                return "Up"
            elif log2fc < -log2fc_thresh and nlog10 > pval_thresh:
                return "Down"
            else:
                return "not DE"

    def map_genes_categories_highlight(
        row: pd.Series,
        log2fc_col: str,
        nlog10_col: str,
        log2fc_thresh: float,
        pval_thresh: float = None,
        s_curve: bool = False,
        symbol_col: str = None,
    ) -> str:
        """
        Map genes to categories based on log2fc and pvalue.
        These categories are used for coloring the dots.
        Used when color_dict is passed, sets DE / not DE for background and user supplied highlight genes.
        """
        log2fc = row[log2fc_col]
        nlog10 = row[nlog10_col]
        symbol = row[symbol_col]

        if color_dict is not None:
            for k in color_dict.keys():
                if symbol in color_dict[k]:
                    return k

        if s_curve:
            # Use S-curve condition for filtering DE
            if nlog10 > pval_reciprocal(abs(log2fc)) and abs(log2fc) > log2fc_thresh:
                return "DE"
            return "not DE"
        else:
            # Use standard condition for filtering DE
            if abs(log2fc) < log2fc_thresh or nlog10 < pval_thresh:
                return "not DE"
            return "DE"

    if isinstance(data, ad.AnnData):
        if varm_key is None:
            raise ValueError("Please pass a .varm key to use for plotting")

        raise NotImplementedError("Anndata not implemented yet")
        df = data.varm[varm_key].copy()

    df = data.copy(deep=True)

    # clean and replace 0s as they would lead to -inf
    if df[[log2fc_col, pvalue_col]].isnull().values.any():
        print("NaNs encountered, dropping rows with NaNs")
        df = df.dropna(subset=[log2fc_col, pvalue_col])

    if df[pvalue_col].min() == 0:
        print("0s encountered for p value, replacing with 1e-323")
        df.loc[df[pvalue_col] == 0, pvalue_col] = 1e-323

    # max for y-axis
    max_log2fc = df[log2fc_col].max()

    # convert p value threshold to nlog10
    pval_thresh = -np.log10(pval_thresh)
    # make nlog10 column
    df["nlog10"] = -np.log10(df[pvalue_col])
    # make a column to pick top genes
    df["top_genes"] = df["nlog10"] * df[log2fc_col]

    # Label everything with assigned color / shape
    if shape_dict or color_dict:
        combined_labels = []
        if isinstance(shape_dict, dict):
            combined_labels.extend(
                [item for sublist in shape_dict.values() for item in sublist]
            )
        if isinstance(color_dict, dict):
            combined_labels.extend(
                [item for sublist in color_dict.values() for item in sublist]
            )
        label_df = df[df[symbol_col].isin(combined_labels)]

    # Label top n_gens
    elif isinstance(to_label, int):
        label_df = pd.concat(
            (
                df.sort_values("top_genes")[-to_label:],
                df.sort_values("top_genes")[0:to_label],
            )
        )

    # assume that a list of genes was passed to label
    else:
        label_df = df[df[symbol_col].isin(to_label)]

    # By default mode colors by up/down if no dict is passed

    if color_dict is None:
        df["color"] = df.apply(
            lambda row: map_genes_categories(
                row,
                log2fc_col=log2fc_col,
                nlog10_col="nlog10",
                log2fc_thresh=log2fc_thresh,
                pval_thresh=pval_thresh,
                s_curve=s_curve,
            ),
            axis=1,
        )

        # order of colors
        hues = ["not DE", "Up", "Down"][: len(df.color.unique())]

    else:
        df["color"] = df.apply(
            lambda row: map_genes_categories_highlight(
                row,
                log2fc_col=log2fc_col,
                nlog10_col="nlog10",
                log2fc_thresh=log2fc_thresh,
                pval_thresh=pval_thresh,
                symbol_col=symbol_col,
                s_curve=s_curve,
            ),
            axis=1,
        )

        user_added_cats = [x for x in df.color.unique() if x not in ["DE", "not DE"]]
        hues = ["DE", "not DE"] + user_added_cats

        # order of colors
        hues = hues[: len(df.color.unique())]
        colors = [
            "dimgrey",
            "lightgrey",
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]

    # coloring if dictionary passed, subtle background + highlight

    # map shapes if dictionary exists

    if shape_dict is not None:
        df["shape"] = df[symbol_col].map(map_shape)
        user_added_cats = [x for x in df["shape"].unique() if x != "other"]
        shape_order = ["other"] + user_added_cats
        if shapes is None:
            shapes = ["o", "^", "s", "X", "*", "d"]
        shapes = shapes[: len(df["shape"].unique())]
        shape_col = "shape"
    else:
        shape_col = None

    # build palette
    colors = colors[: len(df.color.unique())]

    # We want plot highlighted genes on top + at bigger size, split dataframe
    df_highlight = None
    if shape_dict or color_dict:
        label_genes = label_df[symbol_col].unique()
        df_highlight = df[df[symbol_col].isin(label_genes)]
        df = df[~df[symbol_col].isin(label_genes)]

    plt.figure(figsize=figsize)
    # Plot non-highlighted genes
    ax = sns.scatterplot(
        data=df,
        x=log2fc_col,
        y="nlog10",
        hue="color",
        hue_order=hues,
        palette=colors,
        size=size_col,
        sizes=point_sizes,
        style=shape_col,
        style_order=shape_order,
        markers=shapes,
        **kwargs,
    )
    # Plot highlighted genes
    if df_highlight is not None:
        ax = sns.scatterplot(
            data=df_highlight,
            x=log2fc_col,
            y="nlog10",
            hue="color",
            hue_order=hues,
            palette=colors,
            size=size_col,
            sizes=point_sizes,
            style=shape_col,
            style_order=shape_order,
            markers=shapes,
            legend=False,
            edgecolor="black",
            linewidth=1,
            **kwargs,
        )

    # plot vertical and horizontal lines

    if s_curve:
        # log2fc_thresh
        x = np.arange((log2fc_thresh + 0.000001), max_log2fc + 1, 0.01)
        y = pval_reciprocal(x)
        ax.plot(x, y, zorder=1, c="k", lw=2, ls="--")
        ax.plot(-x, y, zorder=1, c="k", lw=2, ls="--")

    else:
        ax.axhline(pval_thresh, zorder=1, c="k", lw=2, ls="--")
        ax.axvline(log2fc_thresh, zorder=1, c="k", lw=2, ls="--")
        ax.axvline(log2fc_thresh * -1, zorder=1, c="k", lw=2, ls="--")
    plt.ylim(0, max_log2fc + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # make labels
    texts = []
    for i in range(len(label_df)):
        txt = plt.text(
            x=label_df.iloc[i][log2fc_col],
            y=label_df.iloc[i].nlog10,
            s=label_df.iloc[i][symbol_col],
            fontsize=fontsize,
        )

        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
        texts.append(txt)

    adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5))

    # make things pretty
    for axis in ["bottom", "left", "top", "right"]:
        ax.spines[axis].set_linewidth(2)

    if not top_right_frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.tick_params(width=2)
    plt.xticks(size=11)
    plt.yticks(size=11)

    # Set default axis titles
    if x_label is None:
        x_label = log2fc_col
    if y_label is None:
        y_label = f"-$log_{{10}}$ {pvalue_col}"

    plt.xlabel(x_label, size=15)
    plt.ylabel(y_label, size=15)

    plt.legend(loc=1, bbox_to_anchor=legend_pos, frameon=False)

    # TODO replace with scanpy save style
    if save:
        files = os.listdir()
        for x in range(100):
            file_pref = "volcano_" + "%02d" % (x,)
            if len([x for x in files if x.startswith(file_pref)]) == 0:
                plt.savefig(file_pref + ".png", dpi=300, bbox_inches="tight")
                plt.savefig(file_pref + ".svg", bbox_inches="tight")
                break
    elif isinstance(save, str):
        plt.savefig(save + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save + ".svg", bbox_inches="tight")

    plt.show()
