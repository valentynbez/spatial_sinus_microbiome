import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

from scipy.stats.mstats import gmean
from scipy.stats import spearmanr
import warnings 
import re
from itertools import combinations
from palettable.colorbrewer.qualitative import Accent_3

from statannotations.Annotator import Annotator


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def rclr(x):  
    dropna_x = [i for i in x if i != 0]
    g = gmean(dropna_x)
    
    # silecing log(0) warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        rclr_row = np.log(x) - np.log(g)
    
    rclr_row[rclr_row < 0] = 0 
    return rclr_row

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = spearmanr(df[col],df[col2], nan_policy="omit")
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def load_gram_stain():
    # Gram lists exported from IMG
    df1 = pd.read_csv("./img_data/gram-positive.txt", sep="\t", index_col = 0)
    df2 = pd.read_csv("./img_data/gram-negative.txt", sep="\t", index_col = 0)
#     df3 = pd.read_csv("./img_data/aerobic.txt", sep="\t", index_col = 0)
#     df4 = pd.read_csv("./img_data/anaerobic.txt", sep="\t", index_col = 0)
#     df5 = pd.read_csv("./img_data/facultative.txt", sep="\t", index_col = 0)
#     df6 = pd.read_csv("./img_data/pathogenic.txt", sep="\t", index_col = 0)
    for df in [df1, df2]:
        df["Species"] = df["Species"].str.split(" ")
        df["Species"] = df["Species"].str.join("_")
    genus_filtered = df1[~df1.Genus.isin(df2.Genus)]
    clean_df1 = genus_filtered[~genus_filtered.Species.isin(df2.Species)]
    genus_filtered = df2[~df2.Genus.isin(df1.Genus)]
    clean_df2 = genus_filtered[~genus_filtered.Species.isin(df1.Species)]
#    df["Sinusitis"] = df["Diseases"].str.contains("[Ss]inusitis", regex=True)
    return clean_df1, clean_df2


def extract_id_pairs(row, *column_pairs):
    dfs = []
    for pair in column_pairs:
        first_second = row[list(pair)]
        first_second["distance"] = "-".join([x for x in pair])
        first_second["distance_type"] = "within"
        first_second["host_subject_id"] = row.name
        dfs.append(first_second)

    return dfs


def get_values_from_distance_matrix(sample_ids, distance_matrix): 
    row, column = sample_ids.to_numpy()
    try:
        val = float(distance_matrix.loc[row, column])
    except KeyError:
        val = np.nan
    
    return val

def extract_bacterial_col_names(df):
    return df.columns[df.columns.str.contains("__")]

def rename_and_extract_columns(df, old_colnames, new_colnames):
    # renames columns to a short name
    # sums the columns with similar names
    new_df = df.rename(columns={k:v for k, v in zip(old_colnames, new_colnames)}).groupby(lambda x:x, axis=1).sum()
    return new_df.loc[:, set(new_colnames)]

def normalize(df, axis):
    return df.div(df.sum(axis=(1 - axis)), axis=axis)

def collapse_low_abundant(df, rel_abd_threshold, collapse_column):
    norm_df = normalize(df, axis=0)
    
    zeroed = norm_df.copy()
    zeroed[zeroed < rel_abd_threshold] = 0
    non_zeros = zeroed.columns[(zeroed != 0).any(axis=0)]
    dct = {k:(k if k in non_zeros else collapse_column) for k in norm_df.columns}
    
    collapsed_df = norm_df.rename(columns=dct).groupby(lambda x:x, axis=1).sum()
    return normalize(collapsed_df, axis=0)

def taxa_barplot(df, order_of_taxa, id_column, ax, xlim, cat_order=None):
    df = df.set_index(id_column).sort_index()[order_of_taxa]
    if df.index.dtype == np.int64:
        df = df.reindex(list(np.arange(df.index.min(), df.index.max()+1)), fill_value=0)
    df.plot(kind="bar", 
            cmap="tab20b",
            stacked=True,
            ax=ax,
            width=1, 
            legend=False)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(xlim);

def draw_annotated_barplot_within_vs_between(df):    
    sns.set_style("whitegrid")
    x = "distance"
    y = "distance_value"
    hue = "distance_type"

    pairs = []
    for cat in df.distance.unique():
        sublist = []
        for color in df.distance_type.unique():
            sublist.append(tuple([cat, color]))
        pairs.append(tuple(sublist))

    g = sns.boxplot(x=x, y=y, hue=hue, data=df, showfliers=False)

    g.set_xlabel("Distance", fontsize=13) 
    g.set_ylabel("Bray-Curtis dissimilarity", fontsize=12) 

    g.set_ylabel("weighted UniFrac", fontsize=12) 
    g.set_ylim((0, 1.6))
    g.set_xticklabels(["Meatus-maxillary sinus", "Meatus-frontal sinus", "Maxillary-frontal sinus"], 
                      fontdict=dict(fontsize=12), rotation=30)
    g.legend(bbox_to_anchor=(1, 1.05))
    annotator = Annotator(g, pairs, data=df, 
                          x=x, y=y, hue=hue)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='outside', comparisons_correction="bonferroni")
    annotator.apply_and_annotate()
    return g

def draw_annotated_alpha_div_boxplot(df):
    pairs = tuple(df.maxillary_ostium_size.dropna().unique())
    pairs = list(combinations(pairs, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True)

    boxplot_df = pd.melt(df, id_vars="maxillary_ostium_size", value_vars="Meatus-maxillary sinus")

    params = dict(order=["narrow", "blocked", "wide"], palette=Accent_3.mpl_colors)

    sns.boxplot(x = "maxillary_ostium_size", y = "value", data = boxplot_df,
                ax=axes[0],
                **params)

    axes[0].set_xlabel("Maxillary ostium size", fontsize=13) 
    axes[0].set_ylabel("Bray-Curtis dissimilarity", fontsize=13)
    axes[0].set_title("Middle nasal meatus - maxillary sinus", pad=10, fontsize=12)
    axes[0].grid(True, axis="y", alpha=0.4)

    annotator = Annotator(axes[0], [pairs[2]], data=boxplot_df, 
                          x="maxillary_ostium_size", y="value",
                          order=params["order"])

    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate();

    boxplot_df = pd.melt(df, id_vars="maxillary_ostium_size", value_vars="Meatus-frontal sinus")

    sns.boxplot(x = "maxillary_ostium_size", y = "value", data = boxplot_df,
                ax=axes[1],
                **params)

    axes[1].set_xlabel("Maxillary ostium size", fontsize=13) 
    axes[1].set_ylabel("", fontsize=13)
    axes[1].set_title("Middle nasal meatus - frontal sinus", pad=10, fontsize=12)
    axes[1].grid(True, axis="y", alpha=0.4)

    annotator = Annotator(axes[1], [pairs[2], pairs[1]], 
                          data=boxplot_df, 
                          order=params["order"],
                          x="maxillary_ostium_size", y="value")

    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()
    return fig

def collect_heatmap_data(metadata_df, dis_matrix):
    heatmap_df = metadata_df[["#SampleID", "host_subject_id", "host_body_site", 
                              "maxillary_ostium_patency", "maxillary_ostium_size",
                              "frontal_ostium_patency"]].copy()

    heatmap_df["#SampleID"] = ['.'.join(x[1:]) for x in heatmap_df["#SampleID"].str.split('.')]

    indexes = []

    for group in heatmap_df.drop(1).groupby("host_subject_id"):
        group_arr = group[1].sort_values(by="host_body_site", ascending=False)
        idxs = np.append(group[0], group_arr["#SampleID"].values)

        indexes.append(idxs)

    heatmap_pairs = pd.DataFrame(indexes, columns=["host_subject_id", "meatus_id", "maxillary_id", "frontal_id"])
    heatmap_pairs["host_subject_id"] = heatmap_pairs["host_subject_id"].astype(int)
    heatmap_pairs = heatmap_pairs.set_index("host_subject_id").sort_index()
    
    heatmap_data = []

    for row in heatmap_pairs.iterrows():
        try:
            meatus_front_dis = dis_matrix[row[1].meatus_id][row[1].frontal_id]
        except KeyError:
            meatus_front_dis = np.nan

        try:
            meatus_max_dis = dis_matrix[row[1].meatus_id][row[1].maxillary_id]
        except KeyError:
            meatus_max_dis = np.nan

        try:
            front_max_dis = dis_matrix[row[1].frontal_id][row[1].maxillary_id]
        except KeyError:
            front_max_dis = np.nan

        heatmap_data.append([int(row[0]), meatus_front_dis, meatus_max_dis, front_max_dis])
        
    return (heatmap_pairs, heatmap_data)

def draw_outliers_heatmap(df):
    fig, g = plt.subplots(figsize=(15,6))


    my_cmap = ListedColormap(sns.color_palette([sns.color_palette("magma", 20)[0], 
                                                sns.color_palette("magma", 20)[-1]]).as_hex())
    c = np.linspace(0, 1, 2)

    cmap = sns.color_palette("binary", 2)

    g = sns.heatmap(df.notnull(), 
                    cmap=my_cmap,
                    cbar_kws=dict(ticks=[0, 1]))

    plt.yticks(rotation=0)
    
    return fig 

def draw_outlier_curve(points):
    fig, ax = plt.subplots(figsize=(15,6))
    plt.plot(points, color=sns.color_palette("magma", 20)[0])
    plt.vlines(9, ymax=150, ymin=0, linestyle="--", color="red")
    plt.ylim(0, 135)
    plt.xlim(0, 135)
    plt.ylabel("# of outliers", fontsize=18)
    plt.xlabel("# of distances exceeding cut-off", fontsize=18)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig

def collect_data(df, outliers_df, metadata_df):
    super_df = pd.DataFrame()

    for distance in df["distance"].unique():
        x = df.sort_values("1st_sample").loc[(df["distance_type"] == "within") & 
                                                    (df["distance"] == distance), 
                                                    "distance_value"].values

        y = df.sort_values("1st_sample").loc[((df["distance_type"] == "between") & 
                                                    (df["distance"] == distance))].groupby("1st_sample")["distance_value"].agg("median").values
        
        outlier = df.sort_values("1st_sample").loc[(df["distance_type"] == "within") & 
                                                          (df["distance"] == distance), 
                                                          ["1st_sample", "2nd_sample"]].isin(outliers_df.index).any(axis=1)

        host = df.sort_values("1st_sample").host_subject_id.loc[(df["distance_type"] == "within") & 
                                                                       (df["distance"] == distance)]
        super_df = pd.concat([super_df, pd.DataFrame([x, y, host, outlier, [distance]*len(x)]).T])

    super_df.columns = ["within", "between", "host", "outlier", "distance"]
    super_df["within"] = super_df.within.astype("float")
    super_df["between"] = super_df.between.astype("float")
    super_df = super_df.set_index("host").join(metadata_df.set_index("host_subject_id")[["frontal_ostium_patency",              
                                                                               "maxillary_ostium_patency"]]).reset_index().drop_duplicates()
    
    return super_df