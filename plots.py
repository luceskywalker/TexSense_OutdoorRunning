import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import get_subject_data
import numpy as np
import pandas as pd
idx = pd.IndexSlice

colors = sns.color_palette(palette="flare", n_colors=6)
linestyles = ["-", ":", "-", ":","-",":" ]
sns.set(style='whitegrid', context='talk')
all_subjects = ['P012', 'P018', 'P024', 'P051', 'P196', 'P213', 'P222', 'P249', 'P424', 'P480', 'P519', 'P561', 'P589', 
                'P608', 'P646', 'P703', 'P742', 'P803', 'P805', 'P807', 'P869', 'P922', 'P935', 'P969', 'P973', 'P981']

axes_dict = {
    "ankle": {
        "frontal": "Inversion (+) / Eversion (-)",
        "transverse": "internal (+) / external Rotation (-)",
        "sagittal": "Dorsiflex (+) / Plantarflex (-)",
    },
    "knee": {
        "frontal": "Adduction (+) / Abduction (-)",
        "transverse": "internal (+) / external Rotation (-)",
        "sagittal": "Extension (+) / Flexion (-)",
    },
    "hip": {
        "frontal": "Adduction (+) / Abduction (-)",
        "transverse": "internal (+) / external Rotation (-)",
        "sagittal": "Flexion (+) / Extension (-)",
    },
    "force": {
        "vertical": "vertical (+)",
        "antero-posterior": "anterior (+) / posterior (-)",
        "medio-lateral": "lateral (+) / medial (-)",
    }
}

def mean_plot(spm_dict):
    
    for joint in ["force", "ankle", "knee", "hip"]:
    
        if joint != "force":
            dims = ["sagittal", "frontal", "transverse"]
            unit = "[Nm/kg]"
            title = f"Speed Influence on {joint.capitalize()} Moments (RM-ANOVA)"
        else:
            dims = ["vertical", "antero-posterior", "medio-lateral"]
            unit = "[N/kg]"
            title = "Speed Influence on Ground Reaction Forces (RM-ANOVA)"
        fig, ax = plt.subplots(nrows=17, ncols=3, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [5] + [.4] * 16})
        fig.suptitle(title, weight="bold", fontsize=22, y=.98)
        
        for dimi, dim in enumerate(dims):
            spm = spm_dict[tuple([joint, dim])]

            # plot zero-line
            ax[0,dimi].hlines(0,0,201, ls="--", color=".3")

            # plot means
            for speedi, (speed, mean_ts) in enumerate(spm["means"].items()):
                ax[0,dimi].plot(np.linspace(0,100,len(mean_ts)), mean_ts, 
                                linestyle=linestyles[speedi],
                                color=colors[speedi],
                                linewidth=3,
                                label=f"{speed} km/h"
                            )
            # Highlight significant ANOVA regions
            for cluster in spm["anova"]["regions"]:
                ax[0,dimi].axvspan(cluster.endpoints[0], cluster.endpoints[1], color='.6', alpha=.3)
            # set labels
            ax[0,dimi].set_title(axes_dict[joint][dim])
            ax[0,0].set_ylabel(unit)

            # Post-hoc Comparison
            ax[1,dimi].axis("off")  # This is an empty subplot to create padding
            ax[1,dimi].text(2, .5, "Post-Hoc Comparisons", ha='left', va='center', fontsize=16, fontweight='bold')

            # plot pairwise comparisons significant regions
            for i, (s1, s2) in enumerate(spm["posthoc"].keys(), start=2):
                for cluster in spm["posthoc"][(s1, s2)]["regions"]:
                    ax[i,dimi].axvspan(cluster.endpoints[0], cluster.endpoints[1], color='.3')

                # set labels
                ax[i,dimi].set_yticks([])  # Hide y-axis labels
                ax[i,0].set_ylabel(f"{s1} vs. {s2}", rotation="horizontal", ha="right", va="center", fontsize=14)
                ax[i,dimi].spines['top'].set_visible(False)
                ax[i,dimi].spines['right'].set_visible(False)
                ax[i,dimi].spines['left'].set_visible(False)
            ax[-1,dimi].set_xlabel("normalized Stance [%]")
        ax[-1,0].set_xlim(0,100)
        ax[0,1].legend(bbox_to_anchor=[.5,1.28], ncols=6, frameon=False, fontsize=18, loc="upper center")
        plt.show(block=False)
    return


def spm_value_plot(spm_dict):
    for joint in ["force", "ankle", "knee", "hip"]:
        if joint != "force":
            dims = ["sagittal", "frontal", "transverse"]
            unit = "[Nm/kg]"
            title = f"SPM Results for {joint.capitalize()} Moments"
        else:
            dims = ["vertical", "antero-posterior", "medio-lateral"]
            unit = "[N/kg]"
            title = "SPM Results for Ground Reaction Forces"

        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(16, 16), sharex=True)
        fig.suptitle(title, weight="bold", y=.93)

        ylim_dict = {}
        for dimi, dim in enumerate(dims):
            ylim_other_rows_min, ylim_other_rows_max = float('inf'), float('-inf')
            spm = spm_dict[tuple([joint, dim])]

            # set title
            ax[0,dimi].set_title(axes_dict[joint][dim])

            # plot anova F-values
            fi = spm["anova"]["Fi"]
            fi.plot(ax=ax[0,dimi], lw=2)
            ax[0,dimi].tick_params(axis='y', which='both', labelleft=True)
            ax[0,dimi].spines['top'].set_visible(False)
            ax[0,dimi].spines['right'].set_visible(False)

            # speed pairs to plot (not all - this would be too big)
            pairs = [(8, 9), (9,10), (10,11), (11,12), (12, 13)]
            for i, (s1, s2) in enumerate(pairs, start=1):
                ti = spm_dict[tuple([joint, dim])]["posthoc"][(s1, s2)]["ti"]
                ti.plot(ax=ax[i,dimi], lw=2)

                # labels and aesthetics
                ax[i,dimi].tick_params(axis='y', which='both', labelleft=True)
                ax[i,dimi].spines['top'].set_visible(False)
                ax[i,dimi].spines['right'].set_visible(False)

                # adjust ylims for all subplots
                ylim_other_rows_min = min(ylim_other_rows_min, ax[i,dimi].get_ylim()[0])
                ylim_other_rows_max = max(ylim_other_rows_max, ax[i,dimi].get_ylim()[1])
            ylim_dict[dim] = (ylim_other_rows_min, ylim_other_rows_max)

        # Set x-labels
        for o in range(3):
            ax[-1,o].set_xlabel("normalized Stance [%]")
            ax[-1,o].set_xlim(0,100)
        
        # Set the shared y-limits for the remaining rows
        for col, dim in enumerate(dims):
            ylim_other_rows_min, ylim_other_rows_max = ylim_dict[dim]
            for row in range(1, ax.shape[0]):
                ax[row, col].set_ylim(ylim_other_rows_min, ylim_other_rows_max)
            # annotate y-labels
            if col == 0:
                ax[0, col].set_ylabel("SPM {F}", fontsize=16)
                for i, (s1, s2) in enumerate(pairs, start=1):
                    ax[i, col].set_ylabel("SPM {t}", fontsize=16)
            else:
                ax[0, col].set_ylabel("")
                for i, (s1, s2) in enumerate(pairs, start=1):
                    ax[i, col].set_ylabel("")
        # Add second y-label for the first column
        ax[0, 0].annotate("RM-ANOVA", xy=(0, 1), xytext=(-0.5, 1), textcoords='axes fraction',
                        rotation="horizontal", ha='left', va='center', weight='bold',)
        ax[1, 0].annotate("Post-Hoc", xy=(0, 1), xytext=(-0.5, 1), textcoords='axes fraction',
                        rotation="horizontal", ha='left', va='center', weight='bold')

        # annotate post-hoc comparisons
        for i, (s1, s2) in enumerate(pairs, start=1):
            ax[i, 0].annotate(f"{s1} vs. {s2}", xy=(0, 0.5), xytext=(-0.32, 0.5), textcoords='axes fraction',
                            rotation="horizontal", ha="right", va="center", fontsize=16)    
        plt.show(block=False)

    return

def plot_subject_data(data, std=False, others=False):
    """
    Plot subject data for a given subject.
    If more than one subject is given, the only the mean is plotted for each subject and speed.
    Higher speeds are darker. Every second speed is a dotted line.
    Individual subjects get a unique base color from colorblind palette in seaborn.
    If only one subject is given, the mean and std are plotted, except std is set to False.
    Other Participants mean +- std (of 10 km/h) can be plotted as reference if only 1 subjcet is specified (sets std to False).

    args:
    data: pd.DataFrame containing the subject data
    std: bool, default False. If True, the standard deviation is plotted as a shaded region around the mean (only for single subjects).
    others: bool, default False. If True, the standard deviation of other subjects plotted as a shaded region around the mean (only for single subjects).
    """
    sns.set(style='whitegrid', context='talk', font_scale=.8)
    # get subject and speed levels
    subjects = data.columns.get_level_values("subject").unique().to_list()
    subjects_str = str(subjects).replace("'", "")
    
    # set others & std to False if more than 1 subject 
    if len(subjects)>1:
        std = False
        others = False

    # get data of others
    if others:
        other_subs = [x for x in all_subjects if not x in subjects]
        df_others = get_subject_data(other_subs)

        # set std to False
        std = False

    # get speeds (8-13 km/h)
    speeds = sorted(data.columns.get_level_values("speed").unique())

    # select base colors
    base_colors = sns.color_palette("colorblind", len(subjects))

    # plot data
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(16, 18), sharex=True)
    fig.suptitle(f"Outdoor Running Kinetics {subjects_str}", weight="bold", fontsize=22, y=.936)

    # loop throuht subjects
    for si, subject in enumerate(subjects):
        # get colors and linestyles
        c_base = base_colors[si]
        colors = sns.dark_palette(c_base, n_colors=len(speeds), reverse=True)

        # loop through joints
        joints = ["hip", "knee", "ankle", "force"]
        for row, joint in enumerate(joints):
            # set labels correctly and choose right dims
            if joint != "force":
                dims = ["sagittal", "frontal", "transverse"]
                unit = "[Nm/kg]"
                joint_label = f"{joint.upper()}\nMoments"
            else:
                dims = ["vertical", "antero-posterior", "medio-lateral"]
                unit = "[N/kg]"
                joint_label = "GRFs"
            # loop through dims
            for col, dim in enumerate(dims):
                # plot zero line
                if si == 0:
                    ax[row, col].hlines(0, 101, 0, color=".3", ls="--")
                    # plot others (if desired)
                    if others:
                        mothers = df_others.loc[:,idx[:,10,:,joint,dim]].mean(axis=1)
                        stdothers = df_others.loc[:,idx[:,10,:,joint,dim]].std(axis=1)
                        ax[row, col].fill_between(range(101), mothers+stdothers, mothers-stdothers, color="orangered", alpha=.5, zorder=1)
                # loop through speeds
                for speedi, speed in enumerate(speeds):
                    # get mean
                    if len(subjects) > 1:
                        m = data.loc[:, idx[subject, speed, :, joint, dim]]
                    else:
                        m = data.loc[:,idx[subject, speed, :, :, :, joint, dim]].mean(axis=1)
                        # get std if std is True
                        if std:
                            s = data.loc[:, idx[subject, speed, :, :, :, joint, dim]].std(axis=1)
                            # plot std
                            ax[row, col].fill_between(np.linspace(0,100,len(m)), m-s, m+s, color=colors[speedi], alpha=.2)
                    
                    # plot mean
                    m.plot(ax=ax[row, col], color=colors[speedi], ls=linestyles[speedi], linewidth=3, legend=False)
                
                # aesthetics
                ax[row, col].set_title(axes_dict[joint][dim])
                # unit
                ax[row, 0].set_ylabel(unit)
                # joint annotation
                ax[row, 0].annotate(joint_label, xy=(0, 0), xytext=(-0.3, 1), textcoords='axes fraction',
                            rotation="horizontal", ha="center", va="top") 
                # xlabel
                ax[-1, col].set_xlabel("normalized Stance [%]")
                ax[-1, col].set_xlim([0,100])

    # legend for speeds
    legend_colors = sns.dark_palette(base_colors[0], n_colors=len(speeds), reverse=True)
    handles = [Line2D([0], [0], color=legend_colors[i], lw=5, linestyle=linestyles[i], label=f'{speed} km/h') for i, speed in enumerate(speeds)]
    ax[0,1].legend(handles = handles, bbox_to_anchor=[.5,1.25], ncols=6, frameon=False, loc="upper center", fontsize=14)

    # in case of multiple subjects: legend for subjects
    if len(subjects) > 1:
        handles_subj = [Line2D([0], [0], color=base_color, lw=5, label=sub) for base_color, sub in zip(base_colors, subjects)]  
        fig.legend(handles=handles_subj,
           bbox_transform=ax[0,2].transAxes, bbox_to_anchor=[1, 1.43], title="Participants")
    # in case of others == True: legend for subjects
    elif others:
        handles_subj = [Line2D([0], [0], color=base_colors[0], lw=5, label=subject)]
        handles_subj.append(Line2D([0], [0], color='orangered', alpha=.5, lw=10, label="others*"))
        fig.legend(handles=handles_subj,
           bbox_transform=ax[0,2].transAxes, bbox_to_anchor=[1, 1.43], title="Participants")
    
    return fig