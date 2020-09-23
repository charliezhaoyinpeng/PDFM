import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_grouped_bar_chart(eval_name, data_order, input_data, std, xaxis, legend_loc, ylim_float, dataname):
    # set width of bar
    barWidth = 0.1

    # Set position of bar on X axis
    r1 = np.arange(len(input_data[list(input_data.keys())[0]]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]
    # r8 = [x + barWidth for x in r7]
    rs = [r1, r2, r3, r4, r5, r6, r7
         # , r8
          ]

    # Add legend
    color_list = ['k',
                  # '#ff7029',
                  'm', 'r', 'y', 'g', 'c', 'b']
    lable_list = data_order
    legend = []
    for i in range(len(input_data.keys())):
        legend.append(mpatches.Patch(color=color_list[i], label=lable_list[i]))

    # Make the plot
    plt.figure(dpi=300)
    plt.grid(zorder=0)
    for i in range(len(input_data.keys())):
        plt.bar(rs[i], input_data[data_order[i]], yerr=std[data_order[i]], color=color_list[i], width=barWidth, zorder=2)
    plt.xticks([r + 3.5 * barWidth for r in range(len(input_data[list(input_data.keys())[0]]))], xaxis)
    plt.tick_params(labelsize=10)
    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    plt.legend(handles=legend, loc=legend_loc, bbox_to_anchor=legend_pos, facecolor='white', framealpha=0.9, fontsize=9)
    axes = plt.gca()
    axes.xaxis.grid(False)
    ys = []
    for values in input_data.values():
        ys += values
    axes.set_ylim([min(ys) - ylim_float, max(ys) + ylim_float])
    plt.suptitle(dataname)

    # Save and Show graphic
    plt.savefig("../../out_figures/%s-%s.png" % (eval_name, dataname), bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":

    xaxis = ['5-shot', '10-shot', '15-shot', '20-shot']
    data_order = ['MAML',
                  # 'Masked MAML',
                  'pretrain', 'fair-MAML', 'FMAML_dp', 'FMAML_eop', 'LAFTR', 'Ours']

    dataname = "Bank Marketing"
    eval_name = 'DBC'

    input_data = {
        'MAML': [],
        'pretrain': [],
        'fair-MAML': [],
        'FMAML_dp': [],
        'FMAML_eop': [],
        'LAFTR': [],
        'Ours': []
    }

    std = {
        'MAML': [],
        'pretrain': [],
        'fair-MAML': [],
        'FMAML_dp': [],
        'FMAML_eop': [],
        'LAFTR': [],
        'Ours': []
    }

    legend_loc = 2
    legend_pos = (0.7, 1)

    ylim_float = 0.007

    # ===========================================================================================
    # ############################################################################################


    plot_grouped_bar_chart(eval_name, data_order, input_data, std, xaxis, legend_loc, ylim_float, dataname)

