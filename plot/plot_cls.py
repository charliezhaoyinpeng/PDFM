import matplotlib.pyplot as plt
import numpy as np


def plot_line(x, y, e, info):
    x = np.array(x)
    y = np.array(y)
    e = np.array(e)
    plt.errorbar(x, y, e, linestyle=info[0], marker=info[3], markersize=8, color=info[1], label=info[2])


def plot(ys, es, dataname, eval_name, ylim):
    x = [5, 10, 15, 20]
    info_sum = [['--', 'k', 'MAML', 'o'],
                ['--', '#ff7029', 'Masked MAML', 'p'],
                ['--', 'm', 'pretrain', '>'],
                ['--', 'r', 'fair-MAML', 'o'],
                ['--', 'y', 'FMAML_dp', 'd'],
                ['--', 'g', 'FMAML_eop', 's'],
                ['--', 'c', 'LAFTR', 'x'],
                ['-', 'b', 'Ours', '*']]
    for i in range(len(info_sum)):
        plot_line(x, ys[i], es[i], info_sum[i])
    plt.legend(loc='best', fontsize=10)
    # plt.gca().set_ylim(ylim)
    plt.xticks(x)
    plt.xlabel("Few Shots", fontsize=11, fontweight='bold')
    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    plt.title("%s " % (dataname))
    plt.savefig("../../out_figures/%s-%s.png" % (eval_name, dataname), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    dataname = 'Bank Marketing'
    eval_name = 'Accuracy'
    ylim = []
    ys = [[],  # MAML
          [],  # Masked MAML
          [],  # pretrain
          [],  # fair-MAML
          [],  # FMAML_dp
          [],  # FMAML_eop
          [],  # LAFTR
          []   # ours
          ]

    es = [[],  # MAML
          [],  # Masked MAML
          [],  # pretrain
          [],  # fair-MAML
          [],  # FMAML_dp
          [],  # FMAML_eop
          [],  # LAFTR
          []   # ours
          ]

    plot(ys, es, dataname, eval_name, ylim)
