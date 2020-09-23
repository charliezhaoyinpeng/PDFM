import matplotlib.pyplot as plt
import numpy as np


def plot_line(x, y, e, info):
    x = np.array(x)
    y = np.array(y)
    e = np.array(e)
    plt.errorbar(x, y, e, linestyle=info[0], marker=info[3], color=info[1], label=info[2])


def plot(xs, ys, es, dataname, maml):
    info_sum = [['--', 'm', 'pretrain', '>'],
                ['--', 'r', 'fair-MAML', 'o'],
                ['--', 'y', 'FMAML_dp', 'd'],
                ['--', 'g', 'FMAML_eop', 's'],
                ['--', 'c', 'LAFTR', 'x'],
                ['-', 'b', 'Ours', '*']]
    for i in range(len(info_sum)):
        plot_line(xs[i], ys[i], es[i], info_sum[i])
    plt.scatter(maml[0], maml[1], label="MAML", color='k')
    plt.legend(loc='best', fontsize=10)
    # plt.gca().set_ylim(ylim)
    # plt.xticks(x)
    plt.xlabel("DBC", fontsize=11, fontweight='bold')
    plt.ylabel("Validation Loss", fontsize=15, fontweight='bold')
    plt.title("%s " % (dataname))


if __name__ == "__main__":
    dataname = 'Bank Marketing'

    maml = []

    xs = [[],  # pretrain
          [],  # fair-MAML
          [],  # FMAML_dp
          [],  # FMAML_eop
          [],  # LAFTR
          []]  # ours

    ys = [[],  # pretrain
          [],  # fair-MAML
          [],  # FMAML_dp
          [],  # FMAML_eop
          [],  # LAFTR
          []]  # ours

    es = [[],  # pretrain
          [],  # fair-MAML
          [],  # FMAML_dp
          [],  # FMAML_eop
          [],  # LAFTR
          []]  # ours

    # =======================================================================================================#

    plot(xs, ys, es, dataname, maml)

    plt.savefig("../../out_figures/trade-off-%s.png" % (dataname), dpi=300, bbox_inches='tight')
    plt.show()
