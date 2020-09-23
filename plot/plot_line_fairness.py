import matplotlib.pyplot as plt
import numpy as np


def plot_line(x, y, std, info):
    x = np.array(x)
    y = np.array(y)
    std = np.array(std)
    plt.plot(x, y, color=info[0], label=info[2])
    plt.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])


def cls_plot(ys, es, dataname, eval_name, ylim):
    x = [5, 10, 15, 20]
    info_sum = [['blue', 'royalblue', 'MAML'],
                # ['black', 'darkgray', 'Masked MAML'],
                ['m', 'violet', 'pretrain'],
                ['olive', 'khaki', 'fair-MAML'],
                ['green', 'greenyellow', 'FMAML_dp'],
                ['darkorange', 'sandybrown', 'FMAML_eop'],
                ['red', 'lightcoral', 'Ours'],
                ['darkturquoise', 'lightskyblue', 'LAFTR']]
    for i in range(len(info_sum)):
        plot_line(x, ys[i], es[i], info_sum[i])
    plt.legend(loc='best', fontsize=10)
    # plt.gca().set_ylim(ylim)
    plt.xticks(x)
    plt.xlabel("Few Shots", fontsize=11, fontweight='bold')
    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    plt.title("%s " % (dataname))
    # plt.savefig("../../out_figures/cls-%s-%s.png" % (eval_name, dataname), dpi=300, bbox_inches='tight')
    plt.show()


def reg_plot(ys, es, dataname, eval_name, ylim):
    x = [5, 10, 15, 20]
    info_sum = [['blue', 'royalblue', 'MAML'],
                ['black', 'darkgray', 'Masked MAML'],
                ['m', 'violet', 'pretrain'],
                ['darkorange', 'sandybrown', 'UP-MAML'],
                ['red', 'lightcoral', 'Ours'],
                ['darkturquoise', 'lightskyblue', 'LAFTR']]
    for i in range(len(info_sum)):
        plot_line(x, ys[i], es[i], info_sum[i])
    plt.legend(loc='best', fontsize=10)
    # plt.gca().set_ylim(ylim)
    plt.xticks(x)
    plt.xlabel("Few Shots", fontsize=11, fontweight='bold')
    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    plt.title("%s " % (dataname))
    # plt.savefig("../../out_figures/cls-%s-%s.png" % (eval_name, dataname), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    dataname = 'Chicago Crime'
    eval_name = 'Mean Difference'
    ylim = []
    ys = [[],  # MAML
          [],  # Masked MAML
          [],  # pretrain
          [],  # UP-MAML
          [],  # ours
          []]  # LAFTR

    es = [[],  # MAML
          [],  # Masked MAML
          [],  # pretrain
          [],  # UP-MAML
          [],  # ours
          []]  # LAFTR

    reg_plot(ys, es, dataname, eval_name, ylim)