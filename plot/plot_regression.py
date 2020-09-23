import matplotlib.pyplot as plt
import pickle


def get_loss_and_fair_list(path, plot_every, start, end):
    with open(path, 'rb') as f:
        train_losses_list = pickle.load(f)
        train_faires_list = pickle.load(f)
        val_losses_list = pickle.load(f)
        val_faires_list = pickle.load(f)
    val_losses_list = val_losses_list[start:end]
    val_faires_list = val_faires_list[start:end]
    val_losses = []
    val_faires = []
    for i in range(len(val_faires_list)):
        if i % plot_every == 0 and i != 0:
            val_losses.append(val_losses_list[i])
            val_faires.append(val_faires_list[i])

    return [val_losses, val_faires]


if __name__ == "__main__":
    #################################### Parameters #####################################################
    c = 0.05
    dpi = 300

    start = 0
    end = 2300
    data = "syn"
    dataname = "SyntheticData"
    k = 10
    plot_every = 10
    ########################################################################################################

    path_base = r'plotting\\' + data + '\k=' + str(k) + '\\base' + '.txt'
    path_fbase = r'plotting\\' + data + '\k=' + str(k) + '\\fbase' + '.txt'
    path_maml = r'plotting\\' + data + '\k=' + str(k) + '\\maml' + '.txt'
    path_fmaml = r'plotting\\' + data + '\k=' + str(k) + '\\fmaml' + '.txt'

    [val_losses_base, val_faires_base] = get_loss_and_fair_list(path_base, plot_every, start, end)
    [val_losses_fbase, val_faires_fbase] = get_loss_and_fair_list(path_fbase, plot_every, start, end)
    [val_losses_maml, val_faires_maml] = get_loss_and_fair_list(path_maml, plot_every, start, end)
    [val_losses_fmaml, val_faires_fmaml] = get_loss_and_fair_list(path_fmaml, plot_every, start, end)

    print(len(val_losses_base))
    print(len(val_losses_fbase))
    print(len(val_losses_maml))
    print(len(val_losses_fmaml))



    plt.rc('ytick', labelsize=20)
    # plot meta-losses
    fig, ax = plt.subplots()
    plt.plot(val_losses_base, 'k', label='Baseline')
    plt.plot(val_losses_fbase, 'm', label='Fair-Baseline')
    plt.plot(val_losses_maml, 'b', label='MAML')
    plt.plot(val_losses_fmaml, 'r', label='Fair-MAML')
    plt.title("K=%s, %s " % (str(k), dataname))
    plt.gca().axes.get_xaxis().set_ticks([])

    plt.gca().set_ylim([0, 5.8])

    plt.xlabel("Iterations", fontsize=15, fontweight='bold')
    plt.ylabel("Validation Loss", fontsize=15, fontweight='bold')
    legend = ax.legend(loc='upper right', fontsize='large')
    plt.savefig("K=%s_%s_loss.png" % (str(k), dataname), dpi=dpi, bbox_inches='tight')
    plt.show()

    # plot mean difference MD
    fig, ax = plt.subplots()
    plt.plot(val_faires_base, 'k', label='Baseline')
    plt.plot(val_faires_fbase, 'm', label='Fair-Baseline')
    plt.plot(val_faires_maml, 'b', label='MAML')
    plt.plot(val_faires_fmaml, 'r',label='Fair-MAML')
    # plt.axhline(y=c, color='r', linestyle='-.')
    plt.title("K=%s, %s " % (str(k), dataname))
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.xlabel("Iterations", fontsize=15, fontweight='bold')
    plt.ylabel("Validation Mean Difference", fontsize=15, fontweight='bold')
    legend = ax.legend(loc='center right', fontsize='large')
    plt.savefig("K=%s_%s_md.png" % (str(k), dataname), dpi=dpi, bbox_inches='tight')
    plt.show()
