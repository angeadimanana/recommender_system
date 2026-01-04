import matplotlib.pyplot as plt

def plot(costs_train, costs_test, rmse_train, rmse_test):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Costs Train + Tests 
    ax1 = axes[0]
    ax1.plot(costs_train, label='costs_train', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax1.plot(costs_test, label='costs_test', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Costs', fontsize=12, fontweight='bold')

    ax1.grid(True, alpha=0.3)
   
    legend1 = ax1.legend(fontsize=6, frameon=False)
    for t in legend1.get_texts():
        t.set_fontweight('bold')


    # Train and Test RMSE
    ax2 = axes[1]
    ax2.plot(rmse_train, label='rmse_train', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax2.plot(rmse_test, label='rmse_test', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')

    ax2.grid(True, alpha=0.3)
    legend2 = ax2.legend(fontsize=6, frameon=False)
    for t in legend2.get_texts():
        t.set_fontweight('bold')

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=10, direction='in')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    #plt.savefig('model_comparison_cost.pdf', dpi=300, bbox_inches='tight')
    #plt.savefig("model_comparison_rmse.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)

    #print("Saved: model_comparison_rmse.png\n")
    plt.show()
