import matplotlib.pyplot as plt


def plot_compare_costs(results_bias, results_embedding, results_features):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Train RMSE
    ax1 = axes[0]
    ax1.plot(results_bias['costs_train'], label='Bias Only', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax1.plot(results_embedding['costs_train'], label='Bias + Embedding', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax1.plot(results_features['costs_train'], label='Bias + Embedding + Features', linewidth=2.5, marker='^', markersize=4, markevery=5)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train costs', fontsize=12, fontweight='bold')

    ax1.grid(True, alpha=0.3)
   
    legend1 = ax1.legend(fontsize=6, frameon=False)
    for t in legend1.get_texts():
        t.set_fontweight('bold')

    # Test RMSE
    ax2 = axes[1]
    ax2.plot(results_bias['costs_test'], label='Bias Only', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax2.plot(results_embedding['costs_test'], label='Bias + Embedding', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax2.plot(results_features['costs_test'], label='Bias + Embedding + Features', linewidth=2.5, marker='^', markersize=4, markevery=5)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test costs', fontsize=12, fontweight='bold')

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

    plt.show()

def plot_compare_rmse(results_bias, results_embedding, results_features):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Train RMSE
    ax1 = axes[0]
    ax1.plot(results_bias['rmse_train'], label='Bias Only', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax1.plot(results_embedding['rmse_train'], label='Bias + Embedding', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax1.plot(results_features['rmse_train'], label='Bias + Embedding + Features', linewidth=2.5, marker='^', markersize=4, markevery=5)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train RMSE', fontsize=12, fontweight='bold')

    ax1.grid(True, alpha=0.3)
   
    legend1 = ax1.legend(fontsize=6, frameon=False)
    for t in legend1.get_texts():
        t.set_fontweight('bold')

    # Test RMSE
    ax2 = axes[1]
    ax2.plot(results_bias['rmse_test'], label='Bias Only', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax2.plot(results_embedding['rmse_test'], label='Bias + Embedding', linewidth=2.5, marker='s', markersize=4, markevery=5)
    ax2.plot(results_features['rmse_test'], label='Bias + Embedding + Features', linewidth=2.5, marker='^', markersize=4, markevery=5)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')

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

    plt.show()
