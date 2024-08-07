import matplotlib.pyplot as plt

LEGEND_FONTSIZE = 14
TICK_SIZE = 16
LABEL_FONTSIZE = 20

LAST_N = 9234
N_FOR_PERF_SCORE_COMPARISON = 9234

IR_MAX_EUCLIDEAN = 46.16
IR_RANDOM_EUCLIDEAN = IR_MAX_EUCLIDEAN / 2
IR_RANDOM_PEARSON_STATISTIC = 0
IR_RANDOM_TARGET_RANK = 30

def save_fig(fbasename):
    plt.gcf().savefig(f'{fbasename}.png', format='png', dpi=300, bbox_inches='tight')
    plt.gcf().savefig(f'{fbasename}.eps', format='eps', dpi=1200, bbox_inches='tight')
    plt.gcf().savefig(f'{fbasename}.svg', format='svg', dpi=1200, bbox_inches='tight')
