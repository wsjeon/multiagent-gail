import matplotlib.pyplot as plt
import numpy as np
import itertools

statistics = np.load("../../results/statistics.npz")

scenario_dict = {
    "simple_tag": (3, 1),
    "simple_spread": (0, 3),
    "simple_speaker_listener": (0, 2),
    "simple_push": (1, 1),
    "simple_crypto": (1, 2),
    "simple_adversary": (1, 2),
    "simple": (0, 1),
}

scenarios = scenario_dict.keys()
max_episode_lens = [25, 50]
seeds = list(range(10))

fig_id = 0
colors = ['r', 'g', 'b']
for scenario, max_episode_len in itertools.product(scenarios, max_episode_lens):
    plt.figure(fig_id)
    plt.title("{}-{}".format(scenario, max_episode_len))

    legend_before = False

    for seed in seeds:
        key = "{}-{}-{}".format(scenario, str(max_episode_len), str(seed))
        x = statistics[key][:, 0]
        y = statistics[key][:, 1:]
        labels = ['adversary'] * scenario_dict[scenario][0] + ['good'] * scenario_dict[scenario][1]
        colors = ['r'] * scenario_dict[scenario][0] + ['b'] * scenario_dict[scenario][1]
        if x.shape[0] == 2751:
            for i in range(y.shape[1]):
                plt.plot(x, y[:, i], alpha=0.3, linewidth=0.5, label=labels[i], color=colors[i])
                if not legend_before:
                    plt.legend()
            legend_before = True
        else:
            continue

    fig_id += 1

plt.show()


