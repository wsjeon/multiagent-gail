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
    # "simple": (0, 1),
}

scenarios = scenario_dict.keys()
max_episode_lens = [25, 50]
seeds = list(range(10))
# seeds = [3]

fig_id = 0
good_colors = ['g', 'b', 'k']
for scenario, max_episode_len in itertools.product(scenarios, max_episode_lens):
    plt.figure(fig_id)
    plt.xlabel("total_timesteps")
    plt.ylabel("V estim. + A estim.(GAE)")

    legend_before = False

    ys = []
    x = np.arange(2751) * 20000
    x[0] = 1000
    for seed in seeds:
        key = "{}-{}-{}".format(scenario, str(max_episode_len), str(seed))
        y = statistics[key][:, 1:]
        if scenario == 'simple_speaker_listener':
            labels = ['speaker', 'listener']
            colors = ['b', 'g']
        else:
            labels = ['adversary'] * scenario_dict[scenario][0] + ['good'] * scenario_dict[scenario][1]
            colors = ['r'] * scenario_dict[scenario][0] + good_colors[:scenario_dict[scenario][1]]
        if y.shape[0] == 2751:
            for i in range(y.shape[1]):
                plt.plot(x, y[:, i], alpha=0.05, linewidth=0.5, label=labels[i], color=colors[i])
                if not legend_before:
                    plt.legend()
            legend_before = True
            ys.append(y)
        else:
            print(scenario, max_episode_len, seed)
            continue

    meany = np.mean(ys, axis=0)
    for i in range(meany.shape[1]):
        plt.plot(x, meany[:, i], linewidth=1, color=colors[i])
    title = "{}-{}, trial {}".format(scenario, max_episode_len, str(len(ys)))
    if scenario in ["simple_tag", "simple_spread", "simple_speaker_listener", "simple_push"] and max_episode_len == 50:
        title += " (in MAGAIL)"
    else:
        title += " (NOT in MAGAIL)"
    plt.title(title)

    fig_id += 1

plt.show()


