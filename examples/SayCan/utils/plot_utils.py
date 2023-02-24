'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

import matplotlib.pyplot as plt
import numpy as np
from heapq import nlargest
from .normalize_scores import normalize_scores

def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
    if show_top:
        top_options = nlargest(show_top, combined_scores, key=combined_scores.get)
        # add a few top llm options in if not already shown
        top_llm_options = nlargest(show_top // 2, llm_scores, key=llm_scores.get)
        for llm_option in top_llm_options:
            if not llm_option in top_options:
                top_options.append(llm_option)
        llm_scores = {option: llm_scores[option] for option in top_options}
        vfs = {option: vfs[option] for option in top_options}
        combined_scores = {option: combined_scores[option] for option in top_options}

    sorted_keys = dict(sorted(combined_scores.items()))
    keys = [key for key in sorted_keys]
    positions = np.arange(len(combined_scores.items()))
    width = 0.3

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    plot_llm_scores = normalize_scores({key: llm_scores[key] for key in sorted_keys})
    plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
    plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
    plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])

    ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")

    score_colors = ["#ea9999ff" for score in plot_affordance_scores]
    ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
    ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
    ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)

    plt.xticks(rotation="vertical")
    ax1.set_ylim(0.0, 1.0)

    ax1.grid(True, which="both")
    ax1.axis("on")

    ax1_llm = ax1.twinx()
    ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
    ax1_llm.set_ylim(0.01, 1.0)
    plt.yscale("log")

    font = {"fontname": "Arial", "size": "16", "color": "k" if correct else "r"}
    plt.title(task, **font)
    key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")", "") for
                   key in keys]
    plt.xticks(positions, key_strings, **font)
    ax1.legend()
    plt.show()