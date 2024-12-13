# coding=utf-8
import os
import sys
import time
import logging
from typing import Union
import natsort
import json
import tqdm
import argparse

import requests
import numpy as np

import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
import json

from datasets import load_dataset


def inference(args):

    # video_dir = "datasets/LMM/OpenDataLab___MovieNet/raw/240P/tt1533117"
    # v_niah_needles_path = "datasets/LMM/lmms-lab/v_niah_needles/"

    # v_niah_needles = load_dataset(v_niah_needles_path)
    v_niah_needles = load_dataset(args.needle_dataset)

    all_filepath = []
    for root, dirs, files in os.walk(args.haystack_dir):
        for filename in files:
            if (filename.endswith("png") or filename.endswith("jpeg")
                    or filename.endswith("jpg")):
                filepath = os.path.join(root, filename)
                all_filepath.append(filepath)

    all_filepath = natsort.natsorted(all_filepath)

    print(f"all_filepath {len(all_filepath)}")

    all_accuries = []
    for num_frames in tqdm.tqdm(range(args.min_num_frames,
                                      args.max_num_frames + 1,
                                      args.frame_interval),
    # for num_frames in tqdm.tqdm(range(args.max_num_frames,
    #                                   args.min_num_frames - 1,
    #                                   -args.frame_interval),
                                position=2):
        for depth in np.arange(0, 1 + args.depth_interval,
                               args.depth_interval):

            accuracies = []
            for i, data in enumerate(v_niah_needles["test"]):
                if i == 2 or i == 4:
                    pass
                else:
                    continue

                # prompt = "<|im_start|>user\n"
                prompt = ""
                image_path_list = []

                query_frame_idx = int(depth * num_frames)
                for j in range(num_frames):
                    if j == query_frame_idx:
                        prompt += "<video>"

                        image = data["image"]
                        image_path = os.path.join(args.output_dir, f"{i}.png")
                        image.save(image_path)
                        image_path_list.append(image_path)
                    else:
                        prompt += "<video>"
                        image_path_list.append(all_filepath[j])

                question = data["question"]
                question = question.replace(
                    "Answer with the option's letter from the given choices directly.",
                    "\nAnswer with the option's letter from the given choices directly."
                )
                prompt += "\n" + question
                # prompt += "<|im_end|>\n<|im_start|>assistant\n"

                print("#" * 100)
                print(f"prompt {prompt}")

                url = os.environ.get('LCVLM_URL',
                                     default='http://127.0.0.1:5001/api')

                headers = {
                    'Content-Type': 'application/json',
                    # 'Request-Id': 'remote-test',
                    # 'Authorization': f'Bearer {self.key}'
                }
                payload = {
                    # 'model': self.model,
                    'prompts': [prompt],
                    # 'image_list': image_list,
                    'image_path_list': image_path_list,
                    'tokens_to_generate': 16,
                }
                response = requests.put(url,
                                        headers=headers,
                                        data=json.dumps(payload),
                                        verify=False)

                if response.status_code != 200:
                    print(
                        f"Error {response.status_code}: {response.json()['message']}"
                    )
                else:
                    answer = response.json()['text'][0]
                    print(f"answer {answer}")
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()

                answer = answer.strip()
                answer = answer[:len(data["answer"])]
                if len(answer) == 0:
                    correct = 0
                elif answer.lower() == data["answer"].lower():
                    correct = 1
                else:
                    correct = 0
                accuracies.append(correct)

                print(f"data['answer'] {data['answer']}")
                print(f"correct {correct}")
                print("#" * 100)

            result = {
                "Num. Frame": num_frames,
                "Frame Depth": round(depth * 100, -1),
                "Score": sum(accuracies) / len(accuracies),
            }
            all_accuries.append(result)

            os.makedirs(f"{args.output_dir}", exist_ok=True)
            # save all_accuries as json
            with open(f"{args.output_dir}/all_accuracies.json", "w") as f:
                json.dump(all_accuries, f, indent=4)

    return all_accuries


def plot(args, all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list("custom_cmap",
                                             ["#F0496E", "#EBB839", "#9ad5b3"])

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(index="Frame Depth",
                                    columns="Num. Frame",
                                    values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5,
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )

    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)

    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [
        thousands_formatter(x, None) for x in context_lengths
    ]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))],
               labels=formatted_context_lengths,
               rotation=45,
               fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0,
               fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save

    plt.savefig(f"{args.output_dir}//heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")


def main(args):

    if args.plot_only:
        # load all_accuracies from json
        with open(f"{args.output_dir}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
    else:
        all_accuracies = inference(args)
    plot(args, all_accuracies)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--max_num_frames", type=int, default=300)
    args.add_argument("--needle_dataset",
                      type=str,
                      default="lmms-lab/v_niah_needles")
    args.add_argument("--min_num_frames", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_dir", type=str, default="output/niah")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--haystack_dir",
                      type=str,
                      default="video_needle_haystack/data/haystack_embeddings")
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--plot_only", action="store_true")

    main(args.parse_args())
