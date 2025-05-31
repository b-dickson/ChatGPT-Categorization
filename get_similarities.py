import argparse
import base64
import csv
import json
import os
from itertools import combinations
from datetime import datetime

import numpy as np
import openai
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

IMAGE_FOLDER = "data\\360 Rocks"
API_KEY = ""
HUMAN_DATA = "data\\rocks_360_similarities.csv"


NUMBER_MAP = "number_map.json"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RAW_RESPONSES = os.path.join(OUTPUT_FOLDER, "chatgpt_raw_responses_360.jsonl")
SIMILARITY_OUTPUT = os.path.join(OUTPUT_FOLDER, "chatgpt_similarity_360.csv")

BASE_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a scale from 1 to 9, with 1 being most dissimilar, "
    "5 being moderately similar, and 9 being most similar. "
    "You only respond with a single number from 1 to 9, without explaining your reasoning. "
)

SCALE_PROMPT = BASE_PROMPT + (
    "You use the full range of the scale. "
    "You only respond with 1 or 2 when the pair is extremely dissimilar. "
    "You only respond with 8 or 9 when the pair is extremely similar. "
)

SIMILAR_ROCK1 = "I_Obsidian_09"
SIMILAR_ROCK2 = "M_Anthracite_12"

DISSIMILAR_ROCK1 = "S_Rock Salt_12"
DISSIMILAR_ROCK2 = "S_Sandstone_12"

MEDIUM_ROCK1 = "I_Andesite_06"
MEDIUM_ROCK2 = "I_Peridotite_11"

QUESTION_PROMPT = "From 1-9, how visually similar are these two rocks?"


def encode_image(image, folder=IMAGE_FOLDER):
    image_path = os.path.join(folder, f"{image}.jpg")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_messages(rocks, folder=IMAGE_FOLDER):
    messages = {
        "role": "user",
        "content": [{"type": "text", "text": QUESTION_PROMPT}]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(rock, folder)}"
                },
            }
            for rock in rocks
        ],
    }
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def get_responses(rock1, rock2, scale, anchors):
    messages = [
        {"role": "system", "content": SCALE_PROMPT if scale else BASE_PROMPT},
    ]
    if anchors:
        messages += [
            create_messages((SIMILAR_ROCK1, SIMILAR_ROCK2)),
            {"role": "assistant", "content": "9"},
            create_messages((DISSIMILAR_ROCK1, DISSIMILAR_ROCK2)),
            {"role": "assistant", "content": "1"},
            create_messages((MEDIUM_ROCK1, MEDIUM_ROCK2)),
            {"role": "assistant", "content": "5"},
        ]
    messages.append(create_messages((rock1, rock2)))
    response = completion_with_backoff(
        model="gpt-4o",
        messages=messages,
        logprobs=True,
        top_logprobs=20,
        seed=args.seed,
        max_tokens=1,
    )
    return response


def get_average_rating(logprobs):
    # We get a weighted average using the logprobs. The logprobs are actually nondeterministic
    # even with a fixed random seed, so different runs might have slightly different results.
    ratings = np.array([i + 1 for i in range(9)])
    weights = np.array([0.0] * 9)
    for lp in logprobs:
        token = token = lp.token.strip().lower()
        i = number_map.get(token, token)  # Convert what we can to numerical digits
        try:
            i = int(i) - 1
            weights[i] += np.exp(lp.logprob)
        except:
            continue
    try:
        av = np.average(ratings, weights=weights)
    except:
        av = 0.0
    return av


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get rock similarity ratings.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the pair to start at (default: 0)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of pairs to process (default: 5)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle the pairs before processing (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for shuffling and model (default: 123)",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Use the full scale prompt (default: False)",
    )
    parser.add_argument(
        "--anchors",
        action="store_true",
        default=False,
        help="Include anchor examples in the prompt (default: False)",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(OUTPUT_FOLDER, run_timestamp)
    os.makedirs(run_folder, exist_ok=True)

    metadata = {"timestamp": run_timestamp, "args": vars(args)}
    metadata_path = os.path.join(run_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)

    RAW_RESPONSES = os.path.join(run_folder, "raw_responses.jsonl")
    SIMILARITY_OUTPUT = os.path.join(run_folder, "similarities.csv")

    human_data = pd.read_csv(HUMAN_DATA, index_col=0)

    client = openai.OpenAI(
        api_key=API_KEY,
    )

    rock_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
    )

    human_data.index = rock_filenames
    human_data.columns = rock_filenames

    with open(NUMBER_MAP, "r", encoding="utf-8") as f:
        number_map = json.load(f)

    pairs = [comb for comb in combinations(human_data.columns, 2)]
    if args.shuffle:
        import random

        random.seed(args.seed)
        random.shuffle(pairs)

    end_idx = min(args.start + args.n_trials, len(pairs))
    pairs = pairs[args.start : end_idx]
    for i, pair in enumerate(pairs, start=args.start):
        rock1, rock2 = pair
        print(
            f"Processing pair {i + 1}/{len(pairs)}: {rock1} and {rock2}",
            end="\r",
            flush=True,
        )
        human = human_data[rock1][rock2]
        try:
            response = get_responses(
                rock1, rock2, scale=args.scale, anchors=args.anchors
            )
        except Exception as e:
            print(f"Error processing pair {rock1} and {rock2}: {e}")
            break
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        chatgpt = get_average_rating(logprobs)

        with open(RAW_RESPONSES, mode="a", encoding="utf-8") as rawfile:
            rawfile.write(
                json.dumps(
                    {
                        "index": i,
                        "response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                    }
                )
                + "\n"
            )

        with open(SIMILARITY_OUTPUT, mode="a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "index",
                "Rock1",
                "Rock2",
                "Rock1 Index",
                "Rock2 Index",
                "Human Rating",
                "ChatGPT Rating",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "index": i,
                    "Rock1": rock1,
                    "Rock2": rock2,
                    "Rock1 Index": rock_filenames.index(rock1),
                    "Rock2 Index": rock_filenames.index(rock2),
                    "Human Rating": human,
                    "ChatGPT Rating": chatgpt,
                }
            )

    completion_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(metadata_path, "r", encoding="utf-8") as metafile:
        metadata = json.load(metafile)
    metadata["completion_timestamp"] = completion_timestamp
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)
