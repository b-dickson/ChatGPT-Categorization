import argparse
import base64
import csv
import json
import os
from itertools import combinations

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
SYSTEM_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a scale from 1 to 9, with 1 being most dissimilar, "
    "5 being moderately similar, and 9 being most similar. "
    "You use the full range of the scale and only respond with a 1 or 2 when the pair is extremely dissimilar. "
    "You only respond with a single number from 1 to 9, without explaining your reasoning."
)
QUESTION_PROMPT = "From 1-9, how visually similar are these two rocks?"
NUMBER_MAP = "number_map.json"
SEED = 123
RAW_RESPONSES = "chatgpt_raw_responses_360.jsonl"
SIMILARITY_OUTPUT = "chatgpt_similarity_360.csv"


def encode_image(image, folder=IMAGE_FOLDER):
    image_path = os.path.join(folder, f"{image}.jpg")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_messages(prompt, rocks, folder=IMAGE_FOLDER):
    messages = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
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


def get_responses(rock1, rock2):
    response = completion_with_backoff(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            create_messages(QUESTION_PROMPT, (rock1, rock2)),
        ],
        logprobs=True,
        top_logprobs=20,
        seed=SEED,
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
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only process 5 pairs (default: True)",
    )
    args = parser.parse_args()

    human_data = pd.read_csv(HUMAN_DATA, index_col=0)

    client = openai.OpenAI(
        api_key=API_KEY,
    )

    # Get sorted list of file names (without extension) from the directory
    rock_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
    )

    # Set the DataFrame's index and columns to these filenames
    human_data.index = rock_filenames
    human_data.columns = rock_filenames

    with open(NUMBER_MAP, "r", encoding="utf-8") as f:
        number_map = json.load(f)

    pairs = [comb for comb in combinations(human_data.columns, 2)]
    # Determine how many pairs to process based on debug flag
    if args.debug:
        end_idx = args.start + 5
    else:
        end_idx = len(pairs)
    for i, pair in enumerate(pairs[args.start : end_idx], start=args.start):
        rock1, rock2 = pair
        print(
            f"Processing pair {i + 1}/{len(pairs)}: {rock1} and {rock2}",
            end="\r",
            flush=True,
        )
        human = human_data[rock1][rock2]
        try:
            response = get_responses(rock1, rock2)
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
