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
    retry_if_exception_type,
)
from PIL import Image
import io
from secret import openai_api_key

IMAGE_FOLDER = "data/360 Rocks"

API_KEY = openai_api_key

ROCKS_30_DATA = "data/30_rocks_similarities.csv"
ROCKS_360_DATA = "data/rocks_360_similarities.csv"

NUMBER_MAP = "number_map.json"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BASE_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a scale from 1 to 9, with 1 being most dissimilar, "
    "5 being moderately similar, and 9 being most similar. "
    "You only respond with a single number from 1 to 9, without explaining your reasoning. "
)

ENCOURAGE_MIDDLE_PROMPT = BASE_PROMPT + (
    "You use the full range of the 1-9 scale and err on the side of using the middle of the scale (4 or 5) when you are unsure. "
    "Rocks that are very similar in almost all respects should be rated as highly similar (8 or 9). "
    "You only use a 1 or 2 when the rocks are truly different in every meaningful visual way. "
    "Most ratings should fall somewhere in the middle of the scale. "
)

DIMENSIONS_PROMPT = BASE_PROMPT + (
    "Here are some dimensions to consider when rating the rocks: "
    "lightness/darkness of color, "
    "average grain size, "
    "smoothness/roughness of texture, "
    "shininess/dullness of surface, "
    "organization of grains (e.g., uniform, layered, haphazard), "
    "chromaticity (i.e., whether the rock is monochromatic, warm-colored, or cool-colored), "
    "hue (i.e., whether the rock is red, blue, green, etc.), "
    "volume (i.e., whether the rock is flat or round/boxy), "
    "porphyritic texture (i.e., whether the rock has small fragments or pebbles that are glued into a separate background texture), "
    "pegmatitic structure (i.e., whether the rock has very large crystals that are embedded in a separate background), "
    "and conchoidal fracture (i.e., whether the rock has a smooth, curved surfaces similar to the inside of a seashell). "
    "These are not the only dimensions to consider, but they are some of the most important. "
    "You use the full range of the 1-9 scale and err on the side of using the middle of the scale (4 or 5) when you are unsure. "
    "Rocks that are very similar in several of these dimensions should be rated as highly similar (8 or 9). "
    "You only use a 1 or 2 when the images are truly different in every meaningful visual way. "
    "Most ratings should fall somewhere in the middle of the scale. "
)

SHORT_DIMENSIONS_PROMPT = BASE_PROMPT + (
    "Here are some dimensions to consider when rating the rocks: "
    "lightness/darkness of color, "
    "average grain size, "
    "smoothness/roughness of texture, "
    "shininess/dullness of surface, "
    "and organization of grains (e.g., uniform, layered, haphazard), "
    "chromaticity (i.e., whether the rock is monochromatic, warm-colored, or cool-colored), "
    "These are not the only dimensions to consider, but they are some of the most important. "
    "You use the full range of the 1-9 scale and err on the side of using the middle of the scale (4 or 5) when you are unsure. "
    "Rocks that are very similar in several of these dimensions should be rated as highly similar (8 or 9). "
    "You only use a 1 or 2 when the images are truly different in every meaningful visual way. "
    "Most ratings should fall somewhere in the middle of the scale. "
)

LONG_PROMPT = BASE_PROMPT + (
    "You only consider the visual appearance of the rocks, not their geological properties or origins. "
    "You consider several visual aspects of the rocks, including their shape, texture, color, and structure. "
    "It is possible for rocks to differ in some aspects while still being highly similar overall. "
    "For example, two rocks may have different shapes and textures, but still be highly similar because they are both dark and shiny. "
    "You use the full range of the 1-9 scale and err on the side of using the middle of the scale (4 or 5) when you are unsure. "
    "You only use a 1 or 2 when the images are truly different in every meaningful visual way. "
)

ELABORATE_PROMPT = BASE_PROMPT + (
    "A rating of 1 means the rocks are extremely dissimilar—they differ in almost every way (e.g., shape, texture, color, and structure). "
    "A rating of 5 means the rocks are moderately similar—they share some key features but have clear differences. "
    "A rating of 9 means the rocks are extremely similar--they are nearly identical in several aspects of their visual appearance. "
    "Most ratings should fall somewhere in the middle of the scale. You should only use a 1 or 2 when the images are truly different in every meaningful visual way."
)

REVERSE_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a scale from 1 to 9, with 1 being most similar, "
    "5 being moderately similar, and 9 being most dissimilar. "
    "You only respond with a single number from 1 to 9, without explaining your reasoning. "
)

DISCOURAGE_LOW_PROMPT = BASE_PROMPT + (
    "You use the full range of the scale. "
    "You only respond with 1 or 2 when the pair is extremely dissimilar. "
)

DISCOURAGE_EXTREME_PROMPT = DISCOURAGE_LOW_PROMPT + (
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
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded


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


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_responses(client, rock1, rock2, prompt_type, anchors):
    if prompt_type == "discourage_low":
        prompt = DISCOURAGE_LOW_PROMPT
    elif prompt_type == "discourage_extreme":
        prompt = DISCOURAGE_EXTREME_PROMPT
    elif prompt_type == "base":
        prompt = BASE_PROMPT
    elif prompt_type == "reverse":
        prompt = REVERSE_PROMPT
    elif prompt_type == "elaborate":
        prompt = ELABORATE_PROMPT
    elif prompt_type == "long":
        prompt = LONG_PROMPT
    elif prompt_type == "dimensions":
        prompt = DIMENSIONS_PROMPT
    elif prompt_type == "short_dimensions":
        prompt = SHORT_DIMENSIONS_PROMPT
    elif prompt_type == "encourage_middle":
        prompt = ENCOURAGE_MIDDLE_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    messages = [
        {"role": "system", "content": prompt},
    ]
    if anchors:
        messages += [
            create_messages((SIMILAR_ROCK1, SIMILAR_ROCK2)),
            {"role": "assistant", "content": "9" if prompt_type != "reverse" else "1"},
            create_messages((DISSIMILAR_ROCK1, DISSIMILAR_ROCK2)),
            {"role": "assistant", "content": "1" if prompt_type != "reverse" else "9"},
            create_messages((MEDIUM_ROCK1, MEDIUM_ROCK2)),
            {"role": "assistant", "content": "5"},
        ]
    messages.append(create_messages((rock1, rock2)))
    
    # GPT-5 models support logprobs and max_completion_tokens
    response = completion_with_backoff(
        client,
        model=args.model,
        messages=messages,
        logprobs=True,
        top_logprobs=20,
        seed=args.seed,
        max_completion_tokens=1,
    )
    return response


def get_average_rating(logprobs, number_map):
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
        except Exception:
            continue
    try:
        av = np.average(ratings, weights=weights)
    except Exception:
        av = 0.0
    return av


def main(args):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = args.run_name if args.run_name else run_timestamp
    run_folder = os.path.join(OUTPUT_FOLDER, run_folder_name)
    os.makedirs(run_folder)

    metadata = {"timestamp": run_timestamp, "args": vars(args)}
    metadata_path = os.path.join(run_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)

    raw_responses = os.path.join(run_folder, "raw_responses.jsonl")
    similarity_output = os.path.join(run_folder, "similarities.csv")

    # Select human data file based on CLI argument
    if args.human_data == "30":
        human_data = ROCKS_30_DATA
    else:
        human_data = ROCKS_360_DATA

    human_data = pd.read_csv(human_data, index_col=0)

    client = openai.OpenAI(
        api_key=API_KEY,
    )

    rock_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
    )

    if args.human_data == "360":
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
            f"\033[KProcessing pair {i + 1}/{len(pairs)}: {rock1} and {rock2}",
            end="\r",
            flush=True,
        )
        human = human_data[rock1][rock2]
        try:
            response = get_responses(
                client, rock1, rock2, prompt_type=args.prompt_type, anchors=args.anchors
            )
        except Exception as e:
            print(f"Error processing pair {rock1} and {rock2}: {e}")
            break
        
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        gpt5_rating = get_average_rating(logprobs, number_map)

        with open(raw_responses, mode="a", encoding="utf-8") as rawfile:
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

        with open(similarity_output, mode="a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "index",
                "Rock1",
                "Rock2",
                "Rock1 Index",
                "Rock2 Index",
                "Human Rating",
                "GPT5 Rating",
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
                    "GPT5 Rating": gpt5_rating,
                }
            )

    completion_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(metadata_path, "r", encoding="utf-8") as metafile:
        metadata = json.load(metafile)
    metadata["completion_timestamp"] = completion_timestamp
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get rock similarity ratings using GPT-5.")
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
        "--prompt_type",
        type=str,
        choices=[
            "base",
            "discourage_low",
            "discourage_extreme",
            "reverse",
            "elaborate",
            "long",
            "dimensions",
            "short_dimensions",
            "encourage_middle",
        ],
        default="base",
        help="Discourage use of low similarity ratings except for extremely dissimilar pairs.",
    )
    parser.add_argument(
        "--anchors",
        action="store_true",
        default=False,
        help="Include anchor examples in the prompt (default: False)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt-5-mini-2025-08-07",
            "gpt-5-2025-08-07", 
            "gpt-5-nano-2025-08-07"
        ],
        default="gpt-5-2025-08-07",
        help="OpenAI GPT-5 model to use (default: gpt-5-2025-08-07)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name for this run; used as output subdirectory name.",
    )
    parser.add_argument(
        "--human_data",
        type=str,
        choices=["30", "360"],
        default="30",
        help="Which human similarity data to use: '30' for 30-rocks, '360' for 360-rocks (default: 30)",
    )
    args = parser.parse_args()

    main(args)