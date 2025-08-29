import argparse
import base64
import csv
import json
import os
from itertools import combinations
from datetime import datetime

import numpy as np
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from PIL import Image
import io
from secret import openai_api_key

# Image folder paths
IMAGE_FOLDER_30 = "data/30_rocks"
IMAGE_FOLDER_360 = "data/360_rocks"

API_KEY = openai_api_key

NUMBER_MAP = "number_map.json"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BASE_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a decimal scale from 1.00 to 9.00, with 1.00 being most dissimilar, "
    "5.00 being moderately similar, and 9.00 being most similar. "
    "You only respond with a single decimal number from 1.00 to 9.00, without explaining your reasoning. "
    "It must be formatted to exactly two decimal places (the hundredths place)"
)

ENCOURAGE_MIDDLE_PROMPT = BASE_PROMPT + (
    "You use the full range of the 1.00-9.00 scale and err on the side of using the middle of the scale (4.00 to 5.00) when you are unsure. "
    "Rocks that are very similar in almost all respects should be rated as highly similar (8.00 to 9.00). "
    "You only use 1.00 to 2.00 when the rocks are truly different in every meaningful visual way. "
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

# Anchor examples - using relative paths that work for both folders
SIMILAR_ROCK1 = "I_Obsidian_09"
SIMILAR_ROCK2 = "M_Anthracite_12"

DISSIMILAR_ROCK1 = "S_Rock Salt_12"
DISSIMILAR_ROCK2 = "S_Sandstone_12"

MEDIUM_ROCK1 = "I_Andesite_06"
MEDIUM_ROCK2 = "I_Peridotite_11"

QUESTION_PROMPT = "From 1-9, how visually similar are these two rocks?"


def encode_image(image, folder):
    image_path = os.path.join(folder, f"{image}.jpg")
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded


def create_messages(rocks, folder):
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


def get_responses(client, rock1, rock2, prompt_type, anchors, model, image_folder, seed):
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
    
    # Note: anchors would need to be in the same folder or handled differently
    # For simplicity, disabling anchors when they don't exist in the folder
    if anchors:
        # Check if anchor images exist in the folder
        anchor_images = [SIMILAR_ROCK1, SIMILAR_ROCK2, DISSIMILAR_ROCK1, 
                        DISSIMILAR_ROCK2, MEDIUM_ROCK1, MEDIUM_ROCK2]
        all_exist = all(os.path.exists(os.path.join(image_folder, f"{img}.jpg")) 
                       for img in anchor_images)
        
        if all_exist:
            messages += [
                create_messages((SIMILAR_ROCK1, SIMILAR_ROCK2), image_folder),
                {"role": "assistant", "content": "9" if prompt_type != "reverse" else "1"},
                create_messages((DISSIMILAR_ROCK1, DISSIMILAR_ROCK2), image_folder),
                {"role": "assistant", "content": "1" if prompt_type != "reverse" else "9"},
                create_messages((MEDIUM_ROCK1, MEDIUM_ROCK2), image_folder),
                {"role": "assistant", "content": "5"},
            ]
    
    messages.append(create_messages((rock1, rock2), image_folder))
    
    # GPT-5 models support logprobs and max_completion_tokens
    response = completion_with_backoff(
        client,
        model=model,
        messages=messages,
#        logprobs=False,
#        top_logprobs=20,
        seed=seed,
        max_completion_tokens=1000,
    )
    return response


def get_average_rating(logprobs, number_map):
    # We get a weighted average using the logprobs. The logprobs are actually nondeterministic
    # even with a fixed random seed, so different runs might have slightly different results.
    ratings = np.array([i + 1 for i in range(9)])
    weights = np.array([0.0] * 9)
    for lp in logprobs:
        token = lp.token.strip().lower()
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

    # Select image folder based on dataset
    if args.dataset == "30":
        image_folder = IMAGE_FOLDER_30
    else:
        image_folder = IMAGE_FOLDER_360
    
    # If n_trials not specified, set it to process all pairs
    if args.n_trials is None:
        if args.dataset == "30":
            # 30 choose 2 = 435 pairs
            args.n_trials = 435 - args.start
        else:
            # 360 choose 2 = 64,620 pairs
            args.n_trials = 64620 - args.start

    metadata = {
        "timestamp": run_timestamp,
        "args": vars(args),
        "image_folder": image_folder
    }
    metadata_path = os.path.join(run_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)

    raw_responses = os.path.join(run_folder, "raw_responses.jsonl")
    similarity_output = os.path.join(run_folder, "similarities.csv")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=API_KEY)

    # Load number map if it exists
    number_map = {}
    if os.path.exists(NUMBER_MAP):
        with open(NUMBER_MAP, "r", encoding="utf-8") as f:
            number_map = json.load(f)

    # Get all rock images from the folder
    rock_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(".jpg")]
    )
    
    print(f"Found {len(rock_filenames)} rock images in {image_folder}")

    # Generate all pairs
    pairs = list(combinations(rock_filenames, 2))
    print(f"Total number of pairs to process: {len(pairs)}")
    
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(pairs)

    # Handle start and n_trials
    end_idx = min(args.start + args.n_trials, len(pairs))
    pairs = pairs[args.start : end_idx]
    print(f"Processing pairs {args.start} to {end_idx - 1} ({len(pairs)} pairs)")
    
    for i, pair in enumerate(pairs, start=args.start):
        rock1, rock2 = pair
        print(
            f"\033[KProcessing pair {i + 1}/{args.start + len(pairs)}: {rock1} and {rock2}",
            end="\r",
            flush=True,
        )
        
        try:
            response = get_responses(
                client, rock1, rock2, prompt_type=args.prompt_type, anchors=args.anchors,
                model=args.model, image_folder=image_folder, seed=args.seed
            )
        except Exception as e:
            print(f"\nError processing pair {rock1} and {rock2}: {e}")
            continue
        
        # Extract rating from response
#        logprobs = response.choices[0].logprobs.content[0].top_logprobs
#        gpt5_rating = get_average_rating(logprobs, number_map)

        with open(raw_responses, mode="a", encoding="utf-8") as rawfile:
            rawfile.write(
                json.dumps(
                    {
                        "index": i,
                        "rock1": rock1,
                        "rock2": rock2,
                        "response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                    }
                )
                + "\n"
            )

#        with open(similarity_output, mode="a", newline="", encoding="utf-8") as csvfile:
#            fieldnames = [
#                "index",
#                "Rock1",
#                "Rock2",
#                "GPT5 Rating",
#            ]
#            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#            if csvfile.tell() == 0:
#                writer.writeheader()
#            writer.writerow(
#                {
#                    "index": i,
#                    "Rock1": rock1,
#                    "Rock2": rock2,
#                    "GPT5 Rating": gpt5_rating,
#                }
#            )

    print(f"\n\nCompleted processing {len(pairs)} pairs")
    
    completion_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(metadata_path, "r", encoding="utf-8") as metafile:
        metadata = json.load(metafile)
    metadata["completion_timestamp"] = completion_timestamp
    metadata["total_pairs_processed"] = len(pairs)
    with open(metadata_path, "w", encoding="utf-8") as metafile:
        json.dump(metadata, metafile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get rock similarity ratings using GPT-5.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["30", "360"],
        required=True,
        help="Which dataset to use: '30' for 30-rocks folder, '360' for 360-rocks folder",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the pair to start at (default: 0)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of pairs to process. If not specified, processes all remaining pairs",
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
        help="Type of prompt to use for similarity ratings.",
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
    args = parser.parse_args()
    
    main(args)
