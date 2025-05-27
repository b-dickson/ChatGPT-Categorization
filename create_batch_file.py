import base64
import json
import os
from itertools import combinations

IMAGE_FOLDER = "data\\120 Rocks"
image_names = [
    os.path.splitext(image)[0]
    for image in os.listdir(IMAGE_FOLDER)
    if image.endswith(".jpg")
]


# Function to encode the image
def encode_image(image):
    image_path = f"data\\120 Rocks\\{image}.jpg"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


SYSTEM_PROMPT = (
    "You are assisting in a study in which you are shown pairs of rocks and "
    "rate how visually similar they are on a scale from 1 to 9, with 1 being most dissimilar, "
    "5 being moderately similar, and 9 being most similar. "
    # "You are shown examples of highly similar, moderately similar, and highly dissimilar pairs. "
    # "You use these examples to anchor your judgments. "
    # ChatGPT is biased to give low similarities. Adding this instruction helps.
    "You use the full range of the scale and only respond with a 1 or 2 when the pair is extremely dissimilar. "
    "You only respond with a single number from 1 to 9, without explaining your reasoning."
)

SIMILAR_EXAMPLE_PROMPT = "This is an example of a highly similar pair of rocks."

DISSIMILAR_EXAMPLE_PROMPT = "This is an example of a highly dissimilar pair of rocks."

MEDIUM_EXAMPLE_PROMPT = "This is an example of a moderately similar pair of rocks."

QUESTION_PROMPT = "From 1-9, how visually similar are these two rocks?"

SIMILAR_ROCK1 = "I_Obsidian_1_120"
SIMILAR_ROCK2 = "M_Anthracite_1_120"

DISSIMILAR_ROCK1 = "I_Obsidian_4_120"
DISSIMILAR_ROCK2 = "M_Marble_3_120"

MEDIUM_ROCK1 = "I_Basalt_1_120"
MEDIUM_ROCK2 = "S_Shale_2_120"

SEED = 123


def create_messages(prompt, rocks):
    messages = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
        + [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(rock)}"},
            }
            for rock in rocks
        ],
    }
    return messages


def prepare_request(rock1, rock2):
    return {
        "custom_id": f"{rock1},{rock2}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # create_messages(SIMILAR_EXAMPLE_PROMPT, (SIMILAR_ROCK1, SIMILAR_ROCK2)),
                # create_messages(MEDIUM_EXAMPLE_PROMPT, (MEDIUM_ROCK1, MEDIUM_ROCK2)),
                # create_messages(
                #     DISSIMILAR_EXAMPLE_PROMPT, (DISSIMILAR_ROCK1, DISSIMILAR_ROCK2)
                # ),
                create_messages(QUESTION_PROMPT, (rock1, rock2)),
            ],
            "logprobs": True,
            "top_logprobs": 20,
            "seed": SEED,
            "max_tokens": 1,
        },
    }


output_folder = "batch_files"
os.makedirs(output_folder, exist_ok=True)
pairs = [comb for comb in combinations(image_names, 2)]

file_index = 1
line_count = 0
batch_size = 100
batch_file = open(f"{output_folder}/batch_requests_{file_index}.jsonl", "w")

for pair in pairs:
    rock1, rock2 = pair
    request = prepare_request(rock1, rock2)
    batch_file.write(json.dumps(request) + "\n")
    line_count += 1

    if line_count >= 100:
        batch_file.close()
        file_index += 1
        line_count = 0
        batch_file = open(f"{output_folder}/batch_requests_{file_index}.jsonl", "w")

batch_file.close()
