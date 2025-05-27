import glob
import os
import time

import openai

client = openai.OpenAI()

TERMINAL = {"completed", "failed", "canceled", "expired"}


def wait_until_no_active_batches(poll_secs: int = 60):
    """Block until every batch is in a terminal state."""
    while True:
        active = [b for b in client.batches.list().data if b.status not in TERMINAL]
        if not active:
            return
        print(f"Waiting for {len(active)} active batch(es) to finish…")
        time.sleep(poll_secs)


batch_files = glob.glob("batch_files/batch_requests_*.jsonl")
batch_ids = []

for batch_file in batch_files:
    wait_until_no_active_batches()

    with open(batch_file, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Started batch {batch.id} from {batch_file}")
    batch_ids.append(batch.id)

    while batch.status not in TERMINAL:
        time.sleep(90)
        batch = client.batches.retrieve(batch.id)
        print(f"{batch.id} → {batch.status}")
        if batch.status == "failed":
            for error in batch.errors.data:
                print(f"Error: {error}")

    print(f"{batch.id} finished with status {batch.status}")
    time.sleep(90)  # Give extra time to clear queue

# Ensure the output folder exists
output_folder = "batch_outputs"
os.makedirs(output_folder, exist_ok=True)

# Download all batch outputs
for batch_id in batch_ids:
    batch = client.batches.retrieve(batch_id)
    file_response = client.files.content(batch.output_file_id)
    output_file = os.path.join(output_folder, f"batch_output_{batch_id}.jsonl")
    with open(output_file, "w") as f:
        f.write(file_response.text)
