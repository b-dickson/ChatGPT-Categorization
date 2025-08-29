#!/bin/bash

# List of Ollama models to test
#"mistral-small3.1:24b-instruct-2503-fp16"
#"llava:13b-v1.6-vicuna-fp16"
#"llava:34b-v1.6-fp16"
#"mistral-small3.2:24b-instruct-2506-fp16"
models=(
"gemma3:12b-it-fp16"
"qwen2.5vl:32b-fp16"
"qwen2.5vl:72b-fp16"
)

# Default parameters - you can override these
START=${START:-0}
# N_TRIALS - if not set, the Python script will process all remaining pairs
ANCHORS=${ANCHORS:-""}
DATASET=${DATASET:-"30"}

# Prompt types to test
prompt_types=("base" "encourage_middle")

echo "Starting Ollama model evaluation..."
if [ -z "$N_TRIALS" ]; then
    echo "Parameters: START=$START, N_TRIALS=ALL (default), DATASET=$DATASET"
else
    echo "Parameters: START=$START, N_TRIALS=$N_TRIALS, DATASET=$DATASET"
fi
echo "Models to test: ${#models[@]}"
echo "Prompt types to test: ${#prompt_types[@]} (${prompt_types[*]})"
echo

for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        # Create a clean model name for the run directory (replace colons and other chars)
        clean_model_name=$(echo "$model" | sed 's/:/_/g' | sed 's/\./_/g')
        
        # Add prompt type to run name
        run_name="${clean_model_name}_${prompt_type}_${DATASET}_integer"
        
        echo "----------------------------------------"
        echo "Testing model: $model"
        echo "Prompt type: $prompt_type"
        echo "Run name: ${run_name}"
        echo "----------------------------------------"
        
        # Build the command
        cmd="python get_similarities_ollama_simple_integer.py --model \"$model\" --run_name \"${run_name}\" --start $START --prompt_type $prompt_type --dataset $DATASET"
        
        # Only add n_trials if it's set
        if [ ! -z "$N_TRIALS" ]; then
            cmd="$cmd --n_trials $N_TRIALS"
        fi
        
        # Add anchors flag if specified
        if [ ! -z "$ANCHORS" ]; then
            cmd="$cmd --anchors"
        fi
        
        echo "Running: $cmd"
        echo
        
        # Execute the command
        eval $cmd
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $model with $prompt_type prompt"
        else
            echo "✗ Failed to complete $model with $prompt_type prompt"
            echo "Continuing with next prompt/model..."
        fi
        
        echo
        sleep 2  # Brief pause between runs
    done
done

echo "All models completed!"
echo "Results saved in output/ directory with model-specific subdirectories"
