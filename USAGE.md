# Rock Similarity Rating Scripts

## Overview

Six scripts to evaluate visual similarity between rock pairs using different model providers:

- `get_similarities.py` - Original OpenAI GPT models
- `get_similarities_anthropic.py` - Anthropic Claude models  
- `get_similarities_gpt5.py` - OpenAI GPT-5 models
- `get_similarities_vllm.py` - Local HuggingFace models via vLLM
- `get_similarities_gemini.py` - Google Gemini models
- `get_similarities_openrouter.py` - OpenRouter unified API access

## Setup

1. Install dependencies:
```bash
pip install openai anthropic vllm google-genai pandas pillow tenacity
```

2. Configure API keys in `secret.py`:
```python
openai_api_key = "your_openai_key_here"
anthropic_api_key = "your_anthropic_key_here"
gemini_api_key = "your_gemini_key_here"
openrouter_api_key = "your_openrouter_key_here"
```

## Basic Usage

All scripts share the same core parameters:

```bash
python script_name.py --model MODEL_NAME --n_trials 10 --prompt_type base
```

### Common Parameters

- `--start` - Starting pair index (default: 0)
- `--n_trials` - Number of pairs to process (default: 5)
- `--shuffle` - Randomize pair order
- `--seed` - Random seed (default: 123)
- `--prompt_type` - Prompt variant (see options below)
- `--anchors` - Include example pairs in prompt
- `--human_data` - Use "30" or "360" rock dataset
- `--run_name` - Custom output folder name

### Prompt Types

- `base` - Standard similarity rating prompt
- `dimensions` - Detailed geological feature analysis
- `short_dimensions` - Simplified feature analysis
- `encourage_middle` - Bias toward middle ratings
- `discourage_low` - Avoid low similarity scores
- `discourage_extreme` - Avoid extreme scores (1,2,8,9)
- `elaborate` - Detailed rating explanations
- `long` - Extended visual analysis prompt
- `reverse` - Inverted scale (1=similar, 9=dissimilar)

## Model-Specific Usage

### OpenAI GPT (Original)
```bash
python get_similarities.py --model gpt-4o --n_trials 50
```

**Supported Models:**
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`
- `o3`

### Anthropic Claude
```bash
python get_similarities_anthropic.py --model claude-sonnet-4-20250514 --n_trials 50
```

**Supported Models:**
- `claude-opus-4-1-20250805`
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-3-7-sonnet-20250219`
- `claude-3-5-haiku-20241022`

### OpenAI GPT-5
```bash
python get_similarities_gpt5.py --model gpt-5-2025-08-07 --n_trials 50
```

**Supported Models:**
- `gpt-5-2025-08-07`
- `gpt-5-mini-2025-08-07`
- `gpt-5-nano-2025-08-07`

### vLLM (Local HuggingFace Models)
```bash
python get_similarities_vllm.py --model llava-hf/llava-1.5-7b-hf --n_trials 50
```

**Popular Models:**
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- `OpenGVLab/InternVL2-8B`
- `OpenGVLab/InternVL2-26B`
- `openbmb/MiniCPM-V-2_6`
- `microsoft/Florence-2-large`
- `HuggingFaceM4/idefics2-8b`
- `Qwen/Qwen2-VL-7B-Instruct`

**Additional vLLM Parameters:**
- `--max_model_len` - Context length (default: 4096)
- `--gpu_memory_utilization` - GPU memory usage (default: 0.9)
- `--temperature` - Sampling temperature (default: 0.0)
- `--max_tokens` - Max generation tokens (default: 10)

## Example Commands

Process 100 rock pairs with dimensional analysis:
```bash
python get_similarities_anthropic.py --model claude-sonnet-4-20250514 --prompt_type dimensions --n_trials 100 --shuffle --run_name claude_dimensions_run
```

Use anchor examples with GPT-5:
```bash
python get_similarities_gpt5.py --model gpt-5-2025-08-07 --anchors --n_trials 200 --human_data 360
```

Local inference with InternVL:
```bash
python get_similarities_vllm.py --model OpenGVLab/InternVL2-8B --gpu_memory_utilization 0.8 --n_trials 50
```

### Google Gemini
```bash
python get_similarities_gemini.py --model gemini-2.5-flash --n_trials 50
```

**Supported Models:**
- `gemini-2.5-flash-lite`
- `gemini-2.5-flash`
- `gemini-2.5-pro`

## Example Commands

Process 100 rock pairs with dimensional analysis:
```bash
python get_similarities_anthropic.py --model claude-sonnet-4-20250514 --prompt_type dimensions --n_trials 100 --shuffle --run_name claude_dimensions_run
```

Use anchor examples with GPT-5:
```bash
python get_similarities_gpt5.py --model gpt-5-2025-08-07 --anchors --n_trials 200 --human_data 360
```

Local inference with InternVL:
```bash
python get_similarities_vllm.py --model OpenGVLab/InternVL2-8B --gpu_memory_utilization 0.8 --n_trials 50
```

Process rocks with Gemini thinking capabilities:
```bash
python get_similarities_gemini.py --model gemini-2.5-pro --prompt_type dimensions --anchors --n_trials 100
```

### OpenRouter
```bash
python get_similarities_openrouter.py --model meta-llama/llama-3.2-11b-vision-instruct:free --n_trials 50
```

**Supported Models:**

*Free Models:*
- `meta-llama/llama-3.2-11b-vision-instruct:free`
- `google/gemma-3-4b-it:free`
- `google/gemma-3-12b-it:free`
- `google/gemma-3-27b-it:free`
- `qwen/qwen2.5-vl-32b-instruct:free`
- `qwen/qwen2.5-vl-72b-instruct:free`
- `moonshotai/kimi-vl-a3b-thinking:free`
- `mistralai/mistral-small-3.1-24b-instruct:free`

*Premium Models:*
- `meta-llama/llama-3.2-90b-vision-instruct`
- `openai/gpt-4o`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-pro-vision`

## Example Commands

Process 100 rock pairs with dimensional analysis:
```bash
python get_similarities_anthropic.py --model claude-sonnet-4-20250514 --prompt_type dimensions --n_trials 100 --shuffle --run_name claude_dimensions_run
```

Use anchor examples with GPT-5:
```bash
python get_similarities_gpt5.py --model gpt-5-2025-08-07 --anchors --n_trials 200 --human_data 360
```

Local inference with InternVL:
```bash
python get_similarities_vllm.py --model OpenGVLab/InternVL2-8B --gpu_memory_utilization 0.8 --n_trials 50
```

Process rocks with Gemini thinking capabilities:
```bash
python get_similarities_gemini.py --model gemini-2.5-pro --prompt_type dimensions --anchors --n_trials 100
```

Compare multiple models via OpenRouter:
```bash
python get_similarities_openrouter.py --model qwen/qwen2.5-vl-72b-instruct:free --prompt_type dimensions --anchors --n_trials 100
```

## Output

Each run creates a timestamped folder in `output/` containing:
- `similarities.csv` - Main results with human and model ratings
- `raw_responses.jsonl` - Complete API/model responses
- `metadata.json` - Run configuration and timing