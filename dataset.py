#!/usr/bin/env python

import argparse
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

# Constants
PGMIX_LANGUAGES: Sequence[str] = (
    "ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "th", 
    "it", "ja", "ko", "pl", "pt", "ru", "es", "sv",
)

PGMIX_SAFETY_CATEGORIES: Sequence[str] = (
    "S1: Violent Crimes", "S2: Non-Violent Crimes", "S3: Sex Crimes", 
    "S4: Child Exploitation", "S5: Defamation", "S6: Specialized Advice", 
    "S7: Privacy", "S8: Intellectual Property", "S9: Indiscriminate Weapons", 
    "S10: Hate", "S11: Self-Harm", "S12: Sexual Content", 
    "S13: Elections", "S14: Code Interpreter Abuse",
)

SYSTEM_PROMPT = """You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.
In addition, identify the policies violated if the interaction is unsafe and harmful according to the safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Sample = Dict[str, Any]

@dataclass
class PGMixPrepConfig:
    dataset_name: str = "ToxicityPrompts/PolyGuardMix"
    split: str = "train"
    target_samples: int = 40000                 # for stratified sampling method
    sample_ratio: float = 0.01                  # for simple sampling method
    save_dir: Path = Path("./data/pgmix_stratified")
    seed: int = 42                                      # for reproducibility
    process_for_training: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)


class PolyGuardFormatter:
    """Formats a sample for supervised fine-tuning."""
    def __call__(self, sample: Sample) -> Sample:
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        formatted_input = f"Human user:\n{prompt}\n\nAI assistant:\n{response}"

        harmful_request_bool = sample.get("prompt_harm_label") == "yes"
        harmful_response_bool = sample.get("response_harm_label") == "yes"
        response_refusal_bool = sample.get("response_refusal_label") == "yes"

        harmful_request = "yes" if harmful_request_bool else "no"
        harmful_response = "yes" if harmful_response_bool else "no"
        response_refusal = "yes" if response_refusal_bool else "no"

        prompt_violations_str = sample.get("prompt_safety_categories")
        response_violations_str = sample.get("response_safety_categories")
        
        prompt_violations = []
        if prompt_violations_str and prompt_violations_str != "None":
            prompt_violations = [v.strip() for v in prompt_violations_str.split(',')]

        response_violations = []
        if response_violations_str and response_violations_str != "None":
            response_violations = [v.strip() for v in response_violations_str.split(',')]

        prompt_violations_output = ", ".join(prompt_violations) if prompt_violations else "None"
        response_violations_output = ", ".join(response_violations) if response_violations else "None"

        target_output = (
            f"Harmful request: {harmful_request}\n"
            f"Request safety violations: {prompt_violations_output}\n"
            f"Response refusal: {response_refusal}\n"
            f"Harmful response: {harmful_response}\n"
            f"Response safety violations: {response_violations_output}"
        )
        return {
            "input": formatted_input,
            "output": target_output,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": formatted_input},
            ],
            "response": target_output,
        }


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def format_for_sft(example):
    """Simple formatter for SFT, operating on original dataset columns."""
    user_prompt = f"""Human user:
{example['prompt']}

AI assistant:
{example['response']}"""

    input_text = f"<s>[INST] {SYSTEM_PROMPT}\n{user_prompt} [/INST]"

    output_text = f"Harmful request: {example['prompt_harm_label']}\n"
    output_text += f"Request safety violations: {example.get('prompt_safety_categories') or 'None'}\n"
    output_text += f"Response refusal: {example['response_refusal_label']}\n"
    output_text += f"Harmful response: {example['response_harm_label']}\n"
    output_text += f"Response safety violations: {example.get('response_safety_categories') or 'None'}"

    return {"input": input_text, "output": output_text}


def prepare_pgmix(config: PGMixPrepConfig) -> Tuple[Dataset, Optional[Dataset], pd.DataFrame]:
    """Loads, stratifies, and optionally processes the PolyGuardMix dataset."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.split, streaming=False)
    
    seed_everything(config.seed)
    
    sampled_dataset, stats = stratify_dataset(dataset, config.target_samples, config.seed)
    
    config.save_dir.mkdir(parents=True, exist_ok=True)
    sampled_dataset.save_to_disk(str(config.save_dir / "train"))
    stats.to_csv(config.save_dir / "stratification_stats.csv", index=False)
    
    processed_dataset: Optional[Dataset] = None
    if config.process_for_training:
        formatter = PolyGuardFormatter()
        processed_dataset = Dataset.from_list([formatter(dict(row)) for row in tqdm(sampled_dataset, desc="Formatting data")])
        processed_dataset.save_to_disk(str(config.save_dir / "processed"))
        
    return sampled_dataset, processed_dataset, stats


def stratify_dataset(dataset: Iterable[Any], target_samples: int, seed: int) -> Tuple[Dataset, pd.DataFrame]:
    """Performs stratified sampling on the dataset to create a balanced subset."""
    buckets: Dict[Tuple, List[Sample]] = defaultdict(list)
    for sample in tqdm(dataset, desc="Building stratification buckets"):
        buckets[build_stratification_key(sample)].append(dict(sample))
        
    total_rows = sum(len(rows) for rows in buckets.values())
    ratio = target_samples / total_rows if total_rows else 0
    rng = random.Random(seed)
    
    sampled_rows: List[Sample] = []
    stats: List[Dict[str, Any]] = []
    
    for key, rows in tqdm(buckets.items(), desc="Sampling from buckets"):
        take = min(max(round(len(rows) * ratio), 1), len(rows))
        chosen = rows if take == len(rows) else rng.sample(rows, take)
        sampled_rows.extend(chosen)
        stats.append({
            "language": key[0],
            "harmful_prompt": key[1],
            "harmful_response": key[2],
            "refusal": key[3],
            "prompt_has_violations": key[4],
            "response_has_violations": key[5],
            "original_count": len(rows),
            "sampled_count": take,
        })
        
    sampled_dataset = Dataset.from_list(sampled_rows)
    stats_df = pd.DataFrame(stats)
    return sampled_dataset, stats_df


def build_stratification_key(sample: Sample) -> Tuple:
    """Builds a key for stratification based on language and safety labels."""
    language = sample.get('metadata', {}).get('language', 'en')

    prompt_violations = sample.get("prompt_safety_categories")
    response_violations = sample.get("response_safety_categories")
    return (
        language,
        sample.get("prompt_harm_label") == "yes",
        sample.get("response_harm_label") == "yes",
        sample.get("response_refusal_label") == "yes",
        bool(prompt_violations and prompt_violations != "None"),
        bool(response_violations and response_violations != "None"),
    )


def seed_everything(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def summarize_dataset(dataset):
    """Generate summary statistics for the dataset."""
    frame = dataset.to_pandas()
    
    # language distribution
    if "metadata" in frame.columns:
        def extract_language(metadata_dict):
            if isinstance(metadata_dict, dict) and 'language' in metadata_dict:
                return metadata_dict['language']
            return 'unknown'
        
        languages = frame["metadata"].apply(extract_language)
        language_count = languages.nunique()
        
        print("\nLanguage distribution:")
        lang_counts = languages.value_counts()
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count}")
    else:
        language_count = "N/A"
    
    # safety-related metrics
    harmful_prompts = 0
    safe_prompts = 0
    harmful_responses = 0
    refusals = 0
    
    if "prompt_harm_label" in frame.columns:
        harmful_prompts = (frame["prompt_harm_label"] == "yes").sum()
        safe_prompts = (frame["prompt_harm_label"] == "no").sum()
    
    if "response_harm_label" in frame.columns:
        harmful_responses = (frame["response_harm_label"] == "yes").sum()
    
    if "response_refusal_label" in frame.columns:
        refusals = (frame["response_refusal_label"] == "yes").sum()
    
    metrics = {
        "total_samples": len(frame),
        "languages": language_count,
        "harmful_prompts": harmful_prompts,
        "safe_prompts": safe_prompts,
        "harmful_responses": harmful_responses,
        "refusals": refusals,
        "avg_length": frame["text"].str.len().mean() if "text" in frame.columns else "N/A",
        "min_length": frame["text"].str.len().min() if "text" in frame.columns else "N/A",
        "max_length": frame["text"].str.len().max() if "text" in frame.columns else "N/A",
    }
    
    print("\nDataset Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return metrics

# for simple sampling method
def simple_dataset_prep(config: PGMixPrepConfig):
    """Simple dataset preparation using fixed sampling ratio instead of stratification."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    pgmix_dataset_dict = load_dataset(config.dataset_name)
    logger.info("Dataset loaded successfully\n")

    train_dataset = cast(Dataset, pgmix_dataset_dict['train'])
    
    train_sample_size = int(config.sample_ratio * len(train_dataset))
    sampled_train_dataset = train_dataset.shuffle(seed=config.seed).select(range(train_sample_size))

    logger.info("Formatting the dataset for SFT...")
    sft_dataset = sampled_train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names, num_proc=os.cpu_count())
    logger.info("Dataset formatted successfully!")

    output_dir = "polyguardmix_sampled"
    os.makedirs(output_dir, exist_ok=True)
    sft_dataset.to_json(os.path.join(output_dir, "polyguardmix_sampled.jsonl"))
    logger.info(f"Sampled dataset saved to {output_dir}")
    return sft_dataset


def run_data_pipeline(config: PGMixPrepConfig, method="stratified") -> None:
    """Runs the main data preparation pipeline."""
    if method == "stratified":
        sampled_dataset, processed_dataset, _stats = prepare_pgmix(config)
        metrics = summarize_dataset(sampled_dataset)
        print(f"Saved stratified training split to {config.save_dir / 'train'}")
        if processed_dataset is not None:
            print(f"Saved formatted dataset to {config.save_dir / 'processed'}")
        print(f"\n--- Stratified Sampling Summary ---")
        print(f"Total sampled rows: {len(sampled_dataset)}")
        print(f"Languages: {metrics['languages']}")
        print(f"Harmful prompts: {metrics['harmful_prompts']}")
        print(f"Safe prompts: {metrics['safe_prompts']}")
        print(f"Harmful responses: {metrics['harmful_responses']}")
        print(f"Refusals: {metrics['refusals']}")
        print(f"Stratification stats written to {config.save_dir / 'stratification_stats.csv'}")
    else:
        simple_dataset_prep(config)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(description="PolyGuard dataset preparation")
    parser.add_argument("--dataset_name", default="ToxicityPrompts/PolyGuardMix", help="Name of the dataset on Hugging Face.")
    parser.add_argument("--data_dir", default="./data/pgmix_stratified", help="Directory to save the processed data.")
    parser.add_argument("--target_samples", type=int, default=40000, help="Number of samples for stratified sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--method", choices=["simple", "stratified"], default="stratified", 
                        help="Dataset preparation method: simple or stratified sampling")
    return parser


def main() -> None:
    """Main execution function."""
    parser = build_parser()
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    prep_config = PGMixPrepConfig(
        dataset_name=args.dataset_name,
        target_samples=args.target_samples,
        save_dir=data_dir,
        seed=args.seed,
        process_for_training=True,
    )
    
    run_data_pipeline(prep_config, method=args.method)


if __name__ == "__main__":
    main()


