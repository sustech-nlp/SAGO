import argparse
import os
import subprocess
import json
from datasets import load_dataset


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def download_rwku_single():
    """Download and process RWKU data for individual targets"""
    print("Loading RWKU datasets...")
    
    # Load all datasets
    forget_target = load_dataset("jinzhuoran/RWKU", 'forget_target')['train']
    forget_level1 = load_dataset("jinzhuoran/RWKU", 'forget_level1')['test']
    forget_level2 = load_dataset("jinzhuoran/RWKU", 'forget_level2')['test']
    forget_level3 = load_dataset("jinzhuoran/RWKU", 'forget_level3')['test']
    neighbor_level1 = load_dataset("jinzhuoran/RWKU", 'neighbor_level1')['test']
    neighbor_level2 = load_dataset("jinzhuoran/RWKU", 'neighbor_level2')['test']
    mia_forget = load_dataset("jinzhuoran/RWKU", 'mia_forget')["test"]
    mia_retain = load_dataset("jinzhuoran/RWKU", 'mia_retain')["test"]
    utility_general = load_dataset("jinzhuoran/RWKU", 'utility_general')['test']
    utility_reason = load_dataset("jinzhuoran/RWKU", 'utility_reason')['test']
    utility_truthfulness = load_dataset("jinzhuoran/RWKU", 'utility_truthfulness')['test']
    utility_factuality = load_dataset("jinzhuoran/RWKU", 'utility_factuality')['test']
    utility_fluency = load_dataset("jinzhuoran/RWKU", 'utility_fluency')['test']
    train_original_passage = load_dataset("jinzhuoran/RWKU", 'train_original_passage')['train']
    train_positive_llama3 = load_dataset("jinzhuoran/RWKU", 'train_positive_llama3')['train']
    train_negative_llama3 = load_dataset("jinzhuoran/RWKU", 'train_negative_llama3')['train']
    train_pair_llama3 = load_dataset("jinzhuoran/RWKU", 'train_pair_llama3')['train']
    train_refusal_llama3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_llama3')['train']
    train_positive_phi3 = load_dataset("jinzhuoran/RWKU", 'train_positive_phi3')['train']
    train_negative_phi3 = load_dataset("jinzhuoran/RWKU", 'train_negative_phi3')['train']
    train_pair_phi3 = load_dataset("jinzhuoran/RWKU", 'train_pair_phi3')['train']
    train_refusal_phi3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_phi3')['train']

    output_dir = 'data/rwku/Target'
    cnt = 0
    
    print("Processing individual targets...")
    for target in forget_target['target']:
        cnt += 1
        target_dir = os.path.join(output_dir, str(cnt) + '_' + target.replace(' ', '_'))
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Processing target {cnt}: {target}")
        
        # Save all data types for this target
        datasets_to_save = [
            (forget_target, 'intro.json', 'target'),
            (forget_level1, 'forget_level1.json', 'subject'),
            (forget_level2, 'forget_level2.json', 'subject'),
            (forget_level3, 'forget_level3.json', 'subject'),
            (neighbor_level1, 'neighbor_level1.json', 'subject'),
            (neighbor_level2, 'neighbor_level2.json', 'subject'),
            (utility_general, 'retain_mmlu.json', 'subject'),
            (utility_reason, 'retain_bbh.json', 'subject'),
            (utility_truthfulness, 'truthful.json', 'subject'),
            (utility_factuality, 'triviaqa.json', 'subject'),
            (utility_fluency, 'fluency.json', 'subject'),
            (train_original_passage, 'passage.json', 'subject'),
            (train_positive_llama3, 'positive_llama3.json', 'subject'),
            (train_negative_llama3, 'negative_llama3.json', 'subject'),
            (train_pair_llama3, 'pair_llama3.json', 'subject'),
            (train_refusal_llama3, 'reject_llama3.json', 'subject'),
            (train_positive_phi3, 'positive_phi3.json', 'subject'),
            (train_negative_phi3, 'negative_phi3.json', 'subject'),
            (train_pair_phi3, 'pair_phi3.json', 'subject'),
            (train_refusal_phi3, 'reject_phi3.json', 'subject'),
            (mia_forget, 'forget_mia.json', 'subject'),
            (mia_retain, 'retain_mia.json', 'subject'),
        ]
        
        for dataset, filename, key_field in datasets_to_save:
            filtered_data = dataset.filter(lambda example: example[key_field] == target).to_list()
            with open(os.path.join(target_dir, filename), 'w') as f:
                json.dump(filtered_data, f, indent=4)
    
    print(f"Completed processing {cnt} targets in {output_dir}")


def download_rwku_batch():
    """Download and process RWKU data in batches"""
    print("Loading RWKU datasets...")
    
    # Load all datasets
    forget_target = load_dataset("jinzhuoran/RWKU", 'forget_target')['train']
    forget_level1 = load_dataset("jinzhuoran/RWKU", 'forget_level1')['test']
    forget_level2 = load_dataset("jinzhuoran/RWKU", 'forget_level2')['test']
    forget_level3 = load_dataset("jinzhuoran/RWKU", 'forget_level3')['test']
    neighbor_level1 = load_dataset("jinzhuoran/RWKU", 'neighbor_level1')['test']
    neighbor_level2 = load_dataset("jinzhuoran/RWKU", 'neighbor_level2')['test']
    mia_forget = load_dataset("jinzhuoran/RWKU", 'mia_forget')["test"]
    mia_retain = load_dataset("jinzhuoran/RWKU", 'mia_retain')["test"]
    train_original_passage = load_dataset("jinzhuoran/RWKU", 'train_original_passage')['train']
    train_positive_llama3 = load_dataset("jinzhuoran/RWKU", 'train_positive_llama3')['train']
    train_negative_llama3 = load_dataset("jinzhuoran/RWKU", 'train_negative_llama3')['train']
    train_pair_llama3 = load_dataset("jinzhuoran/RWKU", 'train_pair_llama3')['train']
    train_refusal_llama3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_llama3')['train']
    train_positive_phi3 = load_dataset("jinzhuoran/RWKU", 'train_positive_phi3')['train']
    train_negative_phi3 = load_dataset("jinzhuoran/RWKU", 'train_negative_phi3')['train']
    train_pair_phi3 = load_dataset("jinzhuoran/RWKU", 'train_pair_phi3')['train']
    train_refusal_phi3 = load_dataset("jinzhuoran/RWKU", 'train_refusal_phi3')['train']

    # Define batch sizes
    batch_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    all_targets = forget_target['target']

    print("Processing batches...")
    for batch_size in batch_sizes:
        batch_name = f"1-{batch_size}"
        output_dir = f'data/rwku/Batch/{batch_name}'
        os.makedirs(output_dir, exist_ok=True)

        # Get targets for this batch
        batch_targets = all_targets[:batch_size]
        batch_targets_set = set(batch_targets)

        print(f"Processing batch {batch_name} with {len(batch_targets)} targets")
        
        # Collect all data using batch filtering
        datasets_to_save = [
            (train_original_passage, 'passage.json', 'subject'),
            (train_positive_llama3, 'positive_llama3.json', 'subject'),
            (train_pair_llama3, 'pair_llama3.json', 'subject'),
            (train_negative_llama3, 'negative_llama3.json', 'subject'),
            (train_refusal_llama3, 'reject_llama3.json', 'subject'),
            (train_positive_phi3, 'positive_phi3.json', 'subject'),
            (train_negative_phi3, 'negative_phi3.json', 'subject'),
            (train_pair_phi3, 'pair_phi3.json', 'subject'),
            (train_refusal_phi3, 'reject_phi3.json', 'subject'),
            (mia_forget, 'forget_mia.json', 'subject'),
            (mia_retain, 'retain_mia.json', 'subject'),
            (forget_target, 'intro.json', 'target'),
            (forget_level1, 'forget_level1.json', 'subject'),
            (forget_level2, 'forget_level2.json', 'subject'),
            (forget_level3, 'forget_level3.json', 'subject'),
            (neighbor_level1, 'neighbor_level1.json', 'subject'),
            (neighbor_level2, 'neighbor_level2.json', 'subject'),
        ]
        
        for dataset, filename, key_field in datasets_to_save:
            filtered_data = dataset.filter(lambda example: example[key_field] in batch_targets_set).to_list()
            
            print(f"  Saving {len(filtered_data)} {filename.replace('.json', '')} entries")
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(filtered_data, f, indent=4)
            
            # Save retain version for train datasets
            retain_targets = all_targets[-batch_size:]
            retain_targets_set = set(retain_targets)

            if filename in [
                'passage.json',
                'positive_llama3.json',
                'pair_llama3.json',
                'negative_llama3.json',
                'reject_llama3.json',
                'positive_phi3.json',
                'negative_phi3.json',
                'pair_phi3.json',
                'reject_phi3.json'
            ]:
                retain_filename = 'retain_' + filename
                retain_filtered_data = dataset.filter(lambda example: example[key_field] in retain_targets_set).to_list()
                print(f"  Saving {len(retain_filtered_data)} retain {retain_filename.replace('.json', '')} entries")
                with open(os.path.join(output_dir, retain_filename), 'w') as f:
                    json.dump(retain_filtered_data, f, indent=4)

        print(f"Completed batch {batch_name}")
    
    print("All batch processing completed!")


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )
    parser.add_argument(
        "--rwku_single",
        action="store_true",
        help="Download and process RWKU dataset for individual targets, saves to data/rwku/Target",
    )
    parser.add_argument(
        "--rwku_batch",
        action="store_true",
        help="Download and process RWKU dataset in batches, saves to data/rwku/Batch",
    )

    args = parser.parse_args()

    if args.wmdp:
        download_wmdp()
    if args.rwku_single:
        download_rwku_single()
    if args.rwku_batch:
        download_rwku_batch()


if __name__ == "__main__":
    main()
