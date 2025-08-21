import os
import json
import torch
import numpy as np
from datetime import datetime
import time
import accelerate

# Import your RLAWM implementation
from Main import RLAWM, RLAWMConfig


def quick_evaluate_rlawm(rlawm_model, test_data, num_samples=20):
    """Quick evaluation with essential metrics"""
    print(f"Running quick evaluation on {num_samples} samples...")

    results = {
        'tpr_at_1': 0.0,
        'best_f1': 0.0,
        'sr': 0.0,
        'detection_accuracy': 0.0
    }

    detection_results = []
    successful_triggers = 0
    successful_generations = 0

    for i, data in enumerate(test_data[:num_samples]):
        prompt = data.get('prompt', data.get('text', ''))[:80]  # Shorter prompts

        if len(prompt.strip()) == 0:
            continue

        try:
            print(f"Sample {i + 1}/{num_samples}: {prompt[:40]}...")

            # Generate watermarked text (shorter)
            watermarked_text = rlawm_model.generate_watermarked_text(prompt, max_new_tokens=20)
            if watermarked_text and len(watermarked_text.strip()) > len(prompt):
                successful_generations += 1
            else:
                continue

            # Generate non-watermarked text (shorter)
            inputs = rlawm_model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(rlawm_model.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = rlawm_model.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=rlawm_model.tokenizer.eos_token_id
                )
            non_watermarked_text = rlawm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Test detection
            water_result = rlawm_model.detect_watermark(watermarked_text, return_dict=True)
            non_water_result = rlawm_model.detect_watermark(non_watermarked_text, return_dict=True)

            # Extract boolean values
            water_detected = water_result['is_watermarked']
            if torch.is_tensor(water_detected):
                water_detected = water_detected.item()

            non_water_detected = non_water_result['is_watermarked']
            if torch.is_tensor(non_water_detected):
                non_water_detected = non_water_detected.item()

            detection_results.extend([
                (True, water_detected),
                (False, non_water_detected)
            ])

            if water_detected:
                successful_triggers += 1

            print(f"  Water: {water_detected}, Non-water: {non_water_detected}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Calculate metrics
    if detection_results:
        true_positives = sum(1 for true_label, pred in detection_results if true_label and pred)
        false_positives = sum(1 for true_label, pred in detection_results if not true_label and pred)
        true_negatives = sum(1 for true_label, pred in detection_results if not true_label and not pred)
        false_negatives = sum(1 for true_label, pred in detection_results if true_label and not pred)

        # TPR@1%
        total_watermarked = true_positives + false_negatives
        if total_watermarked > 0:
            results['tpr_at_1'] = true_positives / total_watermarked

        # F1 Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        results['best_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Detection accuracy
        correct_predictions = true_positives + true_negatives
        total_predictions = len(detection_results)
        results['detection_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Successful trigger ratio
    if successful_generations > 0:
        results['sr'] = successful_triggers / successful_generations

    print(f"\nQuick Evaluation Results:")
    print(f"=" * 40)
    print(f"TPR@1%: {results['tpr_at_1']:.4f}")
    print(f"Best F1: {results['best_f1']:.4f}")
    print(f"SR: {results['sr']:.4f}")
    print(f"Detection Accuracy: {results['detection_accuracy']:.4f}")
    print(f"Successful generations: {successful_generations}/{num_samples}")
    print(f"=" * 40)

    return results


def quick_train_and_test():
    """Quick training and testing with small samples"""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Minimal configuration for quick testing
    config = RLAWMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gamma=0.5,
        hash_key=42,
        z_threshold=2.0,
        prefix_length=5,
        delta_min=0.5,
        delta_max=3.0,
        alpha=0.5,
        beta=0.5,
        lambda_mmd=1.0,
        learning_rate=2e-5,
        n_layers=6,
        n_heads=8,
        d_model=384,
        d_ff=2048
    )

    print("=" * 60)
    print("RLAWM Quick Test - Small Samples & Few Epochs")
    print("=" * 60)

    # Load dataset
    print("1. Loading dataset...")
    dataset_path = 'Dataset/processed_c4.json'
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            dataset = [json.loads(line) for line in lines]
        print(f"   Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   Dataset error: {e}")
        return

    # Use very small dataset
    train_data = dataset[:30]  # Only 30 training samples
    test_data = dataset[100:120]  # Only 20 test samples

    print(f"   Quick training: {len(train_data)} samples")
    print(f"   Quick testing: {len(test_data)} samples")

    # Initialize RLAWM
    print("\n2. Initializing RLAWM...")
    model_path = "/Users/zebaobao/Desktop/Llama-2-7b-chat-hf"
    try:
        rlawm = RLAWM(model_path, config)
        print("   RLAWM initialized")
    except Exception as e:
        print(f"   Initialization failed: {e}")
        return

    # Baseline evaluation
    print("\n3. Baseline evaluation...")
    baseline_results = quick_evaluate_rlawm(rlawm, test_data, num_samples=10)

    # Quick training - 2 stages with very few epochs
    print("\n4. Quick training...")
    training_stages = [
        {"samples": 15, "epochs": 5, "description": "Stage 1: Quick Generation Training", "focus": "generation"},
        {"samples": 30, "epochs": 10, "description": "Stage 2: Quick Collaborative Training", "focus": "collaborative"}
    ]

    best_results = baseline_results

    for stage, config_stage in enumerate(training_stages, 1):
        print(f"\n   {config_stage['description']}")
        print(f"   Using {config_stage['samples']} samples for {config_stage['epochs']} epochs")

        start_time = time.time()

        try:
            stage_train_data = train_data[:config_stage['samples']]

            # Quick training
            rlawm.train_agents_safe(
                stage_train_data,
                num_epochs=config_stage['epochs'],
                focus=config_stage['focus']
            )

            training_time = time.time() - start_time
            print(f"   Training completed in {training_time:.2f} seconds")

            # Quick evaluation
            print(f"   Evaluating Stage {stage}...")
            stage_results = quick_evaluate_rlawm(rlawm, test_data, num_samples=10)

            print(f"\n   Stage {stage} Results:")
            print(f"     TPR@1%: {stage_results['tpr_at_1']:.4f}")
            print(f"     Best F1: {stage_results['best_f1']:.4f}")
            print(f"     SR: {stage_results['sr']:.4f}")
            print(f"     Detection Accuracy: {stage_results['detection_accuracy']:.4f}")

            # Update best if improved
            if stage_results['best_f1'] > best_results['best_f1']:
                best_results = stage_results
                print(f"   *** Stage {stage} shows improvement! ***")

        except Exception as e:
            print(f"   Stage {stage} failed: {e}")
            continue

    # Final evaluation
    print("\n5. Final evaluation...")
    final_results = quick_evaluate_rlawm(rlawm, test_data, num_samples=20)

    print(f"\n6. Summary:")
    print("=" * 50)
    print(f"Baseline vs Final:")
    print(f"TPR@1%:     {baseline_results['tpr_at_1']:.4f} → {final_results['tpr_at_1']:.4f}")
    print(f"Best F1:    {baseline_results['best_f1']:.4f} → {final_results['best_f1']:.4f}")
    print(f"SR:         {baseline_results['sr']:.4f} → {final_results['sr']:.4f}")
    print(f"Det Acc:    {baseline_results['detection_accuracy']:.4f} → {final_results['detection_accuracy']:.4f}")
    print("=" * 50)

    # Show improvements
    improvements = {
        'tpr_improvement': final_results['tpr_at_1'] - baseline_results['tpr_at_1'],
        'f1_improvement': final_results['best_f1'] - baseline_results['best_f1'],
        'sr_improvement': final_results['sr'] - baseline_results['sr'],
        'acc_improvement': final_results['detection_accuracy'] - baseline_results['detection_accuracy']
    }

    print(f"Improvements:")
    for metric, improvement in improvements.items():
        print(f"{metric}: {improvement:+.4f}")

    # Save quick results
    results_file = f"/root/autodl-tmp/WaterMarking/Dataset/quick_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        quick_results = {
            "test_config": "quick_test",
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "baseline_results": baseline_results,
            "final_results": final_results,
            "improvements": improvements
        }

        with open(results_file, 'w') as f:
            json.dump(quick_results, f, indent=2, default=str)
        print(f"\nQuick test results saved to: {results_file}")

    except Exception as e:
        print(f"Save error: {e}")

    print("\nQuick test completed!")
    return final_results


if __name__ == "__main__":
    quick_train_and_test()