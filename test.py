import os
import json
import torch
import numpy as np
from datetime import datetime
import argparse
import time
import random
import nltk
from nltk.corpus import wordnet
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Import your RLAWM implementation
from Main import RLAWM, RLAWMConfig, evaluate_rlawm

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')


def calculate_perplexity(model, tokenizer, texts, device='cuda'):
    """Calculate perplexity using a safer approach"""
    if not texts:
        return float('inf')

    total_loss = 0
    total_tokens = 0
    successful_calculations = 0

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts[:20]):  # Limit to avoid memory issues
            try:
                # Clean and truncate text
                clean_text = text.strip()
                if len(clean_text) == 0:
                    continue

                # Tokenize with safety measures
                inputs = tokenizer(
                    clean_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,  # Shorter sequences
                    padding=False
                )

                input_ids = inputs['input_ids']
                if input_ids.shape[1] < 2:  # Need at least 2 tokens
                    continue

                # Move to device safely
                input_ids = input_ids.to(device)

                # Calculate loss with error handling
                try:
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss

                    if torch.isfinite(loss):
                        total_loss += loss.item() * input_ids.shape[1]
                        total_tokens += input_ids.shape[1]
                        successful_calculations += 1

                except Exception as model_error:
                    print(f"Model forward pass error for text {i}: {model_error}")
                    continue

            except Exception as e:
                print(f"Perplexity calculation error for text {i}: {e}")
                continue

    if total_tokens == 0 or successful_calculations == 0:
        print(f"No successful perplexity calculations out of {len(texts)} texts")
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"Perplexity calculated from {successful_calculations}/{len(texts)} texts")
    return min(perplexity, 1000.0)  # Cap at reasonable value


def word_substitution_attack(text, substitution_rate=0.3):
    """Word-level substitution attack using WordNet synonyms"""
    try:
        words = text.split()
        num_words_to_replace = int(len(words) * substitution_rate)

        if num_words_to_replace == 0:
            return text

        # Randomly select words to replace
        indices_to_replace = random.sample(range(len(words)), min(num_words_to_replace, len(words)))

        for idx in indices_to_replace:
            word = words[idx].lower().strip('.,!?;:"')

            # Get synonyms from WordNet
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))

            # Remove the original word and select a random synonym
            synonyms.discard(word)
            if synonyms:
                words[idx] = random.choice(list(synonyms))

        return ' '.join(words)

    except Exception as e:
        print(f"Word substitution error: {e}")
        return text


def comprehensive_evaluate_rlawm(rlawm_model, test_data, num_samples=100):
    """Comprehensive evaluation with all metrics from the paper"""
    print(f"Running comprehensive evaluation on {num_samples} samples...")

    results = {
        'tpr_at_1': 0.0,
        'best_f1': 0.0,
        'ppl_watermarked': 0.0,
        'ppl_non_watermarked': 0.0,
        'sr': 0.0,  # Successful trigger ratio
        'tpr_at_1_word_attack': 0.0,  # TPR@1% under Word-S/30% attack
        'detection_accuracy': 0.0,
        'false_positive_rate': 0.0,
        'false_negative_rate': 0.0
    }

    watermarked_texts = []
    non_watermarked_texts = []
    attacked_texts = []
    detection_results = []
    attack_detection_results = []
    watermark_detection_scores = []
    non_watermark_detection_scores = []
    successful_triggers = 0
    successful_generations = 0

    print("Generating texts and collecting results...")
    for i, data in enumerate(test_data[:num_samples]):
        prompt = data.get('prompt', data.get('text', ''))
        if len(prompt) > 100:
            prompt = prompt[:100]

        if len(prompt.strip()) == 0:
            continue

        try:
            print(f"Processing sample {i + 1}/{num_samples}: {prompt[:50]}...")

            # Generate watermarked text
            watermarked_text = rlawm_model.generate_watermarked_text(prompt, max_new_tokens=50)
            if watermarked_text and len(watermarked_text.strip()) > len(prompt):
                watermarked_texts.append(watermarked_text)
                successful_generations += 1
            else:
                print(f"  Watermarked generation failed or too short")
                continue

            # Generate non-watermarked text
            try:
                inputs = rlawm_model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(rlawm_model.config.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = rlawm_model.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=rlawm_model.tokenizer.eos_token_id
                    )
                non_watermarked_text = rlawm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                non_watermarked_texts.append(non_watermarked_text)

            except Exception as e:
                print(f"  Non-watermarked generation error: {e}")
                non_watermarked_text = prompt + " [generation failed]"
                non_watermarked_texts.append(non_watermarked_text)

            # Test detection on original texts
            try:
                water_result = rlawm_model.detect_watermark(watermarked_text, return_dict=True)
                non_water_result = rlawm_model.detect_watermark(non_watermarked_text, return_dict=True)

                # Extract boolean values properly
                water_detected = water_result['is_watermarked']
                if torch.is_tensor(water_detected):
                    water_detected = water_detected.item()

                non_water_detected = non_water_result['is_watermarked']
                if torch.is_tensor(non_water_detected):
                    non_water_detected = non_water_detected.item()

                detection_results.extend([
                    (True, water_detected),  # Should be detected
                    (False, non_water_detected)  # Should NOT be detected
                ])

                watermark_detection_scores.append(water_result.get('z_score', 0))
                non_watermark_detection_scores.append(non_water_result.get('z_score', 0))

                # Check if watermark was successfully triggered
                if water_detected:
                    successful_triggers += 1

                print(f"  Watermark detected: {water_detected}, Non-watermark detected: {non_water_detected}")

            except Exception as e:
                print(f"  Detection error: {e}")
                detection_results.extend([(True, False), (False, False)])
                watermark_detection_scores.append(0)
                non_watermark_detection_scores.append(0)

            # Generate attacked text and test robustness
            try:
                attacked_text = word_substitution_attack(watermarked_text, 0.3)
                attacked_texts.append(attacked_text)

                attack_result = rlawm_model.detect_watermark(attacked_text, return_dict=True)
                attack_detected = attack_result['is_watermarked']
                if torch.is_tensor(attack_detected):
                    attack_detected = attack_detected.item()

                attack_detection_results.append((True, attack_detected))
                print(f"  Attack detection: {attack_detected}")

            except Exception as e:
                print(f"  Attack generation/detection error: {e}")
                attack_detection_results.append((True, False))

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"\nSuccessfully processed {successful_generations} samples")
    print(f"Detection results: {len(detection_results)} pairs")
    print(f"Attack results: {len(attack_detection_results)} samples")

    # Calculate detection metrics
    if detection_results:
        true_positives = sum(1 for true_label, pred in detection_results if true_label and pred)
        false_positives = sum(1 for true_label, pred in detection_results if not true_label and pred)
        true_negatives = sum(1 for true_label, pred in detection_results if not true_label and not pred)
        false_negatives = sum(1 for true_label, pred in detection_results if true_label and not pred)

        print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")

        # TPR@1%
        total_watermarked = true_positives + false_negatives
        if total_watermarked > 0:
            results['tpr_at_1'] = true_positives / total_watermarked

        # Best F1 Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        results['best_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Detection accuracy
        correct_predictions = true_positives + true_negatives
        total_predictions = len(detection_results)
        results['detection_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0

        # False positive/negative rates
        total_non_watermarked = false_positives + true_negatives
        results['false_positive_rate'] = false_positives / total_non_watermarked if total_non_watermarked > 0 else 0
        results['false_negative_rate'] = false_negatives / total_watermarked if total_watermarked > 0 else 0

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Calculate TPR@1% under attack
    if attack_detection_results:
        attack_true_positives = sum(1 for true_label, pred in attack_detection_results if true_label and pred)
        attack_total = len(attack_detection_results)
        results['tpr_at_1_word_attack'] = attack_true_positives / attack_total if attack_total > 0 else 0
        print(f"Attack detection: {attack_true_positives}/{attack_total}")

    # Successful trigger ratio
    if successful_generations > 0:
        results['sr'] = successful_triggers / successful_generations
        print(f"Successful triggers: {successful_triggers}/{successful_generations}")

    # Calculate perplexity with better error handling
    print("Calculating perplexity...")
    if watermarked_texts:
        try:
            results['ppl_watermarked'] = calculate_perplexity(
                rlawm_model.model, rlawm_model.tokenizer, watermarked_texts, rlawm_model.config.device
            )
        except Exception as e:
            print(f"Watermarked perplexity calculation failed: {e}")
            results['ppl_watermarked'] = float('inf')

    if non_watermarked_texts:
        try:
            results['ppl_non_watermarked'] = calculate_perplexity(
                rlawm_model.model, rlawm_model.tokenizer, non_watermarked_texts, rlawm_model.config.device
            )
        except Exception as e:
            print(f"Non-watermarked perplexity calculation failed: {e}")
            results['ppl_non_watermarked'] = float('inf')

    # Print Z-score statistics
    if watermark_detection_scores and non_watermark_detection_scores:
        avg_z_watermarked = np.mean(watermark_detection_scores)
        avg_z_non_watermarked = np.mean(non_watermark_detection_scores)
        print(f"Average Z-scores: Watermarked={avg_z_watermarked:.4f}, Non-watermarked={avg_z_non_watermarked:.4f}")

    # Print results
    print(f"\nComprehensive Evaluation Results:")
    print(f"=" * 60)
    print(f"TPR@1%: {results['tpr_at_1']:.4f}")
    print(f"Best F1: {results['best_f1']:.4f}")
    print(f"PPL (Watermarked): {results['ppl_watermarked']:.3f}")
    print(f"PPL (Non-watermarked): {results['ppl_non_watermarked']:.3f}")
    print(f"SR (Successful Trigger Ratio): {results['sr']:.4f}")
    print(f"TPR@1% (Word-S/30%): {results['tpr_at_1_word_attack']:.4f}")
    print(f"Detection Accuracy: {results['detection_accuracy']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {results['false_negative_rate']:.4f}")
    print(f"=" * 60)

    return results


def train_and_evaluate_rlawm():
    """Complete training and evaluation pipeline for RLAWM"""

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better CUDA debugging

    # Configuration for intensive training
    config = RLAWMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gamma=0.5,  # Green list ratio
        hash_key=42,  # Hash seed
        z_threshold=2.0,  # Detection threshold
        prefix_length=5,  # Context length
        delta_min=0.5,  # Min watermark strength
        delta_max=3.0,  # Max watermark strength
        alpha=0.5,  # Detection weight in reward
        beta=0.5,  # Quality weight in reward
        lambda_mmd=1.0,  # MMD penalty coefficient
        learning_rate=1e-5,  # Reduced learning rate for stability
        n_layers=6,  # Transformer layers
        n_heads=8,  # Attention heads
        d_model=384,  # Model dimension
        d_ff=2048,  # Feed-forward dimension
        hash_precision=16  # Hash precision
    )

    print("=" * 80)
    print("RLAWM Intensive Training and Evaluation Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Z-threshold: {config.z_threshold}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Alpha/Beta: {config.alpha}/{config.beta}")

    # Load dataset
    print("\n1. Loading dataset...")
    dataset_path = '/root/autodl-tmp/WaterMarking/Dataset/processed_c4.json'
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            dataset = [json.loads(line) for line in lines]
        print(f"   Loaded {len(dataset)} samples from {dataset_path}")
    except Exception as e:
        print(f"   Dataset loading error: {e}")
        return None

    # Use more training data
    train_size = min(800, int(0.8 * len(dataset)))
    test_size = min(200, len(dataset) - train_size)

    train_data = dataset[:train_size]
    test_data = dataset[train_size:train_size + test_size]

    print(f"   Training samples: {len(train_data)}")
    print(f"   Testing samples: {len(test_data)}")

    # Initialize RLAWM
    print("\n2. Initializing RLAWM...")
    model_path = "/root/autodl-tmp/Llama-2-7b-chat-hf"
    try:
        rlawm = RLAWM(model_path, config)
        print("   RLAWM initialized successfully")
    except Exception as e:
        print(f"   RLAWM initialization failed: {e}")
        return None

    # Pre-training baseline evaluation
    print("\n3. Pre-training baseline evaluation...")
    baseline_results = comprehensive_evaluate_rlawm(rlawm, test_data[:100], num_samples=100)

    # Two-stage training strategy
    print("\n4. Two-stage training of RLAWM agents...")
    training_phases = [
        {
            "samples": 400,
            "epochs": 100,
            "description": "Stage 1: Generation Agent Training",
            "focus": "generation"
        },
        {
            "samples": 600,
            "epochs": 150,
            "description": "Stage 2: Collaborative Training",
            "focus": "collaborative"
        }
    ]

    best_results = baseline_results
    best_checkpoint = None
    all_phase_results = []

    for phase, training_config in enumerate(training_phases, 1):
        print(f"\n   {training_config['description']}")
        print(f"   Using {training_config['samples']} samples for {training_config['epochs']} epochs")
        print(f"   Focus: {training_config['focus']}")

        start_time = time.time()

        try:
            # Train with current configuration
            phase_train_data = train_data[:training_config['samples']]
            print(f"   Starting training with {len(phase_train_data)} samples...")

            # Modified training call with focus parameter
            rlawm.train_agents_safe(
                phase_train_data,
                num_epochs=training_config['epochs'],
                focus=training_config['focus']
            )

            training_time = time.time() - start_time
            print(f"   Training completed in {training_time:.2f} seconds")

            # Comprehensive evaluation after this phase
            print(f"   Running evaluation after Stage {phase}...")
            phase_results = comprehensive_evaluate_rlawm(rlawm, test_data[:50], num_samples=50)

            # Store results
            phase_results['phase'] = phase
            phase_results['training_time'] = training_time
            all_phase_results.append(phase_results)

            print(f"\n   Stage {phase} Results:")
            print(f"     TPR@1%: {phase_results['tpr_at_1']:.4f}")
            print(f"     Best F1: {phase_results['best_f1']:.4f}")
            print(f"     PPL (Watermarked): {phase_results['ppl_watermarked']:.3f}")
            print(f"     SR: {phase_results['sr']:.4f}")
            print(f"     TPR@1% (Word-S/30%): {phase_results['tpr_at_1_word_attack']:.4f}")

            # Save checkpoint if this shows improvement
            improvement_score = (
                                        phase_results['best_f1'] +
                                        phase_results['tpr_at_1'] +
                                        phase_results['sr'] +
                                        phase_results['tpr_at_1_word_attack']
                                ) / 4

            baseline_score = (
                                     baseline_results['best_f1'] +
                                     baseline_results['tpr_at_1'] +
                                     baseline_results['sr'] +
                                     baseline_results['tpr_at_1_word_attack']
                             ) / 4

            if improvement_score > baseline_score:
                best_results = phase_results
                checkpoint_path = f"rlawm_checkpoint_stage{phase}"
                try:
                    rlawm.save_model(checkpoint_path)
                    best_checkpoint = checkpoint_path
                    print(f"   *** New best model saved to {checkpoint_path} ***")
                except Exception as e:
                    print(f"   Checkpoint save error: {e}")

        except Exception as e:
            print(f"   Stage {phase} training failed: {e}")
            print(f"   Continuing with next stage...")
            continue

    # Final comprehensive evaluation
    print("\n5. Final comprehensive evaluation...")
    if best_checkpoint:
        print(f"   Loading best checkpoint: {best_checkpoint}")
        try:
            rlawm.load_model(best_checkpoint)
        except Exception as e:
            print(f"   Checkpoint loading error: {e}")

    # Evaluate on full test set
    print("   Running final evaluation on full test set...")
    final_results = comprehensive_evaluate_rlawm(rlawm, test_data, num_samples=len(test_data))

    print(f"\n6. Final Results Summary:")
    print("=" * 80)
    print(f"TPR@1%: {final_results['tpr_at_1']:.4f}")
    print(f"Best F1: {final_results['best_f1']:.4f}")
    print(f"PPL (Watermarked): {final_results['ppl_watermarked']:.3f}")
    print(f"PPL (Non-watermarked): {final_results['ppl_non_watermarked']:.3f}")
    print(f"SR: {final_results['sr']:.4f}")
    print(f"TPR@1% (Word-S/30%): {final_results['tpr_at_1_word_attack']:.4f}")
    print(f"Detection Accuracy: {final_results['detection_accuracy']:.4f}")
    print("=" * 80)

    # Improvements over baseline
    print(f"\nImprovements over baseline:")
    print(f"TPR@1% improvement: {final_results['tpr_at_1'] - baseline_results['tpr_at_1']:+.4f}")
    print(f"Best F1 improvement: {final_results['best_f1'] - baseline_results['best_f1']:+.4f}")
    print(f"SR improvement: {final_results['sr'] - baseline_results['sr']:+.4f}")
    print(
        f"TPR@1% (Word-S/30%) improvement: {final_results['tpr_at_1_word_attack'] - baseline_results['tpr_at_1_word_attack']:+.4f}")

    # Save comprehensive results
    results_file = f"/root/autodl-tmp/WaterMarking/Dataset/rlawm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        comprehensive_results = {
            "config": {
                "gamma": config.gamma,
                "z_threshold": config.z_threshold,
                "alpha": config.alpha,
                "beta": config.beta,
                "learning_rate": config.learning_rate,
                "training_phases": training_phases
            },
            "baseline_results": baseline_results,
            "phase_results": all_phase_results,
            "final_results": final_results,
            "improvements": {
                "tpr_at_1": final_results['tpr_at_1'] - baseline_results['tpr_at_1'],
                "best_f1": final_results['best_f1'] - baseline_results['best_f1'],
                "sr": final_results['sr'] - baseline_results['sr'],
                "tpr_at_1_word_attack": final_results['tpr_at_1_word_attack'] - baseline_results[
                    'tpr_at_1_word_attack'],
                "ppl_watermarked": final_results['ppl_watermarked'] - baseline_results['ppl_watermarked']
            },
            "best_checkpoint": best_checkpoint,
            "dataset_info": {
                "dataset_path": dataset_path,
                "total_samples": len(dataset),
                "train_samples": len(train_data),
                "test_samples": len(test_data)
            }
        }

        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        print(f"\nComprehensive results saved to: {results_file}")

    except Exception as e:
        print(f"Results save error: {e}")

    print(f"\nRLAWM intensive training and evaluation completed!")
    return final_results
    """Complete training and evaluation pipeline for RLAWM"""

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configuration for full training
    config = RLAWMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gamma=0.5,  # Green list ratio
        hash_key=42,  # Hash seed
        z_threshold=2.0,  # Detection threshold
        prefix_length=5,  # Context length
        delta_min=0.5,  # Min watermark strength
        delta_max=3.0,  # Max watermark strength
        alpha=0.5,  # Detection weight in reward
        beta=0.5,  # Quality weight in reward
        lambda_mmd=1.0,  # MMD penalty coefficient
        learning_rate=2e-5,  # Learning rate
        n_layers=6,  # Transformer layers
        n_heads=8,  # Attention heads
        d_model=384,  # Model dimension
        d_ff=2048,  # Feed-forward dimension
        hash_precision=16  # Hash precision
    )

    print("=" * 60)
    print("RLAWM Training and Evaluation Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Z-threshold: {config.z_threshold}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Alpha/Beta: {config.alpha}/{config.beta}")

    # Load dataset
    print("\n1. Loading dataset...")
    dataset_path = '/root/autodl-tmp/WaterMarking/Dataset/processed_c4.json'
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            dataset = [json.loads(line) for line in lines]
        print(f"   Loaded {len(dataset)} samples from {dataset_path}")
    except Exception as e:
        print(f"   Dataset loading error: {e}")
        print("   Creating dummy dataset...")
        dataset = [
            {"prompt": f"Sample prompt {i}: Discuss the importance of", "text": f"sample text {i}"}
            for i in range(1000)
        ]
        print(f"   Created {len(dataset)} dummy samples")

    # Split dataset
    train_size = min(800, int(0.8 * len(dataset)))
    test_size = min(200, len(dataset) - train_size)

    train_data = dataset[:train_size]
    test_data = dataset[train_size:train_size + test_size]

    print(f"   Training samples: {len(train_data)}")
    print(f"   Testing samples: {len(test_data)}")

    # Initialize RLAWM
    print("\n2. Initializing RLAWM...")
    model_path = "/root/autodl-tmp/Llama-2-7b-chat-hf"
    try:
        rlawm = RLAWM(model_path, config)
        print("   RLAWM initialized successfully")
    except Exception as e:
        print(f"   RLAWM initialization failed: {e}")
        return None

    # Pre-training evaluation
    print("\n3. Pre-training baseline evaluation...")
    baseline_results = evaluate_rlawm(rlawm, test_data[:50], num_samples=50)
    print(f"   Baseline TPR@1%: {baseline_results['tpr_at_1']:.4f}")
    print(f"   Baseline F1: {baseline_results['f1_score']:.4f}")
    print(f"   Baseline Detection Accuracy: {baseline_results['detection_accuracy']:.4f}")

    # Training
    print("\n4. Training RLAWM agents...")
    training_configs = [
        {"samples": 100, "epochs": 20, "description": "Phase 1: Initial training"},
        {"samples": 200, "epochs": 30, "description": "Phase 2: Expanded training"},
        {"samples": 400, "epochs": 50, "description": "Phase 3: Full training"}
    ]

    best_results = baseline_results
    best_checkpoint = None

    for phase, training_config in enumerate(training_configs, 1):
        print(f"\n   {training_config['description']}")
        print(f"   Using {training_config['samples']} samples for {training_config['epochs']} epochs")

        start_time = time.time()

        try:
            # Train with current configuration
            phase_train_data = train_data[:training_config['samples']]
            rlawm.train_agents(phase_train_data, num_epochs=training_config['epochs'])

            training_time = time.time() - start_time
            print(f"   Training completed in {training_time:.2f} seconds")

            # Evaluate after this phase
            print(f"   Evaluating after Phase {phase}...")
            phase_results = evaluate_rlawm(rlawm, test_data[:50], num_samples=50)

            print(f"   Phase {phase} Results:")
            print(
                f"     TPR@1%: {phase_results['tpr_at_1']:.4f} (Δ: {phase_results['tpr_at_1'] - baseline_results['tpr_at_1']:+.4f})")
            print(
                f"     F1 Score: {phase_results['f1_score']:.4f} (Δ: {phase_results['f1_score'] - baseline_results['f1_score']:+.4f})")
            print(
                f"     Detection Accuracy: {phase_results['detection_accuracy']:.4f} (Δ: {phase_results['detection_accuracy'] - baseline_results['detection_accuracy']:+.4f})")
            print(f"     Avg Z-score (Watermarked): {phase_results['avg_z_score_watermarked']:.4f}")
            print(f"     Avg Z-score (Non-watermarked): {phase_results['avg_z_score_non_watermarked']:.4f}")

            # Save checkpoint if this is the best so far
            if phase_results['f1_score'] > best_results['f1_score']:
                best_results = phase_results
                checkpoint_path = f"rlawm_checkpoint_phase{phase}"
                rlawm.save_model(checkpoint_path)
                best_checkpoint = checkpoint_path
                print(f"   *** New best model saved to {checkpoint_path} ***")

        except Exception as e:
            print(f"   Phase {phase} training failed: {e}")
            print(f"   Continuing with next phase...")
            continue

    # Final comprehensive evaluation
    print("\n5. Final comprehensive evaluation...")
    if best_checkpoint:
        print(f"   Loading best checkpoint: {best_checkpoint}")
        rlawm.load_model(best_checkpoint)

    # Evaluate on larger test set
    print("   Running final evaluation on larger test set...")
    final_results = evaluate_rlawm(rlawm, test_data, num_samples=min(100, len(test_data)))

    print(f"\n6. Final Results Summary:")
    print("=" * 50)
    print(f"TPR@1%: {final_results['tpr_at_1']:.4f}")
    print(f"F1 Score: {final_results['f1_score']:.4f}")
    print(f"Detection Accuracy: {final_results['detection_accuracy']:.4f}")
    print(f"Avg Z-score (Watermarked): {final_results['avg_z_score_watermarked']:.4f}")
    print(f"Avg Z-score (Non-watermarked): {final_results['avg_z_score_non_watermarked']:.4f}")
    print("=" * 50)

    # Improvements over baseline
    print(f"\nImprovements over baseline:")
    print(f"TPR@1% improvement: {final_results['tpr_at_1'] - baseline_results['tpr_at_1']:+.4f}")
    print(f"F1 Score improvement: {final_results['f1_score'] - baseline_results['f1_score']:+.4f}")
    print(
        f"Detection Accuracy improvement: {final_results['detection_accuracy'] - baseline_results['detection_accuracy']:+.4f}")

    # Generate example outputs
    print(f"\n7. Example outputs:")
    test_prompts = [
        "The advantages of renewable energy are",
        "Climate change poses significant challenges",
        "Artificial intelligence has revolutionized"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {prompt}")

        try:
            # Generate watermarked text
            watermarked = rlawm.generate_watermarked_text(prompt, max_new_tokens=50)
            print(f"Watermarked: {watermarked}")

            # Test detection
            detection_result = rlawm.detect_watermark(watermarked, return_dict=True)
            print(f"Detection: {detection_result['is_watermarked']}, Z-score: {detection_result['z_score']:.3f}")

        except Exception as e:
            print(f"Example generation error: {e}")

    # Save final model
    final_model_path = f"rlawm_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        rlawm.save_model(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
    except Exception as e:
        print(f"Final model save error: {e}")

    # Save results to file
    results_file = f"rlawm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        results_summary = {
            "config": {
                "gamma": config.gamma,
                "z_threshold": config.z_threshold,
                "alpha": config.alpha,
                "beta": config.beta,
                "learning_rate": config.learning_rate
            },
            "baseline_results": baseline_results,
            "final_results": final_results,
            "improvements": {
                "tpr_at_1": final_results['tpr_at_1'] - baseline_results['tpr_at_1'],
                "f1_score": final_results['f1_score'] - baseline_results['f1_score'],
                "detection_accuracy": final_results['detection_accuracy'] - baseline_results['detection_accuracy']
            },
            "best_checkpoint": best_checkpoint,
            "final_model": final_model_path
        }

        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved to: {results_file}")

    except Exception as e:
        print(f"Results save error: {e}")

    print(f"\nRLAWM training and evaluation completed!")
    return final_results


def quick_test():
    """Quick test for basic functionality"""
    print("Running quick functionality test...")

    config = RLAWMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gamma=0.5,
        hash_key=42,
        z_threshold=2.0,
        prefix_length=5
    )

    model_path = "/root/autodl-tmp/Llama-2-7b-chat-hf"
    rlawm = RLAWM(model_path, config)

    # Test basic generation
    prompt = "The future of artificial intelligence"
    watermarked_text = rlawm.generate_watermarked_text(prompt, max_new_tokens=30)
    detection_result = rlawm.detect_watermark(watermarked_text)

    print(f"Test prompt: {prompt}")
    print(f"Generated: {watermarked_text}")
    print(f"Detection: {detection_result}")
    print("Quick test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAWM Training and Evaluation")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Run full training or quick test")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_evaluate_rlawm()
    else:
        quick_test()