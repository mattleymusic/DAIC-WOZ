import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

def load_chunk_features_and_labels(chunk_length, overlap, data_root="data"):
    """
    Load egemaps features at chunk level with corresponding participant labels.
    Each chunk inherits the label of its participant.
    
    Args:
        chunk_length (str): Chunk length (e.g., "5.0s")
        overlap (str): Overlap length (e.g., "2.5s")
        data_root (str): Root directory for data
    
    Returns:
        dict: Dictionary containing train, dev, and test data at chunk level
    """
    # Construct the feature directory path
    feature_dir = f"{data_root}/features/egemap/{chunk_length}_{overlap}_overlap"
    
    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory not found: {feature_dir}")
    
    # Load meta information for labels and splits
    meta_path = f"{data_root}/daic_woz/meta_info.csv"
    meta_df = pd.read_csv(meta_path)
    
    # Define participants to exclude
    excluded_participants = [
        '324_P', '323_P', '322_P', '321_P',  # String format
        '300_P', '305_P', '318_P', '341_P', '362_P', '451_P', '458_P', '480_P'  # Add _P suffix
    ]
    
    # Filter out excluded participants
    meta_df = meta_df[~meta_df['participant'].isin(excluded_participants)]
    
    print(f"Excluded {len(excluded_participants)} participants")
    print(f"Remaining participants: {len(meta_df)}")
    
    # Create mappings
    participant_to_target = dict(zip(meta_df['participant'], meta_df['target']))
    participant_to_subset = dict(zip(meta_df['participant'], meta_df['subset']))
    
    # Get all available participant directories in features
    available_participants = [d for d in os.listdir(feature_dir) 
                            if os.path.isdir(os.path.join(feature_dir, d))]
    
    print(f"Found {len(available_participants)} participant directories in features")
    print(f"Meta info contains {len(meta_df)} participants (after filtering)")
    
    # Find intersection of available features and meta_info participants
    meta_participants = set(meta_df['participant'])
    available_participants_set = set(available_participants)
    
    common_participants = meta_participants.intersection(available_participants_set)
    missing_from_features = meta_participants - available_participants_set
    missing_from_meta = available_participants_set - meta_participants
    
    print(f"Common participants: {len(common_participants)}")
    if missing_from_features:
        print(f"Participants in meta_info but missing features: {len(missing_from_features)}")
        print(f"Examples: {list(missing_from_features)[:5]}")
    if missing_from_meta:
        print(f"Participants with features but not in meta_info: {len(missing_from_meta)}")
        print(f"Examples: {list(missing_from_meta)[:5]}")
    
    # Initialize data structures for each split
    splits = {'train': [], 'dev': [], 'test': []}
    split_labels = {'train': [], 'dev': [], 'test': []}
    split_chunk_info = {'train': [], 'dev': [], 'test': []}  # Store chunk metadata
    
    # Process only participants that exist in both features and meta_info
    total_chunks = 0
    print("Loading chunk features with progress bar...")
    
    # Create progress bar for participants
    for participant in tqdm(common_participants, desc="Processing participants"):
        participant_dir = os.path.join(feature_dir, participant)
        feature_files = glob.glob(os.path.join(participant_dir, "*_features.csv"))
        
        if not feature_files:
            print(f"Warning: No feature files found for participant {participant}")
            continue
        
        # Get target label and subset for this participant
        target = participant_to_target[participant]
        subset = participant_to_subset[participant]
        
        # Load features from all chunks for this participant
        for file_path in feature_files:
            features_df = pd.read_csv(file_path)
            chunk_features = features_df.iloc[0].values  # Take first row
            
            # Extract chunk information from filename
            filename = os.path.basename(file_path)
            chunk_info = {
                'participant': participant,
                'filename': filename,
                'target': target,
                'subset': subset
            }
            
            # Add to appropriate split
            splits[subset].append(chunk_features)
            split_labels[subset].append(target)
            split_chunk_info[subset].append(chunk_info)
            total_chunks += 1
    
    print(f"Total chunks processed: {total_chunks}")
    
    # Convert to numpy arrays
    data = {}
    for subset in ['train', 'dev', 'test']:
        if splits[subset]:  # Only create arrays if data exists
            data[f'{subset}_features'] = np.array(splits[subset])
            data[f'{subset}_labels'] = np.array(split_labels[subset])
            data[f'{subset}_chunk_info'] = split_chunk_info[subset]
        else:
            data[f'{subset}_features'] = np.array([])
            data[f'{subset}_labels'] = np.array([])
            data[f'{subset}_chunk_info'] = []
    
    # Get feature names from the first file
    if common_participants:
        first_participant = list(common_participants)[0]
        first_file = glob.glob(os.path.join(feature_dir, first_participant, "*_features.csv"))[0]
        feature_names = pd.read_csv(first_file).columns.tolist()
        data['feature_names'] = feature_names
    else:
        data['feature_names'] = []
    
    return data

def train_chunk_level_svm(train_features, train_labels, dev_features, dev_labels, random_state=42):
    """
    Train an SVM model at chunk level with hyperparameter tuning.
    
    Args:
        train_features (np.array): Training feature matrix (chunk level)
        train_labels (np.array): Training labels (chunk level)
        dev_features (np.array): Development feature matrix (chunk level)
        dev_labels (np.array): Development labels (chunk level)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (trained_model, scaler, dev_predictions, dev_probabilities)
    """
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    dev_features_scaled = scaler.transform(dev_features)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weight_dict = dict(zip(np.unique(train_labels), class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    print(f"Training set class distribution: {np.bincount(train_labels)}")
    print(f"Dev set class distribution: {np.bincount(dev_labels)}")
    
    # Define parameter grid for hyperparameter tuning
    # Optimized for chunk-level classification
    param_grid = {
        'C': [0.1, 1, 10 ],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    # Calculate total number of combinations for progress tracking
    total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
    total_fits = total_combinations * 3  # 3-fold CV
    
    print(f"Performing hyperparameter tuning with grid search...")
    print(f"Parameter grid: {param_grid}")
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total model fits (with CV): {total_fits}")
    
    # Base SVM with class weights
    base_svm = SVC(
        random_state=random_state,
        class_weight=class_weight_dict,
        probability=True  # Enable probability estimates for later majority voting
    )
    
    # Grid search with stratified cross-validation
    grid_search = GridSearchCV(
        base_svm, 
        param_grid, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        scoring='f1_macro',  # Use macro F1 for balanced evaluation
        n_jobs=-1,
        verbose=2  # Increased verbosity for more detailed progress
    )
    
    # Fit the grid search with progress tracking
    print("Starting grid search...")
    start_time = time.time()
    
    grid_search.fit(train_features_scaled, train_labels)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Grid search completed in {elapsed_time:.2f} seconds")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Show detailed results with progress bar
    print("\nGrid search results:")
    results_list = []
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        results_list.append((mean_score, params))
    
    # Sort by score for better visualization
    results_list.sort(key=lambda x: x[0], reverse=True)
    
    print("Top 10 parameter combinations:")
    for i, (score, params) in enumerate(results_list[:10]):
        print(f"  {i+1:2d}. {params}: {score:.4f}")
    
    # Make predictions on dev set with best model
    print("Making predictions on dev set...")
    dev_predictions = best_model.predict(dev_features_scaled)
    dev_probabilities = best_model.predict_proba(dev_features_scaled)
    
    return best_model, scaler, dev_predictions, dev_probabilities

def evaluate_chunk_level_model(y_true, y_pred, chunk_info, split_name="dev"):
    """
    Evaluate the model performance at chunk level.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        chunk_info (list): List of chunk metadata dictionaries
        split_name (str): Name of the split being evaluated
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Get detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate per-participant statistics
    participant_stats = {}
    for i, info in enumerate(chunk_info):
        participant = info['participant']
        if participant not in participant_stats:
            participant_stats[participant] = {
                'total_chunks': 0,
                'correct_predictions': 0,
                'true_label': info['target']
            }
        participant_stats[participant]['total_chunks'] += 1
        if y_pred[i] == y_true[i]:
            participant_stats[participant]['correct_predictions'] += 1
    
    # Calculate participant-level accuracy
    participant_accuracies = []
    for participant, stats in participant_stats.items():
        if stats['total_chunks'] > 0:
            acc = stats['correct_predictions'] / stats['total_chunks']
            participant_accuracies.append(acc)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'split': split_name,
        'participant_stats': participant_stats,
        'mean_participant_accuracy': np.mean(participant_accuracies) if participant_accuracies else 0,
        'y_true': y_true,
        'y_pred': y_pred,
        'chunk_info': chunk_info
    }

def save_chunk_level_results(model, scaler, results, chunk_length, overlap, results_dir="data/results"):
    """
    Save the trained model, scaler, and results in a timestamped directory.
    
    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        results (dict): Evaluation results
        chunk_length (str): Chunk length used
        overlap (str): Overlap used
        results_dir (str): Directory to save results
    """
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"chunk_svm_{chunk_length}_{overlap}_overlap_{timestamp}"
    run_dir = os.path.join(results_dir, experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Chunk-Level SVM Training Results for {chunk_length}_{overlap}_overlap\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Split: {results['split']}\n")
        f.write(f"Chunk-level Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Chunk-level F1 Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"Chunk-level F1 Score (macro): {results['f1_macro']:.4f}\n")
        f.write(f"Mean Participant Accuracy: {results['mean_participant_accuracy']:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(
            results['y_true'], 
            results['y_pred'], 
            target_names=['Non-Depressed', 'Depressed']
        ))
        
        f.write("\nParticipant-level Statistics:\n")
        f.write("-" * 30 + "\n")
        for participant, stats in results['participant_stats'].items():
            acc = stats['correct_predictions'] / stats['total_chunks']
            f.write(f"{participant}: {acc:.3f} ({stats['correct_predictions']}/{stats['total_chunks']} chunks)\n")
    
    # Save chunk-level predictions for later majority voting
    predictions_df = pd.DataFrame({
        'participant': [info['participant'] for info in results['chunk_info']],
        'filename': [info['filename'] for info in results['chunk_info']],
        'true_label': results['y_true'],
        'predicted_label': results['y_pred'],
        'subset': [info['subset'] for info in results['chunk_info']]
    })
    
    predictions_path = os.path.join(run_dir, "chunk_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"Results saved to: {run_dir}")
    print(f"Model saved as: {model_path}")
    print(f"Scaler saved as: {scaler_path}")
    print(f"Results saved as: {results_path}")
    print(f"Chunk predictions saved as: {predictions_path}")
    
    return run_dir

def main():
    """
    Main function to run chunk-level SVM training.
    """
    # Configuration - change these parameters as needed
    chunk_length = "5.0s"  # Options: "3.0s", "5.0s", "10.0s", "20.0s", "30.0s"
    overlap = "2.5s"       # Options: "1.5s", "2.5s", "5.0s", "10.0s", "15.0s"
    
    print(f"Training Chunk-Level SVM with {chunk_length} chunks and {overlap} overlap")
    print("Using predefined train/dev/test splits from meta_info.csv")
    print("Handling class imbalance with balanced class weights")
    print("Performing hyperparameter tuning with GridSearchCV")
    print("Excluding specified participants")
    print("=" * 80)
    
    try:
        # Load chunk-level features and labels
        print("Loading chunk-level features and labels...")
        data = load_chunk_features_and_labels(chunk_length, overlap)
        
        # Extract data for each split
        train_features = data['train_features']
        train_labels = data['train_labels']
        dev_features = data['dev_features']
        dev_labels = data['dev_labels']
        test_features = data['test_features']
        test_labels = data['test_labels']
        feature_names = data['feature_names']
        
        print(f"\nLoaded features with {len(feature_names)} dimensions")
        print(f"Train set: {len(train_features)} chunks")
        print(f"Dev set: {len(dev_features)} chunks")
        print(f"Test set: {len(test_features)} chunks")
        
        if len(train_features) > 0:
            print(f"Train label distribution: {np.bincount(train_labels)}")
        if len(dev_features) > 0:
            print(f"Dev label distribution: {np.bincount(dev_labels)}")
        if len(test_features) > 0:
            print(f"Test label distribution: {np.bincount(test_labels)}")
        
        # Check if we have enough data
        if len(train_features) == 0:
            raise ValueError("No training data available!")
        if len(dev_features) == 0:
            raise ValueError("No development data available!")
        
        # Train chunk-level SVM model
        print("\nTraining chunk-level SVM model with hyperparameter tuning...")
        model, scaler, dev_predictions, dev_probabilities = train_chunk_level_svm(
            train_features, train_labels, dev_features, dev_labels
        )
        
        # Evaluate on dev set
        print("Evaluating model on dev set...")
        dev_results = evaluate_chunk_level_model(dev_labels, dev_predictions, data['dev_chunk_info'], "dev")
        
        # Print dev results
        print(f"\nDev Set Results (Chunk Level):")
        print(f"Chunk-level Accuracy: {dev_results['accuracy']:.4f}")
        print(f"Chunk-level F1 Score (weighted): {dev_results['f1_weighted']:.4f}")
        print(f"Chunk-level F1 Score (macro): {dev_results['f1_macro']:.4f}")
        print(f"Mean Participant Accuracy: {dev_results['mean_participant_accuracy']:.4f}")
        
        # Evaluate on test set (if available)
        if len(test_features) > 0:
            print("\nEvaluating model on test set...")
            test_features_scaled = scaler.transform(test_features)
            test_predictions = model.predict(test_features_scaled)
            test_probabilities = model.predict_proba(test_features_scaled)
            test_results = evaluate_chunk_level_model(test_labels, test_predictions, data['test_chunk_info'], "test")
            
            print(f"\nTest Set Results (Chunk Level):")
            print(f"Chunk-level Accuracy: {test_results['accuracy']:.4f}")
            print(f"Chunk-level F1 Score (weighted): {test_results['f1_weighted']:.4f}")
            print(f"Chunk-level F1 Score (macro): {test_results['f1_macro']:.4f}")
            print(f"Mean Participant Accuracy: {test_results['mean_participant_accuracy']:.4f}")
        else:
            test_results = None
        
        # Save results
        print("\nSaving results...")
        run_dir = save_chunk_level_results(model, scaler, dev_results, chunk_length, overlap)
        
        # Save test results separately if available
        if test_results:
            test_results_path = os.path.join(run_dir, "test_results.txt")
            with open(test_results_path, 'w') as f:
                f.write(f"Chunk-Level SVM Test Results for {chunk_length}_{overlap}_overlap\n")
                f.write(f"Run timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Split: {test_results['split']}\n")
                f.write(f"Chunk-level Accuracy: {test_results['accuracy']:.4f}\n")
                f.write(f"Chunk-level F1 Score (weighted): {test_results['f1_weighted']:.4f}\n")
                f.write(f"Chunk-level F1 Score (macro): {test_results['f1_macro']:.4f}\n")
                f.write(f"Mean Participant Accuracy: {test_results['mean_participant_accuracy']:.4f}\n\n")
                f.write("Detailed Classification Report:\n")
                f.write("-" * 50 + "\n")
                f.write(classification_report(
                    test_results['y_true'], 
                    test_results['y_pred'], 
                    target_names=['Non-Depressed', 'Depressed']
                ))
            print(f"Test results saved as: {test_results_path}")
        
        print(f"\nChunk-level training completed successfully!")
        print(f"All results saved in: {run_dir}")
        print(f"Ready for majority voting implementation!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()