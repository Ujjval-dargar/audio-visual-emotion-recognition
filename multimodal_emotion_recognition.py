"""
================================================================================
🎭 MULTIMODAL EMOTION RECOGNITION SYSTEM
================================================================================
Using Facial Images and Speech Signals

Author: Ujjval Dargar
Date: January 2026

This project implements a multimodal emotion recognition system that combines
facial image features and audio speech features to classify emotions into
6 categories: Anger, Disgust, Fear, Happy, Neutral, Sad.

Dataset: CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
================================================================================
"""

# ==============================================================================
# STEP 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import librosa
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import pickle
import time

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': './data/emotion',
    'images_dir': './data/emotion/images',
    'audio_dir': './data/emotion/audio',
    'metadata_path': './data/emotion/dataset_metadata.csv',
    'target_sr': 16000,          # Target sampling rate for audio
    'image_size': (128, 128),    # Image resize dimensions
    'hog_win_size': (128, 128),  # HOG window size
    'test_size': 0.2,            # Train/test split ratio
    'random_state': 42,          # For reproducibility
    'pca_variance_threshold': 0.95,  # Keep 95% variance
}

# Emotion mapping
EMOTION_MAP = {
    'ANG': 'Anger',
    'DIS': 'Disgust', 
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad'
}

print("="*60)
print("🎭 MULTIMODAL EMOTION RECOGNITION SYSTEM")
print("="*60)


# ==============================================================================
# STEP 2: DATA LOADING AND PREPARATION
# ==============================================================================

print("\n" + "="*60)
print("STEP 2: DATA LOADING AND PREPARATION")
print("="*60)

def load_csv(path_to_csv):
    """Load metadata CSV file."""
    df = pd.read_csv(path_to_csv)
    return df

def load_image(path_to_image_file):
    """Load an image and convert to RGB."""
    img = cv2.imread(path_to_image_file)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_audio(path_to_audio_file):
    """Load audio file and return (sampling_rate, audio_data)."""
    audio, sr = librosa.load(path_to_audio_file, sr=None)
    return (sr, audio)

def prepare_dataset(metadata_path, test_size=0.2, random_state=42):
    """
    Load metadata and create train/test splits with stratification.
    
    Returns:
        train_df, test_df: DataFrames for training and testing
    """
    print("\n📂 Loading metadata...")
    metadata = load_csv(metadata_path)
    print(f"   Total samples: {len(metadata)}")
    
    # Display class distribution
    print("\n📊 Class Distribution:")
    class_dist = metadata['emotion_text'].value_counts()
    for emotion, count in class_dist.items():
        print(f"   {emotion}: {count} samples")
    
    # Stratified train/test split
    print(f"\n✂️ Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
    train_df, test_df = train_test_split(
        metadata,
        test_size=test_size,
        stratify=metadata['emotion_text'],
        random_state=random_state
    )
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Testing samples: {len(test_df)}")
    
    return train_df, test_df, metadata

# Load and prepare data
train_df, test_df, full_metadata = prepare_dataset(
    CONFIG['metadata_path'],
    CONFIG['test_size'],
    CONFIG['random_state']
)


# ==============================================================================
# STEP 3: IMAGE FEATURE EXTRACTION
# ==============================================================================

print("\n" + "="*60)
print("STEP 3: IMAGE FEATURE EXTRACTION")
print("="*60)

def resize_image(image, size=(128, 128)):
    """Resize image to specified dimensions."""
    return cv2.resize(image, size)

def convert_to_grayscale(image):
    """Convert RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_histogram(image_array):
    """Compute pixel intensity histogram (256 bins)."""
    hist = cv2.calcHist([image_array], [0], None, [256], [0, 256]).flatten()
    # Normalize histogram
    hist = hist / (np.sum(hist) + 1e-8)
    return hist

def get_lbp_features(img_array):
    """
    Compute Local Binary Pattern features.
    Returns histogram of LBP image (256 bins).
    """
    H, W = img_array.shape
    lbp = np.zeros((H, W), dtype=np.uint8)
    
    # 8 neighbors directions: starting from left, going clockwise
    directions = [(0, -1), (-1, -1), (-1, 0), (-1, 1), 
                  (0, 1), (1, 1), (1, 0), (1, -1)]
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            center = img_array[i, j]
            val = 0
            for idx, (di, dj) in enumerate(directions):
                if img_array[i + di, j + dj] >= center:
                    val += weights[idx]
            lbp[i, j] = val
    
    # Return histogram of LBP image
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
    return lbp_hist

def get_hog_features(img, win_size=(128, 128), block_size=(16, 16),
                     block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    """
    Extract HOG (Histogram of Oriented Gradients) features.
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        gray_img = convert_to_grayscale(img)
    else:
        gray_img = img
    
    # Resize to window size
    gray_img = resize_image(gray_img, win_size)
    
    # Create HOG descriptor
    hog_desc = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog_desc.compute(gray_img)
    
    return hog_features.flatten()

def get_sobel_histogram(img_array):
    """
    Compute Sobel edge features and return histogram of magnitude.
    """
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1)
    
    # Compute magnitude
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Convert to uint8 and compute histogram
    magnitude_uint8 = cv2.convertScaleAbs(magnitude)
    hist = cv2.calcHist([magnitude_uint8], [0], None, [256], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-8)
    
    return hist

def extract_image_features(image_path, image_size=(128, 128)):
    """
    Extract all image features for a single image.
    
    Features:
        - Histogram (256)
        - LBP histogram (256)
        - HOG features (~1764)
        - Sobel histogram (256)
    
    Returns:
        np.ndarray: Concatenated feature vector
    """
    # Load and preprocess image
    img = load_image(image_path)
    if img is None:
        return None
    
    img_resized = resize_image(img, image_size)
    img_gray = convert_to_grayscale(img_resized)
    
    # Extract features
    histogram = get_histogram(img_gray)           # 256
    lbp_hist = get_lbp_features(img_gray)         # 256
    hog_feat = get_hog_features(img_gray, win_size=image_size)  # ~1764
    sobel_hist = get_sobel_histogram(img_gray)    # 256
    
    # Concatenate all features
    features = np.concatenate([histogram, lbp_hist, hog_feat, sobel_hist])
    
    return features.astype(np.float32)

def extract_all_image_features(df, images_dir, image_size=(128, 128)):
    """
    Extract image features for all samples in dataframe.
    
    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_features)
        list: List of valid indices
    """
    print("\n🖼️ Extracting image features...")
    features_list = []
    valid_indices = []
    
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"   Processing image {idx + 1}/{total}...")
        
        filename = row['filename']
        image_path = os.path.join(images_dir, f"{filename}.jpeg")
        
        features = extract_image_features(image_path, image_size)
        
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
    
    print(f"   ✅ Extracted features for {len(features_list)} images")
    print(f"   Feature vector size: {features_list[0].shape[0]}")
    
    return np.array(features_list), valid_indices


# ==============================================================================
# STEP 4: AUDIO FEATURE EXTRACTION
# ==============================================================================

print("\n" + "="*60)
print("STEP 4: AUDIO FEATURE EXTRACTION")
print("="*60)

def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sampling rate."""
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def get_zcr(signal):
    """Compute Zero Crossing Rate."""
    zcr = librosa.feature.zero_crossing_rate(signal)
    return float(zcr.mean())

def get_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Compute Mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return mel_spec

def get_log_mel_spectrogram(mel_spec):
    """Convert Mel spectrogram to log scale (dB)."""
    return librosa.power_to_db(mel_spec, ref=np.max)

def get_cqt(audio, sr, hop_length=512, n_bins=84, bins_per_octave=12):
    """Compute Constant-Q Transform."""
    cqt = librosa.cqt(
        audio, sr=sr, hop_length=hop_length,
        n_bins=n_bins, bins_per_octave=bins_per_octave
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db

def get_mfcc(audio, sr, n_mfcc=20):
    """Compute MFCC features."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def aggregate_temporal_features(feature_2d):
    """
    Aggregate 2D temporal features (freq x time) into fixed-size vector.
    Computes mean and std across time axis.
    """
    mean_feat = np.mean(feature_2d, axis=1)
    std_feat = np.std(feature_2d, axis=1)
    return np.concatenate([mean_feat, std_feat])

def extract_audio_features(audio_path, target_sr=16000):
    """
    Extract all audio features for a single audio file.
    
    Features:
        - ZCR (1)
        - Mel spectrogram stats (128*2 = 256)
        - Log-Mel spectrogram stats (128*2 = 256)
        - CQT stats (84*2 = 168)
        - MFCC stats (20*2 = 40)
    
    Returns:
        np.ndarray: Concatenated feature vector (~721 features)
    """
    # Load audio
    sr, audio = load_audio(audio_path)
    
    if audio is None or len(audio) == 0:
        return None
    
    # Resample to target SR
    audio = resample_audio(audio, sr, target_sr)
    sr = target_sr
    
    # Extract features
    zcr = np.array([get_zcr(audio)])  # 1
    
    mel_spec = get_mel_spectrogram(audio, sr)
    mel_feat = aggregate_temporal_features(mel_spec)  # 256
    
    log_mel = get_log_mel_spectrogram(mel_spec)
    log_mel_feat = aggregate_temporal_features(log_mel)  # 256
    
    cqt = get_cqt(audio, sr)
    cqt_feat = aggregate_temporal_features(cqt)  # 168
    
    mfcc = get_mfcc(audio, sr)
    mfcc_feat = aggregate_temporal_features(mfcc)  # 40
    
    # Concatenate all features
    features = np.concatenate([zcr, mel_feat, log_mel_feat, cqt_feat, mfcc_feat])
    
    return features.astype(np.float32)

def extract_all_audio_features(df, audio_dir, target_sr=16000):
    """
    Extract audio features for all samples in dataframe.
    
    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_features)
        list: List of valid indices
    """
    print("\n🔊 Extracting audio features...")
    features_list = []
    valid_indices = []
    
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"   Processing audio {idx + 1}/{total}...")
        
        filename = row['filename']
        audio_path = os.path.join(audio_dir, f"{filename}.wav")
        
        features = extract_audio_features(audio_path, target_sr)
        
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
    
    print(f"   ✅ Extracted features for {len(features_list)} audio files")
    print(f"   Feature vector size: {features_list[0].shape[0]}")
    
    return np.array(features_list), valid_indices


# ==============================================================================
# STEP 5: FEATURE NORMALIZATION
# ==============================================================================

print("\n" + "="*60)
print("STEP 5: FEATURE NORMALIZATION")
print("="*60)

class MinMaxNormalizer:
    """
    Min-Max Normalization: scales features to [0, 1] range.
    Fits on training data and transforms both train and test.
    """
    
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.fitted = False
    
    def fit(self, X):
        """Learn min and max from training data."""
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using learned min/max."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        range_val = self.max_val - self.min_val
        # Avoid division by zero
        range_val[range_val == 0] = 1
        
        X_norm = (X - self.min_val) / range_val
        return X_norm
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# ==============================================================================
# STEP 6: DIMENSIONALITY REDUCTION (PCA)
# ==============================================================================

print("\n" + "="*60)
print("STEP 6: DIMENSIONALITY REDUCTION (PCA)")
print("="*60)

class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    Implemented from scratch using NumPy.
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Args:
            n_components: Number of components to keep. If None, use variance_threshold.
            variance_threshold: Keep components that explain this much variance (0-1).
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        self.n_components_final = None
    
    def fit(self, X):
        """Fit PCA on training data."""
        # Mean center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Compute explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_var
        
        # Determine number of components
        if self.n_components is not None:
            self.n_components_final = self.n_components
        else:
            # Use variance threshold
            cumsum = np.cumsum(self.explained_variance_ratio)
            self.n_components_final = np.argmax(cumsum >= self.variance_threshold) + 1
        
        # Store components
        self.components = eigenvectors[:, :self.n_components_final]
        
        return self
    
    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_variance_retained(self):
        """Return total variance retained by selected components."""
        return np.sum(self.explained_variance_ratio[:self.n_components_final])


# ==============================================================================
# STEP 7: FEATURE FUSION
# ==============================================================================

print("\n" + "="*60)
print("STEP 7: FEATURE FUSION")
print("="*60)

def fuse_features(image_features, audio_features):
    """
    Early fusion: concatenate image and audio features.
    
    Args:
        image_features: np.ndarray of shape (n_samples, n_img_features)
        audio_features: np.ndarray of shape (n_samples, n_audio_features)
    
    Returns:
        np.ndarray: Fused features of shape (n_samples, n_img + n_audio)
    """
    return np.hstack([image_features, audio_features])


# ==============================================================================
# STEP 8: TRAIN EMOTION CLASSIFIER
# ==============================================================================

print("\n" + "="*60)
print("STEP 8: TRAIN EMOTION CLASSIFIER")
print("="*60)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train multiple classifiers and return results.
    
    Returns:
        dict: Results containing trained models and metrics
    """
    print(f"\n🤖 Training classifiers for {model_name}...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        train_time = time.time() - start_time
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'train_time': train_time
        }
        
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      F1 Score: {f1:.4f}")
        print(f"      Time: {train_time:.2f}s")
    
    return results


# ==============================================================================
# STEP 9: MODEL EVALUATION
# ==============================================================================

print("\n" + "="*60)
print("STEP 9: MODEL EVALUATION")
print("="*60)

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def evaluate_model(y_true, y_pred, labels, model_name="Model"):
    """
    Comprehensive model evaluation.
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n📊 Evaluation for {model_name}")
    print("-" * 40)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    }

def compare_models(results_dict, title="Model Comparison"):
    """
    Compare multiple models and create visualization.
    """
    print(f"\n📈 {title}")
    print("=" * 50)
    
    # Create comparison table
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'F1 Score': results['f1_score'],
            'Train Time (s)': results['train_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    """
    Main execution pipeline for the Multimodal Emotion Recognition System.
    """
    print("\n" + "="*60)
    print("🚀 STARTING MAIN PIPELINE")
    print("="*60)
    
    # ------------------------------------------------------------------
    # Extract features for training data
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("EXTRACTING TRAINING FEATURES")
    print("-"*60)
    
    # Image features
    X_train_img, train_img_indices = extract_all_image_features(
        train_df, CONFIG['images_dir'], CONFIG['image_size']
    )
    
    # Audio features
    X_train_audio, train_audio_indices = extract_all_audio_features(
        train_df, CONFIG['audio_dir'], CONFIG['target_sr']
    )
    
    # Get common valid indices (samples where both image and audio exist)
    train_valid_idx = list(set(train_img_indices) & set(train_audio_indices))
    train_valid_idx.sort()
    
    # Filter features to common samples
    img_idx_map = {idx: i for i, idx in enumerate(train_img_indices)}
    audio_idx_map = {idx: i for i, idx in enumerate(train_audio_indices)}
    
    X_train_img_valid = np.array([X_train_img[img_idx_map[idx]] for idx in train_valid_idx])
    X_train_audio_valid = np.array([X_train_audio[audio_idx_map[idx]] for idx in train_valid_idx])
    
    # Get labels
    train_df_reset = train_df.reset_index(drop=True)
    y_train = train_df_reset.loc[train_valid_idx, 'emotion_text'].values
    
    print(f"\n✅ Training set: {len(train_valid_idx)} valid samples")
    print(f"   Image features shape: {X_train_img_valid.shape}")
    print(f"   Audio features shape: {X_train_audio_valid.shape}")
    
    # ------------------------------------------------------------------
    # Extract features for test data
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("EXTRACTING TEST FEATURES")
    print("-"*60)
    
    # Image features
    X_test_img, test_img_indices = extract_all_image_features(
        test_df, CONFIG['images_dir'], CONFIG['image_size']
    )
    
    # Audio features
    X_test_audio, test_audio_indices = extract_all_audio_features(
        test_df, CONFIG['audio_dir'], CONFIG['target_sr']
    )
    
    # Get common valid indices
    test_valid_idx = list(set(test_img_indices) & set(test_audio_indices))
    test_valid_idx.sort()
    
    # Filter features
    img_idx_map = {idx: i for i, idx in enumerate(test_img_indices)}
    audio_idx_map = {idx: i for i, idx in enumerate(test_audio_indices)}
    
    X_test_img_valid = np.array([X_test_img[img_idx_map[idx]] for idx in test_valid_idx])
    X_test_audio_valid = np.array([X_test_audio[audio_idx_map[idx]] for idx in test_valid_idx])
    
    # Get labels
    test_df_reset = test_df.reset_index(drop=True)
    y_test = test_df_reset.loc[test_valid_idx, 'emotion_text'].values
    
    print(f"\n✅ Test set: {len(test_valid_idx)} valid samples")
    print(f"   Image features shape: {X_test_img_valid.shape}")
    print(f"   Audio features shape: {X_test_audio_valid.shape}")
    
    # ------------------------------------------------------------------
    # STEP 5: Normalize features
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("NORMALIZING FEATURES")
    print("-"*60)
    
    # Normalize image features
    img_normalizer = MinMaxNormalizer()
    X_train_img_norm = img_normalizer.fit_transform(X_train_img_valid)
    X_test_img_norm = img_normalizer.transform(X_test_img_valid)
    print("✅ Image features normalized")
    
    # Normalize audio features
    audio_normalizer = MinMaxNormalizer()
    X_train_audio_norm = audio_normalizer.fit_transform(X_train_audio_valid)
    X_test_audio_norm = audio_normalizer.transform(X_test_audio_valid)
    print("✅ Audio features normalized")
    
    # ------------------------------------------------------------------
    # STEP 6: Apply PCA
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("APPLYING PCA")
    print("-"*60)
    
    # PCA on image features
    pca_img = PCA(variance_threshold=CONFIG['pca_variance_threshold'])
    X_train_img_pca = pca_img.fit_transform(X_train_img_norm)
    X_test_img_pca = pca_img.transform(X_test_img_norm)
    print(f"✅ Image PCA: {X_train_img_norm.shape[1]} → {X_train_img_pca.shape[1]} components")
    print(f"   Variance retained: {pca_img.get_variance_retained()*100:.2f}%")
    
    # PCA on audio features
    pca_audio = PCA(variance_threshold=CONFIG['pca_variance_threshold'])
    X_train_audio_pca = pca_audio.fit_transform(X_train_audio_norm)
    X_test_audio_pca = pca_audio.transform(X_test_audio_norm)
    print(f"✅ Audio PCA: {X_train_audio_norm.shape[1]} → {X_train_audio_pca.shape[1]} components")
    print(f"   Variance retained: {pca_audio.get_variance_retained()*100:.2f}%")
    
    # ------------------------------------------------------------------
    # STEP 7: Feature Fusion
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("FUSING FEATURES")
    print("-"*60)
    
    X_train_fused = fuse_features(X_train_img_pca, X_train_audio_pca)
    X_test_fused = fuse_features(X_test_img_pca, X_test_audio_pca)
    print(f"✅ Fused feature shape: {X_train_fused.shape}")
    
    # ------------------------------------------------------------------
    # STEP 8: Train Classifiers
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("TRAINING CLASSIFIERS")
    print("-"*60)
    
    # Train image-only models
    print("\n📷 IMAGE-ONLY MODELS")
    results_img = train_and_evaluate_models(
        X_train_img_pca, X_test_img_pca, y_train, y_test, "Image-Only"
    )
    
    # Train audio-only models
    print("\n🔊 AUDIO-ONLY MODELS")
    results_audio = train_and_evaluate_models(
        X_train_audio_pca, X_test_audio_pca, y_train, y_test, "Audio-Only"
    )
    
    # Train multimodal models
    print("\n🎭 MULTIMODAL MODELS")
    results_fused = train_and_evaluate_models(
        X_train_fused, X_test_fused, y_train, y_test, "Multimodal"
    )
    
    # ------------------------------------------------------------------
    # STEP 9: Comprehensive Evaluation
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("FINAL EVALUATION")
    print("-"*60)
    
    # Get emotion labels
    emotion_labels = sorted(list(set(y_train)))
    
    # Find best model for each modality
    best_img_model = max(results_img.items(), key=lambda x: x[1]['accuracy'])
    best_audio_model = max(results_audio.items(), key=lambda x: x[1]['accuracy'])
    best_fused_model = max(results_fused.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n🏆 BEST MODELS:")
    print(f"   Image-Only: {best_img_model[0]} - {best_img_model[1]['accuracy']*100:.2f}%")
    print(f"   Audio-Only: {best_audio_model[0]} - {best_audio_model[1]['accuracy']*100:.2f}%")
    print(f"   Multimodal: {best_fused_model[0]} - {best_fused_model[1]['accuracy']*100:.2f}%")
    
    # Detailed evaluation for best multimodal model
    print("\n" + "="*60)
    print("DETAILED EVALUATION - BEST MULTIMODAL MODEL")
    print("="*60)
    evaluate_model(y_test, best_fused_model[1]['predictions'], 
                   emotion_labels, f"Multimodal ({best_fused_model[0]})")
    
    # ------------------------------------------------------------------
    # Create comparison summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    summary_data = [
        {'Modality': 'Image-Only', 'Best Model': best_img_model[0], 
         'Accuracy': best_img_model[1]['accuracy'], 'F1': best_img_model[1]['f1_score']},
        {'Modality': 'Audio-Only', 'Best Model': best_audio_model[0], 
         'Accuracy': best_audio_model[1]['accuracy'], 'F1': best_audio_model[1]['f1_score']},
        {'Modality': 'Multimodal', 'Best Model': best_fused_model[0], 
         'Accuracy': best_fused_model[1]['accuracy'], 'F1': best_fused_model[1]['f1_score']},
    ]
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Calculate improvement
    img_acc = best_img_model[1]['accuracy']
    audio_acc = best_audio_model[1]['accuracy']
    fused_acc = best_fused_model[1]['accuracy']
    
    print(f"\n📈 IMPROVEMENT FROM FUSION:")
    print(f"   vs Image-Only: +{(fused_acc - img_acc)*100:.2f}%")
    print(f"   vs Audio-Only: +{(fused_acc - audio_acc)*100:.2f}%")
    
    # ------------------------------------------------------------------
    # Plot confusion matrices
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60)
    
    # Create figure with 3 confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Image-only confusion matrix
    cm_img = confusion_matrix(y_test, best_img_model[1]['predictions'])
    sns.heatmap(cm_img, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    axes[0].set_title(f"Image-Only ({best_img_model[0]})\nAccuracy: {img_acc*100:.2f}%")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Audio-only confusion matrix
    cm_audio = confusion_matrix(y_test, best_audio_model[1]['predictions'])
    sns.heatmap(cm_audio, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    axes[1].set_title(f"Audio-Only ({best_audio_model[0]})\nAccuracy: {audio_acc*100:.2f}%")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # Multimodal confusion matrix
    cm_fused = confusion_matrix(y_test, best_fused_model[1]['predictions'])
    sns.heatmap(cm_fused, annot=True, fmt='d', cmap='Purples', ax=axes[2],
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    axes[2].set_title(f"Multimodal ({best_fused_model[0]})\nAccuracy: {fused_acc*100:.2f}%")
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("✅ Saved confusion_matrices.png")
    
    # Accuracy comparison bar chart
    plt.figure(figsize=(10, 6))
    modalities = ['Image-Only', 'Audio-Only', 'Multimodal']
    accuracies = [img_acc * 100, audio_acc * 100, fused_acc * 100]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    bars = plt.bar(modalities, accuracies, color=colors, edgecolor='black')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved accuracy_comparison.png")
    
    # ------------------------------------------------------------------
    # Save models and results
    # ------------------------------------------------------------------
    print("\n" + "-"*60)
    print("SAVING MODELS AND RESULTS")
    print("-"*60)
    
    # Save best multimodal model
    best_model = best_fused_model[1]['model']
    with open('best_multimodal_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'pca_img': pca_img,
            'pca_audio': pca_audio,
            'img_normalizer': img_normalizer,
            'audio_normalizer': audio_normalizer,
            'config': CONFIG
        }, f)
    print("✅ Saved best_multimodal_model.pkl")
    
    # Save results summary
    results_summary = {
        'image_only': {name: {'accuracy': r['accuracy'], 'f1': r['f1_score']} 
                       for name, r in results_img.items()},
        'audio_only': {name: {'accuracy': r['accuracy'], 'f1': r['f1_score']} 
                       for name, r in results_audio.items()},
        'multimodal': {name: {'accuracy': r['accuracy'], 'f1': r['f1_score']} 
                       for name, r in results_fused.items()},
    }
    
    with open('results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print("✅ Saved results_summary.pkl")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return results_img, results_audio, results_fused


# ==============================================================================
# RUN THE PIPELINE
# ==============================================================================

if __name__ == "__main__":
    results_img, results_audio, results_fused = main()
    plt.show()
