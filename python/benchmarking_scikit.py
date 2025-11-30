#!/usr/bin/env python3
"""
Scikit-learn Benchmarking Module for CoreAI3D
Provides comprehensive benchmarking against scikit-learn baseline models
"""

import numpy as np
import pandas as pd
import json
import time
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        roc_auc_score, confusion_matrix
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

@dataclass
class BenchmarkMetrics:
    """Performance metrics for a model"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    auc_roc: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: int = 0
    parameters: int = 0

@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model"""
    model_name: str
    metrics: BenchmarkMetrics
    predictions: List[float]
    targets: List[float]
    additional_metrics: Dict[str, float]
    total_time: float
    training_time: float
    inference_time: float
    cross_val_scores: List[float] = None
    feature_importance: List[float] = None

class ScikitLearnBenchmarker:
    """Benchmarker using scikit-learn models"""

    def __init__(self, random_state: int = 42, verbose: bool = False):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ScikitLearnBenchmarker")

        self.random_state = random_state
        self.verbose = verbose
        self.scaler = StandardScaler()

        # Define regression models
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=0.1, random_state=random_state),
            'Lasso Regression': Lasso(alpha=0.01, random_state=random_state),
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=random_state
            ),
            'Gradient Boosting Regressor': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=random_state
            ),
            'SVR': SVR(kernel='rbf', C=1.0),
            'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree Regressor': DecisionTreeRegressor(
                max_depth=10, random_state=random_state
            ),
            'MLP Regressor': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state
            )
        }

        # Define classification models
        self.classification_models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_state
            ),
            'Gradient Boosting Classifier': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=random_state
            ),
            'SVC': SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state),
            'K-Neighbors Classifier': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree Classifier': DecisionTreeClassifier(
                max_depth=10, random_state=random_state
            ),
            'Naive Bayes': GaussianNB(),
            'MLP Classifier': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state
            )
        }

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def _detect_problem_type(self, targets: np.ndarray) -> str:
        """Detect if this is a regression or classification problem"""
        unique_targets = np.unique(targets)
        if len(unique_targets) <= 10 and np.all(unique_targets == unique_targets.astype(int)):
            return 'classification'
        else:
            return 'regression'

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: np.ndarray = None, problem_type: str = 'regression') -> BenchmarkMetrics:
        """Calculate comprehensive metrics"""
        metrics = BenchmarkMetrics()

        try:
            if problem_type == 'classification':
                # Classification metrics
                metrics.accuracy = accuracy_score(y_true, y_pred)
                metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                if y_prob is not None and len(np.unique(y_true)) == 2:
                    try:
                        metrics.auc_roc = roc_auc_score(y_true, y_prob[:, 1])
                    except:
                        metrics.auc_roc = 0.0

                # For classification, set regression metrics to 0
                metrics.rmse = 0.0
                metrics.mae = 0.0
                metrics.r2_score = 0.0

            else:
                # Regression metrics
                metrics.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics.mae = mean_absolute_error(y_true, y_pred)
                metrics.r2_score = r2_score(y_true, y_pred)

                # For regression, set classification metrics to 0
                metrics.accuracy = 0.0
                metrics.precision = 0.0
                metrics.recall = 0.0
                metrics.f1_score = 0.0
                metrics.auc_roc = 0.0

        except Exception as e:
            if self.verbose:
                print(f"Error calculating metrics: {e}")

        return metrics

    def benchmark_single_model(self, model_name: str, model, X_train: np.ndarray,
                             X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                             problem_type: str = 'regression') -> BenchmarkResult:
        """Benchmark a single model"""
        result = BenchmarkResult(
            model_name=model_name,
            metrics=BenchmarkMetrics(),
            predictions=[],
            targets=[],
            additional_metrics={},
            total_time=0.0,
            training_time=0.0,
            inference_time=0.0
        )

        start_time = time.time()

        try:
            # Training phase
            train_start = time.time()
            model.fit(X_train, y_train)
            train_end = time.time()
            result.training_time = train_end - train_start

            # Get parameter count
            if hasattr(model, 'coef_'):
                result.metrics.parameters = len(model.coef_.flatten()) + (1 if hasattr(model, 'intercept_') else 0)
            elif hasattr(model, 'feature_importances_'):
                result.metrics.parameters = len(model.feature_importances_)
                result.feature_importance = model.feature_importances_.tolist()
            elif hasattr(model, 'n_features_in_'):
                result.metrics.parameters = model.n_features_in_
            else:
                result.metrics.parameters = 1  # Default

            # Inference phase
            inference_start = time.time()
            if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
                y_prob = None
            inference_end = time.time()
            result.inference_time = inference_end - inference_start

            # Store predictions and targets
            result.predictions = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
            result.targets = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)

            # Calculate metrics
            result.metrics = self._calculate_metrics(y_test, y_pred, y_prob, problem_type)
            result.metrics.training_time = result.training_time
            result.metrics.inference_time = result.inference_time
            result.metrics.memory_usage = self._get_memory_usage()

            # Cross-validation scores
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error' if problem_type == 'regression' else 'accuracy')
                result.cross_val_scores = (-cv_scores if problem_type == 'regression' else cv_scores).tolist()
            except:
                result.cross_val_scores = []

        except Exception as e:
            if self.verbose:
                print(f"Error benchmarking {model_name}: {e}")
            result.additional_metrics['error'] = str(e)

        result.total_time = time.time() - start_time

        return result

    def benchmark_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                           random_state: int = None) -> List[BenchmarkResult]:
        """Benchmark all available models"""
        if random_state is None:
            random_state = self.random_state

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Detect problem type
        problem_type = self._detect_problem_type(y_train)

        if self.verbose:
            print(f"Detected problem type: {problem_type}")
            print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Test set: {X_test.shape[0]} samples")

        # Select appropriate models
        models_to_test = self.regression_models if problem_type == 'regression' else self.classification_models

        results = []
        for model_name, model in models_to_test.items():
            if self.verbose:
                print(f"Benchmarking {model_name}...")

            result = self.benchmark_single_model(
                model_name, model, X_train_scaled, X_test_scaled,
                y_train, y_test, problem_type
            )
            results.append(result)

        return results

    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save benchmark results to JSON file"""
        data = {
            'timestamp': time.time(),
            'results': [asdict(result) for result in results]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_results(self, filename: str) -> List[BenchmarkResult]:
        """Load benchmark results from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        results = []
        for result_data in data['results']:
            result = BenchmarkResult(**result_data)
            results.append(result)

        return results

    def generate_comparison_report(self, results: List[BenchmarkResult],
                                 output_file: str = None) -> str:
        """Generate a comparison report"""
        if not results:
            return "No results to compare"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SCIKIT-LEARN BENCHMARK COMPARISON REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models compared: {len(results)}")
        report_lines.append("")

        # Sort by performance (R² for regression, accuracy for classification)
        sorted_results = sorted(results,
                              key=lambda x: x.metrics.r2_score if x.metrics.r2_score > 0 else x.metrics.accuracy,
                              reverse=True)

        report_lines.append("MODEL PERFORMANCE RANKING:")
        report_lines.append("-" * 40)

        for i, result in enumerate(sorted_results, 1):
            report_lines.append(f"{i}. {result.model_name}")
            if result.metrics.r2_score > 0:
                report_lines.append(f"   R² Score: {result.metrics.r2_score:.4f}")
            else:
                report_lines.append(f"   Accuracy: {result.metrics.accuracy:.4f}")
            report_lines.append(f"   Training Time: {result.training_time:.2f}s")
            report_lines.append(f"   Inference Time: {result.inference_time:.2f}s")
            report_lines.append("")

        # Detailed metrics table
        report_lines.append("DETAILED METRICS:")
        report_lines.append("-" * 80)

        header = f"{'Model':<25} {'R²/Acc':<10} {'RMSE':<8} {'MAE':<8} {'Train(s)':<10} {'Infer(s)':<10}"
        report_lines.append(header)
        report_lines.append("-" * 80)

        for result in sorted_results:
            metrics = result.metrics
            if metrics.r2_score > 0:
                perf_metric = f"{metrics.r2_score:<10.4f}"
            else:
                perf_metric = f"{metrics.accuracy:<10.4f}"
            line = f"{result.model_name:<25} {perf_metric} {metrics.rmse:<8.2f} {metrics.mae:<8.2f} {result.training_time:<10.2f} {result.inference_time:<10.2f}"
            report_lines.append(line)

        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            if self.verbose:
                print(f"Report saved to {output_file}")

        return report

def benchmark_from_csv(csv_file: str, target_column: str = 'target',
                      test_size: float = 0.2, output_file: str = None) -> List[BenchmarkResult]:
    """Convenience function to benchmark from CSV file"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required")

    # Load data
    df = pd.read_csv(csv_file)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Create benchmarker and run
    benchmarker = ScikitLearnBenchmarker(verbose=True)
    results = benchmarker.benchmark_all_models(X, y, test_size=test_size)

    # Save results if requested
    if output_file:
        benchmarker.save_results(results, output_file)

    return results

if __name__ == "__main__":
    # Example usage
    if SKLEARN_AVAILABLE:
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = X.sum(axis=1) + np.random.randn(1000) * 0.1  # Linear relationship with noise

        benchmarker = ScikitLearnBenchmarker(verbose=True)
        results = benchmarker.benchmark_all_models(X, y)

        report = benchmarker.generate_comparison_report(results)
        print(report)

        # Save results
        benchmarker.save_results(results, 'benchmark_results.json')
    else:
        print("scikit-learn not available. Install with: pip install scikit-learn")