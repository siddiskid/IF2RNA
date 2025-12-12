

import os
import json
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import logging
from datetime import datetime

from .model import IF2RNA, fit, predict, evaluate
from .data import IF2RNADataset, IF2RNATileDataset, create_synthetic_data
from .config import (
    DEFAULT_MODEL_CONFIG, 
    DEFAULT_TRAINING_CONFIG, 
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG
)


class IF2RNAExperiment:
    
    def __init__(self, config_path=None, experiment_name=None, save_dir='experiments'):
        
        self.experiment_name = experiment_name or f"if2rna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
            
        self.model_config = {**DEFAULT_MODEL_CONFIG, **config.get('model', {})}
        self.training_config = {**DEFAULT_TRAINING_CONFIG, **config.get('training', {})}
        self.data_config = {**DEFAULT_DATA_CONFIG, **config.get('data', {})}
        self.experiment_config = {**DEFAULT_EXPERIMENT_CONFIG, **config.get('experiment', {})}
        
        self._setup_logging()
        self.model = None
        self.results = {}
        
        self.logger.info(f"Initialized IF2RNA experiment: {self.experiment_name}")
        
    def _setup_logging(self):
        log_file = self.save_dir / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'IF2RNA.{self.experiment_name}')
        
    def save_config(self):
        config = {
            'model': self.model_config,
            'training': self.training_config,
            'data': self.data_config,
            'experiment': self.experiment_config
        }
        
        config_file = self.save_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_file}")
        
    def create_model(self, input_dim=None, output_dim=None):
        config = self.model_config.copy()
        
        if input_dim is not None:
            config['input_dim'] = input_dim
        if output_dim is not None:
            config['output_dim'] = output_dim
            
        self.model = IF2RNA(**config)
        self.logger.info(f"Created model: input_dim={config['input_dim']}, output_dim={config['output_dim']}")
        return self.model
        
    def run_synthetic_experiment(self, n_samples=200, n_genes=100):
        """Run complete experiment with synthetic data."""
        self.logger.info("Starting synthetic data experiment")
        
        # Generate synthetic data
        X, y, patients, projects = create_synthetic_data(
            n_samples=n_samples,
            n_tiles=self.data_config['n_tiles'],
            feature_dim=self.model_config['input_dim'],
            n_genes=n_genes
        )
        
        # Create dataset
        genes = [f"gene_{i:04d}" for i in range(n_genes)]
        dataset = IF2RNADataset(genes, patients, projects, X, y)
        
        # Create model
        self.create_model(output_dim=n_genes)
        
        # Run experiment
        results = self._run_cross_validation(dataset)
        
        # Save results
        self._save_results(results)
        
        self.logger.info("Synthetic experiment completed")
        return results
        
    def _run_cross_validation(self, dataset):
        """Run k-fold cross-validation."""
        n_folds = self.experiment_config['cv_folds']
        kfold = KFold(n_splits=n_folds, shuffle=True, 
                     random_state=self.experiment_config['random_seed'])
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            self.logger.info(f"Running fold {fold + 1}/{n_folds}")
            
            # Create data splits
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            
            # Get validation projects
            val_projects = np.array([dataset.projects[i] for i in val_idx])
            
            # Create fresh model for this fold
            self.create_model(output_dim=len(dataset.genes))
            
            # Train model
            model_path = self.save_dir / f'fold_{fold}'
            model_path.mkdir(exist_ok=True)
            
            preds, labels = fit(
                model=self.model,
                train_set=train_set,
                valid_set=val_set,
                valid_projects=val_projects,
                params=self.training_config,
                path=str(model_path),
                logdir=str(self.save_dir / f'logs_fold_{fold}')
            )
            
            # Compute metrics
            fold_result = self._compute_fold_metrics(preds, labels, val_projects, fold)
            fold_results.append(fold_result)
            
        # Aggregate results
        results = self._aggregate_cv_results(fold_results)
        return results
        
    def _compute_fold_metrics(self, preds, labels, projects, fold):
        """Compute metrics for a single fold."""
        from .model import compute_correlations
        
        # Overall correlation
        overall_corr = compute_correlations(labels, preds, projects)
        
        # Per-gene correlations
        gene_corrs = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) > 1:
                corr = np.corrcoef(labels[:, i], preds[:, i])[0, 1]
                gene_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                gene_corrs.append(0.0)
        
        fold_result = {
            'fold': fold,
            'overall_correlation': overall_corr if not np.isnan(overall_corr) else 0.0,
            'gene_correlations': gene_corrs,
            'mean_gene_correlation': np.mean(gene_corrs),
            'median_gene_correlation': np.median(gene_corrs),
            'predictions': preds,
            'labels': labels
        }
        
        self.logger.info(f"Fold {fold} - Overall correlation: {fold_result['overall_correlation']:.4f}")
        return fold_result
        
    def _aggregate_cv_results(self, fold_results):
        """Aggregate results across folds."""
        overall_corrs = [r['overall_correlation'] for r in fold_results]
        mean_gene_corrs = [r['mean_gene_correlation'] for r in fold_results]
        
        results = {
            'cv_folds': len(fold_results),
            'overall_correlation_mean': np.mean(overall_corrs),
            'overall_correlation_std': np.std(overall_corrs),
            'gene_correlation_mean': np.mean(mean_gene_corrs),
            'gene_correlation_std': np.std(mean_gene_corrs),
            'fold_results': fold_results
        }
        
        self.logger.info(f"CV Results - Overall: {results['overall_correlation_mean']:.4f} ± {results['overall_correlation_std']:.4f}")
        self.logger.info(f"CV Results - Gene-wise: {results['gene_correlation_mean']:.4f} ± {results['gene_correlation_std']:.4f}")
        
        return results
        
    def _save_results(self, results):
        """Save experiment results."""
        # Save main results (without predictions to save space)
        results_summary = {k: v for k, v in results.items() if k != 'fold_results'}
        
        results_file = self.save_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
            
        # Save detailed results
        detailed_file = self.save_dir / 'detailed_results.pkl'
        with open(detailed_file, 'wb') as f:
            pkl.dump(results, f)
            
        self.logger.info(f"Results saved to {self.save_dir}")
        
        # Save configuration
        self.save_config()


def create_experiment_config(
    experiment_name="if2rna_test",
    model_config=None,
    training_config=None,
    data_config=None,
    experiment_config=None
):
    """Create experiment configuration file."""
    
    config = {
        'model': model_config or DEFAULT_MODEL_CONFIG,
        'training': training_config or DEFAULT_TRAINING_CONFIG,
        'data': data_config or DEFAULT_DATA_CONFIG,
        'experiment': experiment_config or DEFAULT_EXPERIMENT_CONFIG
    }
    
    config_file = f"{experiment_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Configuration saved to {config_file}")
    return config_file
