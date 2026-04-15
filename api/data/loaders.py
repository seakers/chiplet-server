"""
Data loading utilities for chiplet design optimization.
Centralizes all CSV parsing and data loading logic.
"""
import csv
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from api.config.settings import get_points_file, EVALUATOR_BASE_PATHS
from api.config.evaluators import get_evaluator_config, EvaluatorConfig
from .models import DesignPoint, ParetoFront, RunStatistics


class CSVLoader:
    """Generic CSV loader with deduplication support."""
    
    @staticmethod
    def load_csv(file_path: str, has_header: bool = False) -> List[List[str]]:
        """Load a CSV file and return rows as lists of strings."""
        rows = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            if has_header:
                next(csv_reader, None)  # Skip header
            for row in csv_reader:
                if row:  # Skip empty rows
                    rows.append(row)
        return rows
    
    @staticmethod
    def deduplicate_rows(rows: List[List[Any]]) -> List[List[Any]]:
        """Remove duplicate rows based on all columns."""
        seen = set()
        unique_rows = []
        for row in rows:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)
        return unique_rows


class PointsLoader:
    """
    Loads design points from CSV files with evaluator-aware parsing.
    Consolidates the repeated CSV loading logic from views.py [1].
    """
    
    def __init__(self, evaluator: str = 'cascade', run_id: str = None):
        self.evaluator = evaluator.lower()
        self.run_id = run_id
        self.config = get_evaluator_config(evaluator)
    
    def get_file_path(self, backup_filename: str = None) -> str:
        """Get the appropriate file path for loading points."""
        if backup_filename:
            # Loading from backup file
            base_path = EVALUATOR_BASE_PATHS.get(self.evaluator)
            return os.path.join(base_path, 'dse/results', backup_filename)
        return get_points_file(self.evaluator, self.run_id)
    
    def load_points(self, file_path: str = None, 
                    algorithm: str = '', 
                    trace: str = '') -> List[DesignPoint]:
        """
        Load design points from CSV file.
        
        Args:
            file_path: Optional custom file path. If None, uses default for evaluator.
            algorithm: Algorithm name to attach to points.
            trace: Trace name to attach to points.
            
        Returns:
            List of DesignPoint objects.
        """
        if file_path is None:
            file_path = self.get_file_path()
        
        rows = CSVLoader.load_csv(file_path, has_header=self.config.csv_has_header)
        
        points = []
        for row in rows:
            try:
                point = DesignPoint.from_csv_row(
                    row, 
                    evaluator=self.evaluator,
                    algorithm=algorithm,
                    trace=trace
                )
                points.append(point)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid row: {row}. Error: {e}")
                continue
        
        return points
    
    def load_points_as_dicts(self, file_path: str = None,
                             algorithm: str = '',
                             trace: str = '') -> List[Dict[str, Any]]:
        """
        Load design points and return as list of dictionaries.
        This maintains backward compatibility with existing frontend expectations [1].
        """
        if file_path is None:
            file_path = self.get_file_path()
        
        if not os.path.exists(file_path):
            return []
        
        points = []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            if self.config.csv_has_header:
                next(csv_reader, None)
            
            for row in csv_reader:
                if len(row) < 6:
                    continue
                
                if self.evaluator == 'cascade':
                    points.append({
                        'x': float(row[0]),  # exe_time
                        'y': float(row[1]),  # energy
                        'gpu': int(float(row[2])),
                        'attn': int(float(row[3])),
                        'sparse': int(float(row[4])),
                        'conv': int(float(row[5])),
                        'algorithm': algorithm,
                        'trace': trace,
                    })
                elif self.evaluator == 'pistil':
                    points.append({
                        'x': float(row[9]),   # latency_ms
                        'y': float(row[10]),  # energy_mJ
                        'num_cus': int(float(row[0])),
                        'num_tmacs': int(float(row[1])),
                        'mem_buf_cap': int(float(row[2])),
                        'net_buf_cap': int(float(row[3])),
                        'mem_banks_per_group': int(float(row[4])),
                        'mem_ranks': int(float(row[5])),
                        'mem_frac_bank_cap': float(row[6]),
                        'batch_size': int(float(row[7])),
                        'kv_cache': int(float(row[8])),
                        'algorithm': algorithm,
                        'trace': trace,
                    })
        
        return points
    
    def load_points_as_numpy(self, file_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load points and return as numpy arrays for analysis.
        
        Returns:
            Tuple of (objective_values, decision_values) as numpy arrays.
        """
        if file_path is None:
            file_path = self.get_file_path()
        
        points = self.load_points_as_dicts(file_path)
        
        if self.evaluator == 'cascade':
            objectives = np.array([[p['x'], p['y']] for p in points])
            decisions = np.array([[p['gpu'], p['attn'], p['sparse'], p['conv']] for p in points])
        elif self.evaluator == 'pistil':
            objectives = np.array([[p['x'], p['y']] for p in points])
            decisions = np.array([
                [p['num_cus'], p['num_tmacs'], p['mem_buf_cap'], p['net_buf_cap'],
                 p['mem_banks_per_group'], p['mem_ranks'], p['mem_frac_bank_cap'],
                 p['batch_size'], p['kv_cache']]
                for p in points
            ])
        else:
            raise ValueError(f"Unknown evaluator: {self.evaluator}")
        
        return objectives, decisions
    
    def load_deduplicated_data(self, file_path: str = None) -> np.ndarray:
        """
        Load and deduplicate data for analysis (rule mining, etc.).
        Returns numpy array with objectives and decisions combined.
        """
        if file_path is None:
            file_path = self.get_file_path()
        
        full_data_list = []
        
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if not row:
                    continue
                # Skip header for PISTIL
                if self.evaluator == "pistil" and row[0] == "num_cus":
                    continue
                numeric_row = [float(val) for val in row]
                full_data_list.append(numeric_row)
        
        if not full_data_list:
            return np.array([])
        
        # Convert to numpy and get unique rows
        full_data = np.array(full_data_list)
        total_cols = self.config.num_objectives + len(self.config.decision_columns)
        full_data = full_data[:, :total_cols]
        
        # Deduplicate
        full_data = np.array(list(set(tuple(row) for row in full_data)))
        
        return full_data


class RunDataLoader:
    """
    Loads data for specific optimization runs, including backup files.
    Consolidates the repeated run loading logic from views.py [1].
    """
    
    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.points_loader = PointsLoader(evaluator)
        self.base_path = EVALUATOR_BASE_PATHS.get(evaluator)
        self.results_dir = os.path.join(self.base_path, 'dse/results')
    
    def load_run_data(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load data for a specific run, handling different run_id formats.
        
        Supports:
        - 'loaded_run_<timestamp>' - Load from backup file
        - 'previous_<filename>' - Load from previous run file
        - Regular run_id - Load from current points.csv
        """
        print(f"Loading data for run_id: {run_id}")
        
        if run_id.startswith('loaded_run_'):
            timestamp_str = run_id.replace('loaded_run_', '')
            backup_filename = f"points_backup_{timestamp_str}.csv"
            file_path = os.path.join(self.results_dir, backup_filename)
        elif run_id.startswith('previous_'):
            backup_filename = run_id.replace('previous_', '')
            file_path = os.path.join(self.results_dir, backup_filename)
        else:
            file_path = os.path.join(self.results_dir, "points.csv")
        
        print(f"File path: {file_path}, exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return None
        
        return self.points_loader.load_points_as_dicts(file_path)
    
    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load metadata from zip file if available."""
        timestamp_str = run_id.replace('loaded_run_', '').replace('previous_', '')
        zip_filename = f"run_{timestamp_str}.zip"
        zip_path = os.path.join(self.results_dir, zip_filename)
        
        default_params = {
            'model': 'CASCADE',
            'algorithm': 'Genetic Algorithm',
            'objectives': ['Energy', 'Runtime'],
            'population_size': 50,
            'generations': 100,
            'trace_name': 'gpt-j-65536-weighted'
        }
        
        if os.path.exists(zip_path):
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    if 'metadata.json' in zipf.namelist():
                        metadata_content = zipf.read('metadata.json').decode('utf-8')
                        metadata = json.loads(metadata_content)
                        default_params.update(metadata)
            except Exception as e:
                print(f"Error extracting metadata from zip: {e}")
        
        return default_params