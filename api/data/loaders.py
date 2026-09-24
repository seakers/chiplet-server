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


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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
    
    def __init__(self, evaluator: str = 'cascade', run_id: str = None, num_objs: int = None):
        self.evaluator = evaluator.lower()
        self.run_id = run_id
        self.config = get_evaluator_config(evaluator, num_objs=num_objs)
    
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
                             trace: str = '') -> List[Dict[str, Any]]:
        """
        Load design points and return as list of dictionaries.
        This maintains backward compatibility with existing frontend expectations [1].
        """
        if file_path is None:
            file_path = self.get_file_path()
        
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return []
        
        # print("B", file_path)  # Debug: print file path being loaded
        
        points = []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            if self.evaluator == 'pistil':
                header = next(csv_reader, None)
                if header is None:
                    return []
                # Build column-name → index map from the actual file header
                col_index = {name.strip(): i for i, name in enumerate(header)}
                
                for row in csv_reader:
                    if not row or row[0].strip() == 'num_cus':
                        continue  # skip any duplicate header rows
                    if len(row) < 2:
                        continue
                    
                    point = {}
                    
                    # Decisions — look up by name
                    for dec_name in self.config.decision_columns:
                        if dec_name in col_index:
                            idx = col_index[dec_name]
                            try:
                                val = float(row[idx])
                                point[dec_name] = int(val) if val.is_integer() else val
                            except (ValueError, IndexError):
                                pass
                    
                    # Objectives — look up by name
                    for obj_name in self.config.objective_columns:
                        if obj_name in col_index:
                            idx = col_index[obj_name]
                            try:
                                point[obj_name] = float(row[idx])
                            except (ValueError, IndexError):
                                pass
                    
                    # Also pull any extra named columns the frontend uses
                    # (num_chiplets, system_power_W, etc.) the same way
                    for col_name, idx in col_index.items():
                        if col_name not in point:
                            try:
                                point[col_name] = float(row[idx])
                            except (ValueError, IndexError):
                                point[col_name] = row[idx]  # keep as string if not numeric
                    
                    # Set x/y from the first two selected objective columns
                    obj_cols = self.config.objective_columns
                    if len(obj_cols) >= 2:
                        point['x'] = point.get(obj_cols[0], 0.0)
                        point['y'] = point.get(obj_cols[1], 0.0)
                    
                    # Algorithm / trace / model
                    if 'algorithm' in col_index:
                        point['algorithm'] = row[col_index['algorithm']].strip() or 'Genetic Algorithm'
                    else:
                        point['algorithm'] = 'Genetic Algorithm'
                    point['trace'] = trace
                    point['model'] = 'PISTIL'
                    points.append(point)
                
                return points
        
            if self.config.csv_has_header:
                next(csv_reader, None)
            # print(f"Columns Needed: {len(self.config.decision_columns) + len(self.config.objective_columns)}")  # Debug: count rows
            # print(f"Num Decision Columns: {len(self.config.decision_columns)}, Num Objective Columns: {len(self.config.objective_columns)}")  # Debug: count columns
            for row in csv_reader:
                # if len(row) < (len(self.config.decision_columns) + len(self.config.objective_columns)):
                #     print(f"Warning: Skipping row with insufficient columns: {row}")
                #     continue

                point = {}
                # Parse all objectives by name
                for obj_name in self.config.objective_columns:
                    try:
                        idx = self.config.get_objective_index(obj_name)
                        if idx < len(row):
                            point[obj_name] = float(row[idx])
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse objective '{obj_name}' in row: {row}")
                        continue

                # Parse all decisions by name
                for dec_name in self.config.decision_columns:
                    try:
                        idx = self.config.get_decision_index(dec_name)
                        if idx < len(row):
                            val = float(row[idx])
                            # CASCADE keys are lowercase in frontend; PISTIL keys match config
                            key = self._frontend_decision_key(dec_name)
                            point[key] = int(val) if val.is_integer() else val
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse decision '{dec_name}' in row: {row}")
                        continue

                # Backward-compat: keep x/y as the first two objectives
                if self.config.objective_columns:
                    point['x'] = point.get(self.config.objective_columns[0], 0)
                    if len(self.config.objective_columns) > 1:
                        point['y'] = point.get(self.config.objective_columns[1], 0)

                # Algorithm column (last column for CASCADE, configurable for PISTIL)
                last_val = row[-1].strip() if isinstance(row[-1], str) else ''
                if last_val and not _is_numeric(last_val):
                    point['algorithm'] = last_val          # preserves 'User', 'Custom', 'Full-Factorial', etc.
                else:
                    point['algorithm'] = 'Genetic Algorithm'

                point['trace'] = trace
                point['model'] = self.evaluator.upper()
                points.append(point)

        return points
    
    
    def _frontend_decision_key(self, dec_name: str) -> str:
        """Map config decision names to frontend-expected keys (CASCADE compatibility)."""
        cascade_map = {
            'GPU': 'gpu', 'Attention': 'attn',
            'Sparse': 'sparse', 'Convolution': 'conv'
        }
        return cascade_map.get(dec_name, dec_name)
    
    
    def load_points_as_numpy(self, file_path: str = None) -> np.ndarray:
        """
        Load the entire CSV file as a single numpy array.
        """
        if file_path is None:
            file_path = self.get_file_path()

        if not os.path.exists(file_path):
            return np.array([])

        rows = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            if self.config.csv_has_header:
                next(csv_reader, None)
            for row in csv_reader:
                if row:
                    rows.append(row[:-1])  # Exclude last column (algorithm) for numpy array

        if not rows:
            return np.array([])

        return np.array(rows, dtype=float)

    
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
                numeric_row = [float(val) for val in row if _is_numeric(val)]
                full_data_list.append(numeric_row)


        if not full_data_list:
            return np.array([])
        
        # Convert to numpy and get unique rows
        full_data = np.array(full_data_list)
        # total_cols = self.config.num_objectives + len(self.config.decision_columns)
        # full_data = full_data[:, :total_cols]

        
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