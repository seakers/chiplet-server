# Run Storage and Export System

This document describes the new database-centric approach for storing optimization runs and exporting results as zip files.

## Overview

The system now stores all optimization runs in a SQLite database with the following components:

### Database Models

1. **OptimizationRun**: Stores metadata for each optimization run
2. **DesignPoint**: Stores individual design points with objectives and chiplet configurations
3. **AnalyticsResult**: Stores analytics results (rule mining, distance correlation, etc.)
4. **ComparativeStudy**: Stores metadata for comparative studies between runs

### Key Features

- **Persistent Storage**: All runs are stored in the database and can be retrieved later
- **Time-tagged Exports**: Results are exported as zip files with timestamped names
- **CSV + Text Reports**: Each zip contains CSV data and text reports
- **Pareto Front Tracking**: Automatic identification and storage of Pareto optimal designs
- **Comparative Studies**: Support for comparing multiple runs

## File Structure

### Zip File Contents

Each exported zip file contains:

```
myrun_2025Aug02_205153.zip
├── myrun_2025Aug02_205153_designs.csv      # All design points
├── myrun_2025Aug02_205153_report.txt       # Text report (placeholder)
└── myrun_2025Aug02_205153_metadata.json    # Run metadata
```

### CSV Format

The CSV file contains:
- `execution_time_ms`: Execution time in milliseconds
- `energy_mj`: Energy consumption in millijoules
- `gpu_count`: Number of GPU chiplets
- `attention_count`: Number of Attention chiplets
- `sparse_count`: Number of Sparse chiplets
- `convolution_count`: Number of Convolution chiplets
- `total_chiplets`: Total number of chiplets
- `is_pareto_optimal`: Whether the design is Pareto optimal

## API Endpoints

### Run Management

- `GET /api/runs/` - List all runs
- `GET /api/runs/statistics/` - Get overall statistics
- `GET /api/runs/search/` - Search runs by criteria
- `GET /api/runs/{run_id}/` - Get run details
- `GET /api/runs/{run_id}/plot/` - Get run data for plotting
- `DELETE /api/runs/{run_id}/delete/` - Delete a run

### Export Endpoints

- `POST /api/runs/{run_id}/export/` - Export single run to zip
- `POST /api/exports/comparative-study/` - Export comparative study
- `POST /api/exports/multiple-runs/` - Export multiple runs
- `POST /api/exports/cleanup/` - Clean up old exports

### Plotting and Comparison

- `GET /api/runs/plot/` - Get multiple runs for plotting
- `GET /api/runs/compare/` - Compare two runs
- `GET /api/runs/pareto-fronts/` - Get Pareto fronts for multiple runs

## Usage Examples

### Export a Single Run

```bash
curl -X POST http://localhost:8000/api/runs/run_20250802_205153/export/ \
  -H "Content-Type: application/json" \
  -d '{"base_name": "my_optimization"}'
```

Response:
```json
{
  "status": "success",
  "message": "Run my_optimization exported successfully",
  "download_url": "/media/exports/my_optimization_2025Aug02_205153.zip",
  "file_path": "/path/to/media/exports/my_optimization_2025Aug02_205153.zip"
}
```

### Export Comparative Study

```bash
curl -X POST http://localhost:8000/api/exports/comparative-study/ \
  -H "Content-Type: application/json" \
  -d '{
    "run_a_id": "run_20250802_205153",
    "run_b_id": "run_20250802_205154",
    "study_name": "gpt_vs_resnet_comparison"
  }'
```

### Export Multiple Runs

```bash
curl -X POST http://localhost:8000/api/exports/multiple-runs/ \
  -H "Content-Type: application/json" \
  -d '{
    "run_ids": ["run_20250802_205153", "run_20250802_205154", "run_20250802_205155"],
    "base_name": "batch_optimization"
  }'
```

## Migration from Existing Data

To migrate existing CSV data to the database:

```bash
curl -X POST http://localhost:8000/api/migrate-data/
```

This will:
1. Read the existing `points.csv` file
2. Create a new `OptimizationRun` record
3. Create `DesignPoint` records for each row
4. Calculate and mark Pareto optimal designs

## File Naming Convention

Zip files are named using the pattern:
```
{base_name}_{YYYY}{MMM}{DD}_{HHMMSS}.zip
```

Examples:
- `myrun_2025Aug02_205153.zip`
- `gpt_optimization_2025Aug02_205154.zip`
- `comparative_study_2025Aug02_205155.zip`

## Storage Locations

- **Database**: `chiplet-server/db.sqlite3`
- **Exports**: `chiplet-server/media/exports/`
- **Analytics**: `chiplet-server/analytics/`
- **Detailed Results**: `chiplet-server/api/Evaluator/cascade/chiplet_model/dse/results/`

## Future Enhancements

1. **Enhanced Text Reports**: Detailed analysis and insights in text reports
2. **Visualization Files**: Include plots and charts in zip files
3. **Automated Cleanup**: Scheduled cleanup of old export files
4. **Export Scheduling**: Automatic export after run completion
5. **Compression Options**: Different compression levels for zip files

## Database Schema

### OptimizationRun
- `run_id`: Unique identifier
- `name`: Human-readable name
- `description`: Run description
- `algorithm`: Optimization algorithm used
- `model`: Model type (CASCADE, CUSTOM)
- `population_size`: Population size
- `generations`: Number of generations
- `objectives`: List of objective names
- `trace_name`: Trace used for optimization
- `status`: Run status (running, completed, failed, cancelled)
- `total_designs_evaluated`: Number of designs evaluated
- `pareto_front_size`: Number of Pareto optimal designs

### DesignPoint
- `run`: Foreign key to OptimizationRun
- `gpu_count`, `attention_count`, `sparse_count`, `convolution_count`: Chiplet configuration
- `execution_time_ms`, `energy_mj`: Objective values
- `is_pareto_optimal`: Whether design is Pareto optimal
- `additional_metrics`: JSON field for additional data

### AnalyticsResult
- `run`: Foreign key to OptimizationRun
- `analytics_type`: Type of analytics (rule_mining, distance_correlation, etc.)
- `results_data`: JSON field containing analytics results
- `file_path`: Path to analytics file

### ComparativeStudy
- `study_id`: Unique study identifier
- `run_a`, `run_b`: Foreign keys to OptimizationRun
- `shared_parameters`: JSON field with shared parameters
- `name`, `description`: Study metadata 