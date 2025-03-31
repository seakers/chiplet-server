#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import glob


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    # clear data
    # Clear all files in the specified directory
    point_context_dir = "api/Evaluator/cascade/chiplet_model/dse/results/pointContext"
    files = glob.glob(os.path.join(point_context_dir, "*"))
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

    # Clear the points.csv file
    points_csv_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
    try:
        with open(points_csv_path, "w") as file:
            file.truncate(0)
    except Exception as e:
        print(f"Error clearing file {points_csv_path}: {e}")
        
    main()
