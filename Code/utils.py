# utilities for package

# libraries
import os

# package to define input output paths
def paths():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)

    data_dir = os.path.join(parent_dir, 'Data/')
    data_output_dir = os.path.join(parent_dir, 'Output', 'Data/')
    plots_dir = os.path.join(parent_dir, 'Output', 'Plots/')
    tables_dir = os.path.join(parent_dir, 'Output', 'Tables/')
    models_dir = os.path.join(parent_dir, 'Output', 'Models/')

    return {
        'parent_dir': parent_dir,
        'data_input': data_dir,
        'data': data_output_dir,
        'plots': plots_dir,
        'tables': tables_dir,
        'models': models_dir
    }