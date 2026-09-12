#!/usr/bin/env python3
"""
Simple test script for the DataVisualizer module.
"""
import numpy as np
import pandas as pd
from src.musikalyze.visualizer import DataVisualizer

def create_test_dataframe():
    """Create a sample DataFrame similar to MusicBatch.analyze() output."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample data that resembles music analysis results
    data = {
        'track_id': [f'track_{i}' for i in range(n_samples)],
        'duration': np.random.uniform(120, 300, n_samples),  # 2-5 minutes
        'loudness': np.random.uniform(-30, 0, n_samples),    # -30 to 0 dB
        'tempo': np.random.uniform(60, 200, n_samples),      # 60-200 BPM
        'energy': np.random.uniform(0, 1, n_samples),       # 0-1 scale
        'danceability': np.random.uniform(0, 1, n_samples),  # 0-1 scale
        'valence': np.random.uniform(0, 1, n_samples),      # 0-1 scale
        'acousticness': np.random.uniform(0, 1, n_samples), # 0-1 scale
    }
    
    return pd.DataFrame(data)

def main():
    # Create test data
    df = create_test_dataframe()
    print(f"Created test DataFrame with shape: {df.shape}")
    print(df.head())
    
    # Test 1: Plotly backend
    print("\n=== Testing Plotly Backend ===")
    try:
        visualizer = DataVisualizer(backend='plotly')
        visualizer.set_data(df)  # Set data before plotting
        print("✓ DataVisualizer instantiated with plotly backend")
        
        # Test distribution plot with a specific column
        fig = visualizer.plot_distribution('loudness')
        print("✓ Distribution plot created successfully")
        
        # Test correlation matrix
        fig = visualizer.plot_correlation_matrix()
        print("✓ Correlation matrix created successfully")
        
    except Exception as e:
        print(f"✗ Plotly backend failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Matplotlib backend
    print("\n=== Testing Matplotlib Backend ===")
    try:
        visualizer = DataVisualizer(backend='matplotlib')
        visualizer.set_data(df)  # Set data before plotting
        print("✓ DataVisualizer instantiated with matplotlib backend")
        
        # Test distribution plot with a specific column
        fig = visualizer.plot_distribution('loudness')
        print("✓ Distribution plot created successfully")
        
        # Test correlation matrix
        fig = visualizer.plot_correlation_matrix()
        print("✓ Correlation matrix created successfully")
        
    except Exception as e:
        print(f"✗ Matplotlib backend failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()