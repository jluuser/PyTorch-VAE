#!/usr/bin/env python3
"""
Curve Visualization Script

This script reads curve npy files, prints array information, 
and visualizes curves with different coloring schemes.
Attention: This file is used for the raw npy files
Author: Claude Sonnet 4
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def load_curve_data(npy_path: Path) -> dict:
    """
    Load curve data from npy file.
    
    Args:
        npy_path: Path to npy file
        
    Returns:
        Dictionary containing curve data
    """
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None


def _print_ss_info(ss_labels: list) -> None:
    """
    Helper to print SS statistics and aligned labels.
    """
    total = len(ss_labels)
    if not total:
        print("  - Not available")
        return
        
    # Print statistics in a compact format
    h_count = ss_labels.count('h')
    s_count = ss_labels.count('s')
    l_count = ss_labels.count('l')
    print(f"  Stats: H {h_count*100/total:.1f}% | S {s_count*100/total:.1f}% | L {l_count*100/total:.1f}%")
    
    # Print the full sequence in chunks for readability
    print("  Sequence:")
    chunk_size = 80
    for i in range(0, total, chunk_size):
        chunk = ss_labels[i:i+chunk_size]
        start_idx = i + 1
        end_idx = i + len(chunk)
        print(f"    [{start_idx:4d}-{end_idx:4d}]: {''.join(chunk)}")


def print_curve_info(data: dict, npy_path: Path) -> None:
    """
    Print detailed information about curve data.
    
    Args:
        data: Curve data dictionary
        npy_path: Path to npy file
    """
    print(f"\n?? File: {npy_path.name}")
    print(f"?? Path: {npy_path.parent}")
    print(f"?? Original PDB: {data.get('original_pdb', 'Unknown')}")
    
    # Extract data with new structure
    curve_coords = data['curve_coords']
    ca_coords = data.get('ca_coords')
    
    # Metadata
    print(f"\n?? Metadata:")
    print(f"  - Residues: {data.get('num_residues', 'N/A')}")
    print(f"  - Active Curve Points: {data.get('num_curve_points', 'N/A')}")
    print(f"  - Padded Curve Points: {data.get('total_curve_points_with_padding', 'N/A')}")
    print(f"  - Interpolation: {data.get('interpolation_method', 'N/A')}")

    # Secondary structure analysis for both levels
    print(f"\n?? Original Residue-Level SS ({len(data.get('residue_ss_labels', []))} residues):")
    _print_ss_info(data.get('residue_ss_labels', []))
    
    print(f"\n?? Mapped Curve-Level SS ({len(data.get('ss_labels', []))} points):")
    _print_ss_info(data.get('ss_labels', []))

    # Filter out zero-padded points for coordinate analysis
    active_coords = curve_coords[np.any(curve_coords != 0, axis=1)]
    
    # Raw coordinate data preview
    print(f"\n?? Raw Data Preview (first 5):")
    if ca_coords is not None:
        print(f"  - Original CA Coords: {len(ca_coords)} points")
        for i, coord in enumerate(ca_coords[:5]):
            print(f"    [{i+1:3d}] {np.array2string(coord, precision=3)}")

    print(f"  - Active Curve Coords: {len(active_coords)} points")
    for i, coord in enumerate(active_coords[:5]):
        print(f"    [{i+1:3d}] {np.array2string(coord, precision=3)}")


def find_correspondence(bb_coords: np.ndarray, curve_coords: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find correspondence between backbone and curve coordinates using k-nearest neighbors.
    
    Args:
        bb_coords: Backbone CA coordinates [num_residues, 3]
        curve_coords: Curve coordinates [num_curve_points, 3]
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (distances, indices) for each curve point to its k nearest backbone residues
    """
    tree = cKDTree(bb_coords)
    distances, indices = tree.query(curve_coords, k=k)
    return distances, indices


def visualize_curve(data: dict, npy_path: Path, save_dir: Optional[Path] = None, show_rainbow: bool = False) -> None:
    """
    Visualize curve with different coloring schemes.
    
    Args:
        data: Curve data dictionary
        npy_path: Path to npy file
        save_dir: Directory to save plots (optional)
        show_rainbow: Whether to show rainbow colors in curve visualization
    """
    curve_coords = data['curve_coords']
    # Distinguish between residue-level and curve-level SS labels
    residue_ss_labels = data.get('residue_ss_labels', [])
    curve_ss_labels = data.get('ss_labels', [])
    bb_coords = data.get('bb_coords', data.get('ca_coords', None))
    
    # Filter out zero-padded points
    non_zero_mask = np.any(curve_coords != 0, axis=1)
    active_coords = curve_coords[non_zero_mask]
    
    if len(active_coords) < 2:
        print("? Not enough active curve points for visualization")
        return
    
    if bb_coords is None:
        print("? No backbone coordinates found in data")
        return
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: PDB CA coordinates colored by secondary structure
    ax1 = fig.add_subplot(131, projection='3d')
    
    ss_color_map = {'h': 'red', 's': 'green', 'l': 'blue'}
    
    # Generate colors for backbone using residue_ss_labels
    if len(residue_ss_labels) != len(bb_coords):
        print(f"?? Warning: Mismatch between backbone coordinates ({len(bb_coords)}) and residue SS labels ({len(residue_ss_labels)}). Using fallback grey color.")
        bb_ss_colors = ['grey'] * len(bb_coords)
    else:
        bb_ss_colors = [ss_color_map.get(ss, 'grey') for ss in residue_ss_labels]

    # Plot backbone as connected points
    for i in range(len(bb_coords) - 1):
        ax1.plot(bb_coords[i:i+2, 0], 
                bb_coords[i:i+2, 1], 
                bb_coords[i:i+2, 2], 
                color=bb_ss_colors[i], linewidth=2, alpha=0.8)
    
    ax1.scatter(bb_coords[:, 0], bb_coords[:, 1], bb_coords[:, 2], 
               c=bb_ss_colors, s=20, alpha=0.7)
    
    ax1.set_title('PDB CA Coordinates\n(Red=Helix, Green=Sheet, Blue=Loop)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (?)')
    ax1.set_ylabel('Y (?)')
    ax1.set_zlabel('Z (?)')
    
    # Plot 2: Curve coordinates with SS coloring (default) or rainbow + SS overlay
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Generate colors for curve using curve_ss_labels, with fallback to interpolation
    if len(curve_ss_labels) != len(active_coords):
        print(f"?? Warning: Mismatch between active curve points ({len(active_coords)}) and curve SS labels ({len(curve_ss_labels)}). Falling back to interpolation from residue labels.")
        if len(residue_ss_labels) > 0:
            ss_indices = np.linspace(0, len(residue_ss_labels) - 1, len(active_coords))
            curve_ss = [residue_ss_labels[min(int(np.round(idx)), len(residue_ss_labels) - 1)] for idx in ss_indices]
        else:
            curve_ss = ['l'] * len(active_coords)  # Default to loop
    else:
        curve_ss = curve_ss_labels

    curve_ss_colors = [ss_color_map.get(ss, 'grey') for ss in curve_ss]
    
    if show_rainbow:
        # Create rainbow colors for curve points
        rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(active_coords)))
        
        # Plot curve with rainbow colors
        for i in range(len(active_coords) - 1):
            ax2.plot(active_coords[i:i+2, 0], 
                    active_coords[i:i+2, 1], 
                    active_coords[i:i+2, 2], 
                    color=rainbow_colors[i], linewidth=3, alpha=0.8)
        
        # Overlay SS-colored points
        ax2.scatter(active_coords[:, 0], active_coords[:, 1], active_coords[:, 2], 
                   c=curve_ss_colors, s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Curve Coordinates\n(Rainbow + SS Overlay)', 
                     fontsize=12, fontweight='bold')
    else:
        # Plot curve with SS colors only
        for i in range(len(active_coords) - 1):
            ax2.plot(active_coords[i:i+2, 0], 
                    active_coords[i:i+2, 1], 
                    active_coords[i:i+2, 2], 
                    color=curve_ss_colors[i], linewidth=3, alpha=0.8)
        
        ax2.scatter(active_coords[:, 0], active_coords[:, 1], active_coords[:, 2], 
                   c=curve_ss_colors, s=40, alpha=0.7)
        
        ax2.set_title('Curve Coordinates\n(SS Colored)', 
                     fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (?)')
    ax2.set_ylabel('Y (?)')
    ax2.set_zlabel('Z (?)')
    
    # Plot 3: Correspondence between backbone and curve
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Find correspondence using k-nearest neighbors
    distances, indices = find_correspondence(bb_coords, active_coords, k=3)
    
    # Plot backbone coordinates with a lower zorder
    ax3.scatter(bb_coords[:, 0], bb_coords[:, 1], bb_coords[:, 2], 
               c='lightblue', s=15, alpha=0.6, label='Backbone CA', zorder=1)
    
    # Plot curve coordinates with a medium zorder
    ax3.scatter(active_coords[:, 0], active_coords[:, 1], active_coords[:, 2], 
               c='orange', s=30, alpha=0.8, label='Curve Points', zorder=2)
    
    # Draw correspondence lines for a sample of points to improve clarity and visibility
    num_connections = min(15, len(active_coords))
    # Sample points evenly along the curve instead of just taking the first N
    sample_indices = np.linspace(0, len(active_coords) - 1, num_connections, dtype=int)

    for i in sample_indices:
        curve_point = active_coords[i]
        # Connect to nearest backbone residue
        nearest_bb_idx = indices[i] if np.isscalar(indices[i]) else indices[i][0]
        nearest_bb_point = bb_coords[nearest_bb_idx]
        
        # Use thicker, more opaque dashed lines and a higher zorder to ensure they are visible
        ax3.plot([curve_point[0], nearest_bb_point[0]], 
                [curve_point[1], nearest_bb_point[1]], 
                [curve_point[2], nearest_bb_point[2]], 
                'k--', alpha=0.7, linewidth=1.2, zorder=3)
    
    ax3.set_title(f'Backbone-Curve Correspondence\n({num_connections} connections shown)', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (?)')
    ax3.set_ylabel('Y (?)')
    ax3.set_zlabel('Z (?)')
    ax3.legend()
    
    # Add overall legend for SS coloring
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Helix'),
        Line2D([0], [0], color='green', lw=2, label='Sheet'),
        Line2D([0], [0], color='blue', lw=2, label='Loop')
    ]
    
    # Add legend to the second subplot
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle(f'Protein Structure Visualization: {npy_path.stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'{npy_path.stem}_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"?? Saved plot to: {save_path}")
    
    plt.show()


def get_npy_files(input_source: str, regex_pattern: Optional[str] = None) -> List[Path]:
    """
    Get list of npy files from input source.
    
    Args:
        input_source: Directory path or single file path
        regex_pattern: Regex pattern to filter files
        
    Returns:
        List of npy file paths
    """
    input_path = Path(input_source)
    
    if input_path.is_file() and input_path.suffix == '.npy':
        return [input_path]
    
    elif input_path.is_dir():
        pattern = regex_pattern or r'.*_curve\.npy$'
        npy_files = []
        
        for file_path in input_path.rglob('*.npy'):
            if re.match(pattern, file_path.name, re.IGNORECASE):
                npy_files.append(file_path)
                
        return npy_files
    
    else:
        raise ValueError(f"Input source {input_source} is not a valid file or directory")


def sample_files(files: List[Path], n_samples: int, seed: Optional[int] = None) -> List[Path]:
    """
    Randomly sample n files from the list.
    
    Args:
        files: List of file paths
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled file paths
    """
    if seed is not None:
        random.seed(seed)
    
    n_samples = min(n_samples, len(files))
    return random.sample(files, n_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize curve data from npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single file (default: SS colored)
  python visualize_curve.py --input curve_001.npy

  # Visualize with rainbow colors
  python visualize_curve.py --input curve_001.npy --rainbow

  # Visualize all curves in directory
  python visualize_curve.py --input ./curves --batch

  # Sample 5 random curves
  python visualize_curve.py --input ./curves --sample 5

  # Batch process with regex filter
  python visualize_curve.py --input ./curves --batch --regex ".*protein.*_curve\.npy$"

  # Save plots to directory
  python visualize_curve.py --input ./curves --sample 3 --save-plots ./plots
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input: npy file OR directory containing npy files')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all files in directory')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Randomly sample N files from directory')
    parser.add_argument('--regex', '-r', default=None,
                        help='Regex pattern to filter npy files')
    parser.add_argument('--save-plots', '-p', default=None,
                        help='Directory to save plot images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--print-only', action='store_true',
                        help='Only print array info, skip visualization')
    parser.add_argument('--rainbow', action='store_true',
                        help='Show rainbow colors in curve visualization (default: SS colors only)')
    
    args = parser.parse_args()
    
    # Get npy files
    print("?? Discovering npy files...")
    npy_files = get_npy_files(args.input, args.regex)
    
    if not npy_files:
        print("? No npy files found!")
        return
    
    print(f"?? Found {len(npy_files)} npy files")
    
    # Handle sampling or batch processing
    if args.sample is not None:
        npy_files = sample_files(npy_files, args.sample, args.seed)
        print(f"?? Sampled {len(npy_files)} files")
    elif not args.batch and len(npy_files) > 1:
        # If multiple files found but no batch flag, process only the first one
        npy_files = npy_files[:1]
        print(f"?? Processing first file only. Use --batch to process all files.")
    
    # Create save directory if specified
    save_dir = None
    if args.save_plots:
        save_dir = Path(args.save_plots)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"?? Plots will be saved to: {save_dir}")
    
    # Process files
    for i, npy_path in enumerate(npy_files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(npy_files)}: {npy_path.name}")
        print('='*60)
        
        # Load data
        data = load_curve_data(npy_path)
        if data is None:
            continue
        
        # Print information
        print_curve_info(data, npy_path)
        
        # Visualize if not print-only mode
        if not args.print_only:
            print(f"\n?? Generating visualization...")
            try:
                visualize_curve(data, npy_path, save_dir, show_rainbow=args.rainbow)
            except Exception as e:
                print(f"? Visualization failed: {e}")
        
        # Add separator for multiple files
        if len(npy_files) > 1 and i < len(npy_files) - 1:
            continue
            # input("Press Enter to continue to next file...")
    
    print(f"\n? Completed processing {len(npy_files)} files")


if __name__ == '__main__':
    main() 