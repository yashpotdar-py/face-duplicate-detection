"""
Advanced visualization module for face duplicate detection results.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
from loguru import logger

class ResultVisualizer:
    """
    Create advanced visualizations for face duplicate detection results.
    """
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization results
        """
        self.viz_dir = Path(output_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"ResultVisualizer initialized, saving to {self.viz_dir}")
    
    def plot_similarity_heatmap(
        self, 
        similarity_matrix: np.ndarray, 
        face_database: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """Create a heatmap of face similarities."""
        if save_path is None:
            save_path = self.viz_dir / "similarity_heatmap.png"
        
        # Create labels from face database
        labels = []
        for i, face in enumerate(face_database):
            source = Path(face['source_path']).stem
            face_id = face['face_id']
            labels.append(f"{source[:10]}...\n{face_id}")
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        
        # Use a diverging colormap
        mask = np.zeros_like(similarity_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        sns.heatmap(
            similarity_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            xticklabels=labels,
            yticklabels=labels,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            center=0.5,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Face Similarity Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Face Index', fontsize=12)
        plt.ylabel('Face Index', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity heatmap saved to {save_path}")
        return str(save_path)
    
    def plot_cluster_visualization(
        self,
        duplicate_groups: List[Dict],
        face_database: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """Create a cluster visualization showing duplicate groups."""
        if save_path is None:
            save_path = self.viz_dir / "cluster_visualization.png"
        
        # Prepare data for plotting
        face_data = []
        cluster_id = 0
        color_palette = sns.color_palette("husl", len(duplicate_groups) + 1)
        
        # Add clustered faces - handle both old and new data structures
        for group in duplicate_groups:
            # Handle both list of dicts and dict with 'faces' key
            if isinstance(group, dict) and 'faces' in group:
                faces_in_group = group['faces']
            else:
                faces_in_group = group
            
            for face_info in faces_in_group:
                face_idx = next((i for i, f in enumerate(face_database) 
                               if f['face_id'] == face_info['face_id']), None)
                if face_idx is not None:
                    face_data.append({
                        'x': np.random.normal(cluster_id, 0.3),
                        'y': np.random.normal(0, 0.1),
                        'face_id': face_info['face_id'],
                        'source': Path(face_database[face_idx]['source_path']).stem,
                        'cluster': cluster_id,
                        'similarity': face_info.get('avg_similarity', face_info.get('similarity', 0))
                    })
            cluster_id += 1
        
        # Add unclustered faces
        clustered_ids = set()
        for group in duplicate_groups:
            if isinstance(group, dict) and 'faces' in group:
                faces_in_group = group['faces']
            else:
                faces_in_group = group
            for item in faces_in_group:
                clustered_ids.add(item['face_id'])
        
        unclustered_x = cluster_id
        
        for face in face_database:
            if face['face_id'] not in clustered_ids:
                face_data.append({
                    'x': np.random.normal(unclustered_x, 0.3),
                    'y': np.random.normal(0, 0.1),
                    'face_id': face['face_id'],
                    'source': Path(face['source_path']).stem,
                    'cluster': -1,
                    'similarity': 0
                })
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        df = pd.DataFrame(face_data)
        
        # Plot points
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            if cluster == -1:
                plt.scatter(cluster_data['x'], cluster_data['y'], 
                          c='lightgray', s=100, alpha=0.7, 
                          label='Unique Faces')
            else:
                plt.scatter(cluster_data['x'], cluster_data['y'], 
                          c=[color_palette[cluster]], s=150, alpha=0.8,
                          label=f'Duplicate Group {cluster + 1}')
        
        # Add annotations
        for _, row in df.iterrows():
            plt.annotate(f"{row['source'][:8]}\n{row['face_id']}", 
                        (row['x'], row['y']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.title('Face Clustering Results', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster Groups', fontsize=12)
        plt.ylabel('Faces', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualization saved to {save_path}")
        return str(save_path)
    
    def create_duplicate_montage(
        self,
        duplicate_groups: List[Dict],
        face_database: List[Dict],
        max_groups: int = 5,
        save_path: Optional[str] = None
    ) -> str:
        """Create a montage showing duplicate face groups side by side."""
        if save_path is None:
            save_path = self.viz_dir / "duplicate_montage.png"
        
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available for montage creation")
            return ""
        
        # Limit number of groups to display
        display_groups = duplicate_groups[:max_groups]
        
        if not display_groups:
            logger.warning("No duplicate groups to display")
            return ""
        
        # Calculate grid dimensions
        max_faces_per_group = max(len(group.get('faces', [])) for group in display_groups)
        rows = len(display_groups)
        cols = max_faces_per_group
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        # Process each group
        for group_idx, group in enumerate(display_groups):
            faces_in_group = group.get('faces', [])
            group_id = group.get('group_id', group_idx)
            
            for face_idx, face_info in enumerate(faces_in_group):
                ax = axes[group_idx][face_idx]
                
                # Find face in database
                face_data = next((f for f in face_database 
                                if f['face_id'] == face_info['face_id']), None)
                
                if face_data and 'source_path' in face_data:
                    try:
                        # Load and display image
                        img_path = face_data['source_path']
                        if Path(img_path).exists():
                            img = cv2.imread(img_path)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            ax.imshow(img_rgb)
                            
                            # Add title with similarity info
                            similarity = face_info.get('avg_similarity', 0)
                            title = f"{Path(img_path).stem}\n{face_info['face_id']}\nSim: {similarity:.3f}"
                            ax.set_title(title, fontsize=8)
                        else:
                            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                            ax.set_title(f"{face_info['face_id']}", fontsize=8)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error\n{str(e)[:20]}', ha='center', va='center')
                        ax.set_title(f"{face_info['face_id']}", fontsize=8)
                
                ax.axis('off')
            
            # Hide unused axes in the row
            for face_idx in range(len(faces_in_group), cols):
                axes[group_idx][face_idx].axis('off')
            
            # Add group label
            axes[group_idx][0].text(-0.1, 0.5, f'Group {group_id + 1}', 
                                  rotation=90, ha='center', va='center',
                                  transform=axes[group_idx][0].transAxes,
                                  fontsize=12, fontweight='bold')
        
        plt.suptitle('Duplicate Face Groups', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Duplicate montage saved to {save_path}")
        return str(save_path)
    
    def plot_performance_comparison(
        self,
        benchmark_results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """Create performance comparison charts from benchmark data."""
        if save_path is None:
            save_path = self.viz_dir / "performance_comparison.png"
        
        if not benchmark_results:
            logger.warning("No benchmark results to plot")
            return ""
        
        # Handle both old and new benchmark data structures
        if 'results' in benchmark_results:
            # Old structure with nested results
            results = benchmark_results['results']
            categories = []
            cpu_times = []
            gpu_times = []
            speedups = []
            
            for category, data in results.items():
                if 'cpu' in data and 'gpu' in data:
                    categories.append(category.replace('_', ' ').title())
                    cpu_times.append(data['cpu']['mean_time'])
                    gpu_times.append(data['gpu']['mean_time'])
                    speedups.append(data.get('speedup', data['cpu']['mean_time'] / data['gpu']['mean_time']))
        else:
            # New structure with direct arrays
            if 'gpu_results' in benchmark_results and 'cpu_results' in benchmark_results:
                categories = ['Face Detection']
                gpu_times = [np.mean(benchmark_results['gpu_results'])]
                cpu_times = [np.mean(benchmark_results['cpu_results'])]
                speedups = [benchmark_results.get('speedup', cpu_times[0] / gpu_times[0])]
            else:
                logger.warning("No valid benchmark data found")
                return ""
        
        if not categories:
            logger.warning("No valid benchmark data found")
            return ""
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Execution time comparison
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, gpu_times, width, label='GPU', alpha=0.8, color='lightcoral')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('CPU vs GPU Execution Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (cpu_time, gpu_time) in enumerate(zip(cpu_times, gpu_times)):
            ax1.text(i - width/2, cpu_time + max(cpu_times) * 0.01, f'{cpu_time:.3f}s', 
                    ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, gpu_time + max(gpu_times) * 0.01, f'{gpu_time:.3f}s', 
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Speedup chart
        colors = ['green' if s >= 1.0 else 'red' for s in speedups]
        bars = ax2.bar(categories, speedups, alpha=0.8, color=colors)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('GPU Speedup Factor')
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            color = 'green' if speedup >= 1.0 else 'red'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{speedup:.2f}x', ha='center', va='bottom', color=color, fontweight='bold')
        
        # 3. Detailed timing comparison
        if 'gpu_results' in benchmark_results and 'cpu_results' in benchmark_results:
            gpu_results = benchmark_results['gpu_results']
            cpu_results = benchmark_results['cpu_results']
            iterations = range(1, len(gpu_results) + 1)
            
            ax3.plot(iterations, cpu_results, 'o-', label='CPU', color='skyblue', linewidth=2, markersize=6)
            ax3.plot(iterations, gpu_results, 's-', label='GPU', color='lightcoral', linewidth=2, markersize=6)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Performance by Iteration')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            cpu_mean = np.mean(cpu_results)
            gpu_mean = np.mean(gpu_results)
            ax3.axhline(y=cpu_mean, color='skyblue', linestyle='--', alpha=0.7, label=f'CPU Mean: {cpu_mean:.3f}s')
            ax3.axhline(y=gpu_mean, color='lightcoral', linestyle='--', alpha=0.7, label=f'GPU Mean: {gpu_mean:.3f}s')
        else:
            ax3.text(0.5, 0.5, 'Iteration data\nnot available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Performance by Iteration')
        
        # 4. System information and statistics
        system_info = benchmark_results.get('system_info', {})
        test_params = benchmark_results.get('test_parameters', {})
        
        info_text = []
        if system_info:
            cpu_info = system_info.get('cpu', {})
            gpu_info = system_info.get('gpu', {})
            
            info_text.append("=== SYSTEM INFO ===")
            info_text.append(f"CPU: {cpu_info.get('cpu_count', 'N/A')} cores")
            info_text.append(f"RAM: {cpu_info.get('memory_total', 'N/A'):.1f} GB")
            info_text.append(f"GPU: {gpu_info.get('gpu_name', 'N/A')}")
            info_text.append(f"GPU Memory: {gpu_info.get('gpu_memory', 'N/A'):.1f} GB")
            info_text.append(f"CUDA: {gpu_info.get('cuda_version', 'N/A')}")
            info_text.append("")
            
        if test_params:
            info_text.append("=== TEST PARAMETERS ===")
            info_text.append(f"Test Size: {test_params.get('test_size', 'N/A')} images")
            info_text.append(f"Iterations: {test_params.get('iterations', 'N/A')}")
            info_text.append("")
        
        if categories:
            info_text.append("=== RESULTS ===")
            for i, cat in enumerate(categories):
                info_text.append(f"{cat}:")
                info_text.append(f"  CPU: {cpu_times[i]:.3f}s")
                info_text.append(f"  GPU: {gpu_times[i]:.3f}s")
                info_text.append(f"  Speedup: {speedups[i]:.2f}x")
        
        info_text.append("")
        info_text.append(f"Test Date: {benchmark_results.get('timestamp', 'N/A')}")
        
        ax4.text(0.05, 0.95, '\n'.join(info_text), transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Benchmark Details')
        ax4.axis('off')
        
        plt.suptitle('Performance Benchmark Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison saved to {save_path}")
        return str(save_path)
    
    def plot_statistical_summary(
        self,
        duplicate_groups: List[Dict],
        face_database: List[Dict],
        similarity_matrix: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create statistical summary charts."""
        if save_path is None:
            save_path = self.viz_dir / "statistical_summary.png"
        
        # Extract statistics
        total_faces = len(face_database)
        total_groups = len(duplicate_groups)
        total_duplicates = sum(len(group.get('faces', [])) for group in duplicate_groups)
        unique_faces = total_faces - total_duplicates
        
        # Group sizes
        group_sizes = [len(group.get('faces', [])) for group in duplicate_groups]
        
        # Similarity statistics
        similarities = []
        for group in duplicate_groups:
            faces_in_group = group.get('faces', [])
            for face in faces_in_group:
                sim = face.get('avg_similarity', face.get('similarity', 0))
                if sim > 0:
                    similarities.append(sim)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Face distribution pie chart
        labels = ['Duplicate Faces', 'Unique Faces']
        sizes = [total_duplicates, unique_faces]
        colors = ['lightcoral', 'lightblue']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=(0.05, 0))
        ax1.set_title('Face Distribution')
        
        # 2. Group size histogram
        if group_sizes:
            ax2.hist(group_sizes, bins=max(1, len(set(group_sizes))), alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Number of Faces per Group')
            ax2.set_ylabel('Number of Groups')
            ax2.set_title('Duplicate Group Size Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No duplicate\ngroups found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Group Size Distribution')
        
        # 3. Similarity distribution
        if similarities:
            ax3.hist(similarities, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Similarity Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Similarity Score Distribution')
            ax3.axvline(np.mean(similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(similarities):.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No similarity\ndata available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Similarity Distribution')
        
        # 4. Summary statistics table
        stats_data = [
            ['Total Faces', total_faces],
            ['Duplicate Groups', total_groups],
            ['Duplicate Faces', total_duplicates],
            ['Unique Faces', unique_faces],
            ['Duplication Rate', f'{(total_duplicates/total_faces*100):.1f}%' if total_faces > 0 else 'N/A']
        ]
        
        if similarities:
            stats_data.extend([
                ['Avg Similarity', f'{np.mean(similarities):.3f}'],
                ['Min Similarity', f'{np.min(similarities):.3f}'],
                ['Max Similarity', f'{np.max(similarities):.3f}']
            ])
        
        # Create table
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.suptitle('Face Duplicate Detection - Statistical Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical summary saved to {save_path}")
        return str(save_path)

    def generate_all_visualizations(
        self,
        similarity_matrix: Optional[np.ndarray] = None,
        duplicate_groups: Optional[List[Dict]] = None,
        face_database: Optional[List[Dict]] = None,
        benchmark_results: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate all available visualizations."""
        generated_files = {}
        
        # Similarity heatmap
        if similarity_matrix is not None and face_database is not None:
            try:
                path = self.plot_similarity_heatmap(similarity_matrix, face_database)
                generated_files['Similarity Heatmap'] = path
            except Exception as e:
                logger.error(f"Failed to generate similarity heatmap: {e}")
        
        # Cluster visualization
        if duplicate_groups is not None and face_database is not None:
            try:
                path = self.plot_cluster_visualization(duplicate_groups, face_database)
                generated_files['Cluster Visualization'] = path
            except Exception as e:
                logger.error(f"Failed to generate cluster visualization: {e}")
        
        # Duplicate montage
        if duplicate_groups is not None and face_database is not None and len(duplicate_groups) > 0:
            try:
                path = self.create_duplicate_montage(duplicate_groups, face_database)
                if path:  # Only add if successful
                    generated_files['Duplicate Montage'] = path
            except Exception as e:
                logger.error(f"Failed to generate duplicate montage: {e}")
        
        # Statistical summary
        if duplicate_groups is not None and face_database is not None:
            try:
                path = self.plot_statistical_summary(duplicate_groups, face_database, similarity_matrix)
                generated_files['Statistical Summary'] = path
            except Exception as e:
                logger.error(f"Failed to generate statistical summary plot: {e}")
        
        # Performance comparison
        if benchmark_results is not None:
            try:
                path = self.plot_performance_comparison(benchmark_results)
                if path:  # Only add if successful
                    generated_files['Performance Comparison'] = path
            except Exception as e:
                logger.error(f"Failed to generate performance comparison plot: {e}")
        
        logger.info(f"Generated {len(generated_files)} visualizations")
        return generated_files
