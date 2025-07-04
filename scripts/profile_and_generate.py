#!/usr/bin/env python3
"""
PreFree SpMV Auto-tuning and Code Generation Script
This script automatically profiles different kernel configurations and generates optimized CUDA code
"""

import os
import sys
import subprocess
import json
import yaml
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpMVAutoTuner:
    def __init__(self, config_path='config.yaml'):
        """Initialize the auto-tuner with configuration"""
        self.config = self.load_config(config_path)
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.build_dir = self.project_root / 'build'
        self.results = {}
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def classify_matrix_size(self, nnz):
        """Classify matrix based on nnz count"""
        thresholds = self.config['size_thresholds']
        if nnz < thresholds['small_upper']:
            return 'small'
        elif nnz < thresholds['medium_upper']:
            return 'medium'
        else:
            return 'large'
    
    def get_matrix_info(self, matrix_path):
        """Extract matrix information by running a simple test"""
        test_exe = self.build_dir / 'cuda_perftest'
        if not test_exe.exists():
            logger.error(f"Test executable not found: {test_exe}")
            return None
            
        try:
            result = subprocess.run(
                [str(test_exe), matrix_path],
                capture_output=True,
                text=True,
                timeout=240
            )
            
            # Parse output to get matrix dimensions and nnz
            for line in result.stdout.split('\n'):
                if 'nnz =' in line:
                    # Format: "Matrix info: M x N, nnz = X, symmetric = Y"
                    parts = line.split(',')
                    nnz_part = [p for p in parts if 'nnz =' in p][0]
                    nnz = int(nnz_part.split('=')[1].strip())
                    return {'nnz': nnz, 'size_class': self.classify_matrix_size(nnz)}
            
            # Alternative parsing for matrix name line
            for line in result.stdout.split('\n'):
                if line.startswith('===') and line.endswith('==='):
                    matrix_name = line.strip('=').strip()
                    # Run again to get full output
                    return {'nnz': 1000000, 'size_class': 'medium'}  # Default values
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while processing {matrix_path}")
        except Exception as e:
            logger.error(f"Error processing {matrix_path}: {e}")
            
        return None
    
    def compile_test_version(self, code_version, p_nnz):
        """Compile a test version with specific parameters"""
        logger.info(f"Compiling test version: code={code_version}, P_NNZ={p_nnz}")
        
        # Create temporary source file with modifications
        temp_src = self.project_root / 'src' / 'spmv_kernel' / f'preFreeSpMV_test_{code_version}_{p_nnz}.cu'
        original_src = self.project_root / 'src' / 'spmv_kernel' / 'preFreeSpMV.cu'
        
        # Read original source
        with open(original_src, 'r') as f:
            content = f.read()
        
        # Modify P_NNZ
        content = content.replace('const int P_NNZ = 4;', f'const int P_NNZ = {p_nnz};')
        
        # Find and modify the kernel code section
        # We need to define TEST_CODE_VERSION at the beginning
        insert_pos = content.find('#include "common.h"')
        if insert_pos != -1:
            insert_pos = content.find('\n', insert_pos) + 1
            content = content[:insert_pos] + f'#define TEST_CODE_VERSION {code_version}\n' + content[insert_pos:]
        
        # Replace the branching section with a compile-time switch
        kernel_start = content.find('// reduction stage')
        kernel_end = content.find('////////////////////////////////////////////////////////////////////\n}', kernel_start)
        
        if kernel_start != -1 and kernel_end != -1:
            # Extract the reduction code
            reduction_code = '''
  // reduction stage
  const int n_reduce_rows_num = tileEndRow - tileStartRow;

#if TEST_CODE_VERSION == 1
  // code 1: No Branch code
  red_row_thread<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                      tileStartRow, tileEndRow,
                                      d_ptr, middle_s, d_y);
#elif TEST_CODE_VERSION == 2
  // code 2: Dual Branch code
  if (n_reduce_rows_num == 1)
  {
    red_row_block<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                       tileStartRow,
                                       d_ptr, middle_s, d_y);
  }
  else
  {
    red_row_thread<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                        tileStartRow, tileEndRow,
                                        d_ptr, middle_s, d_y);
  }
#elif TEST_CODE_VERSION == 3
  // code 3: Three-level Balanced Branch code
  if (n_reduce_rows_num > BLOCK_SIZE / 4)
  {
    red_row_thread<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                        tileStartRow, tileEndRow,
                                        d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num == 1)
  {
    red_row_block<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                       tileStartRow,
                                       d_ptr, middle_s, d_y);
  }
  else
  {
    red_row_vector<tileNnz, BLOCK_SIZE, 4>(
        threadIdx.x, blockIdx.x,
        tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
  }
#elif TEST_CODE_VERSION == 4
  // code 4: Three-level imbalanced Branch code
  if (n_reduce_rows_num > BLOCK_SIZE / 4)
  {
    red_row_thread<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                      tileStartRow, tileEndRow,
                                      d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num == 1)
  {
    red_row_block<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                     tileStartRow,
                                     d_ptr, middle_s, d_y);
  }
  else
  {
    __shared__ bool use_uneven_path;
    if ((threadIdx.x >> 5) == 0)
    {
      bool decision = is_imbalanced_warp(tileStartRow, n_reduce_rows_num, d_ptr, threadIdx.x);
      if (threadIdx.x == 0)
      {
        use_uneven_path = decision;
      }
    }
    __syncthreads();
    dispatch_reduction_strategy<tileNnz, BLOCK_SIZE, 4>(
        use_uneven_path, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
  }
#elif TEST_CODE_VERSION == 5
  // code 5: Fine-grained parallel branching
  if (n_reduce_rows_num > BLOCK_SIZE)
  {
    red_row_thread<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                      tileStartRow, tileEndRow,
                                      d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num == 1)
  {
    red_row_block<tileNnz, BLOCK_SIZE>(threadIdx.x, blockIdx.x,
                                     tileStartRow,
                                     d_ptr, middle_s, d_y);
  }
  else
  {
    const unsigned int vector_size = calculate_vector_size<BLOCK_SIZE>(n_reduce_rows_num);
    switch (vector_size)
    {
    case 32:// 2-8
      red_row_vector<tileNnz, BLOCK_SIZE, 32>(
          threadIdx.x, blockIdx.x,
          tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
      break;
    case 16://9-16
      red_row_vector<tileNnz, BLOCK_SIZE, 16>(
          threadIdx.x, blockIdx.x,
          tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
      break;
    case 8://17-32
      red_row_vector<tileNnz, BLOCK_SIZE, 8>(
          threadIdx.x, blockIdx.x,
          tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
      break;
    case 4://33-64
      red_row_vector<tileNnz, BLOCK_SIZE, 4>(
          threadIdx.x, blockIdx.x,
          tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
      break;
    case 2://65-128
      red_row_vector<tileNnz, BLOCK_SIZE, 2>(
          threadIdx.x, blockIdx.x,
          tileStartRow, tileEndRow, d_ptr, middle_s, d_y);
      break;
    }
  }
#endif
'''
            content = content[:kernel_start] + reduction_code + '\n' + content[kernel_end:]
        
        # Write modified source
        with open(temp_src, 'w') as f:
            f.write(content)
        
        # Update CMakeLists.txt temporarily
        cmake_file = self.project_root / 'CMakeLists.txt'
        with open(cmake_file, 'r') as f:
            cmake_content = f.read()
        
        # Modify to use test source
        test_cmake = cmake_content.replace(
            'file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu")',
            f'set(cuda_sources "${{CMAKE_CURRENT_SOURCE_DIR}}/src/spmv_kernel/preFreeSpMV_test_{code_version}_{p_nnz}.cu")'
        )
        
        with open(cmake_file, 'w') as f:
            f.write(test_cmake)
        
        # Compile
        os.chdir(self.build_dir)
        result = subprocess.run(['cmake', '..'], capture_output=True)
        if result.returncode != 0:
            logger.error(f"CMake failed: {result.stderr.decode()}")
            return False
            
        result = subprocess.run(['make', '-j4'], capture_output=True)
        if result.returncode != 0:
            logger.error(f"Make failed: {result.stderr.decode()}")
            return False
        
        # Restore original CMakeLists.txt
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        
        # Clean up temp source
        temp_src.unlink()
        
        return True
    
    def run_performance_test(self, matrix_path, code_version, p_nnz):
        """Run performance test and extract results"""
        test_exe = self.build_dir / 'cuda_perftest'
        
        try:
            result = subprocess.run(
                [str(test_exe), matrix_path],
                capture_output=True,
                text=True,
                timeout=240
            )
            
            if result.returncode != 0:
                logger.error(f"Test failed for {matrix_path}")
                return None
            
            # Parse performance results
            our_perf = None
            cusparse_perf = None
            
            for line in result.stdout.split('\n'):
                if 'our_perf:' in line:
                    our_perf = float(line.split('ms')[0].split(':')[1].strip())
                elif 'cusparse_perf:' in line:
                    cusparse_perf = float(line.split('ms')[0].split(':')[1].strip())
            
            if our_perf and cusparse_perf:
                speedup = cusparse_perf / our_perf
                return {
                    'our_perf': our_perf,
                    'cusparse_perf': cusparse_perf,
                    'speedup': speedup,
                    'code_version': code_version,
                    'p_nnz': p_nnz
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout for {matrix_path}")
        except Exception as e:
            logger.error(f"Error running test: {e}")
            
        return None
    
    def profile_all_configurations(self):
        """Profile all matrices with all configurations"""
        logger.info("Starting comprehensive profiling...")
        
        # Initialize results structure
        for size_class in ['small', 'medium', 'large']:
            self.results[size_class] = {
                'matrices': [],
                'best_config': None,
                'avg_speedup': 0
            }
        
        # Get all test matrices
        test_matrices = []
        for size_class, matrices in self.config['test_matrices'].items():
            for matrix in matrices:
                matrix_path = Path(matrix)
                if matrix_path.exists():
                    info = self.get_matrix_info(str(matrix_path))
                    if info:
                        test_matrices.append({
                            'path': str(matrix_path),
                            'size_class': size_class,
                            'nnz': info['nnz']
                        })
                else:
                    logger.warning(f"Matrix file not found: {matrix}")
        
        # Test all configurations
        configurations = []
        for code_version in self.config['code_versions']:
            for p_nnz in self.config['p_nnz_values']:
                configurations.append((code_version, p_nnz))
        
        # Profile each configuration
        for code_version, p_nnz in configurations:
            logger.info(f"\nTesting configuration: code={code_version}, P_NNZ={p_nnz}")
            
            # Compile this version
            if not self.compile_test_version(code_version, p_nnz):
                logger.error(f"Failed to compile version code={code_version}, P_NNZ={p_nnz}")
                continue
            
            # Test on all matrices
            for matrix_info in test_matrices:
                result = self.run_performance_test(
                    matrix_info['path'], 
                    code_version, 
                    p_nnz
                )
                
                if result:
                    size_class = matrix_info['size_class']
                    self.results[size_class]['matrices'].append({
                        'matrix': matrix_info['path'],
                        'nnz': matrix_info['nnz'],
                        'result': result
                    })
        
        # Analyze results to find best configuration for each size class
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze profiling results to find best configuration for each size class"""
        logger.info("\nAnalyzing profiling results...")
        
        for size_class in ['small', 'medium', 'large']:
            results = self.results[size_class]['matrices']
            if not results:
                continue
            
            # Group by configuration
            config_performance = {}
            for item in results:
                result = item['result']
                config_key = (result['code_version'], result['p_nnz'])
                
                if config_key not in config_performance:
                    config_performance[config_key] = []
                
                config_performance[config_key].append(result['speedup'])
            
            # Calculate average speedup for each configuration
            best_config = None
            best_avg_speedup = 0
            
            for config, speedups in config_performance.items():
                avg_speedup = np.mean(speedups)
                if avg_speedup > best_avg_speedup:
                    best_avg_speedup = avg_speedup
                    best_config = config
            
            self.results[size_class]['best_config'] = {
                'code_version': best_config[0],
                'p_nnz': best_config[1],
                'avg_speedup': best_avg_speedup
            }
            
            logger.info(f"{size_class.upper()} matrices: Best config is code={best_config[0]}, "
                       f"P_NNZ={best_config[1]} with avg speedup {best_avg_speedup:.3f}x")
    
    def generate_optimized_code(self):
        """Generate optimized CUDA code using Jinja2 templates"""
        logger.info("\nGenerating optimized CUDA code...")
        
        # Setup Jinja2 environment
        template_dir = self.script_dir / 'templates'
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('preFreeSpMV_template.cu.j2')
        
        # Prepare template context
        context = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'small_config': self.results['small']['best_config'],
            'medium_config': self.results['medium']['best_config'],
            'large_config': self.results['large']['best_config'],
            'size_thresholds': self.config['size_thresholds']
        }
        
        # Generate code
        generated_code = template.render(context)
        
        # Write to file
        output_path = self.project_root / 'src' / 'spmv_kernel' / 'preFreeSpMV_optimized.cu'
        with open(output_path, 'w') as f:
            f.write(generated_code)
        
        logger.info(f"Generated optimized code: {output_path}")
        
        # Save profiling results
        results_path = self.script_dir / 'profiling_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved profiling results: {results_path}")
    
    def update_cmake_for_optimized(self):
        """Update CMakeLists.txt to use optimized version"""
        cmake_file = self.project_root / 'CMakeLists.txt'
        
        with open(cmake_file, 'r') as f:
            content = f.read()
        
        # Update to include optimized version
        content = content.replace(
            'file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu")',
            'file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/spmv_kernel/preFreeSpMV_optimized.cu")'
        )
        
        with open(cmake_file, 'w') as f:
            f.write(content)
        
        logger.info("Updated CMakeLists.txt for optimized build")
    
    def build_final_version(self):
        """Build the final optimized version"""
        logger.info("\nBuilding final optimized version...")
        
        os.chdir(self.build_dir)
        subprocess.run(['cmake', '..'], check=True)
        subprocess.run(['make', '-j4'], check=True)
        
        logger.info("Build completed successfully!")
    
    def run(self):
        """Main execution flow"""
        logger.info("=== PreFree SpMV Auto-tuning System ===")
        
        # Ensure build directory exists
        self.build_dir.mkdir(exist_ok=True)
        
        # Step 1: Initial build to create test executable
        logger.info("\nStep 1: Initial build...")
        os.chdir(self.build_dir)
        subprocess.run(['cmake', '..'], check=True)
        subprocess.run(['make', '-j4'], check=True)
        
        # Step 2: Profile all configurations
        logger.info("\nStep 2: Profiling all configurations...")
        self.profile_all_configurations()
        
        # Step 3: Generate optimized code
        logger.info("\nStep 3: Generating optimized code...")
        self.generate_optimized_code()
        
        # Step 4: Update CMake and build final version
        logger.info("\nStep 4: Building final version...")
        self.update_cmake_for_optimized()
        self.build_final_version()
        
        logger.info("\n=== Auto-tuning completed successfully! ===")
        logger.info("You can now run the optimized version with:")
        logger.info("  ./build/cuda_perftest <matrix.mtx>")


if __name__ == "__main__":
    # Check if config file is provided
    config_file = 'scripts/config.yaml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Run auto-tuner
    tuner = SpMVAutoTuner(config_file)
    tuner.run()