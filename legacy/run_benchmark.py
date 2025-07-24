#!/usr/bin/env python3
"""
Model Benchmark Runner Script
ANN vs LightGBM 성능 비교 벤치마크 실행 스크립트

사용법:
1. 합성 데이터로 벤치마크: python run_benchmark.py --synthetic
2. 실제 데이터로 벤치마크: python run_benchmark.py --real
3. 둘 다 실행: python run_benchmark.py --both
4. 반복 횟수 지정: python run_benchmark.py --real --iterations 500000
"""

import argparse
import sys
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='ANN vs LightGBM Model Benchmark')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Run benchmark with synthetic data')
    parser.add_argument('--real', action='store_true', 
                       help='Run benchmark with real data')
    parser.add_argument('--both', action='store_true', 
                       help='Run both synthetic and real data benchmarks')
    parser.add_argument('--iterations', type=int, default=1000000,
                       help='Number of iterations (default: 1000000)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with 10000 iterations')
    
    args = parser.parse_args()
    
    # 기본값 설정
    if not any([args.synthetic, args.real, args.both]):
        print("No benchmark type specified. Running both synthetic and real data benchmarks.")
        args.both = True
    
    # Quick 모드 설정
    if args.quick:
        args.iterations = 10000
        print(f"Quick mode: Using {args.iterations} iterations")
    
    print(f"Benchmark Configuration:")
    print(f"  Iterations: {args.iterations:,}")
    print(f"  Synthetic: {args.synthetic or args.both}")
    print(f"  Real: {args.real or args.both}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 필요한 모듈 import 확인
    try:
        import torch
        import lightgbm as lgb
        import sklearn
        import matplotlib
        import seaborn
        print("✓ All required packages are available")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install torch lightgbm scikit-learn matplotlib seaborn")
        sys.exit(1)
    
    # 합성 데이터 벤치마크
    if args.synthetic or args.both:
        print("\n" + "="*60)
        print("RUNNING SYNTHETIC DATA BENCHMARK")
        print("="*60)
        try:
            from model_benchmark import main as synthetic_main
            # iterations 파라미터를 전달하기 위해 모듈 수정
            import model_benchmark
            model_benchmark.NUM_ITERATIONS = args.iterations
            synthetic_main()
        except Exception as e:
            print(f"Error in synthetic benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    # 실제 데이터 벤치마크
    if args.real or args.both:
        print("\n" + "="*60)
        print("RUNNING REAL DATA BENCHMARK")
        print("="*60)
        try:
            from real_data_benchmark import main as real_main
            # iterations 파라미터를 전달하기 위해 모듈 수정
            import real_data_benchmark
            real_data_benchmark.NUM_ITERATIONS = args.iterations
            real_main()
        except Exception as e:
            print(f"Error in real data benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nBenchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 결과 파일 확인
    result_files = []
    if os.path.exists('benchmark_results.json'):
        result_files.append('benchmark_results.json')
    if os.path.exists('real_data_benchmark_results.json'):
        result_files.append('real_data_benchmark_results.json')
    if os.path.exists('model_benchmark_results.png'):
        result_files.append('model_benchmark_results.png')
    
    if result_files:
        print(f"\nGenerated result files:")
        for file in result_files:
            print(f"  - {file}")

if __name__ == "__main__":
    main() 