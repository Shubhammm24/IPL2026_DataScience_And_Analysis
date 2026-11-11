"""
Automated Experiment Runner & Terminal Output Exporter
======================================================
Course: Data Science and Analytics (IT 411)

This utility executes each experiment script, captures stdout and stderr,
and saves the full console output into an 'outputs/' directory as text files.
"""

import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Discover all experiment scripts sorted sequentially
exp_files = sorted([f for f in os.listdir(BASE_DIR) if f.startswith("Exp") and f.endswith(".py")])

print("="*70)
print(f"DSA LAB FILE (IT 411) — BATCH EXPERIMENT RUNNER")
print(f"Total experiments found: {len(exp_files)}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)

for idx, script_name in enumerate(exp_files, 1):
    script_path = os.path.join(BASE_DIR, script_name)
    base_name = os.path.splitext(script_name)[0]
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_output.txt")
    
    print(f"\n[{idx}/{len(exp_files)}] Executing: {script_name} ...", end=" ", flush=True)
    start_time = time.time()
    
    try:
        # Run python script and capture stdout + stderr
        res = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        duration = time.time() - start_time
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(f"======================================================================\n")
            out_f.write(f"EXPERIMENT LOG: {script_name}\n")
            out_f.write(f"Executed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_f.write(f"Exit Code: {res.returncode} (Elapsed: {duration:.2f}s)\n")
            out_f.write(f"======================================================================\n\n")
            
            if res.stdout:
                out_f.write(res.stdout)
            if res.stderr:
                out_f.write("\n--- [STDERR / LOGS] ---\n")
                out_f.write(res.stderr)
                
        status = "COMPLETED" if res.returncode == 0 else f"EXIT CODE {res.returncode}"
        print(f"[{status}] in {duration:.1f}s -> Saved to outputs/{os.path.basename(output_file)}")
        
    except Exception as e:
        print(f"[ERROR]: {e}")

print("\n" + "="*70)
print(f"All experiment executions finished! Logs available in: {OUTPUT_DIR}")
print("="*70)

