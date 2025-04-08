import os
import subprocess
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SAC training with TensorBoard on Vast.ai")
    parser.add_argument('--config', '-c', type=str, default='default', 
                        help="Configuration name to use (default: vast)")
    parser.add_argument('--port', '-p', type=int, default=6006, 
                        help="Port to use for TensorBoard (default: 6006)")
    parser.add_argument('--device', '-d', type=str, default=None,
                        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    args = parser.parse_args()
    
    # Start TensorBoard in background
    print(f"Starting TensorBoard server on port {args.port}...")
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", f"--logdir=C:/Users/Pedro/Documents/MAI/ATCI/ATCI-P1/runs", f"--port={args.port}", "--bind_all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)  # Give TensorBoard time to start
    
    if tb_process.poll() is not None:
        print("ERROR: TensorBoard failed to start!")
        stdout, stderr = tb_process.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        sys.exit(1)
    
    print(f"TensorBoard running on port {args.port}. Access via Vast.ai port forwarding.")
    
    try:
        # Start training with device option
        print(f"Starting training with config: {args.config}")
        train_cmd = [sys.executable, "src/main.py", f"--config={args.config}"]
        if args.device:
            train_cmd.extend(["--device", args.device])
        subprocess.run(train_cmd)
    finally:
        # Cleanup
        print("Training complete or interrupted. Stopping TensorBoard...")
        tb_process.terminate()
        try:
            tb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb_process.kill()

if __name__ == "__main__":
    main()