import subprocess
import sys
import os

def install_requirements():
    # Install requirements for dataset_build
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "dataset_build/requirements.txt"])
    
    # Install requirements for mixture_training
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "mixture_training/requirements.txt"])

def run_dataset_build(args):
    os.chdir("dataset_build")
    subprocess.run([sys.executable, "src/builder.py"] + args)
    os.chdir("..")

def run_mixture_training(args):
    os.chdir("mixture_training")
    subprocess.run([sys.executable, "src/training.py"] + args)
    os.chdir("..")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and run dataset_build and mixture_training")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing requirements")
    parser.add_argument("--dataset-build-args", nargs=argparse.REMAINDER, help="Arguments for dataset_build")
    parser.add_argument("--mixture-training-args", nargs=argparse.REMAINDER, help="Arguments for mixture_training")
    
    args = parser.parse_args()
    
    if not args.skip_install:
        print("Installing requirements...")
        install_requirements()
    
    if args.dataset_build_args:
        print("Running dataset_build...")
        run_dataset_build(args.dataset_build_args)
    
    if args.mixture_training_args:
        print("Running mixture_training...")
        run_mixture_training(args.mixture_training_args)

    if not args.dataset_build_args and not args.mixture_training_args:
        print("No arguments provided for dataset_build or mixture_training. Skipping execution.")
