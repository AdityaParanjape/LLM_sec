# manage.py for v6
import argparse, subprocess, sys, os

def run(cmd, env=None):
    print(f"\n$ {cmd}")
    proc = subprocess.run(cmd, shell=True, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def simulate(threshold: float):
    env = os.environ.copy()
    env["SAFE_THRESHOLD"] = str(threshold)
    run(f"python v6_run.py {threshold}", env=env)

def harvest():
    run("python v6_harvest_misses.py")

def retrain():
    run("python v6_blue_sklearn_train.py")

def sweep():
    run("python v6_threshold_sweep.py")

def cycle(threshold: float):
    simulate(threshold)
    harvest()
    retrain()
    simulate(threshold)
    print("\n✅ Cycle complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate")
    s.add_argument("--threshold", type=float, default=0.9)

    sub.add_parser("harvest")
    sub.add_parser("retrain")
    sub.add_parser("sweep")

    c = sub.add_parser("cycle")
    c.add_argument("--threshold", type=float, default=0.9)

    args = p.parse_args()
    if args.cmd == "simulate":
        simulate(args.threshold)
    elif args.cmd == "harvest":
        harvest()
    elif args.cmd == "retrain":
        retrain()
    elif args.cmd == "sweep":
        sweep()
    elif args.cmd == "cycle":
        cycle(args.threshold)
