"""Print scalar curves from a TensorBoard event file (so a backgrounded run can
be monitored without its stdout)."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/pilot/tb")
    ap.add_argument("--tags", default="")
    ap.add_argument("--last", type=int, default=12)
    args = ap.parse_args()

    ea = EventAccumulator(str(Path(args.run)), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    want = [t.strip() for t in args.tags.split(",") if t.strip()] or tags
    want = [t for t in want if t in tags]

    for t in sorted(want):
        ev = ea.Scalars(t)
        pts = ev[-args.last:]
        s = "  ".join(f"{e.step}:{e.value:.4g}" for e in pts)
        print(f"{t:<28} n={len(ev):4d}  {s}")


if __name__ == "__main__":
    main()
