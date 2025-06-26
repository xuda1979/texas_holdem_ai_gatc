import re
import sys


def parse_losses(log_file: str):
    pattern = re.compile(r"Loss: ([0-9.eE+-]+)")
    losses = []
    with open(log_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "poker_ai.log"
    losses = parse_losses(path)
    if not losses:
        print("No losses found in log file")
    else:
        print(f"Read {len(losses)} loss values")
        print(f"Most recent 5: {losses[-5:]}")
        avg = sum(losses) / len(losses)
        print(f"Average loss: {avg:.6f}")
