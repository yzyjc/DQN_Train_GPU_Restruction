"""GPT6-Astra: read-only, named outcome metrics from a recorded training timeline.

Usage: python tests/report_outcome_metrics.py <logs/metric_outcome_raw.bin>
No external packages. A partial last row is ignored; invalid headers/values fail.
"""
import argparse
import collections
import math
import struct

NAMES = ("black_win_avg_steps", "black_loss_avg_steps", "white_win_avg_steps",
         "white_loss_avg_steps", "draw_avg_steps", "avg_game_steps")


def report(path):
    totals = [[0, 0] for _ in NAMES]
    recent = collections.deque(maxlen=1000)
    last = -1
    with open(path, "rb") as source:
        header = source.read(24)
        if len(header) != 24:
            raise ValueError("Incomplete header")
        magic, fields, version, *_ = struct.unpack("<6I", header)
        if magic != 0x44514E4C or fields != 14 or version > 1:
            raise ValueError("Unsupported metric schema")
        while chunk := source.read(64):
            if len(chunk) != 64:
                print("Ignoring incomplete trailing record")
                break
            episode, _, *values = struct.unpack("<2i14f", chunk)
            steps, winner = values[:2]
            if (not all(map(math.isfinite, values)) or steps != int(steps)
                    or not 1 <= steps <= 25 or winner not in (0, 1, 2)):
                raise ValueError("Invalid outcome record")
            if episode <= last:
                raise ValueError("Non-increasing timeline; select one recovered run")
            last = episode
            selected = (winner == 1, winner == 2, winner == 2,
                        winner == 1, winner == 0, True)
            recent.append((steps, selected))
            for index, keep in enumerate(selected):
                if keep:
                    totals[index][0] += steps
                    totals[index][1] += 1
    for index, name in enumerate(NAMES):
        total, count = totals[index]
        window = [steps for steps, mask in recent if mask[index]]
        mean = f"{total/count:.4f}" if count else "N/A"
        rolling = f"{sum(window)/len(window):.4f}" if window else "N/A"
        print(f"{name}={mean} count={count} rolling1000={rolling} rolling_count={len(window)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    report(parser.parse_args().path)
