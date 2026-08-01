import argparse
import shutil
import subprocess
from pathlib import Path


SOURCES = [
    "color=c=0xE63946:s=320x180:r=25", "testsrc2=s=320x180:r=25",
    "color=c=0x457B9D:s=320x180:r=25", "smptebars=s=320x180:r=25",
    "color=c=0x2A9D8F:s=320x180:r=25", "testsrc=s=320x180:r=25",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg không có trong PATH")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        shot_count = 4 + index % 3
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for shot in range(shot_count):
            duration = 0.8 + 0.2 * ((index + shot) % 4)
            command += ["-f", "lavfi", "-t", f"{duration:.1f}", "-i", SOURCES[(index + shot) % len(SOURCES)]]
        inputs = "".join(f"[{i}:v]" for i in range(shot_count))
        destination = args.output / f"L01_V{index:03d}.mp4"
        command += ["-filter_complex", f"{inputs}concat=n={shot_count}:v=1:a=0,format=yuv420p[v]",
                    "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", str(destination)]
        subprocess.run(command, check=True)
        print(f"created {destination}")


if __name__ == "__main__":
    main()
