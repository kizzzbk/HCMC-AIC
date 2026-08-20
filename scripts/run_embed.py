import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from shotlab.cli import embed_main

if __name__ == "__main__":
    sys.argv = ['', '--output', 'output', '--device', 'cpu']
    embed_main()
