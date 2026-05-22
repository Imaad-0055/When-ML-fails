from __future__ import annotations

import sys

from run_experiments import main


if __name__ == "__main__":
    sys.argv.extend(["--pipeline", "reference"])
    main()
