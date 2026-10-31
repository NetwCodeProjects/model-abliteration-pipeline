"""Run the CLI with the test fakes in a separate process (``Env.run_killable``).

Used by tests in which a fake tool hard-kills the pipeline process; in-process runs would take
pytest down with it. argv: <test tmp dir> <CLI arguments...>.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import FakeProbe, fake_hub, fake_services  # noqa: E402

from model_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    os.environ["FAKE_PIPELINE_PID"] = str(os.getpid())
    sys.exit(main(sys.argv[2:], services=fake_services(fake_hub(Path(sys.argv[1])), FakeProbe())))
