"""Compile the bundled amalgamation into a loadable SQLite extension and ship it
inside the wheel. `src/` (the amalgamation + SQLite ext headers) is populated by
`make python-src` from the repo root, and is present in the sdist."""
import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")


class Build(build_py):
    def run(self):
        src = os.path.join(HERE, "src", "sqlite-predict.c")
        if not os.path.exists(src):
            raise SystemExit(
                "src/sqlite-predict.c is missing; run `make python-src` from the"
                " repo root to generate the amalgamation into this package")
        out = os.path.join(HERE, "sqlite_predict", "predict0." + EXT)
        cc = os.environ.get("CC", "cc")
        flags = ["-O2", "-shared"] if sys.platform == "win32" \
            else ["-O3", "-fPIC", "-shared"]
        subprocess.check_call([cc, *flags, "-I", os.path.join(HERE, "src"),
                               src, "-o", out])
        super().run()


setup(cmdclass={"build_py": Build})
