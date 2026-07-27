"""Compile the bundled amalgamation into a loadable SQLite extension and ship it
inside the wheel. `src/` (the amalgamation + SQLite ext headers) is populated by
`make python-src` from the repo root, and is present in the sdist."""
import os
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:  # setuptools >= 70.1 vendors bdist_wheel; older builds use the wheel package
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

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


class BinaryDistribution(Distribution):
    """The package ships a prebuilt loadable. Declaring ext modules makes
    setuptools install it to platlib (not purelib, which auditwheel rejects)
    and marks the wheel platform-specific."""

    def has_ext_modules(self):
        return True


class BDistWheel(_bdist_wheel):
    """The loadable is not a CPython extension, so the wheel is platform-specific
    but ABI-agnostic. Tag it py3-none-<platform> rather than cp3x-cp3x-<platform>
    (any Python on that platform can load it), and not py3-none-any (which would
    claim it is pure Python and makes cibuildwheel reject it)."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution,
      cmdclass={"build_py": Build, "bdist_wheel": BDistWheel})
