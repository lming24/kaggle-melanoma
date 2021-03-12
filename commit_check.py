#!/usr/bin/env python3
"""
Run precommit checks.

Precommit checks for all languages. All flags are enabled by default. Otherwise only the specified flags are enabled.
"""

import argparse
import pathlib
import re
import subprocess


def py_yapf(ci=False):
    print("-----------------------")
    print("Yapfing...")

    repo_root = pathlib.Path(__file__).parent
    if not ci:
        subprocess.check_call(["yapf", "--in-place", "--style=style.yapf", "--recursive", "src"], cwd=str(repo_root))
    else:
        subprocess.check_call(["yapf", "--quiet", "--style=style.yapf", "--recursive", "src"], cwd=str(repo_root))


def py_pylint():
    print("-----------------------")
    print("pylinting...")

    repo_root = pathlib.Path(__file__).parent.resolve()
    subprocess.check_call(["pylint", "--rcfile=pylint.rc", "src"], cwd=str(repo_root))


def py_unit_test():
    print("-----------------------")
    print("Unit testing...")
    subprocess.check_call(["python3", "-m", "unittest", "discover", "src", "test_*.py"])


def main():
    """Main routine."""

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--yapf",
        help="Run YAPF for python code.",
        action="store_true",
    )
    parser.add_argument(
        "--pylint",
        help="Run Pylint for python code.",
        action="store_true",
    )
    parser.add_argument(
        "--unit-test-py",
        help="Run python unit tests.",
        action="store_true",
    )
    parser.add_argument("--ci", help="Do not change any files. Only emit errors.", action="store_true")

    args = parser.parse_args()

    arguments = [
        "yapf",
        "pylint",
        "unit_test_py",
    ]

    none_set = True
    for arg in arguments:
        if getattr(args, arg):
            none_set = False

    do_all = False
    if none_set:
        do_all = True

    if do_all:
        args.yapf = True
        args.pylint = True
        args.unit_test_py = True

    if args.yapf:
        py_yapf(args.ci)

    if args.pylint:
        py_pylint()

    if args.unit_test_py:
        py_unit_test()

    print("Finished!")


if __name__ == "__main__":
    main()
