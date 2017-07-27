#! /usr/bin/env python
import argparse as ap
import os
import sys

import numpy as np


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-o", "--output", type=ap.FileType("w"),
                      default=sys.stdout)

    args = argp.parse_args(args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
