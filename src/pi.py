#!/usr/bin/env python3
# Pi Calculator using the Chudnovsky algorithm
# binary_split function from Wikipedia (2023-11-28)
#   link: https://en.wikipedia.org/wiki/Chudnovsky_algorithm
# Required modules: mpmath, gmpy (optional)
# SPDX-License-Identifier: CC-BY-SA-4.0

import mpmath
import logging
import os
import multiprocessing
import argparse
from math import ceil


def binary_split(args):
    a, b = args
    if b == a + 1:
        Pab = -(6 * a - 5) * (2 * a - 1) * (6 * a - 1)
        Qab = 10939058860032000 * a**3
        Rab = Pab * (545140134 * a + 13591409)
    else:
        m = (a + b) // 2
        Pam, Qam, Ram = binary_split((a, m))
        Pmb, Qmb, Rmb = binary_split((m, b))

        Pab = Pam * Pmb
        Qab = Qam * Qmb
        Rab = Qmb * Ram + Pam * Rmb
    return Pab, Qab, Rab


def binary_split_worker(args):
    logging.debug("process %s started", multiprocessing.current_process().name)
    result = binary_split(args)
    logging.debug("process %s finished", multiprocessing.current_process().name)
    return result


def make_ranges(start, end, n):
    # make n tuple ranges. each tuple's end is the next tuple's start
    interval = (end - start) // n
    ranges = []
    for i in range(0, n - 1):
        ranges.append((i * interval + start, (i + 1) * interval + start))

    ranges.append(((n - 1) * interval + start, end))
    logging.debug("ranges: %s", str(ranges))
    return ranges


def combine_pairs(pair):
    if len(pair) == 1:
        return pair[0]
    a, b = pair
    return (a[0] * b[0], a[1] * b[1], b[1] * a[2] + a[0] * b[2])


def chudnovsky(n, threads=1):
    logging.debug("threads: %d", threads)

    intervals = make_ranges(1, n + 1, threads)
    with multiprocessing.Pool(threads) as p:
        results = p.map(binary_split_worker, intervals)

    logging.debug("combining results")

    while len(results) > 1:
        with multiprocessing.Pool(threads) as p:
            results = p.map(
                combine_pairs, [results[i:i+2] for i in range(0, len(results), 2)]
            )

    P1n, Q1n, R1n = results[0]

    # assert (P1n, Q1n, R1n) == binary_split((1, n))
    logging.debug("calculating final result")
    return (426880 * mpmath.mpf(10005).sqrt() * Q1n) / (13591409 * Q1n + R1n)


def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Calculate Pi to n digits with multiprocessing",
    )
    parser.add_argument("precision", type=int, default=1)
    parser.add_argument("-n", "--ncpu", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("-o", "--output", type=str, default="-")
    args = parser.parse_args()

    if args.precision > 14:
        iters = ceil(args.precision / 14)
        # iters = args.precision
    else:
        iters = multiprocessing.cpu_count() * 2
    mpmath.mp.dps = args.precision + 1
    logging.debug("precision: %d", args.precision)
    logging.debug("iters: %d", iters)

    final_result = chudnovsky(iters, threads=args.ncpu)
    if args.output == "-":
        print(final_result)
    else:
        with open(args.output, "w") as f:
            f.write(str(final_result))


if __name__ == "__main__":
    main()
