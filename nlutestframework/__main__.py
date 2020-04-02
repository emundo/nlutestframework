import argparse
import asyncio
import logging
from signal import SIGINT, SIGTERM
import sys
from typing import Any

import yaml

from .nlu_benchmarker import NLUBenchmarker

def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NLU frameworks.")

    parser.add_argument(
        "-c", "--config-file",
        dest    = "CONFIG",
        type    = str,
        default = "config.yml",
        help    = (
            "Path to the configuration file."
            " Defaults to \"config.yml\" in the current working directory."
            "\nWARNING: Always verify externally supplied configurations before using them."
        )
    )

    parser.add_argument(
        "-v", "--verbose",
        dest    = "VERBOSE",
        action  = "store_const",
        const   = True,
        default = False,
        help    = "Enable verbose/debug output."
    )

    parser.add_argument(
        "-p", "--python",
        dest = "python",
        type = str,
        help = (
            "Path to the python executable to use instead of the current one."
            " Overrides the corresponding setting in the configuration file."
            "\nWARNING: Always verify externally supplied configurations before using them."
        )
    )

    parser.add_argument(
        "-i", "--iterations",
        dest = "iterations",
        type = int,
        help = (
            "The number of iterations to benchmark."
            " Overrides the corresponding setting in the configuration file."
        )
    )

    parser.add_argument(
        "--ignore-cache",
        dest   = "ignore_cache",
        action = "store_const",
        const  = True,
        help   = (
            "If set, ignore all sorts of cached data."
            " Overrides the corresponding setting in the configuration file."
        )
    )

    args = parser.parse_args()

    # Set the general log level to DEBUG or INFO
    logging.basicConfig(level = logging.DEBUG if args.VERBOSE else logging.INFO)

    # Prevent various modules from spamming the log
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("docker").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("snips_nlu").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("msrest").setLevel(logging.INFO)

    global_config_override = {}
    for label, value in vars(args).items():
        if label.islower() and value is not None:
            global_config_override[label] = value

    async def main_runner() -> None:
        def cancel(_sig: int) -> None:
            print(
                "Aborting execution in the next possible situation (this may take a while, as the"
                " current operation has to finish first)."
            )

            NLUBenchmarker.getInstance().cancel()

        loop = asyncio.get_event_loop()
        for sig in (SIGINT, SIGTERM):
            loop.add_signal_handler(sig, cancel, sig)

        try:
            await NLUBenchmarker.getInstance().runFromConfigFile(
                args.CONFIG,
                **global_config_override
            )
        except OSError as e:
            eprint("Error reading the config file: {}".format(e))
        except yaml.YAMLError as e:
            eprint("Malformed YAML in the config file: {}".format(e))

    asyncio.run(main_runner())

if __name__ == "__main__":
    main()
