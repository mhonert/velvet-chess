import logging as log
from typing import List


def parse_final_results(file_name: str) -> (List[str], bool):
    with open(file_name, 'r') as result_file:
        return parse_final_results_file(result_file)


def parse_final_results_file(result_file) -> (List[str], bool):
    lines = result_file.readlines()[-3:]

    if "Finished match" not in lines[-1]:
        log.info(lines)
        raise RuntimeError("cutechess result output does not contain 'Finished match' line")

    if "H1 was accepted" in lines[-2]:
        return lines[:-1], True
    elif "H0 was accepted" in lines[-2]:
        return lines[:-1], False
    else:
        log.info("Unexpected cutechess output:", lines)
        return lines[:-1], False


def parse_ongoing_results(file_name: str) -> (str, str):
    with open(file_name, 'r') as result_file:
        previous_line = ''
        for line in reversed(result_file.readlines()):
            if line.startswith('Elo difference:'):
                return line.lstrip('Elo difference: ').strip(), previous_line.lstrip('SPRT: ').strip()
            previous_line = line

    return '-', '-'

