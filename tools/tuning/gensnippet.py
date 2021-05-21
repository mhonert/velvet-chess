# Velvet Chess Engine
# Copyright (C) 2020 mhonert (https://github.com/mhonert)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging as log
import sys
import yaml


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file = open("./genetic_evals.yml", "r")

    programs = yaml.safe_load(file)
    print(programs)
    file.close()

    out = open("./gen_program_snippet.rs", "w")
    for program in programs:
        out.write(f'''eval.add_program(GeneticProgram::new(U512::from_str_radix("{program['code']}", 10).unwrap(), {program['data']}, {program['score_increment']}, {program['score_raise']}));
        ''')

    out.close()


# Main
if __name__ == "__main__":
    main()
