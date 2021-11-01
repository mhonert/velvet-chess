import argparse
import glob
import json
import logging as log
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from common import parse_final_results, parse_ongoing_results

BASE_PATH = os.getcwd()


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Automatic Patch Verifier")

    parser.add_argument('--tb_path', type=str, help='Path to Syzygy table base folder', required=True)
    parser.add_argument('--book_file', type=str, help='Path to opening book file in EPD format', required=True)
    parser.add_argument('--cpu_cores', type=int, help='Number of physical cores to use', required=True)

    config = parser.parse_args()

    Path('inbox').mkdir(exist_ok=True)
    Path('accepted').mkdir(exist_ok=True)
    Path('rejected').mkdir(exist_ok=True)
    Path('work').mkdir(exist_ok=True)
    if not Path('bin').exists():
        log.error("Path 'bin' must exist and contain the cutechess-cli and velvet-baseline executables")
        exit(-1)
    if not Path('repo').exists():
        log.error("Path 'repo' must exist and contain the velvet-chess Git Repo")
        exit(-1)

    scan(config)


def read_config(patch_id: str) -> Dict:
    with open('inbox/%s.json' % patch_id) as file:
        return json.load(file)


def scan(server_config):
    log.info('Scanning inbox for new patches ...')
    while True:
        for patch_file in sorted(glob.glob('inbox/*.patch')):
            patch_id = patch_file.lstrip('inbox/').rstrip('.patch')
            patch_descr = Path('inbox/%s.txt' % patch_id).read_text()
            patch_config = read_config(patch_id)

            if not patch_config['stc'] and not patch_config['ltc']:
                log.error('Patch contains no options for STC or LTC verification')
                exit(-1)

            log.info('Found patch %s - %s' % (patch_id, patch_descr))

            if not Path('bin/velvet-%s' % patch_id).exists():
                build_patch(patch_file, patch_id)

            (stc_elo_result, stc_sprt_result, stc_accepted) = verify(server_config, patch_config['stc'], 'STC', patch_id)
            if not stc_accepted:
                reject_patch(patch_id)
                log.info('Scanning inbox for new patches ...')
                continue

            (ltc_elo_result, ltc_sprt_result, ltc_accepted) = verify(server_config, patch_config['ltc'], 'LTC', patch_id)
            if not ltc_accepted:
                reject_patch(patch_id)
                log.info('Scanning inbox for new patches ...')
                continue

            accept_patch(patch_id, patch_descr, stc_elo_result, stc_sprt_result, ltc_elo_result, ltc_sprt_result)
            log.info('Scanning inbox for new patches ...')

        time.sleep(5)


def build_patch(patch_file, patch_id):
    patch_branch = get_acc_patch_branch()
    if patch_branch is None:
        patch_branch = 'master'
    try:
        log.info('Checkout base branch')
        subprocess.run(['git', 'checkout', patch_branch], cwd='repo/velvet-chess', check=True)

        log.info('Create branch')
        subprocess.run(['git', 'checkout', '-b', '%s' % patch_id], cwd='repo/velvet-chess', check=True)

        log.info('Apply patch')
        subprocess.run(['git', 'apply', '../../%s' % str(patch_file)], cwd='repo/velvet-chess', check=True)

        log.info('Build binary')
        cargo_env = dict(os.environ, RUSTFLAGS='-Ctarget-feature=+crt-static,-bmi2 -Ctarget-cpu=x86-64-v3')
        subprocess.run(['cargo', 'build', '--release', '--target', 'x86_64-unknown-linux-musl', '--bin', 'velvet'],
                       cwd='repo/velvet-chess', env=cargo_env, check=True)

        subprocess.run(['strip', 'target/x86_64-unknown-linux-musl/release/velvet'], cwd='repo/velvet-chess',
                       check=True)
    except subprocess.CalledProcessError:
        cleanup_repo(patch_id)
        exit(-1)
    log.info('Move binary')
    shutil.move('repo/velvet-chess/target/x86_64-unknown-linux-musl/release/velvet', 'bin/velvet-%s' % patch_id)
    log.info('Generate cutechess-cli config')
    create_cutechess_config(patch_id)


def verify(server_config, patch_config, mode: str, patch_id: str) -> (str, str, bool):
    if not patch_config:
        return None, None, True

    completed = False
    if Path('work/%s.%s.result' % (patch_id, mode)).exists():
        (_, sprt) = parse_ongoing_results('work/%s.%s.result' % (patch_id, mode))
        completed = 'was accepted' in sprt

    if not completed:
        run_cutechess(server_config, patch_config, mode, patch_id)
    else:
        log.info('%s verification was already completed' % mode)

    (results, accepted) = parse_final_results('work/%s.%s.result' % (patch_id, mode))

    elo_result = results[0].strip()
    sprt_result = results[1].strip()

    log.info(elo_result)
    log.info(sprt_result)

    return elo_result, sprt_result, accepted


def get_acc_patch_branch() -> Optional[str]:
    subprocess.run(['git', 'checkout', 'master'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'pull'], cwd='repo/velvet-chess', check=True)

    result = subprocess.run(['git', 'branch', '--no-merged'], stdout=subprocess.PIPE, cwd='repo/velvet-chess', check=True)

    patch_branches = []
    for entry in result.stdout.decode('utf-8').splitlines():
        if entry[2:].startswith('patches_'):
            branch = entry[2:]
            patch_branches.append(branch)

    patch_branches.sort(reverse=True)
    if len(patch_branches) == 0:
        return None

    return patch_branches[0]


def new_patch_branch() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    branch_name = "patches_" + timestamp

    subprocess.run(['git', 'checkout', 'master'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'pull'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'checkout', '-b', branch_name], cwd='repo/velvet-chess', check=True)

    return branch_name


def accept_patch(patch_id: str, descr: str, stc_elo_result: str, stc_sprt_result: str, ltc_elo_result: str, ltc_sprt_result: str):
    log.info("Patch was accepted!")
    shutil.move('inbox/%s.patch' % patch_id, 'work/%s.patch' % patch_id)
    shutil.move('inbox/%s.json' % patch_id, 'work/%s.json' % patch_id)
    shutil.move('inbox/%s.txt' % patch_id, 'work/%s.txt' % patch_id)

    subprocess.run(['git', 'add', '-A'], cwd='repo/velvet-chess', check=True)

    commit_message = descr
    if stc_elo_result is not None:
        commit_message += '\n\nSTC results:\n%s\n%s' % (stc_elo_result, stc_sprt_result)

    if ltc_elo_result is not None:
        commit_message += '\n\nLTC results:\n%s\n%s' % (ltc_elo_result, ltc_sprt_result)

    subprocess.run(['git', 'commit', '-m', commit_message], cwd='repo/velvet-chess', check=True)

    target_branch = get_acc_patch_branch()
    if target_branch is None:
        target_branch = new_patch_branch()

    subprocess.run(['git', 'checkout', target_branch], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'merge', '--ff', patch_id], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'push', '-u', 'origin', target_branch], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'branch',  '-D', '%s' % patch_id], cwd='repo/velvet-chess', check=True)

    log.info('Package files')
    subprocess.run(['tar cvf %s.tar.gz %s.*' % (patch_id, patch_id)], cwd='work', shell=True, check=True)

    log.info('Move to "accepted" folder')
    shutil.move('work/%s.tar.gz' % patch_id, 'accepted/%s.tar.gz' % patch_id)

    log.info('Use this version as new baseline')
    os.unlink('bin/velvet-baseline')
    shutil.copy2('bin/velvet-%s' % patch_id, 'bin/velvet-baseline')

    log.info('Clean up work dir')
    shutil.rmtree('work')
    Path('work').mkdir()


def reject_patch(patch_id: str):
    log.info("Patch was rejected")
    cleanup_repo(patch_id)
    shutil.move('inbox/%s.patch' % patch_id, 'work/%s.patch' % patch_id)
    shutil.move('inbox/%s.json' % patch_id, 'work/%s.json' % patch_id)
    shutil.move('inbox/%s.txt' % patch_id, 'work/%s.txt' % patch_id)

    log.info('Package files')
    subprocess.run(['tar cvf %s.tar.gz %s.*' % (patch_id, patch_id)], cwd='work', shell=True, check=True)

    log.info('Move to "rejected" folder')
    shutil.move('work/%s.tar.gz' % patch_id, 'rejected/%s.tar.gz' % patch_id)

    log.info('Clean up work dir')
    shutil.rmtree('work')
    Path('work').mkdir()


def run_cutechess(server_config, patch_config, mode: str, patch_id: str):

    tc = patch_config['tc']
    log.info('Start %s (%s) verification ...' % (mode, tc))

    options = ['-tb', server_config.tb_path,
               '-rounds', '10000',
               '-concurrency', str(server_config.cpu_cores),
               '-ratinginterval', '10',
               '-games', '2',
               '-openings', 'file=%s' % server_config.book_file, 'format=epd', 'order=random',
               '-each', 'option.Hash=256', 'option.Threads=1', 'restart=off', 'tc=%s' % tc]

    engines = ['-engine', 'conf=challenger',
               '-engine', 'conf=baseline']

    low_elo = patch_config['low_elo']
    high_elo = patch_config['high_elo']

    sprt = ['-sprt', 'elo0=%s' % str(low_elo), 'elo1=%s' % str(high_elo), 'alpha=0.05', 'beta=0.05']

    pgn_out = ['-pgnout', '../work/%s.%s.pgn' % (patch_id, mode)]

    result_file_name = 'work/%s.%s.result' % (patch_id, mode)

    with open(result_file_name, 'w') as result_file:
        subprocess.run(['./cutechess-cli', *engines, *sprt, *options, *pgn_out], stdout=result_file, cwd='bin', check=True)


def create_cutechess_config(patch_id: str):
    config = [
        {
            "name": "baseline",
            "command": "%s/bin/velvet-baseline" % BASE_PATH,
            "protocol": "uci"
        },
        {
            "name": "challenger",
            "command": "%s/bin/velvet-%s" % (BASE_PATH, patch_id),
            "protocol": "uci"
        }
    ]

    with open('bin/engines.json', 'w') as config_file:
        json.dump(config, config_file)


def cleanup_repo(patch_id: str):
    log.info("Clean-Up after error")
    subprocess.run(['git', 'checkout', '.'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'clean', '-dfq'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'checkout', 'master'], cwd='repo/velvet-chess', check=True)
    subprocess.run(['git', 'branch',  '-D', '%s' % patch_id], cwd='repo/velvet-chess', check=True)


main()
