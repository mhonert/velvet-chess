import argparse
import datetime
import glob
import io
import json
import logging as log
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import List

from common import (parse_final_results, parse_final_results_file, parse_ongoing_results)

# patch-verifier must be called from the repository root using the 'verify-patch' shell script
BASE_DIR = 'pytools/patch-verifier/tmp'


STC = {
    'tc': '40/8+0.08',
    'low_elo': 0,
    'high_elo': 5
}

LTC = {
    'tc': '40/50+0.5',
    'low_elo': 0,
    'high_elo': 5
}

QUICKFIX = {
    'tc': '40/5+0.05',
    'low_elo': -3,
    'high_elo': 1
}


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Automatic Patch Verifier")
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    upload_parser = subparsers.add_parser('upload', help='Uploads a new patch to the server for verification')
    upload_parser.add_argument('description', type=str, help='Description of the patch. Will be included in the commit message')

    skips = upload_parser.add_mutually_exclusive_group()
    skips.add_argument('--skip-ltc', help='Skips long time control verification', action='store_true')
    skips.add_argument('--skip-stc', help='Skips short time control verification', action='store_true')
    skips.add_argument('--quickfix', help='Quick verification for refactorings or minor bug-fixes', action='store_true')

    subparsers.add_parser('report', help='Fetches all available verification results')

    args = parser.parse_args()

    if args.command == 'upload':
        upload(args.description, args.skip_stc, args.skip_ltc, args.quickfix)
    elif args.command == 'report':
        report()
    else:
        parser.print_usage()


def upload(description: str, skip_stc: bool, skip_ltc: bool, quickfix: bool):
    if skip_stc:
        log.info("Skipping STC verification")

    if skip_ltc:
        log.info("Skipping LTC verification")

    Path('%s/patches' % BASE_DIR).mkdir(parents=True, exist_ok=True)

    log.info('Upload patch "%s" to server ...' % description)

    patch_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log.info('Generated patch ID: "%s"' % patch_id)

    patch_dir = '%s/patches/%s' % (BASE_DIR, patch_id)

    log.info('Preparing Git patch')
    os.mkdir(patch_dir)

    try:
        with open('%s/%s.patch' % (patch_dir, patch_id), 'w') as patch_file:
            subprocess.run(['git', 'diff'], stdout=patch_file, check=True)

        with open('%s/%s.txt' % (patch_dir, patch_id), 'w') as description_file:
            description_file.write(description)

        with open('%s/%s.json' % (patch_dir, patch_id), 'w') as config_file:
            json.dump(create_config(skip_stc, skip_ltc, quickfix), config_file)

        log.info('Uploading to server')
        subprocess.run(['scp -q %s/%s.* server:chess/patch-verifier/inbox/' % (patch_dir, patch_id)], shell=True, check=True)

    except subprocess.CalledProcessError:
        upload_error(patch_dir, 'patch upload failed')

    log.info('Upload successful')


def create_config(skip_stc, skip_ltc, quickfix):
    config = {}
    if not skip_stc:
        config['stc'] = STC

    if not skip_ltc:
        config['ltc'] = LTC

    if quickfix:
        config['stc'] = QUICKFIX  # Only perform a short time control verification

    return config


def report():
    log.info('Verification report')
    sync()

    log.info('Fetching latest results ...')
    subprocess.run(["rsync -az --delete --exclude='*.pgn' server:chess/patch-verifier/{inbox,work,accepted,rejected} %s" % BASE_DIR], shell=True, check=True)

    create_report()


def sync():
    shutil.rmtree('%s/work' % BASE_DIR, ignore_errors=True)
    Path('%s' % BASE_DIR).mkdir(parents=True, exist_ok=True)


def create_report():
    inbox = find_patch_ids('%s/inbox/' % BASE_DIR, '.patch')
    in_progress = find_patch_ids('%s/work/' % BASE_DIR, '.STC.result')
    accepted = find_patch_ids('%s/accepted/' % BASE_DIR, '.tar.gz')[-10:]
    rejected = find_patch_ids('%s/rejected/' % BASE_DIR, '.tar.gz')[-10:]

    for patch_id in in_progress:
        inbox.remove(patch_id)

    hl()
    if len(inbox) > 0:
        report_inbox(inbox)
    else:
        log.info("No patches waiting in Inbox")

    if len(accepted) > 0:
        hl()
        report_finished('accepted', accepted)

    if len(rejected) > 0:
        hl()
        report_finished('rejected', rejected)

    if len(in_progress) > 0:
        hl()
        report_in_progress(in_progress)


def report_inbox(ids: List[str]):
    log.info("Patches waiting in Inbox:")

    for patch_id in ids:
        description = Path('%s/inbox/%s.txt' % (BASE_DIR, patch_id)).read_text()
        log.info(" -> %s - %s" % (patch_id, description))


def report_in_progress(ids: List[str]):
    log.info("Currently running verification:")
    for patch_id in ids:
        description = Path('%s/inbox/%s.txt' % (BASE_DIR, patch_id)).read_text()
        log.info(" - %s - %s" % (patch_id, description))
        log.info("")

        if Path('%s/work/%s.LTC.result' % (BASE_DIR, patch_id)).exists():
            report_final_result('STC',  patch_id)
            hl()
            report_ongoing_result('LTC', patch_id)
        else:
            report_ongoing_result('STC', patch_id)


def report_finished(result: str, ids: List[str]):
    log.info("%s patches:" % result.capitalize())
    for patch_id in ids:
        with tarfile.open('%s/%s/%s.tar.gz' % (BASE_DIR, result, patch_id)) as tar:
            descr_file = io.TextIOWrapper(tar.extractfile('%s.txt' % patch_id))
            description = descr_file.read()
            log.info(" - %s - %s" % (patch_id, description))

            stc_file = io.TextIOWrapper(tar.extractfile('%s.STC.result' % patch_id))
            (stc_results, accepted) = parse_final_results_file(stc_file)
            if accepted:
                ltc_file = io.TextIOWrapper(tar.extractfile('%s.LTC.result' % patch_id))
                (ltc_results, accepted) = parse_final_results_file(ltc_file)
                log.info('  > STC: %s' % stc_results[0].strip())
                log.info('  > LTC: %s' % ltc_results[0].strip())
            else:
                log.info('  > STC: %s' % stc_results[0].strip())

            log.info("")


def report_final_result(mode: str, patch_id: str):
    (results, _) = parse_final_results('%s/work/%s.%s.result' % (BASE_DIR, patch_id, mode))
    log.info('%s verification successful!' % mode)
    log.info('%s Elo result : %s' % (mode, results[0].strip()))
    log.info('%s SPRT result: %s' % (mode, results[1].strip()))


def report_ongoing_result(mode: str, patch_id: str):
    (elo, sprt) = parse_ongoing_results('%s/work/%s.%s.result' % (BASE_DIR, patch_id, mode))
    log.info('%s verification ongoing ...' % mode)
    log.info('%s Elo result : %s' % (mode, elo))
    log.info('%s SPRT result: %s' % (mode, sprt))


def find_patch_ids(folder: str, file_ending: str) -> List[str]:
    patch_ids = []
    for file in sorted(glob.glob('%s/*%s' % (folder, file_ending))):
        patch_id = file.lstrip(folder).rstrip(file_ending)
        patch_ids.append(patch_id)
    return patch_ids


def hl():
    log.info("___________________________________________________________________________________________________")
    log.info("")


def upload_error(patch_dir: str, msg: str):
    print(' - %s' % msg)
    shutil.rmtree(patch_dir)
    exit(-1)


main()
