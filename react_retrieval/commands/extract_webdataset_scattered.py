import os
import argparse
from multiprocessing.pool import ThreadPool as Pool

import subprocess
from tqdm import tqdm


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Extract features from Vision-Language datasets.')
    parser.add_argument('--dataset_dir', required=True, help='Dataset directory.', type=str)
    parser.add_argument('--scatter_dir', required=True, help='Output directory.', type=str)
    parser.add_argument('--tsv_padzero', default=5, help='TSV file name padding.', type=int)
    return parser


def untar(args):
    src, dst = args

    os.makedirs(dst, exist_ok=True)

    untar_command = f"tar -xf {src} -C {dst}"

    p = subprocess.Popen(untar_command, stdout=subprocess.PIPE, shell=True)

    p.wait()        


def main(args):
    tarfile_list = sorted(f for f in os.listdir(args.dataset_dir) if f.endswith('.tar'))

    TASKS = []
    pool = Pool(36)

    for tsv_idx, tarfile_name in enumerate(tarfile_list):
        assert tarfile_name == f'{tsv_idx:0{args.tsv_padzero}d}.tar'

        src_file = os.path.join(args.dataset_dir, tarfile_name)
        dst_dir = os.path.join(args.scatter_dir, tarfile_name.replace('.tar', ''))

        TASKS.append((src_file, dst_dir))

    return list(tqdm(pool.imap(untar, TASKS), total=len(TASKS)))

if __name__ == "__main__":
    args = get_argument_parser().parse_args()

    main(args)
