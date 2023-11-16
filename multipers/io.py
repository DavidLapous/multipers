from multipers.simplex_tree_multi import SimplexTreeMulti
import os

mpfree_path = None
mpfree_in_path = "multipers_mpfree_input.scc"
mpfree_out_path = "multipers_mpfree_output.scc"


def scc_parser(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    # Find scc2020
    while lines[0].strip() != "scc2020":
        lines = lines[1:]
    lines = lines[1:]
    # stripped scc2020 we can start

    def pass_line(line):
        return len(line) == 0 or line[0] == "#"

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        num_parameters = int(line)
        lines = lines[i + 1:]
        break

    block_sizes = []

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        block_sizes = [int(i) for i in line.split(" ")]
        lines = lines[i + 1:]
        break
    blocks = []
    for block_size in block_sizes:
        counter = block_size
        block_filtrations = []
        block_boundaries = []
        for i, line in enumerate(lines):
            if counter == 0:
                lines = lines[i:]
                break
            line = line.strip()
            if pass_line(line):
                continue
            filtration, boundary = line.split(";")
            block_filtrations.append(
                [float(x) for x in filtration.split(" ") if len(x) > 0]
            )
            block_boundaries.append(
                [int(x) for x in boundary.split(" ") if len(x) > 0])
            counter -= 1
        blocks.append((block_filtrations, block_boundaries))

    return blocks


def _init_mpfree():
    from shutil import which

    global mpfree_path, mpfree_in_path, mpfree_out_path
    if mpfree_path is None:
        a = which("./mpfree")
        b = which("mpfree")
        if a:
            mpfree_path = a
        elif b:
            mpfree_path = b
    else:
        return
    if not mpfree_path:
        raise Exception(
            "mpfree not found. Install it from https://bitbucket.org/mkerber/mpfree/, or use `mpfree_path`"
        )

    shm_memory = "/dev/shm/"  # on unix, we can write in RAM instead of disk.
    if os.access(shm_memory, os.W_OK):
        mpfree_in_path = shm_memory + mpfree_in_path
        mpfree_out_path = shm_memory + mpfree_out_path


def minimal_presentation_from_mpfree(
    simplextree: SimplexTreeMulti, full_resolution: bool = True, dimension: int = 1
):
    global mpfree_path, mpfree_in_path, mpfree_out_path
    if not mpfree_path:
        _init_mpfree()

    simplextree.to_scc(
        path=mpfree_in_path,
        rivet_compatible=False,
        strip_comments=False,
        ignore_last_generators=False,
        overwrite=True,
        reverse_block=True,
    )
    resolution_str = "--resolution" if full_resolution else ""
    if os.path.exists(mpfree_out_path):
        os.remove(mpfree_out_path)
    os.system(
        f"{mpfree_path} {resolution_str} --dim={dimension} {mpfree_in_path} {mpfree_out_path} >/dev/null 2>&1"
    )
    blocks = scc_parser(mpfree_out_path)
    clear_io()
    return blocks


def clear_io():
    global mpfree_in_path, mpfree_out_path
    if os.path.exists(mpfree_in_path):
        os.remove(mpfree_in_path)
    if os.path.exists(mpfree_out_path):
        os.remove(mpfree_out_path)
