import os


def exasim_path():
    cdir = os.getcwd()
    idx = cdir.rfind("Exasim")
    if idx < 0:
        raise RuntimeError(
            f"Current directory {cdir} does not contain 'Exasim' in its path.")
    return cdir[:idx + len("Exasim")]
