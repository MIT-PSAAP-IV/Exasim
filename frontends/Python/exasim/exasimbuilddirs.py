import os

from .exasim_path import exasim_path


def exasimbuilddirs(sharedbuild=1):
    if sharedbuild == 1:
        sharedroot = os.path.join(exasim_path(), "examples", "exasimfe")
        if sharedroot:
            return sharedroot, sharedroot

    localroot = os.path.join(os.getcwd(), "exasim")
    return localroot, localroot
