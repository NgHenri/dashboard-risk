import pkg_resources
from pip._internal.operations import freeze

# Liste des packages à exclure
conda_only = {
    "aom",
    "liblightgbm",
    "uvicorn-standard",
    "pip",
    "python",
    "setuptools",
    "wheel",
    "sqlite",
    "libstdcxx-ng",
    "libgcc-ng",
    "ca-certificates",
    "openssl",
    "ld_impl_linux-64",
    "zlib",
    "libffi",
    "tk",
    "xz",
    "ncurses",
}

# Obtenir seulement les packages installés via pip
installed = [
    pkg
    for pkg in freeze.freeze()
    if not pkg.startswith(("@", "-e", "file://"))
    and pkg.split("==")[0].lower() not in conda_only
]

with open("requirements_pip_only.txt", "w") as f:
    f.write("\n".join(installed))

print(f"✅ Generated requirements_pip_only.txt with {len(installed)} packages")
