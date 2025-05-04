import re

# Liste de paquets à exclure car conda-only ou inutiles pour pip
conda_only = {
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

with open("full_list.txt", "r") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    pkg = line.strip()
    name = pkg.split("==")[0]
    if name and name not in conda_only and re.match(r"^[a-zA-Z0-9_.-]+==[0-9]", pkg):
        cleaned.append(pkg)

with open("requirements.txt", "w") as f:
    f.write("\n".join(cleaned))

print("✅ requirements.txt generated.")
