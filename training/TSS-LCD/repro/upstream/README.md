# Upstream TSS-LCD source (NOT redistributed)

These files are the authors' own code from `github.com/Xlab2024/TSS-LCD`.
The repository ships **no LICENSE**, so redistribution is not granted and this
directory is gitignored. Re-fetch instead of committing:

    SHA=b156d0a007        # HEAD, 2025-12-03
    curl -sL -o tsslcd.tgz https://codeload.github.com/Xlab2024/TSS-LCD/tar.gz/$SHA
    tar xzf tsslcd.tgz --strip-components=1 -C upstream '*.py'

The deleted training split lives at commit `d7f1e51263`; see
`data/tsslcd/PROVENANCE.md`.

Files: Context2CondNew.py, F2Cnet.py, NoiseNet.py, DataSetPrepare.py,
data_utils.py, config.py, train.py, test.py
