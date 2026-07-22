import json
from pathlib import Path


def test_colab_bootstraps_make_src_importable_without_kernel_restart():
    notebooks = sorted(Path('notebooks').glob('*.ipynb'))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding='utf-8'))
        sources = [
            ''.join(cell.get('source', []))
            for cell in notebook.get('cells', [])
            if cell.get('cell_type') == 'code'
        ]
        bootstrap = next(
            (source for source in sources if 'CPS_REPO_URL' in source and 'os.chdir(repo)' in source),
            None,
        )
        assert bootstrap is not None, path
        assert 'sys.path.insert(0, str(src_dir))' in bootstrap, path
        assert 'importlib.invalidate_caches()' in bootstrap, path
        assert 'import cps' in bootstrap, path
        assert '[BOOT] CPS import verified from' in bootstrap, path
