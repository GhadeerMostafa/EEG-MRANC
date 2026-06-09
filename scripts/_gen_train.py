from pathlib import Path
TRAIN = r"""
MRANC training — physics-locked, no ref_eeg.
"""
Path('train.py').write_text(TRAIN, encoding='utf-8')
