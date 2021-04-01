from setuptools import setup, find_packages
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as fid:
    requires = [line.strip() for line in fid]

console_scripts = {
    'console_scripts': [
        'w2l-color-syncnet-train=w2l.scripts.color_syncnet_train:main',
        'w2l-create-filelists=w2l.scripts.create_filelists:main',
        'w2l-dump-face=w2l.scripts.dump_face:main',
        'w2l-generate-video=w2l.scripts.generate_video:main',
        'w2l-hq-wav2lip-train=w2l.scripts.hq_wav2lip_train:main',
        'w2l-inference=w2l.scripts.inference:main',
        'w2l-preprocess=w2l.scripts.preprocess:main',
        'w2l-wav2lip-train=w2l.scripts.wav2lip_train:main',
    ]
}

setup(
    name="w2l",
    version=subprocess.check_output(
        ['git', 'describe', '--tags']).strip().decode('ascii'),
    url="https://github.com/mychiux413/Wav2Lip",
    packages=find_packages(include=['w2l']),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=requires,
)
