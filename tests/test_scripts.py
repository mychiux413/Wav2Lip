import subprocess
import os
import shutil
import glob

pngs_dir = "temp/pngs"
video_dir = "tests/data/videos/"
video_path = "tests/data/videos/test_dataset/dictator.mp4"
audio_path = "tests/data/dictator.wav"
filelists_dir = "temp/filelists"
syncnet_dir = "temp/syncnet-test"
syncnet_path = os.path.join(syncnet_dir, "checkpoint_step000000020.pth")
wav2lip_dir = "temp/wav2lip-test"
wav2lip_path = os.path.join(wav2lip_dir, "checkpoint_step000000100.pth")
wav2lip_gan_dir = "temp/wav2lip-gan-test"
wav2lip_gan_path = os.path.join(
    wav2lip_gan_dir, "checkpoint_step000000100.pth")
disc_path = os.path.join(wav2lip_gan_dir, "disc_checkpoint_step000000100.pth")
face_dump_dir = "temp/face_dump/"
face_config_path = "temp/face_dump/face.tsv"
generated_video_path = "temp/generated.mp4"
generated_by_inference_video_path = "temp/generated-by-inference.mp4"


def clear(paths):
    for path in paths:
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            print("remove dir:", path)
            shutil.rmtree(path)
            continue
        print("remove file:", path)
        os.remove(path)


clear([
    pngs_dir,
    filelists_dir,
    syncnet_dir,
    wav2lip_dir,
    wav2lip_gan_dir,
    face_dump_dir,
    generated_video_path,
    generated_by_inference_video_path])

for dataset in os.listdir(video_dir):
    dataset_dir = os.path.join(video_dir, dataset)
    videos = os.listdir(dataset_dir)
    if len(videos) < 5:
        print("create duplicate videos")
        for i in range(5):
            src = os.path.join(dataset_dir, videos[0])
            dest = src[:-4] + str(i) + ".mp4"
            shutil.copyfile(src, dest)


def test_preprocess():

    subprocess.call(
        """python preprocess.py \
            --batch_size 1 \
            --data_root {video_dir} \
            --preprocessed_root {pngs_dir}""".format(
                video_dir=video_dir,
                pngs_dir=pngs_dir,
            ),
        shell=True)
    assert len(glob.glob(os.path.join(pngs_dir, "**/**/*.png"))) > 0, "must generate pngs"


def test_create_filelists():
    subprocess.call(
        """python create_filelists.py \
            --data_root {pngs_dir} \
            --train_ratio 0.5 \
            --filelists_dir {filelists_dir}""".format(
                pngs_dir=pngs_dir,
                filelists_dir=filelists_dir,
            ),
        shell=True
    )
    train_path = os.path.join(filelists_dir, "train.txt")
    val_path = os.path.join(filelists_dir, "val.txt")
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)

    for path in [train_path, val_path]:
        for row in open(path).read().strip().split('\n'):
            dirpath = os.path.join(pngs_dir, row)
            assert os.path.exists(dirpath)
            assert len(os.listdir(dirpath)) > 0


def test_color_syncnet_train():
    subprocess.call(
        """
    W2L_SYNCNET_BATCH_SIZE=16 \
    W2L_NEPOCHS=1 \
    W2L_SYNCNET_EVAL_INTERVAL=10 \
    W2L_SYNCNET_CHECKPOINT_INTERVAL=10 \
    python color_syncnet_train.py --data_root {pngs_dir} \
        --checkpoint_dir {syncnet_dir} \
        --filelists_dir {filelists_dir}""".format(
            pngs_dir=pngs_dir,
            syncnet_dir=syncnet_dir,
            filelists_dir=filelists_dir,
        ), shell=True)
    assert os.path.exists(syncnet_dir)
    assert len(os.listdir(syncnet_dir)) > 0
    assert os.path.exists(syncnet_path)


def test_wav2lip_train():
    subprocess.call(
        """
        W2L_NEPOCHS=1 \
        W2L_CHECKPOINT_INTERVAL=100 \
        W2L_EVAL_INTERVAL=100 \
        W2L_BATCH_SIZE=4 \
        python wav2lip_train.py --data_root {pngs_dir} \
        --checkpoint_dir {wav2lip_dir} \
        --syncnet_checkpoint_path {syncnet_path} \
        --filelists_dir {filelists_dir} \
        --train_limit 2 --val_limit 2""".format(
            pngs_dir=pngs_dir,
            wav2lip_dir=wav2lip_dir,
            syncnet_path=syncnet_path,
            filelists_dir=filelists_dir,
        ),
        shell=True,
    )
    assert os.path.exists(wav2lip_dir)
    assert os.path.exists(wav2lip_path)


def test_hq_wav2lip_train():
    subprocess.call(
        """
        W2L_NEPOCHS=1 \
        W2L_CHECKPOINT_INTERVAL=100 \
        W2L_EVAL_INTERVAL=100 \
        W2L_BATCH_SIZE=4 \
        python hq_wav2lip_train.py --data_root {pngs_dir} \
        --checkpoint_dir {wav2lip_gan_dir} \
        --syncnet_checkpoint_path {syncnet_path} \
        --filelists_dir {filelists_dir} \
        --train_limit 2 --val_limit 2""".format(
            pngs_dir=pngs_dir,
            wav2lip_gan_dir=wav2lip_gan_dir,
            syncnet_path=syncnet_path,
            filelists_dir=filelists_dir,
        ),
        shell=True,
    )
    assert os.path.exists(wav2lip_gan_dir)
    assert os.path.exists(wav2lip_gan_path)
    assert os.path.exists(disc_path)


def test_dump_face():
    subprocess.call(
        """
        python dump_face.py --face {video_path} \
            --pads 0 20 0 0 \
            --temp_face_dir {face_dump_dir}
        """.format(
            video_path=video_path,
            face_dump_dir=face_dump_dir,
        ), shell=True,
    )
    assert os.path.exists(face_dump_dir)
    assert os.path.exists(face_config_path)


def test_generate_video():
    subprocess.call(
        """python generate_video.py \
        --checkpoint_path {wav2lip_path} \
        --face_config_path {face_config_path} \
        --audio {audio_path} --batch_size 4 \
        --outfile {generated_video_path}""".format(
            wav2lip_path=wav2lip_path,
            face_config_path=face_config_path,
            audio_path=audio_path,
            generated_video_path=generated_video_path,
        ),
        shell=True,
    )
    assert os.path.exists(generated_video_path)


def test_inference():
    subprocess.call(
        """python inference.py \
        --face {video_path} \
        --audio {audio_path} --pads 0 20 0 0 \
        --face_det_batch_size 1 --wav2lip_batch_size 4 \
        --outfile {generated_by_inference_video_path} \
        --checkpoint_path {wav2lip_gan_path}""".format(
            video_path=video_path,
            audio_path=audio_path,
            generated_by_inference_video_path=generated_by_inference_video_path,
            wav2lip_gan_path=wav2lip_gan_path,
        ),
        shell=True,
    )
    assert os.path.exists(generated_by_inference_video_path)


def test_remove_files():
    clear([
        pngs_dir,
        filelists_dir,
        syncnet_dir,
        wav2lip_dir,
        wav2lip_gan_dir,
        face_dump_dir,
        generated_video_path,
        generated_by_inference_video_path])
