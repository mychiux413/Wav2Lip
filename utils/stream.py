import cv2


def stream_video_as_batch(filepath, batch_size, steps=1, infinite_loop=False):
    video_stream = cv2.VideoCapture(filepath)
    batch = []
    assert steps > 0

    while True:
        if len(batch) == batch_size:
            yield batch
            for _ in range(steps):
                batch.pop(0)

        still_reading, frame = video_stream.read()
        if not still_reading:
            if infinite_loop:
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if len(batch) < batch_size:
                    # reset
                    batch = []
            else:
                video_stream.release()
                break
        batch.append(frame)
