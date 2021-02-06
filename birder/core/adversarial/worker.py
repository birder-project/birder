import logging
import os
import random

import cv2


def write_worker(queue) -> None:
    while True:
        deq = queue.get()
        if deq is None:
            break

        (adv_count, image_path, img) = deq
        image_name = os.path.basename(image_path)
        image_name = f"adv_{image_name}"
        adv_path = os.path.join(os.path.dirname(image_path), image_name)
        jpeg_quality = random.randint(85, 95)
        img = img.asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(adv_path, img, (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))

        if adv_count % 1000 == 999:
            logging.info(f"Adversarial image no. {adv_count + 1}...")
