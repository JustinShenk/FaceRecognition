from icrawler.builtin import GoogleImageCrawler
from icrawler.downloader import ImageDownloader
import json
import os
from io import BytesIO
from skimage import io, transform
from more_itertools import unique_everseen
import cv2


class FaceDetector:

    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        if not os.path.exists(cascade_path):
            raise ValueError('File not found:', cascade_path)

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """
        Find faces using Haar cascade.
        """
        # Convert frame to grayscale.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
        )
        return faces


class RGBFaceDownloader(ImageDownloader):

    def __init__(self, thread_num, signal, session, storage):
        """Init Parser with some shared variables."""
        self._face_detector = FaceDetector()
        super().__init__(thread_num, signal, session, storage)

    def keep_file(self, response, min_size=None, max_size=None):
        """Returns False if the image should not be kept."""
        img = io.imread(BytesIO(response.content))
        print(img.shape)
        # Discard gray value images of dimension 2.
        if len(img.shape) != 3 or img.shape[-1] != 3:
            return False

        # Only keep images with one and only one face.
        # Resize so that opencv won't freak out.
        img = transform.resize(img, output_shape=(500, 500),
                               mode='reflect', preserve_range=True)
        img = img.astype('uint8')

        try:
            faces = self._face_detector.detect(img)
        except:
            return False
        # If the facedetector does not detect anything it returns an empty tuple.
        if isinstance(faces, tuple):
            return False
        print('faces', faces.shape)
        if faces.shape[0] != 1:
            return False
        return super().keep_file(response, min_size, max_size)


if __name__ == '__main__':
    with open('scientist_curated.json', 'r') as fp:
        scientists = json.load(fp)
    for scientist in unique_everseen(scientists['scientists']):

        google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                            storage={'root_dir': os.path.join('history_crawled', '-'.join(scientist) + '-1')},
                                            downloader_cls=RGBFaceDownloader)
        scientist_full_name = ' '.join(scientist)
        google_crawler.crawl(keyword=scientist_full_name, offset=0, max_num=50,
                             date_min=None, date_max=None,
                             min_size=(200, 200), max_size=None)
