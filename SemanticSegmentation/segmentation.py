import time
t0 = time.time()
import argparse
import logging
import pathlib
import cv2
import torch
from torchvision import transforms
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results

t = time.time()


class SemanticSegmentation:

    def __init__(self, threshold=0.5, save=False, display=True, gpu=0, img_path="Pictures",
                 model="pretrained/model_segmentation_skin_30.pth", model_type="FCNResNet101", verbose=False):
        self.threshold = threshold
        self.save = save
        self.display = display
        self.model_type = model_type
        self.model = model
        self.img_path = img_path
        self.gpu = gpu
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        if gpu == -1:
            self.device = 'cpu'
        else:
            if self.gpu >= 0 and torch.cuda.is_available():
                self.device = f"cuda:{self.gpu}"
            else:
                self.device = "cpu"
                if self.verbose:
                    logging.error('no gpu found!')
        if self.verbose:
            logging.info(f'running inference on {self.device}')
            logging.info(f'loading {self.model_type} from {self.model}')
        torch_model = torch.load(self.model, map_location=self.device)
        torch_model = load_model(models[self.model_type], torch_model)
        torch_model.to(self.device).eval()
        self.torch_model = torch_model

    @staticmethod
    def find_files(dir_path: pathlib.Path, file_exts):
        assert dir_path.exists()
        assert dir_path.is_dir()

        for file_ext in file_exts:
            yield from dir_path.rglob(f'*{file_ext}')

    @staticmethod
    def _load_image(image_path: pathlib.Path):
        image = cv2.imread(str(image_path))
        assert image is not None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_width = (image.shape[1] // 32) * 32
        image_height = (image.shape[0] // 32) * 32

        image = image[:image_height, :image_width]
        return image

    @staticmethod
    def _load_cv2_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_width = (image.shape[1] // 32) * 32
        image_height = (image.shape[0] // 32) * 32

        image = image[:image_height, :image_width]
        return image

    @staticmethod
    def scale_image(img, scale=60):
        scale = np.sqrt(scale/100)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        logging.info(f'Resized Dimensions : {resized.shape}')
        return resized

    def run(self, image, scale=60):
        oh, ow, _ = image.shape
        img = self.scale_image(image, scale)
        fn_image_transform = transforms.Compose([
                transforms.Lambda(lambda image_path: self._load_cv2_image(img)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        if self.verbose:
            start_time = time.time()
        img = fn_image_transform(img)
        with torch.no_grad():
            img = img.to(self.device).unsqueeze(0)
            results = self.torch_model(img)['out']
            results = torch.sigmoid(results)
            results = results > self.threshold
        # squeezing to remove batch channel
        result_mask = results[0].cpu().numpy().squeeze().astype(np.uint8)*255
        segmenting_mask = cv2.resize(result_mask, (ow, oh))
        mask_image = cv2.bitwise_and(image, image, mask=segmenting_mask)
        if self.verbose:
            logging.info(f'Execution time: {time.time() - start_time} seconds')
        if self.display:
            cv2.imshow(f'segmented_skin', mask_image)
        return mask_image

    def test_run(self):
        logging.info(f'evaluating images from {self.img_path}')
        image_dir = pathlib.Path(self.img_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fn_image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda image_path: self._load_image(image_path)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        for image_file in self.find_files(image_dir, ['.png', '.jpg', '.jpeg']):
            logging.info(f'segmenting {image_file} with threshold of {self.threshold}')
            start_time = time.time()
            image = fn_image_transform(image_file)

            with torch.no_grad():
                image = image.to(device).unsqueeze(0)
                results = self.torch_model(image)['out']
                results = torch.sigmoid(results)
                results = results > self.threshold

            for category, category_image, mask_image in draw_results(image[0], results[0],
                                                                     categories=self.torch_model.categories):
                if self.save:
                    output_name = f'results_{category}_{image_file.name}'
                    logging.info(f'writing output to {output_name}')
                    cv2.imwrite('output/' + str(output_name), category_image)
                    cv2.imwrite(f'output/mask_{category}_{image_file.name}', mask_image)
                logging.info(f'Execution time: {time.time() - start_time} seconds')
                if self.display:
                    cv2.imshow(category, category_image)
                    cv2.imshow(f'mask_{category}', mask_image)

            if self.display:
                if cv2.waitKey(0) == ord('q'):
                    logging.info('exiting...')
                    exit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="Pictures")
    parser.add_argument('--model', type=str, default="pretrained/model_segmentation_skin_30.pth")
    parser.add_argument('--model-type', type=str, choices=models, default="FCNResNet101")
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def limit(n, low, high):
    if n < low:
        return low
    elif n > high:
        return high
    return n


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    args = parse_args()
    print(f"Importing Dependencies took {t-t0}s")
    t0 = time.time()
    print("Initializing Model")
    model = SemanticSegmentation(
        args.threshold, args.save, args.display, args.device,
        args.img_path, args.model, args.model_type, args.verbose)
    t = time.time()
    print(f"Model Initialization took: {t-t0:.2}s")
    scale = 16
    fps = 0
    while cap.isOpened():
        t0 = time.time()
        _, frame = cap.read()
        segmented_skin = model.run(frame, scale)
        scale = limit(scale, 5, 100)
        segmented_skin = cv2.putText(segmented_skin,
                             f"FPS: {fps:.2f} - Scale: {scale}%.",
                             (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 255), thickness=2,
                             lineType=cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.imshow('segmented_skin', segmented_skin)
        key = cv2.waitKey(1)
        t = time.time()
        fps = (t-t0)**-1
        print(f"FPS: {fps:.2f} Scale: {scale}")
        if key == ord("q"):
            cap.release()
            break
        elif key == ord(","):
            scale -= 1
        elif key == ord("."):
            scale += 1
    cv2.destroyAllWindows()
