import argparse
import logging
import pathlib
import functools
import time

import cv2
import torch
from torchvision import transforms

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results


class SemanticSegmentation:
    threshold = 0.5
    save = False
    display = True
    model = "pretrained/model_segmentation_skin_30.pth"
    model_type = "FCNResNet101"
    img_path = "Pictures"

    def __init__(self, threshold=0.5, save=False, display=True, img_path="Pictures",
                 model="pretrained/model_segmentation_skin_30.pth", model_type="FCNResNet101"):
        self.threshold = threshold
        self.save = save
        self.display = display
        self.model_type = model_type
        self.model = model
        self.img_path = img_path

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

    def run(self):
        logging.basicConfig(level=logging.INFO)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f'running inference on {device}')

        logging.info(f'loading {self.model_type} from {self.model}')
        torch_model = torch.load(self.model, map_location=device)
        torch_model = load_model(models[self.model_type], torch_model)
        torch_model.to(device).eval()

        logging.info(f'evaluating images from {self.img_path}')
        image_dir = pathlib.Path(self.img_path)

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
                results = torch_model(image)['out']
                results = torch.sigmoid(results)

                results = results > self.threshold

            for category, category_image, mask_image in draw_results(image[0], results[0],
                                                                     categories=torch_model.categories):
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

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = SemanticSegmentation(
        args.threshold, args.save, args.display,
        args.img_path, args.model, args.model_type)
    model.run()