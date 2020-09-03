import argparse
import os

from PIL import ImageFont
from PIL import Image

from handwriting_preparation.preprocessing.crop_characters import char_img_iter
from model.preprocessing_helper import CHAR_SIZE, CANVAS_SIZE, draw_example_src_only
from package import save_train_valid_data

parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', default="data/raw_fonts/NotoSansCJKtc-Regular.otf", help='path of the source font')
parser.add_argument('--image_basename_path',
                    default="kai/liuGongCyuan",
                    help='path of the handwritten image (box file should be in the same folders)')
parser.add_argument('--embedding_id', type=int, default=136, help='embedding id')
parser.add_argument('--sample_dir', default='data/paired_images_kai', help='directory to save examples')
parser.add_argument('--resample', type=int, default=1, help='sample with replacement')

# These two are for package.py
parser.add_argument('--split_ratio', type=float, default=0.1, help='split ratio between train and val')
parser.add_argument('--save_dir', default="experiments/data", help='path to save pickled files')

args = parser.parse_args()

if __name__ == '__main__':
    assert os.path.isfile(args.src_font), "src file doesn't exist:%s" % args.src_font
    src_font = ImageFont.truetype(args.src_font, size=CHAR_SIZE)
    count = 0

    image_path = args.image_basename_path

    if not os.path.isdir(args.sample_dir):
        os.makedirs(args.sample_dir)

    for dirpath, dirnames, filenames in os.walk(image_path):

        print(dirpath)
        for f in filenames:
            if not f.endswith('.jpg'):
                continue
            
            ch = f.split('_')[0].split('-')[2]
            dst_path = os.path.join(dirpath, f)
            dst_img = Image.open(dst_path).convert('L')
            dst_img.thumbnail((CANVAS_SIZE,CANVAS_SIZE), Image.ANTIALIAS)
            
            e = draw_example_src_only(ch, src_font, dst_img, CANVAS_SIZE, CHAR_SIZE)
            if e:
                for _ in range(args.resample):
                    e.save(os.path.join(args.sample_dir, "%d_%04d.jpg" % (args.embedding_id, count)), mode='F')
                    count += 1
                    if count % 100 == 0:
                        print("processed %d chars" % count)

    # Save as pickled file
    save_train_valid_data(save_dir=args.save_dir, sample_dir=args.sample_dir, split_ratio=args.split_ratio)
