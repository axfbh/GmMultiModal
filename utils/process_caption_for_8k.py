import os.path
from pathlib import Path


def process_caption(root, file, caption_path):
    with open(file, 'r', encoding='utf8') as f:
        lines_file = f.readlines()

    with open(caption_path, 'r', encoding='utf8') as f:
        lines_caption = f.readlines()

    lines_file = [l.strip() for l in lines_file]
    lines_caption = [l.strip() for l in lines_caption]

    result = []

    for info in lines_caption:
        image_split = info.split(' ')
        image_info = image_split[0]
        image_cap = image_split[1:]
        image_path, translate_mode, image_ids = image_info.split('#')
        if image_path in lines_file:
            result.append([os.path.join(root, image_path), ''.join(image_cap)])

    return result


if __name__ == '__main__':
    # 将caption处理成统一格式
    root = r'G:\cgm\dataset\Flickr8k and Flickr8kCN\Flicker8k_Dataset'

    caption_path = r"G:\cgm\dataset\Flickr8k and Flickr8kCN\flickr8kcn\data\flickr8kzhb.caption.txt"

    files = [r"G:\cgm\dataset\Flickr8k and Flickr8kCN\Flickr_8k.trainImages.txt",
             r"G:\cgm\dataset\Flickr8k and Flickr8kCN\Flickr_8k.devImages.txt",
             r"G:\cgm\dataset\Flickr8k and Flickr8kCN\Flickr_8k.testImages.txt"]

    modes = ['train', 'val', 'test']

    output_path = Path('../datasets')

    output_path.mkdir()

    for file, mode in zip(files, modes):
        result = process_caption(root, file, caption_path)

        num = len(result)

        with open(os.path.join(output_path, f'{mode}.txt'), 'w', encoding='utf8') as f:
            for i, (image_path, caption) in enumerate(result):
                if (i + 1) == num:
                    f.write('{}\t{}'.format(image_path, caption))
                else:
                    f.write('{}\t{}\n'.format(image_path, caption))
