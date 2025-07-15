import os.path
from pathlib import Path


def process_caption(root, file, caption_path):
    with open(file, 'r', encoding='utf8') as f:
        lines_file = f.readlines()

    with open(caption_path, 'r', encoding='utf8') as f:
        lines_caption = f.readlines()

    lines_file = [l.strip() for l in lines_file]
    lines_caption = [l.strip() for l in lines_caption]

    result = {k: [] for k in lines_file}

    for info in lines_caption:
        image_split = info.split('\t')
        image_info = image_split[0]
        image_cap = image_split[1:]
        split_info = image_info.split('#')
        if len(split_info) == 3:
            image_path, translate_mode, image_ids = split_info
        else:
            image_path, image_ids = split_info
        if image_path in result.keys():
            result[image_path].append(''.join(image_cap) + '<sep>')

    return result


if __name__ == '__main__':
    # 将caption处理成统一格式
    root = r'G:\cgm\dataset\Flickr8k and Flickr8kCN\Flicker8k_Dataset'

    caption_path = r"G:\cgm\dataset\Flickr8k and Flickr8kCN\Flickr8k.lemma.token.txt"

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
            for i, (image_path, caption) in enumerate(result.items()):
                if (i + 1) == num:
                    f.write('{}\t'.format(os.path.join(root, image_path), caption))
                    f.writelines(caption)
                else:
                    f.write('{}\t'.format(os.path.join(root, image_path), caption))
                    f.writelines(caption + ['\n'])
