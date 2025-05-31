import json
from collections import defaultdict


def create_flickr8_json(token_path, train_path, val_path, test_path):
    flick_8k_json = {
        "images": []
    }
    with open(train_path, 'r') as fp:
        train_info = fp.readlines()
        train_info = [path.rstrip() for path in train_info]

    with open(val_path, 'r') as fp:
        val_info = fp.readlines()
        val_info = [path.rstrip() for path in val_info]

    with open(test_path, 'r') as fp:
        test_info = fp.readlines()
        test_info = [path.rstrip() for path in test_info]

    caption_res = defaultdict(list)
    with open(token_path, 'r') as fp:
        caption_info = fp.readlines()
        for caption in caption_info:
            g, token = caption.split('\t')
            filename, token_id = g.split('#')
            token = token.rstrip()
            caption_res[filename].append([int(token_id), token])

    for i, (k, v) in enumerate(caption_res.items()):
        is_train = None
        if k in val_info:
            is_train = 'val'
        elif k in test_info:
            is_train = 'test'
        elif k in train_info:
            is_train = 'train'
        if is_train is not None:
            tmp = {
                'imgid': i,
                "split": is_train,
                "filename": k,
            }
            sentids = []
            sentences = []
            for r in v:
                sentid, token = r
                sentids.append(sentid)
                sentences.append({
                    "tokens": token.split(),
                    "raw": token,
                    "imgid": i,
                    "sentid": sentid,
                })
            tmp['sentids'] = sentids
            tmp['sentences'] = sentences
            flick_8k_json['images'].append(tmp)

    with open("caption_data/dataset_flick_8k.json", "w") as f:
        json.dump(flick_8k_json, f)


if __name__ == '__main__':
    create_flickr8_json(
        token_path=r"E:\dataset\Flickr8k and Flickr8kCN\Flickr8k.lemma.token.txt",
        train_path=r"E:\dataset\Flickr8k and Flickr8kCN\Flickr_8k.trainImages.txt",
        val_path=r"E:\dataset\Flickr8k and Flickr8kCN\Flickr_8k.devImages.txt",
        test_path=r"E:\dataset\Flickr8k and Flickr8kCN\Flickr_8k.testImages.txt"

    )
