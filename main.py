from models import NIC

if __name__ == '__main__':
    # model = NIC("nica.yaml", task='caption')
    # model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=100, batch=32)

    model = NIC("./runs/caption/train/version_2/checkpoints/last.pt", task='caption')
    model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=100, batch=32, resume=True)
