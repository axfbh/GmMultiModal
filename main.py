from models import NIC

if __name__ == '__main__':
    model = NIC("nica.yaml", task='caption')
    model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=30, batch=7)

    # model = NIC("./runs/caption/train/version_4/checkpoints/last.pt", task='caption')
    # model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=50, batch=7, resume=True)
