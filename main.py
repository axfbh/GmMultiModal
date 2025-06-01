from models import NIC

if __name__ == '__main__':
    model = NIC("nic.yaml", task='caption')
    model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=100, batch=32)
