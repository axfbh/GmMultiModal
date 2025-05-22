from models import SAT

if __name__ == '__main__':
    # model = SAT("sat.yaml", task='caption')
    # model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=100, batch=32)

    model = SAT("sat.yaml", task='caption')
    model.train(data="./cfg/datasets/flickr8k.yaml", device='0', imgsz=256, epochs=100, batch=32)
