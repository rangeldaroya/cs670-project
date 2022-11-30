import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from lime import lime_image
from skimage.segmentation import mark_boundaries

RANDOM_SEED = 0
MODEL_PATH = "/home/rdaroya/Documents/cs670-project/models/resnet50_oxford102_acc0.80.pth"
idx2label = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
device = "cpu"
NUM_SAMPLES = 15    # number of samples to generate

np.random.seed(RANDOM_SEED)

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.ToTensor(),normalize])


    return trans  

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


if __name__=="__main__":

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_feats, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 102),   # 102 Oxford102 Flower categories
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    testset = torchvision.datasets.Flowers102(
        root='./data', split='test', download=True, transform=trans)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False#, num_workers=2
    )


    for i, (inputs, targets) in enumerate(test_dl):
        # print(inputs, inputs.shape)
        # print(targets)
        model.eval()
        logits = model(inputs)
        # print(f"logits: {logits}")
        probs = F.softmax(logits, dim=1)
        probs5 = probs.topk(5)
        # print(tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy())))
        
        img = inputs.cpu().detach().numpy()
        img = img[0,:,:,:]
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        img = Image.fromarray(np.uint8((img-np.min(img))*255/(np.max(img)-np.min(img)))).convert('RGB')
        
        # predict on a single image
        test_pred = batch_predict([pill_transf(img)])
        pred_idx = test_pred.squeeze().argmax()
        # print(f"pred: {pred_idx}, target: {targets}")
        pred_class = idx2label[pred_idx]
        target_class = idx2label[targets]

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(pill_transf(img)), 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        plt.imshow(img_boundry1)      
        plt.savefig(f"../outputs/lime/oxford/{i:02d}_t{target_class}_p{pred_class}_boundary.jpg")
        plt.close()

        # Shade areas that contribute to top prediction
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        plt.imshow(img_boundry2)
        plt.savefig(f"../outputs/lime/oxford/{i:02d}_t{target_class}_p{pred_class}_shade.jpg")
        plt.close()
        print(f"Done marking img {i+1:02d}/{len(test_dl)}")
        if (i+1) == NUM_SAMPLES:
            break