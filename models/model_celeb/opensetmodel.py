import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pickle

class openSetClassifier(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super(openSetClassifier, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(512, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        # self.encoder = BaseEncoder(3,init_weights,dropout)
        # self.classify = nn.Linear(128*14*14, num_classes)
        self.classify = nn.Linear(1024, num_classes)

        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)

        # self.cpu()

    def forward(self, x, skip_distance=False):
        batch_size = len(x)
        x = x.view(batch_size, -1)

        out1 = self.fc1(x)
        out1 = self.relu1(out1)
        out2 = self.fc2(out1)
        out2 = self.relu2(out2)
        outLinear = self.classify(out2)

        if skip_distance:
            return outLinear, None

        outDistance = self.distance_classifier(outLinear)

        return outLinear, outDistance
        # return outLinear

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad=False)
        #self.cuda()

    def distance_classifier(self, x):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)
        # dists = (1 + torch.cosine_similarity(x,anchors,dim=2) ) / 2
        return dists

def predict_person_from_image(embed_result,model_celeb,thresh):
    model_celeb.eval()
    out_celeb = model_celeb(embed_result)
    logits = out_celeb[0]
    distances = out_celeb[1]
    softmax = torch.nn.Softmax(dim=1)
    softmin = softmax(-distances)
    invScores = 1 - softmin
    scores = distances * invScores
    fn_scores = scores.cpu().detach().numpy()
    score_pred = np.min(fn_scores, axis=1)[0]
    predicted = np.argmin(fn_scores, axis=1)[0]
    if score_pred > thresh:
        predicted = -999
    return predicted,score_pred




if __name__ == "__main__":
    ckpt_facereg = torch.load("models/model_celeb/FAR_1percent_90DIR_0.1percent_80DIR_best.pth", map_location='cpu')
    model_celeb = openSetClassifier(num_classes=103)
    model_celeb.to('cpu')
    net_dict = model_celeb.state_dict()
    pretrained_dict = {k: v for k, v in ckpt_facereg['net'].items() if k in net_dict}
    if 'anchors' not in pretrained_dict.keys():
        pretrained_dict['anchors'] = ckpt_facereg['net']['means']
    model_celeb.load_state_dict(pretrained_dict)
    anchor_mean = pickle.loads(open('models/model_celeb/anchor_mean.pickle', "rb").read())
    model_celeb.set_anchors(torch.Tensor(anchor_mean))
    model_celeb.eval()

    embed_result = torch.rand((1,512))
    th = 3.600184650310259
    pred = predict_image(embed_result,model_celeb,th)
    print(f'pred : {pred}')
