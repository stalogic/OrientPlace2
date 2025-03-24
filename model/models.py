import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self):
        super(MyCNNCoarse, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn.fc = torch.nn.Linear(512, 16 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),  # 28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1),  # 56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=1),  # 112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1),  # 224
        )

    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.cnn = MyCNN()
        self.cnn_coarse = MyCNNCoarse()
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)
        self.grid = 224

    def forward(
        self,
        canvas: torch.Tensor,
        wire_img: torch.Tensor,
        pos_mask: torch.Tensor,
        soft_coefficient: float = 1.0,
    ):
        cnn_in = torch.concat([wire_img, pos_mask], dim=1)
        cnn_out = self.cnn(cnn_in)

        coarse_input = torch.concat([canvas, wire_img, pos_mask], axis=1)
        coarse_out = self.cnn_coarse(coarse_input)

        merge_out = self.merge(torch.cat((cnn_out, coarse_out), dim=1))
        x = merge_out.reshape(-1, self.grid * self.grid)

        mask = wire_img + 10 * pos_mask
        mask = mask.reshape(-1, self.grid * self.grid)
        threshold = mask.min(dim=1, keepdim=True)[0] + soft_coefficient
        mask = mask.le(threshold).logical_not().float()  # mask.gt(threshold).float()
        x = torch.where(mask >= 1.0, -1.0e10, x.double())
        x = self.softmax(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.ori_emb = nn.Embedding(8, 64)

    def forward(self, macro_id: torch.Tensor, oritnt: torch.Tensor):
        x0 = torch.concat([self.pos_emb(macro_id.long()), self.ori_emb(oritnt.long())], axis=1)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value
    


class OrientCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 1), # N, 8, 224 224
            nn.ReLU(),
            nn.AvgPool2d(2),    # N, 8, 112, 112
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),    # N, 8, 56, 56
            nn.Conv2d(8, 1, 1), # N, 1, 56, 56
            nn.ReLU(),
            nn.AvgPool2d(2),    # N, 1, 28, 28
            nn.Flatten(),       # N, 784
        )

    def forward(self, x):
        return self.cnn(x)
    

class OrientResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 784)

    def forward(self, x):
        return self.resnet(x)



class OrientActor(nn.Module):

    def __init__(self):
        super(OrientActor, self).__init__()
        self.cnn = OrientCNN()
        self.resnet = OrientResnet()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(784*2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, canvas: torch.Tensor, wire_img: torch.Tensor, pos_mask: torch.Tensor):
        batch_size = canvas.shape[0]
        canvas = torch.tile(canvas, (1, 8, 1, 1))
        assert canvas.shape == (batch_size, 8, 224, 224)
        assert wire_img.shape == (batch_size, 8, 224, 224)
        assert pos_mask.shape == (batch_size, 8, 224, 224)

        reshaped_canvas = canvas.reshape(-1, 224, 224) # 8N, 224, 224
        reshaped_wire_img = wire_img.reshape(-1, 224, 224)  # 8N, 224, 224
        reshaped_pos_mask = pos_mask.reshape(-1, 224, 224)  # 8N, 224, 224

        stacked_input = torch.stack((reshaped_canvas, reshaped_wire_img, reshaped_pos_mask), dim=1) # 8N, 3, 224, 224
        cnn_out = self.cnn(stacked_input)   # 8N, 784
        resnet_out = self.resnet(stacked_input) # 8N, 784
        stacked_out = torch.concat((cnn_out, resnet_out), dim=1) # 8N, 2*784
        mlp_out = self.mlp(stacked_out) # 8N, 1
        logit = mlp_out.reshape((batch_size, 8)) # N, 8
        prob = self.softmax(logit)
        return prob


class OrientCritic(nn.Module):
    def __init__(self):
        super(OrientCritic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)

    def forward(self, macro_id: torch.Tensor):
        emb = self.pos_emb(macro_id.long())
        x1 = F.relu(self.fc1(emb))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value