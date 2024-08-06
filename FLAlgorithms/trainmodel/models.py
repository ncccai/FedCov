import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_

#################################
##### Neural Network model #####
#################################
class Net(nn.Module):
    def __init__(self, dataset='cmpass', dataset_all_name='cmapss-biid-u5c126-FD004'):
        super(Net, self).__init__()
        # define network layers
        self.feature_ext, self.feature_fc, self.reg_block, self.cls_block, self.cls_act = None, None, None, None, None
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim = CONFIGS_[dataset]

        self.output_dim = int((dataset_all_name.split('-')[2]).split('c')[1])

        print('Network configs:', configs)
        self.build_network()
        self.n_parameters = len(list(self.parameters()))

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def build_network(self):
        self.feature_ext = nn.Sequential(
            nn.Conv1d(14, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 3, 2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 3, 1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 3, 2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.reg_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
        # add uncertainty
        self.reg_v = nn.Linear(256, 1)
        self.reg_alpha = nn.Linear(256, 1)
        self.reg_beta = nn.Linear(256, 1)

        self.cls_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=self.output_dim),
        )

    def evidence(self, x):
        return F.softplus(x)

    def split(self, mu, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta

    def forward(self, x, latent=False):
        if latent:
            return self.reg_block(x), self.cls_block(x)
        else:
            z = torch.transpose(x, 1, 2)
            z = self.feature_ext(z)
            z = z.view(z.shape[0], -1)
            shared_feature = self.feature_fc(z)

            output_reg = self.reg_block(shared_feature)
            output_cls = self.cls_block(shared_feature)

            # add uncertainty
            output_v = self.reg_v(shared_feature)
            output_alpha = self.reg_alpha(shared_feature)
            output_beta = self.reg_beta(shared_feature)
            output_reg, output_v, output_alpha, output_beta = self.split(output_reg, output_v, output_alpha, output_beta)

            return output_reg, output_cls, output_v, output_alpha, output_beta

