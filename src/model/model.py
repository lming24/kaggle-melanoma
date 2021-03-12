"""
Module containing all available neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from base.base_model import BaseModel
import model.modules as modules


class MnistModel(BaseModel):
    """
    Simple CNN for the Mnist dataset
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResNet(BaseModel):
    """
    Implementation of ResNet model
    """
    def __init__(self, variant_str, num_classes, additional_features=0, pretrained=True):
        super().__init__()
        variant_dict = {
            '101': models.resnet101,
            '50': models.resnet50,
            '34': models.resnet34,
            '18': models.resnet18,
            '152': models.resnet152
        }

        resnet = variant_dict[variant_str](pretrained=pretrained)
        resnet_layers = list(resnet.children())
        in_features = resnet_layers[-1].in_features
        self.num_classes = num_classes
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Linear(in_features + additional_features, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ
        # Append additional features here
        features = x["features"]
        x = self.resnet(x["img"])
        x = torch.flatten(x, 1)
        x = torch.cat([x, features], axis=1)
        if self.num_classes == 1:
            return self.final(x).view(-1)
        else:
            return self.final(x)


class EfficientNetwork(BaseModel):
    """
    Implementation of Efficient model
    """
    def __init__(self, variant_str, num_classes, additional_features=0, extra_layer=False):
        # pylint: disable=too-many-arguments
        super().__init__()
        variant_dict = {
            'b0': ('efficientnet-b0', 1280),
            'b1': ('efficientnet-b1', 1280),
            'b2': ('efficientnet-b2', 1408),
            'b3': ('efficientnet-b3', 1536),
            'b4': ('efficientnet-b4', 1792),
            'b5': ('efficientnet-b5', 2048)
        }

        variant_str, out_size = variant_dict[variant_str]
        self.num_classes = num_classes

        self.model = EfficientNet.from_pretrained(variant_str)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.other = nn.Linear(additional_features, 100)

        if extra_layer:
            self.fc = nn.Sequential(nn.Linear(out_size + 100, 3000), modules.Swish(), nn.Linear(3000, num_classes))
        else:
            self.fc = nn.Linear(out_size + 100, num_classes)  # pylint: disable=invalid-name

    def _forward_impl(self, x):
        # Append additional features here
        features = x["features"]
        batch_size = features.size(0)

        img = x["img"]

        x = self.model.extract_features(img)
        x = self.avg(x).view(batch_size, -1)

        features = self.other(features)

        x = torch.cat([x, features], axis=1)
        if self.num_classes == 1:
            return self.fc(x).view(-1)
        else:
            return self.fc(x)

    def forward(self, x):  # pylint: disable=arguments-differ
        # Append additional features here
        img = x["img"]

        # 4D tensor. first dim is batch, followed by channels, height, width
        if len(img.size()) == 4:
            return self._forward_impl(x)
        elif len(img.size()) == 5:  # test time augmentations
            tta = img.size(1)
            image_size = img.size()[2:]
            x["img"] = x["img"].view(-1, *image_size)
            x["features"] = torch.repeat_interleave(x["features"], repeats=tta, dim=0)
            result = self._forward_impl(x)
            sizes = result.size()[1:]
            result = result.view(-1, tta, *sizes)
            result = torch.mean(result, dim=1)
            return result
        else:
            raise ValueError("Incorrect input img shape")


class MILEfficientNetwork(BaseModel):
    """
    Implementation of Efficient model
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 variant_str,
                 zero_mean_per_channel=False,
                 one_std_per_channel=False,
                 mil_hidden=100,
                 mil_n_attentions=10,
                 mil_gated=False,
                 outputs=1,
                 checkpoint=None,
                 load_mode='all',
                 freeze_context_extractor=False):
        # pylint: disable=too-many-arguments
        super().__init__()
        variant_dict = {
            'b0': ('efficientnet-b0', 1280),
            'b1': ('efficientnet-b1', 1280),
            'b2': ('efficientnet-b2', 1408),
            'b3': ('efficientnet-b3', 1536),
            'b4': ('efficientnet-b4', 1792),
            'b5': ('efficientnet-b5', 2548)
        }

        self.zero_mean_per_channel = zero_mean_per_channel
        self.one_std_per_channel = one_std_per_channel

        variant_str, out_size = variant_dict[variant_str]

        self.model = EfficientNet.from_pretrained(variant_str)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.mil = modules.MILAttentionPool(out_size, mil_hidden, mil_n_attentions, gated=mil_gated)

        self.fc1 = nn.Linear(mil_n_attentions * out_size, out_size)  # pylint: disable=invalid-name
        if outputs == 0:
            self.fc2 = None
        else:
            self.fc2 = nn.Linear(out_size, outputs)  # pylint: disable=invalid-name
        self.outputs = outputs

        if checkpoint:
            state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
            if load_mode == 'all':
                self.load_state_dict(state_dict)
            elif load_mode == 'main':
                for key in list(state_dict.keys()):
                    if key.startswith("model."):
                        state_dict[key[6:]] = state_dict.pop(key)
                    else:
                        state_dict.pop(key)
                self.model.load_state_dict(state_dict)

        if freeze_context_extractor:
            self.model.requires_grad_(False)

    def _per_instance_forward(self, img):
        # total_instances, channel, height, width

        batch_size = img.size(0)

        mean = 0.0
        if self.zero_mean_per_channel:
            mean = torch.mean(img, dim=(2, 3), keepdim=True)

        std = 1.0
        if self.one_std_per_channel:
            std = torch.std(img, dim=(2, 3), keepdim=True) + 1e-8

        img = (img - mean) / std

        x = self.model.extract_features(img)
        x = self.avg(x).view(batch_size, -1)

        return x

    def extract_context(self, imgs, bag_lengths):
        """
        Extract representation for the whole bag
        """
        imgs = self._per_instance_forward(imgs)

        bags = self.mil(imgs, bag_lengths)
        return self.fc1(bags)

    def forward(self, x):  # pylint: disable=arguments-differ
        imgs = x["imgs"]
        imgs = self._per_instance_forward(imgs)

        bag_lengths = x["len"]

        bags = self.mil(imgs, bag_lengths)
        bags = F.relu(self.fc1(bags), inplace=True)

        if self.outputs == 1:
            return self.fc2(bags).view(-1)
        else:
            return self.fc2(bags)


class EfficientNetworkWithContext(BaseModel):
    """
    Implementation of Efficient model
    """

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self,
                 variant_str,
                 num_classes,
                 additional_features=0,
                 zero_mean_per_channel=False,
                 one_std_per_channel=False,
                 mil_hidden=100,
                 mil_n_attentions=1,
                 mil_gated=True,
                 concat_method='concat',
                 hidden_unit_multiplier=3,
                 checkpoint=None,
                 load_mode='main',
                 load_single_image=False,
                 freeze_context_extractor=False):
        # pylint: disable=too-many-arguments
        super().__init__()
        variant_dict = {
            'b0': ('efficientnet-b0', 1280),
            'b1': ('efficientnet-b1', 1280),
            'b2': ('efficientnet-b2', 1408),
            'b3': ('efficientnet-b3', 1536),
            'b4': ('efficientnet-b4', 1792),
            'b5': ('efficientnet-b5', 2548)
        }

        self.zero_mean_per_channel = zero_mean_per_channel
        self.one_std_per_channel = one_std_per_channel

        variant_str_net, out_size = variant_dict[variant_str]
        self.num_classes = num_classes

        self.model = EfficientNet.from_pretrained(variant_str_net)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.other = nn.Sequential(nn.Linear(additional_features, 200), nn.LeakyReLU(inplace=True), nn.Dropout(p=0.1),
                                   nn.BatchNorm1d(200), nn.Linear(200, 500), nn.LeakyReLU(inplace=True))

        self.context = MILEfficientNetwork(variant_str,
                                           zero_mean_per_channel=zero_mean_per_channel,
                                           one_std_per_channel=one_std_per_channel,
                                           mil_hidden=mil_hidden,
                                           mil_n_attentions=mil_n_attentions,
                                           mil_gated=mil_gated,
                                           outputs=0,
                                           checkpoint=checkpoint,
                                           load_mode=load_mode,
                                           freeze_context_extractor=freeze_context_extractor)

        assert concat_method in ['concat', 'subtract']
        self.concat_method = concat_method

        if concat_method == 'concat':
            input_size = 2 * out_size
        else:
            input_size = out_size

        self.image_fc = nn.Sequential(nn.Linear(input_size, int(hidden_unit_multiplier * input_size)),
                                      nn.LeakyReLU(inplace=True),
                                      nn.BatchNorm1d(int(hidden_unit_multiplier * input_size)),
                                      nn.Linear(int(hidden_unit_multiplier * input_size), out_size),
                                      nn.LeakyReLU(inplace=True))

        self.fc = nn.Linear(out_size + 500, num_classes)  # pylint: disable=invalid-name

        if checkpoint and load_single_image:
            state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    state_dict[key[6:]] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
            self.model.load_state_dict(state_dict)

    def _forward_impl(self, x):
        # Append additional features here
        features = x["features"]
        batch_size = features.size(0)

        img = x["img"]
        context = x["context"]
        bag_lengths = x["context_lengths"]

        mean = 0.0
        if self.zero_mean_per_channel:
            mean = torch.mean(img, dim=(2, 3), keepdim=True)

        std = 1.0
        if self.one_std_per_channel:
            std = torch.std(img, dim=(2, 3), keepdim=True) + 1e-8

        img = (img - mean) / std

        # Features from target image
        x = self.model.extract_features(img)
        x = self.avg(x).view(batch_size, -1)

        # Other patient features
        features = self.other(features)

        # Other pictures of patient
        if len(context.size()) == 5:
            context = context.unbind(dim=1)
            res = []
            for context_tmp in context:
                res.append(self.context.extract_context(context_tmp, bag_lengths))
            # res is list of batch x output (length tta)
            res = torch.stack(res, dim=0)
            tta, batch_size, _ = res.size()
            context = res.transpose(0, 1).contiguous().view(batch_size * tta, -1)
        else:
            context = self.context.extract_context(context, bag_lengths)

        if self.concat_method == 'concat':
            x = torch.cat([x, context], axis=1)
        else:
            x = x - context

        x = self.image_fc(x)
        x = torch.cat([x, features], axis=1)

        if self.num_classes == 1:
            return self.fc(x).view(-1)
        else:
            return self.fc(x)

    def forward(self, x):  # pylint: disable=arguments-differ
        # Append additional features here
        img = x["img"]

        # 4D tensor. first dim is batch, followed by channels, height, width
        if len(img.size()) == 4:
            return self._forward_impl(x)
        elif len(img.size()) == 5:  # test time augmentations
            tta = img.size(1)
            image_size = img.size()[2:]
            x["img"] = x["img"].view(-1, *image_size)
            x["features"] = torch.repeat_interleave(x["features"], repeats=tta, dim=0)
            result = self._forward_impl(x)
            sizes = result.size()[1:]
            result = result.view(-1, tta, *sizes)
            result = torch.mean(result, dim=1)
            return result
        else:
            raise ValueError("Incorrect input img shape")


class EfficientNetworkWithContext2(BaseModel):
    """
    Implementation of Efficient model
    """

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self,
                 variant_str,
                 num_classes,
                 additional_features=0,
                 concat_method='concat',
                 hidden_unit_multiplier=3,
                 checkpoint=None):
        # pylint: disable=too-many-arguments
        super().__init__()
        variant_dict = {
            'b0': ('efficientnet-b0', 1280),
            'b1': ('efficientnet-b1', 1280),
            'b2': ('efficientnet-b2', 1408),
            'b3': ('efficientnet-b3', 1536),
            'b4': ('efficientnet-b4', 1792),
            'b5': ('efficientnet-b5', 2548)
        }

        variant_str_net, out_size = variant_dict[variant_str]
        self.num_classes = num_classes

        self.model = EfficientNet.from_pretrained(variant_str_net)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.other = nn.Sequential(nn.Linear(additional_features, 200), modules.Swish(), nn.Dropout(p=0.2),
                                   nn.BatchNorm1d(200), nn.Linear(200, 500), modules.Swish())

        assert concat_method in ['concat', 'subtract']
        self.concat_method = concat_method

        if concat_method == 'concat':
            input_size = 2 * out_size
        else:
            input_size = out_size

        self.image_fc = nn.Sequential(nn.Linear(input_size, int(hidden_unit_multiplier * input_size)), modules.Swish(),
                                      nn.Dropout(p=0.2), nn.BatchNorm1d(int(hidden_unit_multiplier * input_size)),
                                      nn.Linear(int(hidden_unit_multiplier * input_size), out_size), modules.Swish())

        self.fc = nn.Linear(out_size + 500, num_classes)  # pylint: disable=invalid-name

        if checkpoint:
            state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    state_dict[key[6:]] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
            self.model.load_state_dict(state_dict)

    def _forward_impl(self, x):
        # Append additional features here
        features = x["features"]
        batch_size = features.size(0)

        img = x["img"]
        context = x["context"]
        total_context_length = context.size(0)
        bag_lengths = x["context_lengths"]

        # Features from target image
        x = self.model.extract_features(torch.cat([img, context], dim=0))
        x = self.avg(x).view(batch_size + total_context_length, -1)

        img_x = x[:batch_size]

        img_context = x[batch_size:]
        indices = torch.arange(batch_size, device=bag_lengths.device)
        target = torch.repeat_interleave(indices, bag_lengths)

        context_representation = torch.zeros((batch_size, img_x.size(1)), device=target.device)
        context_representation = context_representation.index_add(0, target, img_context)
        context_representation = context_representation / bag_lengths.view(batch_size, -1)

        # Other patient features
        features = self.other(features)

        if self.concat_method == 'concat':
            x = torch.cat([img_x, context_representation], axis=1)
        else:
            x = img_x - context_representation

        x = self.image_fc(x)
        x = torch.cat([x, features], axis=1)

        if self.num_classes == 1:
            return self.fc(x).view(-1)
        else:
            return self.fc(x)


#batch x img, tta, c, h, w

    def forward(self, x):  # pylint: disable=arguments-differ
        # Append additional features here
        img = x["img"]

        # 4D tensor. first dim is batch, followed by channels, height, width
        if len(img.size()) == 4:
            return self._forward_impl(x)
        elif len(img.size()) == 5:  # test time augmentations
            tta = img.size(1)
            image_size = img.size()[2:]
            x["img"] = x["img"].view(-1, *image_size)
            x["features"] = torch.repeat_interleave(x["features"], repeats=tta, dim=0)
            x["context"] = list(torch.split(x["context"], x["context_lengths"].tolist(), dim=0))
            for i in range(len(x["context"])):
                x["context"][i] = x["context"][i].transpose(0, 1).flatten(0, 1)
            x["context"] = torch.cat(x["context"], dim=0)
            x["context_lengths"] = torch.repeat_interleave(x["context_lengths"], repeats=tta, dim=0)
            result = self._forward_impl(x)
            sizes = result.size()[1:]
            result = result.view(-1, tta, *sizes)
            result = torch.mean(result, dim=1)
            return result
        else:
            raise ValueError("Incorrect input img shape")

if __name__ == '__main__':
    ResNet('101', 10)
