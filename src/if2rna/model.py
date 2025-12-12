

import numpy as np
import torch
import time
import os
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm


class MultiChannelResNet50(nn.Module):
    
    def __init__(self, n_channels=50, pretrained=True):
        super(MultiChannelResNet50, self).__init__()
        
        self.resnet = models.resnet50(pretrained=pretrained)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            n_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        if pretrained and n_channels != 3:
            with torch.no_grad():
                original_weight = original_conv1.weight
                
                if n_channels > 3:
                    if n_channels >= 50:
                        new_weight = torch.randn(64, n_channels, 7, 7) * 0.01
                        new_weight[:, :3, :, :] = original_weight
                    else:
                        weight = original_weight.repeat(1, n_channels // 3 + 1, 1, 1)[:, :n_channels, :, :]
                        new_weight = weight
                else:
                    new_weight = original_weight.mean(dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
                
                self.resnet.conv1.weight = nn.Parameter(new_weight)
        
        self.resnet.fc = nn.Identity()
        
    def forward(self, x):
        return self.resnet(x)


class IF2RNA(nn.Module):
    def __init__(self, input_dim=2048, output_dim=18815,
                 layers=[1024, 512], nonlin=nn.ReLU(), ks=[10],
                 dropout=0.5, device='cpu',
                 bias_init=None, **kwargs):
        super(IF2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if self.training:
            k = int(np.random.choice(self.ks))
            return self.forward_fixed_k(x, k)
        else:
            pred = 0
            for k in self.ks:
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred

    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x

    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x


def training_epoch(model, dataloader, optimizer):
    model.train()
    loss_fn = nn.MSELoss()
    train_loss = []
    for x, y in tqdm(dataloader):
        x = x.float().to(model.device)
        y = y.float().to(model.device)
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += [loss.detach().cpu().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_loss)
    return train_loss


def compute_correlations(labels, preds, projects):
    metrics = []
    for project in np.unique(projects):
        for i in range(labels.shape[1]):
            y_true = labels[projects == project, i]
            if len(np.unique(y_true)) > 1:
                y_prob = preds[projects == project, i]
                metrics.append(np.corrcoef(y_true, y_prob)[0, 1])
    metrics = np.asarray(metrics)
    return np.mean(metrics)


def evaluate(model, dataloader, projects):
    model.eval()
    loss_fn = nn.MSELoss()
    valid_loss = []
    preds = []
    labels = []
    for x, y in dataloader:
        pred = model(x.float().to(model.device))
        labels += [y]
        loss = loss_fn(pred, y.float().to(model.device))
        valid_loss += [loss.detach().cpu().numpy()]
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
    valid_loss = np.mean(valid_loss)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = compute_correlations(labels, preds, projects)
    return valid_loss, metrics


def predict(model, dataloader):
    model.eval()
    labels = []
    preds = []
    for x, y in dataloader:
        pred = model(x.float().to(model.device))
        labels += [y]
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def fit(model,
        train_set,
        valid_set,
        valid_projects,
        params={},
        optimizer=None,
        test_set=None,
        path=None,
        logdir='./exp'):

    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 200,
        'patience': 20,
        'batch_size': 16,
        'num_workers': 0}
    default_params.update(params)
    batch_size = default_params['batch_size']
    patience = default_params['patience']
    max_epochs = default_params['max_epochs']
    num_workers = default_params['num_workers']

    writer = SummaryWriter(log_dir=logdir)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_set is not None:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer is None:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                                     weight_decay=0.)

    metrics = 'correlations'
    epoch_since_best = 0
    start_time = time.time()

    if valid_set is not None:
        valid_loss, best = evaluate(
            model, valid_loader, valid_projects)
        print(f'{metrics}: {best:.3f}')
        if np.isnan(best):
            best = 0
        if test_set is not None:
            preds, labels = predict(model, test_loader)
        else:
            preds, labels = predict(model, valid_loader)

    try:

        for e in range(max_epochs):

            epoch_since_best += 1

            train_loss = training_epoch(model, train_loader, optimizer)
            dic_loss = {'train_loss': train_loss}

            print(f'Epoch {e + 1}/{max_epochs} - {time.time() - start_time:.2f}s')
            start_time = time.time()

            if valid_set is not None:
                valid_loss, scores = evaluate(
                    model, valid_loader, valid_projects)
                dic_loss['valid_loss'] = valid_loss
                score = np.mean(scores)
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                writer.add_scalar('data/metrics', score, e)
                print(f'loss: {train_loss:.4f}, val loss: {valid_loss:.4f}')
                print(f'{metrics}: {score:.3f}')
            else:
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                print(f'loss: {train_loss:.4f}')

            if valid_set is not None:
                criterion = (score > best)

                if criterion:
                    epoch_since_best = 0
                    best = score
                    if path is not None:
                        torch.save(model, os.path.join(path, 'model.pt'))
                    elif test_set is not None:
                        preds, labels = predict(model, test_loader)
                    else:
                        preds, labels = predict(model, valid_loader)

                if epoch_since_best == patience:
                    print(f'Early stopping at epoch {e + 1}')
                    break

    except KeyboardInterrupt:
        pass

    if path is not None and os.path.exists(os.path.join(path, 'model.pt')):
        model = torch.load(os.path.join(path, 'model.pt'), weights_only=False)

    elif path is not None:
        torch.save(model, os.path.join(path, 'model.pt'))

    if test_set is not None:
        preds, labels = predict(model, test_loader)
    elif valid_set is not None:
        preds, labels = predict(model, valid_loader)
    else:
        preds = None
        labels = None

    writer.close()

    return preds, labels


# Factory functions for different configurations

def create_if2rna_model_6_channel(n_genes=18815, device='cpu'):
    feature_extractor = MultiChannelResNet50(n_channels=6, pretrained=True)
    model = IF2RNA(
        input_dim=2048,
        output_dim=n_genes,
        layers=[1024, 512],
        device=device
    )
    return feature_extractor, model


def create_if2rna_model_50_channel(n_genes=18815, device='cpu'):
    feature_extractor = MultiChannelResNet50(n_channels=50, pretrained=True)
    model = IF2RNA(
        input_dim=2048,
        output_dim=n_genes,
        layers=[1024, 512],
        dropout=0.3,
        device=device
    )
    return feature_extractor, model


def create_complete_if2rna_pipeline(n_channels=50, n_genes=18815, device='cpu'):
    if n_channels <= 10:
        return create_if2rna_model_6_channel(n_genes, device)
    else:
        return create_if2rna_model_50_channel(n_genes, device)
