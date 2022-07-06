import torch
import copy
import numpy as np
from torch import nn
from torch import optim 
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from models.dec.dec import CAE_5
from models.dec.dataloader import DCECDataset


def get_criterion(type_:str="pretrain"):
  if type_=="pretrain":
    # Reconstruction loss
    criterion = nn.MSELoss(size_average=True)
  else:
    # Clustering loss
    criterion = nn.KLDivLoss(size_average=False)
  return criterion


def get_optimizer(model, rate, weight):
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)
  return optimizer


def get_scheduler(optimizer, step, gamma):
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
  return scheduler


def get_model_base(model, rate=0.001, weight=0.0, step=200, gamma=0.1):
  criterion = get_criterion()
  optimizer = get_optimizer(model, rate, weight)
  scheduler = get_scheduler(optimizer, step, gamma)

  return criterion, optimizer, scheduler


def pretrain(model, device, dataloader, num_epochs, dataset_size):
  rate = 0.001
  weight = 0.0
  step = 200
  gamma = 0.1

  criterion = get_criterion()
  optimizer = get_optimizer(model, rate, weight)
  scheduler = get_scheduler(optimizer, step, gamma)

  best_model_weights = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in tqdm((range(num_epochs)), desc="Pretrain ..."):
    scheduler.step()
    model.train(True)
    running_loss = 0.0
      
    batch_num = 1

    # Iterate over data.
    for data in dataloader:
      inputs = data.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      with torch.set_grad_enabled(True):
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

      # For keeping statistics
      running_loss += loss.item() * inputs.size(0)

      # Some current stats
      batch_num = batch_num + 1
      
    epoch_loss = running_loss / dataset_size
    if epoch == 0: first_loss = epoch_loss
    if epoch == 4 and epoch_loss / first_loss > 1:
      print("Loss not converging, starting pretraining again")
      return False

    # If wanted to add some criterium in the future
    if epoch_loss < best_loss or epoch_loss >= best_loss:
      best_loss = epoch_loss
      best_model_weights = copy.deepcopy(model.state_dict())
    
    
  # load best model weights
  model.load_state_dict(best_model_weights)
  model.pretrained = True
  torch.save(model.state_dict(), "./pretrained.pt")

  return model


# K-means clusters initialisation
def kmeans(model, device, dataloader):
  km = KMeans(n_clusters=model.num_clusters, n_init=20)
  output_array = None
  model.eval()
  # Itarate throught the data and concatenate the latent space representations of images
  for data in dataloader:
    inputs = data.to(device)
    _, _, outputs = model(inputs)
    if output_array is not None:
      output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
    else:
      output_array = outputs.cpu().detach().numpy()
    # print(output_array.shape)
    if output_array.shape[0] > 50000: break

  # Perform K-means
  km.fit_predict(output_array)
  # Update clustering layer weights
  weights = torch.from_numpy(km.cluster_centers_)
  model.clustering.set_weight(weights.to(device))
  # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, device, dataloader):
  output_array = None
  # label_array = None
  model.eval()
  for data in dataloader:
    # inputs, labels = data
    inputs = data
    inputs = inputs.to(device)
    # labels = labels.to(device)
    _, outputs, _ = model(inputs)
    if output_array is not None:
      output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
      # label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
    else:
      output_array = outputs.cpu().detach().numpy()
      # label_array = labels.cpu().detach().numpy()

  print(f'[D] output. {output_array.shape}')

  preds = np.argmax(output_array.data, axis=1)
  # print(output_array.shape)
  # return output_array, label_array, preds
  return output_array, preds


# Calculate target distribution
def target(out_distr):
  tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
  tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
  return tar_dist


def dcec_train(model, device, dataloader, dataset_size, batch, num_epochs, update_interval, tol,
               pretrained, pretrain_path:str, pretrain_epochs:int=200):
  rate = 0.001
  weight = 0.0
  step = 200
  gamma = 0.1

  criterion = get_criterion()
  optimizer = get_optimizer(model, rate, weight)
  scheduler = get_scheduler(optimizer, step, gamma)

  if pretrained:
    pretrained_model = pretrain(model, device, copy.deepcopy(dataloader), 
                                pretrain_epochs, dataset_size)
    model = pretrained_model
  else:
    try:
      model.load_state_dict(torch.load(pretrain_path))
    except:
      print(f'[W] Couldn\'t load pretrained weights')

  kmeans(model, device, copy.deepcopy(dataloader))

  # Prep variables for weights and accuracy of the best model
  best_model_weights = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  # Initial target distribution
  # output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params)
  output_distribution, preds_prev = \
      calculate_predictions(model, device, copy.deepcopy(dataloader))
  target_distribution = target(output_distribution)
    
  dataset_size = len(dataloader)
  finished = False
  # Go through all epochs
  for epoch in tqdm(range(num_epochs), desc="Main train ..."):
    scheduler.step()
    model.train(True)  # Set model to training mode

    running_loss = 0.0
    # running_loss_rec = 0.0
    # running_loss_clust = 0.0

    # Keep the batch number for inter-phase statistics
    batch_num = 1

    # Iterate over data.
    for data in dataloader:
      inputs = data.to(device)
      
      # Uptade target distribution, chack and print performance
      if (batch_num - 1) % update_interval == 0 and \
          not (batch_num == 1 and epoch == 0):
        # output_distribution, labels, preds = calculate_predictions(model, dataloader, params)
        output_distribution, preds = \
            calculate_predictions(model, device, copy.deepcopy(dataloader))
        target_distribution = target(output_distribution)

        print(f'[D] prediction.\n\tprev({len(preds_prev)}): {preds_prev}\n\tpreds({len(preds)}):{preds}')

        # check stop criterion
        delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
        preds_prev = np.copy(preds)
        if delta_label < tol:
          print('Label divergence ' + str(delta_label) + ' < tol ' + str(tol))
          print('Reached tolerance threshold. Stopping training.')
          finished = True
          break

      tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
      tar_dist = torch.from_numpy(tar_dist).to(device)
      # print(tar_dist)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Calculate losses and backpropagate
      with torch.set_grad_enabled(True):
          outputs, clusters, _ = model(inputs)
          loss_rec = criterion(outputs, inputs)
          loss_clust = gamma * criterion(torch.log(clusters), tar_dist) / batch
          loss = loss_rec + loss_clust
          loss.backward()
          optimizer.step()

      # For keeping statistics
      running_loss += loss.item() * inputs.size(0)
      # running_loss_rec += loss_rec.item() * inputs.size(0)
      # running_loss_clust += loss_clust.item() * inputs.size(0)

    if finished: break

    epoch_loss = running_loss / dataset_size

    # If wanted to do some criterium in the future (for now useless)
    if epoch_loss < best_loss or epoch_loss >= best_loss:
        best_loss = epoch_loss
        best_model_weights = copy.deepcopy(model.state_dict())

  # load best model weights
  model.load_state_dict(best_model_weights)
  torch.save(model.state_dict(), "./full-trained.pt")
  return model


if __name__=="__main__":
  cae = CAE_5(input_shape=[512, 512, 3], num_clusters=20)
  train_data = DCECDataset('/data/kwkim/dataset/bladder/dcec_testset',
                           img_size=[512, 512, 3])
  dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
  device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")
  cae.to(device)
  dcec_train(cae, device, dataloader, len(train_data), 1, 200, 10, 1e-2,
      True, "./pretrained.pt", 200)

  inference = False
  if inference:
    test_data = DCECDataset('/data/kwkim/dataset/bladder/dcec_testset',
                            img_size=[512, 512, 3])
    dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    cae.eval()
    
    with cae.no_grad():
      cluster_idx = list()
      for data in dataloader:
        _, output, _ = cae(data.to(device))
        cluster_idx.append(output)

    print(f'[D] inference result: {cluster_idx}')

