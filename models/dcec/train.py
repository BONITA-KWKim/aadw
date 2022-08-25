import torch
import copy
import numpy as np
from time import time
from torch import nn
from torch import optim 
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dcec.dcec import CAE_5
from models.dcec.dataloader import DCECDataset
from models.dcec.dataloader import DCECBatchDataset


def get_criterion():
  # Reconstruction loss
  criterion_1 = nn.MSELoss(size_average=True)
  # Clustering loss
  criterion_2 = nn.KLDivLoss(size_average=False)

  criteria = [criterion_1, criterion_2]
  return criteria


def get_optimizer(model, rate, weight):
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)
  return optimizer


def get_scheduler(optimizer, step, gamma):
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
  return scheduler


def pretrain(model, device, dataloader, dataset_size, num_epochs:int=2000):
# def pretrain(model, device, dataloader, num_epochs, dataset_size):
  rate = 0.001
  weight = 0.0
  step = 200
  gamma = 0.1

  criteria = get_criterion()
  criterion = criteria[0]
  optimizer = get_optimizer(model, rate, weight)
  scheduler = get_scheduler(optimizer, step, gamma)

  best_model_weights = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  previous_loss = 0.0

  for epoch in tqdm((range(num_epochs)), desc="Pretrain ..."):
    scheduler.step()
    model.train(True)
    running_loss = 0.0
      
    batch_num = 1

    # Iterate over data.
    for data in dataloader:
      data = data[0]
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
    print(f'[D] epoch_loss({epoch}th): {epoch_loss}')
    if epoch == 0:
      first_loss = epoch_loss
    if epoch == 4 and epoch_loss / first_loss > 1:
      print("Loss not converging, starting pretraining again")
      return False
    
    if epoch % 5 == 0:
      torch.save(model.state_dict(), f"./pretrained-tmp-20220721/pretrained-tmp-{epoch}-{round(epoch_loss,4)}.pt")
    
    # If wanted to add some criterium in the future
    # if epoch_loss < best_loss or epoch_loss >= best_loss:
    if epoch_loss < best_loss:
      best_loss = epoch_loss
      best_model_weights = copy.deepcopy(model.state_dict())
    
  # load best model weights
  model.load_state_dict(best_model_weights)
  model.pretrained = True
  torch.save(model.state_dict(), "./pretrained.pt")

  return model


# K-means clusters initialisation
def kmeans(model, device, dataloader):
  # km = KMeans(n_clusters=model.num_clusters, n_init=20)
  output_array = None
  model.eval()
  # Itarate throught the data and concatenate the latent space representations of images
  for data in dataloader:
    data = data[0]
    inputs = data.to(device)
    _, _, outputs = model(inputs)
    if output_array is not None:
      output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
    else:
      output_array = outputs.cpu().detach().numpy()
    # print(output_array.shape)
    # if output_array.shape[0] > 50000: break

  split_count = 1000
  if output_array.shape[0] > split_count:
    n, _ = divmod(len(output_array), split_count)
    split_list = np.array_split(output_array, n)
  else:
    split_list = np.array_split(output_array, 1)

  kmeans_kwargs = {
    # "init": "random",
    "init": "k-means++",
    "max_iter": 1000,
    "random_state": 0,
    # "random_state": 42,
    "n_init": 20,
    "tol": 1e-04
  }
  # Get Model and learning
  if len(split_list) == 1:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=model.num_clusters, **kmeans_kwargs)
  else:
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=model.num_clusters, **kmeans_kwargs)

  # train kmeans
  if len(split_list) == 1:
    km.fit(split_list[0])          
  else:
    for s in split_list:
      km.partial_fit(s) ## Partially fitting data in batches

  # Perform K-means
  # km.fit_predict(output_array)
  # Update clustering layer weights
  weights = torch.from_numpy(km.cluster_centers_)
  print(f'[D] weights({weights.shape}): {weights}')
  model.clustering.set_weight(weights.to(device))
  # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output 
# and returning prediciotns and labels
def calculate_predictions(model, device, dataloader):
  output_array = None
  model.eval()
  for data in dataloader:
    inputs = data[0]
    inputs = inputs.to(device)
    _, outputs, _ = model(inputs)
    if output_array is not None:
      output_array = \
          np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
    else:
      output_array = outputs.cpu().detach().numpy()

  preds = np.argmax(output_array.data, axis=1)
  return output_array, preds


# Calculate target distribution
def target(out_distr):
  tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
  tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
  return tar_dist


def dcec_train(model, device, dataloader, dataset_size, batch:int=64, 
               num_epochs:int=200, update_interval:int=140, tol=1e3, 
               pretrained:bool=False, pretrain_path:str="./pretrained.pt", 
               pretrain_epochs:int=2000):
  print('[D] Start DCEC train')
  rate = 0.001
  weight = 0.0
  step = 200
  gamma = 0.1

  criteria = get_criterion()
  optimizer = get_optimizer(model, rate, weight)
  scheduler = get_scheduler(optimizer, step, gamma)

  if pretrained is False:
    pretrained_model = pretrain(model, device, copy.deepcopy(dataloader), 
                                dataset_size, pretrain_epochs)
    model = pretrained_model
  else:
    try:
      model.load_state_dict(torch.load(pretrain_path))
    except:
      print(f'[W] Couldn\'t load pretrained weights')
  
  t0 = time()
  kmeans(model, device, copy.deepcopy(dataloader))
  t1 = time()
  print(f'[D] Clustering time: {t1 - t0}')

  # Prep variables for weights and accuracy of the best model
  best_model_weights = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  # Initial target distribution
  # output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params)
  t0 = time()
  output_distribution, preds_prev = \
      calculate_predictions(model, device, copy.deepcopy(dataloader))
  target_distribution = target(output_distribution)
  t1 = time()
  print(f'[D] Prediction time: {t1 - t0}')
  
  dataset_size = len(dataloader)
  print(f'[D] Dataset Size: {dataset_size}')
  # finished = False
  # Go through all epochs
  for epoch in tqdm(range(num_epochs), desc="Main train ..."):
    t0 = time() # Epoch start time
    
    scheduler.step()
    model.train(True)  # Set model to training mode

    # Stop criterion
    # if epoch % update_interval == 0 and not epoch == 0:
    if epoch % update_interval == 0 and epoch > 500:
      c_t0 = time()
      output_distribution, preds = \
          calculate_predictions(model, device, copy.deepcopy(dataloader))
      target_distribution = target(output_distribution)

      print(f'[D] prediction.\n\tprev({len(preds_prev)}): {preds_prev}\
            \n\tpreds({len(preds)}):{preds}')

      c_t1 = time()
      print(f'[D] Prediction for model checking time: {c_t1 - c_t0}')
      # check stop criterion
      delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
      preds_prev = np.copy(preds)
      if delta_label < tol:
        torch.save(model.state_dict(), f"./full-traied/full-trained-tolerance-{epoch}.pt")
        print(f'[D] epochs: {epoch}')
        print('Label divergence ' + str(delta_label) + ' < tol ' + str(tol))
        print('Reached tolerance threshold. Stopping training.')
        # finished = True
        # break

    # variables for an epoch
    running_loss = 0.0
    batch_num = 1

    # Iterate over data.
    for dl in dataloader:
      data = dl[0]
      inputs = data.to(device)

      # if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 or epoch == 0):
      #   output_distribution, preds = \
      #       calculate_predictions(model, device, copy.deepcopy(dataloader))
      #   target_distribution = target(output_distribution)

      #   print(f'[D] prediction.\n\tprev({len(preds_prev)}): {preds_prev}\
      #         \n\tpreds({len(preds)}):{preds}')

      #   # check stop criterion
      #   delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
      #   preds_prev = np.copy(preds)
      #   if delta_label < tol:
      #     print(f'[D] epochs: {epoch}, batch_num: {batch_num}')
      #     print('Label divergence ' + str(delta_label) + ' < tol ' + str(tol))
      #     print('Reached tolerance threshold. Stopping training.')
      #     finished = True
      #     break

      tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
      tar_dist = torch.from_numpy(tar_dist).to(device)
      
      # zero the parameter gradients
      optimizer.zero_grad()

      # Calculate losses and backpropagate
      with torch.set_grad_enabled(True):
        outputs, clusters, _ = model(inputs)
        loss_rec = criteria[0](outputs, inputs)
        loss_clust = gamma * criteria[1](torch.log(clusters), tar_dist) / batch
        loss = loss_rec + loss_clust
        loss.backward()
        optimizer.step()

      # For keeping statistics
      running_loss += loss.item() * inputs.size(0)

      batch_num = batch_num + 1

    # early stopping
    # if finished: 
    #   print('[D] Early Stopping')
    #   break

    t1 = time() # Epoch start time
    print(f'[D] Training time(an Epoch): {t1 - t0}')

    epoch_loss = running_loss / dataset_size
    print(f'[D] loss({epoch}): {epoch_loss}')

    if epoch == 0: first_loss = epoch_loss
    if epoch == 4 and epoch_loss / first_loss > 1:
      print("\nLoss not converging, starting pretraining again\n")
      return False
    
    if epoch % 5 == 0:
      torch.save(model.state_dict(), f"./full-traied/full-trained-tmp-{epoch}-{round(epoch_loss,4)}.pt")
  
    # If wanted to do some criterium in the future (for now useless)
    # if epoch_loss < best_loss or epoch_loss >= best_loss:
    if epoch_loss >= best_loss:
      best_loss = epoch_loss
      best_model_weights = copy.deepcopy(model.state_dict())

  # load best model weights
  model.load_state_dict(best_model_weights)
  torch.save(model.state_dict(), "./full-trained.pt")
  return model


#  nohup python train.py > dcec-20220719-001.log & : cuda102-v100
#  nohup python train.py > dcec-20220719-002.log & : cuda113-3080
import os
if __name__=="__main__":
  # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
  device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")
  cae = CAE_5(input_shape=[512, 512, 3], num_clusters=20)
  
  if os.path.exists("./full-trained.pt"):
    cae.load_state_dict(torch.load("./full-trained.pt"))

  workers = 8
  batch_size = 128
  # workers = 1
  # batch_size = 2

  # train = True
  train = False
  pretrained = True
  # pretrained = False
  if train:
    cae.to(device)
    train_data = DCECDataset('/data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1',
                            img_size=[512, 512, 3])
    # train_data = DCECDataset('/data/kwkim/dataset/bladder/dcec_testset',
    #                         img_size=[512, 512, 3])
    # train_data = DCECDataset('/data/kwkim/dataset/bladder/test_patches',
    #                          img_size=[512, 512, 3])
    # train_batchdir = '/data/kwkim/dataset/bladder/patch-v4.1/trainset-v4.1/batch'
    # train_data = DCECBatchDataset(train_batchdir, img_size=[512, 512, 3])
    dataloader = DataLoader(train_data, batch_size=batch_size, 
                            num_workers=workers, shuffle=False)
    dcec_train(cae, device, dataloader, len(train_data), batch=batch_size, 
               num_epochs=2000, update_interval=140, tol=1e-3, 
               pretrained=pretrained, pretrain_path="./pretrained.pt", 
               pretrain_epochs=2000)

  inference = True
  # inference = False
  if inference:
    t0 = time()
    cae.load_state_dict(torch.load("./full-trained.pt"))
    cae.eval()
    cae.to(device)
    t1 = time()
    print(f'[D] Ready to model: {t1-t0}')

    t0 = time()
    test_data = DCECDataset('/data/kwkim/dataset/bladder/patch-v4.1/testset-v4.1',
                            img_size=[512, 512, 3])
    # test_data = DCECDataset('/data/kwkim/dataset/bladder/test_patches',
    #                         img_size=[512, 512, 3])
    # test_data = DCECDataset('/data/kwkim/dataset/bladder/dcec_testset',
    #                         img_size=[512, 512, 3])
    # test_batchdir = '/data/kwkim/dataset/bladder/patch-v4.1/testset-v4.1/batch'
    # test_data = DCECBatchDataset(test_batchdir, img_size=[512, 512, 3])
    dataloader = DataLoader(test_data, batch_size=batch_size, 
                            num_workers=workers, shuffle=False)
    t1 = time()
    print(f'[D] Ready to Dataloader: {t1-t0}')
    
    output_arr = None
    path_arr = []
    t0 = time()
    cnt = 0 # tmp
    for dl in dataloader:
      data = dl[0]
      _, outputs, _ = cae(data.to(device))
      if output_arr is not None:
        output_arr = np.concatenate((output_arr, outputs.cpu().detach().numpy()), 0)
      else:
        output_arr = outputs.cpu().detach().numpy()
      
      path_arr.extend(dl[1])

    path_arr = np.array(path_arr)
    preds = np.argmax(output_arr.data, axis=1)
    t1 = time() 
    print(f'[D] Prediction time: {t1 - t0}')
    print(f'[D] length. path: {len(path_arr)}, prediction: {len(preds)}')
    print(f'[D] Path result: {path_arr}')
    print(f'[D] inference result: {preds}')
    for idx, p in enumerate(preds):
      print(f'({idx}th) prediction: {p}')
