import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import Model
from utils import precision, psp


class Runner(object):
    def __init__(self, num_labels, dim_data, dim_hidden, lr,
                 weight_decay, drop_val, model_path, sampling_type):

        self.num_labels = num_labels
        self.model = Model(num_labels, dim_data, dim_hidden, drop_val)
        self.weight_decay = weight_decay
        self.optimizer = AdamW(self._get_params(), lr=lr)
        self.model_path = model_path
        self.sampling_type = sampling_type
        self.criterion = nn.BCEWithLogitsLoss()


    def train(self, train_loader, test_loader, inv_prop, epoch, log_step=5000):      


        print("\n****************** Training the model ******************")

        self.model.cuda()

        for epoch_idx in range(epoch):
            
            for step, batch in enumerate(train_loader, 1):
                
                self.model.train()
                self.model.zero_grad()

                data = batch[0].squeeze(0).cuda()

                if self.sampling_type == 'uniform':
                    lbl_one_hot, lbl_indices = batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
                    scores = self.model(data, lbl_indices, uniform=True)
                    loss = self.criterion(scores, lbl_one_hot)

                elif self.sampling_type == 'in-batch':
                    targets_in_batch, all_labels_batch = batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
                    scores = self.model(data, all_labels_batch)
                    loss = self.criterion(scores, targets_in_batch)

                elif self.sampling_type == 'full':
                    targets = batch[1].squeeze(0).cuda()
                    lbl_indices = torch.arange(self.num_labels, device='cuda')
                    scores = self.model(data, lbl_indices)

                    loss = self.criterion(scores, targets)                                
                
                loss.backward()

                self.optimizer.step()

                iter = epoch_idx * len(train_loader) + step + 1
                if step % log_step == 0:
                    print(F"Epoch:   {epoch_idx}, Step:   {step}, Loss:   {loss.item():.8f}")
                    self.predict(test_loader, inv_prop)                
                    # self.save_model(iter)
        self.save_model()
    


    def predict(self, data_loader, inv_prop):
        self.model.eval()
        with torch.no_grad():
            lbl_indices = torch.arange(self.num_labels, device='cuda')
            all_scores, all_labels = [], []
            for step, batch in enumerate(data_loader, 1):         
                data = batch[0].squeeze(0).cuda()
                targets_sample = batch[1]
                scores = self.model(data, lbl_indices)
                top_idx = torch.topk(scores.cpu().detach(), 100)[1]
                all_scores.append(top_idx)
                all_labels.extend(targets_sample)
        
        all_scores = torch.cat(all_scores).numpy()
        p_k = precision(pred_mat=all_scores, true_mat=all_labels, k=5)
        psp_k = psp(pred_mat=all_scores, true_mat=all_labels, inv_prop=inv_prop, k=5)
        print(F"P@1:     {p_k[0]:.4f}, P@3:     {p_k[2]:.4f}, P@5:     {p_k[4]:.4f}")
        print(F"PSP@1:   {psp_k[0]:.4f}, PSP@3:   {psp_k[2]:.4f}, PSP@5:   {psp_k[4]:.4f}\n")


    def save_model(self, iter=''):
        torch.save(self.model.state_dict(), 
                   ('{}{}').format(self.model_path, '_' + str(iter) if iter!='' else ''))

    def load_model(self, model_path):
        self.model.load_state_dict(model_path)
    

    def _get_params(self):
        grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if 'bias' not in n], 'weight_decay': self.weight_decay},
        {'params': [p for n, p in self.model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
        ]
        return grouped_parameters