# Provides generic logic for training and evaluation of a Pytorch neural network model using mini-batches
# Assumes that we have separate train, validation and test sets

import numpy as np
import matplotlib.pyplot as plt
import torch

class Trainer:
    
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.best_model = None
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.model.train()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):

        min_loss = 1e6  # Keep track of the minimum validation loss during training, so we can save the best model
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            i = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                #print("Train x_batch:", x_batch.shape, "y_batch:", y_batch.shape)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
                if i % (len(train_loader)//10 + 1) == 0:
                    print(i, end="...")
                i += 1
            print()
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    #print("Eval x_val:", x_val.shape, "y_val:", y_val.shape)
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(yhat, y_val).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")
                
                if validation_loss < min_loss:
                    print("Best so far, saving")
                    min_loss = validation_loss
                    torch.save(self.model.state_dict(), "best_model.dict")
                    self.best_model = self.model
        
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        # Apply the best model (that was found during training) to the test set, return the result for each input
        
        with torch.no_grad():
            
            # Do this only to find out how many time steps there are in each sequence:
            for _, y in test_loader:
                output_length = y.size()[1]
                break
            N_OUTPUTS = batch_size*output_length

            preds = np.zeros((len(test_loader), N_OUTPUTS))
            targets = np.zeros((len(test_loader), N_OUTPUTS))
            i = 0
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                self.best_model.eval()
                y_pred = self.best_model(x_test)
                preds[i,:] = y_pred.detach().cpu().numpy().reshape(1,-1)
                targets[i,:] = y_test.detach().cpu().numpy().reshape(1,-1)
                i += 1

        return preds.reshape(1,-1)[0], targets.reshape(1,-1)[0]
    
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses during training")
        plt.show()
