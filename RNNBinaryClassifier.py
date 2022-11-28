# RNN-based binary classifier, provides:
# - Training initialization including selection of loss function and optimizer and ingestion of prepared data
# - Application of the trained model to the given test dataset and storage of the result
# - Functions for evaluation of the trained model (plotting and metrics calculation)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from Trainer import Trainer

class RNNBinaryClassifier:
    
    def __init__(self, device):
        self.device = device
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.n_features = 1
    
    def load_data(self, datasets):
        self.train_set = datasets[0]
        self.val_set = datasets[1]
        self.test_set = datasets[2]
        self.logits = None    # Will hold the results from evaluating on the internal test set
        self.y_true = None
        self.sigmoids = None

    def set_test_set(self, test_set):
        self.test_set = test_set
        self.logits = None
        self.y_true = None
        self.sigmoids = None

    def train(self, model, n_features=1, n_epochs=10, lr=0.01, batch_size=100):
        
        self.n_features = n_features
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=True)
        
        loss_fn =  nn.BCEWithLogitsLoss()   # The NN output is logits so we use a loss function with built-in sigmoid
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        self.trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=self.device)
        self.trainer.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=self.n_features)
        self.trainer.plot_losses()

        return self.trainer.best_model
    
    def run_evaluation(self, batch_size=100):
        # Extract predictions of class labels for all items in the internal test set, store for later use 
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, drop_last=True)
        self.logits, self.y_true = self.trainer.evaluate(test_loader, n_features=self.n_features, batch_size=batch_size)
        self.sigmoids = torch.sigmoid(torch.tensor(self.logits)).numpy()
        self.y_true = self.y_true.astype(bool)

    def classify_with_threshold(self, r_value = 0.5):
        # Calculate the classic classification metrics, given a specific threshold (r_value)
        preds = self.sigmoids > r_value
        metrics = [np.mean(self.y_true == preds)]  # Accuracy
        metrics.append(precision_recall_fscore_support(self.y_true, preds, average='binary', pos_label=True)) # Precision, recall, F1
        
        return metrics, preds

    def calculate_signal_eff_noise_red(self):
        # Calculation of signal efficiency vs noise reduction, as defined in the Arianna paper
        nr = []
        ns = []
        NN = np.sum(self.y_true == False)  # Number of noise events
        NS = np.sum(self.y_true == True)   # Number of signal events
        print("Total signal events:", NS, "Total noise events:", NN)
        r_set = np.linspace(0.001, 0.999, 100)
        for r_value in r_set:
          y_pred = self.sigmoids > r_value
          correct_noise_events = np.sum((self.y_true == y_pred) & (self.y_true == False))
          correct_signal_events = np.sum((self.y_true == y_pred) & (self.y_true == True))
          if correct_noise_events < NN:
              cne_ratio = np.log10(1.0/(1.0 - correct_noise_events/NN))
          else:
              cne_ratio = np.log10(NN)
          nr.append(cne_ratio)
          ns.append(correct_signal_events/NS)
        
        return ns, nr 
    
    def plot_signal_eff_vs_noise_red(self):
        ns, nr = self.calculate_signal_eff_noise_red()
        plt.scatter(ns, nr)
        plt.xlabel("Signal efficiency")
        plt.ylabel("Noise reduction (log10)")
        plt.grid()
        plt.show()

    def plot_output_distribution(self):
        # Show the relative frequency of sigmoid output levels for the two classes
        h_noise = self.sigmoids[(self.y_true == False)]
        h_signal = self.sigmoids[(self.y_true == True)]
        plt.title("RNN output distribution")
        plt.hist(h_signal, density=True, bins=20, alpha=0.6, label="event")
        plt.hist(h_noise, density=True, bins=20, alpha=0.6, label="noise")
        plt.legend()
        plt.show()

    def plot_samples(self, n_samples=5):
        # Select some random examples from the current test set, show the logit output and the signal for visual verification
        dl = DataLoader(self.test_set, batch_size=1, shuffle=True)
        count = 1
        with torch.no_grad():
          for x, y in dl:
              x_input = x.view([1, -1, self.n_features]).to(self.device)
              logit = self.trainer.best_model(x_input)
              for i in range(self.n_features):
                  plt.plot(x[0,i].cpu().numpy().ravel())
                  plt.show()  
              print("Logit output:", logit.cpu().numpy())
              print("True label:", y.cpu().numpy())
              print()      
              count += 1
              if count > n_samples:
                  break
