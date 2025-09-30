import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryConfusionMatrix, MatthewsCorrCoef, BinaryF1Score
import sklearn
from sklearn import metrics 
from sklearn.metrics import roc_curve, auc
import os


def test(model, test_loader, device, save_dir, df):
    print('testing is started')
    # Test phase
    test_loss = 0.0
    batch_out_test = []
    batch_labels_test = []
    batch_names_test = []

    model.eval()    
    with torch.no_grad():
        for batch in test_loader:
            spectrograms, labels, names = batch
            batch_labels_test.append(labels)              # append labels of all batches
            batch_names_test.extend(names)              # append names of all batches
            with torch.autocast(device_type="cuda"):
                out_T = model(spectrograms)
                batch_out_test.append(out_T)

            torch.cuda.empty_cache()

    # concatenate output and labels of each epoch
    epoch_labels_test = torch.cat(batch_labels_test)
    epoch_names_test = batch_names_test
    epoch_out_test = torch.cat(batch_out_test) 

    # calculate evaluation metrics
    # prediction probability and labels
    _ , predicted_labels = torch.max(epoch_out_test, dim=1)      # use soft max for prediction labels
    probabilities = torch.softmax(epoch_out_test, dim=1)[:, 1]
   
    # Accuracy
    acc_metrics = BinaryAccuracy().to(device)
    acc = acc_metrics(probabilities, epoch_labels_test)
    # AUC
    auc_metrics = BinaryAUROC().to(device)
    auc = auc_metrics(probabilities, epoch_labels_test)
 
    confmat = BinaryConfusionMatrix().to(device)
    confmat(predicted_labels, epoch_labels_test)
    conf_matrix = confmat.compute()
    tn = conf_matrix[0, 0].item()
    fp = conf_matrix[0, 1].item()
    fn = conf_matrix[1, 0].item()
    tp = conf_matrix[1, 1].item()
    print('TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn)

    f1score_metrics = BinaryF1Score().to(device)
    f1score = f1score_metrics(predicted_labels, epoch_labels_test)

    print('val_acc:', round(acc.item()*100, 3) ,'val_AUC', round(auc.item()*100, 3), 'prediction', probabilities, 'target', epoch_labels_test )

    
    # Matthews correlation coefficient 
    matthews_corrcoef = MatthewsCorrCoef(task='binary').to(device)
    MCC = matthews_corrcoef(predicted_labels, epoch_labels_test)
    print('MCC = ', MCC)

    # Plot test AUC
    fpr, tpr, thresholds  = roc_curve(epoch_labels_test.cpu(), np.ravel(probabilities.cpu()))    #compare annotaion file labels with classifier prediction result 
    sensitivity = tpr
    specificity = 1-fpr

    plt.figure(figsize=(10,5))
    plt.plot(specificity, sensitivity, marker='.')
    plt.title(f'ROC Curve')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig(f'{save_dir}/AUC.png')
    # plt.show()

    # save prediction and target of test set in a dataframe
    test_dict = {'ID': epoch_names_test, 'prediction': probabilities.cpu().numpy(), 'pred_labels':predicted_labels.cpu().numpy(), 'target': epoch_labels_test.cpu().numpy()}
    testdf = pd.DataFrame(test_dict)
    testdf.to_csv(f'{save_dir}/model_pred.csv', index=False)

        # save results in a text file
    result_dict = {
                   'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                   'AUC': round(auc.item()*100, 4), 
                   'acc': round(acc.item()*100, 3),
                   'F1Score': round(f1score.item()*100, 3),
                   'MCC': MCC.item()
                   }
    
    df = pd.concat([df, pd.DataFrame([result_dict])], axis=1)
    df.to_csv(f'{save_dir}/results.csv', index=False)
       