# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:00:08 2023

@author: kaueu
"""

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, recall_score, precision_score
import nibabel as nib

class Metrics:
    def __init__(self, pred_volume, gt_volume=None):
        self.pred_volume = pred_volume
        self.gt_volume = gt_volume

    def calculate_metrics(self):
        if self.gt_volume is None:
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

        # Flatten volumes to 1D arrays
        pred_flat = np.ravel(self.pred_volume)
        gt_flat = np.ravel(self.gt_volume)

        f_measure = self.calculate_f_measure(gt_flat, pred_flat)
        iou = self.calculate_iou()
        accuracy = self.calculate_accuracy(gt_flat, pred_flat)
        recall = self.calculate_recall(gt_flat, pred_flat)
        precision = self.calculate_precision(gt_flat, pred_flat)
        specificity = self.calculate_specificity(gt_flat, pred_flat)
        dice = self.calculate_dice()
        hausdorff_dist = self.calculate_hausdorff_distance()
        wmh_count = self.calculate_wmh_count()
        wmh_size_per_region = self.calculate_wmh_size_per_region()

        return f_measure, iou, accuracy, recall, precision, specificity, dice, hausdorff_dist, wmh_count, wmh_size_per_region

    def calculate_f_measure(self, gt_flat, pred_flat):
        return f1_score(gt_flat, pred_flat)

    def calculate_iou(self):
        intersection = np.sum(np.logical_and(self.pred_volume, self.gt_volume))
        union = np.sum(np.logical_or(self.pred_volume, self.gt_volume))
        return intersection / union

    def calculate_accuracy(self, gt_flat, pred_flat):
        return accuracy_score(gt_flat, pred_flat)

    def calculate_recall(self, gt_flat, pred_flat):
        return recall_score(gt_flat, pred_flat)

    def calculate_precision(self, gt_flat, pred_flat):
        return precision_score(gt_flat, pred_flat)

    def calculate_specificity(self, gt_flat, pred_flat):
        return np.sum(np.logical_and(np.logical_not(self.pred_volume), np.logical_not(self.gt_volume))) / \
               np.sum(np.logical_not(self.gt_volume))

    def calculate_dice(self):
        intersection = np.sum(np.logical_and(self.pred_volume, self.gt_volume))
        return 2 * intersection / (np.sum(self.pred_volume) + np.sum(self.gt_volume))

    def calculate_hausdorff_distance(self):
        distance_gt = distance_transform_edt(self.gt_volume)
        hausdorff_dist_1 = np.max(distance_gt[self.pred_volume > 0])
        hausdorff_dist_2 = np.max(distance_transform_edt(self.pred_volume)[self.gt_volume > 0])
        return max(hausdorff_dist_1, hausdorff_dist_2)

    def calculate_wmh_count(self):
        return np.sum(self.gt_volume)

    def calculate_wmh_size_per_region(self):
        labeled_gt, num_components = label(self.gt_volume)
        wmh_size_per_region = np.zeros(num_components)
        for i in range(1, num_components + 1):
            wmh_size_per_region[i - 1] = np.sum(labeled_gt == i)
        return wmh_size_per_region
    
    #For Bazil
    def calculate_roc_auc_curve(self):
        pass

def main():
    
    threshold = 0.5
    gt_volume = np.load('pred_mask/FLAIR_mask_gt_3.npy') 
    pred_volume = np.load('pred_mask/FLAIR_mask_pred_3.npy')
    pred_volume = pred_volume > threshold
    
    metrics_calculator = Metrics(pred_volume, gt_volume)
    f_measure, iou, accuracy, recall, precision, specificity, dice, hausdorff_dist, wmh_count, wmh_size_per_region = \
        metrics_calculator.calculate_metrics()
    
    print("F-measure:", f_measure)
    print("IoU:", iou)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Specificity:", specificity)
    print("Dice:", dice)
    print("Hausdorff Distance:", hausdorff_dist)
    print("WMH Count:", wmh_count)
    print("WMH Size per Region:", wmh_size_per_region)

if __name__ == "__main__":
    main()