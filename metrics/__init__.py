import numpy as np

from scipy import ndimage as ndi

from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max, match_descriptors
from skimage.segmentation import watershed
import skimage.measure as skmeas


def find_peaks(img, threshold_rel=0.05, min_distance=1, eps=7.5, min_samples=1, max_val=255):

    peaks = peak_local_max(
      img, threshold_rel=threshold_rel,
      min_distance=min_distance, exclude_border=False
    )

    if len(peaks) > 0:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(peaks)
        peaks_clustered = np.array(
          [
            np.mean(peaks[clustering.labels_ == i], axis=0)
            for i in range(np.max(clustering.labels_) + 1)
          ]
        )

        # make an image with peaks at 1
        peak_im = np.zeros_like(img)
        for p in peaks_clustered.astype(int):
            peak_im[p[0], p[1]] = 1

        # label peaks
        peak_label, n_labels = ndi.label(peak_im)

        # propagate peak labels with watershed
        labels = watershed(max_val - img, peak_label)

        # limit watershed labels to area where the image is intense enough
        labels = labels * (img > threshold_rel * max_val)
        centers = np.array([[dim.mean() for dim in np.where(labels == l)] for l in np.unique(labels) if l != 0])

        return labels, centers
    else:
        return np.array([]), np.array([])


def peak_based_f1(true, pred, thT = 0.05, thP = 0.05, eps=8):
    labels_true, centers_true = find_peaks(
        true, threshold_rel=thT, eps=eps
    )

    labels_pred, centers_pred = find_peaks(
        pred, threshold_rel=thP, eps=eps
    )

    TP = 0
    FP = 0
    FN = 0

    TP_centers = []
    FP_centers = []
    FN_centers = []

    if centers_true.shape[0] == 0:
        FP = centers_pred.shape[0]
        FP_centers = centers_pred
    elif centers_pred.shape[0] == 0:
        FN = centers_true.shape[0]
        FN_centers = centers_true
    else:
        matches = match_descriptors(
            centers_true, centers_pred,
            metric='minkowski', p=2,
            max_distance=eps,
            cross_check=True, max_ratio=1.0
        )

        TP = matches.shape[0]
        FP = centers_pred.shape[0] - TP
        FN = centers_true.shape[0] - TP

        TP_centers = np.array([[list(centers_true[m[0]]), list(centers_pred[m[1]])] for m in matches])
        FP_centers = np.array([list(cp) for p, cp in enumerate(centers_pred) if p not in matches[:, 1]])
        FN_centers = np.array([list(ct) for p, ct in enumerate(centers_true) if p not in matches[:, 0]])

    precision = np.divide(TP, TP + FP) if TP + FP > 0 else 1.
    recall = np.divide(TP, TP + FN) if TP + FN > 0 else 1.
    f1 = np.divide(2 * precision * recall, precision + recall) if precision + recall > 0 else 0.

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP_centers": TP_centers,
        "FP_centers": FP_centers,
        "FN_centers": FN_centers
    }