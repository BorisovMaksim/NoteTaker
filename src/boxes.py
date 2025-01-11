from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import colors as mcolors
from ultralytics import YOLOv10
import matplotlib.pyplot as plt







class BoxCounter:
    def __init__(self):
        self.total_area = 0
        self.mid_points = []
        self.colors = {i - 1: c for i, c in enumerate(mcolors.CSS4_COLORS.keys())}
        self.clustering = DBSCAN(eps=190, min_samples=3)
        self.boxes = []
        self.labels = None
        self.label2cluster_box = None
        
    def add(self, box):
        self.total_area += box.area
        self.mid_points.append(box.mid_point)
        self.boxes.append(box)

    def plot_mid_points(self, axis):
        if self.labels is not None:
            for i, l in enumerate(self.labels):
                mid_point = self.mid_points[i]
                axis.scatter(mid_point[0], mid_point[1], c=self.colors[l], s=1, label=l)
    
    def calculate_cluster_box(self, plot=True):
        label2cluster_box = {}
        for label, boxes in self.label2box.items():
            if label == -1:
                continue
            all_x = []
            all_y = []
            for b in boxes:
                all_y.append(b.x0[0])
                all_y.append(b.x0[0] + b.height)
                all_x.append(b.x0[1])
                all_x.append(b.x0[1] + b.width)
            
            x0 = (min(all_y), min(all_x))
            height = max(all_y) - x0[0]
            width = max(all_x) - x0[1]
            cluster_box = Box(x0, height, width)
            
            if plot:
                cluster_box.plot(color='r')

            label2cluster_box[label] = cluster_box
            
        self.label2cluster_box = label2cluster_box
        
        
    def cluster(self):
        X = np.array(self.mid_points)
        clustering = self.clustering.fit(X)
        labels = clustering.labels_
        label2box = {i: [] for i in set(labels) }
        
        for i, box in enumerate(self.boxes):
            label2box[labels[i]].append(box)

        self.labels = labels
        self.label2box = label2box
        
        
class Box:
    def __init__(self, x0, height, width):
        self.x0 = x0
        self.x1 = (x0[0] + height, x0[1])
        self.x2 = (x0[0] + height, x0[1] + width)
        self.x3 = (x0[0], x0[1] + width)

        self.height = height
        self.width = width
        self.mid_point = (x0[0] +  height / 2, x0[1] +  width / 2 )
        self.area = height * width
        
    def plot(self, color='k', axis=None):
        if not axis:
            axis = plt.gca()
        axis.add_patch(Rectangle(self.x0, self.height, self.width,  facecolor="none", ec=color, lw=2))
        
        