import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import region as A
import SpatialGraph as B
import PRM as  C
import mrf as D
import readData as E


class MovingObjectDetection():
    def __init__(self):
        testdata = E.readfolder()
        index = 1
        for data in testdata:
            firstframe = data[0]
            secondframe = data[1]
            self.detection(index, firstframe, secondframe)
            index += 1

    def detection(self, curtest, firstframe, secondframe):
        self.height, self.width, _ = firstframe.shape
        alpha = 8
        beta = 32
        object = A.RegionGrow(self.height, self.width, alpha, beta)

        # Color Segmentation phase1
        colorFrame, regionFrame, totalRegion = object.applyRegionGrow(firstframe)

        # Spatial Graph phase2
        spatialGraph = B.SpatialGraph(totalRegion, regionFrame, colorFrame, firstframe)

        # Motion Estimation phase3
        motionObject = C.PartitionRegionMatching(self.height, self.width, spatialGraph, secondframe)

        # update graph after motion estimation
        currentGraph = motionObject.updateGraph()

        # Region Merging Phase 4
        obj = D.MRF(currentGraph)
        regionMap, totalNodes, currentGraph = obj.graphOptmization()

        # Background Substraction
        optmizedGraph = self.createGraph(totalNodes, currentGraph, regionMap)
        background = self.getBackground(optmizedGraph)

        movingObject = []
        index = 0
        for node in optmizedGraph:
            if background != index:
                if node.area > 64 and node.pr > 0: movingObject.append(node)
            index += 1

        resultframe = self.removeBackground(movingObject, firstframe, regionFrame)

        print('Test %d : ', curtest)
        self.printFrame(firstframe)
        self.printFrame(secondframe)
        print('Result')
        self.printFrame(resultframe)

    def removeBackground(self, movingObject, frame, regionFrame):
        hashset = set()
        for object in movingObject:
            for region in object.regions:
                hashset.add(region)
        resultFrame = np.zeros((self.height, self.width, 3), dtype='uint8')
        for i in range(0, self.height):
            for j in range(0, self.width):
                regionnum = int(regionFrame[i, j])
                if hashset.__contains__(regionnum):
                    resultFrame[i, j] = frame[i, j, 0], frame[i, j, 1], frame[i, j, 2]
                else:
                    resultFrame[i, j] = 255, 255, 255
        return resultFrame

    def createBoxes(self, movingObject, frame):
        fig, ax = plt.subplots(1)
        ax.imshow(frame)
        for object in movingObject:
            rect = patch.Rectangle((object.topy, object.topx), object.length - 1, object.height - 1, linewidth=1.5,
                                   edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def getBackground(self, graph):
        maxarea = 0
        index = 0
        maxindex = -1
        for node in graph:
            if node.area > maxarea:
                maxarea = node.area
                maxindex = index
            index += 1
        return maxindex

    class Node():
        def __init__(self):
            self.regions = set()
            self.topx, self.topy, self.bottomx, self.bottomy = -1, -1, -1, -1
            self.length, self.height, self.area = 0, 0, 0
            self.pr = 1000000
            self.mvx = -10
            self.mvy = -10

    def createGraph(self, totalNodes, graph, label):
        graphNodes = []
        for num in range(1, totalNodes):
            node = self.Node()
            nodes = []
            for reg in label[num]:
                nodetemp = graph.graphNode[reg]
                node.regions.add(reg)
                nodes.append(nodetemp)
                if nodetemp.pr < node.pr:
                    node.pr = nodetemp.pr
                if node.topx == -1:
                    node.topx = nodetemp.topX
                else:
                    node.topx = self.min(node.topx, nodetemp.topX)

                if node.topy == -1:
                    node.topy = nodetemp.topY
                else:
                    node.topy = self.min(node.topy, nodetemp.topY)

                if node.bottomx == -1:
                    node.bottomx = nodetemp.bottomX
                else:
                    node.bottomx = self.max(node.bottomx, nodetemp.bottomX)

                if node.bottomy == -1:
                    node.bottomy = nodetemp.bottomY
                else:
                    node.bottomy = self.max(node.bottomy, nodetemp.bottomY)

            height = (int)(math.fabs(node.topx - node.bottomx)) + 1
            length = (int)(math.fabs(node.topy - node.bottomy)) + 1
            node.area = (int)(length * height)
            node.height = height
            node.length = length
            node.mvx, node.mvy = self.probabilityVector(nodes)
            node.regions = label[num]
            graphNodes.append(node)

        return graphNodes

    def probabilityVector(self, nodes):
        maxprob = 0
        x, y = 0, 0
        for i in range(-4, 5):
            for j in range(-4, 5):
                sum = 0
                for n in nodes:
                    if i == n.mvx and j == n.mvy:
                        sum += 1
                temp = sum / 81
                if temp > maxprob:
                    maxprob = temp
                    x, y = i, j
        return x, y

    def min(self, a, b):
        if a > b: return b
        return a

    def max(self, a, b):
        if a > b: return a
        return b

    def printFrame(self, frame):
        plt.imshow(frame)
        plt.show()
