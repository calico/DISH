import os, cv2, sys
import tensorflow as tf
import argparse, os, numpy as np
import scipy.stats
import pathlib
import os


# writes the output boxes
class DishScorer:
    def __init__(self, objDetMod, classifyMod, numBoxes, minScr):
        self._boxMod = objDetMod
        self._classMod = classifyMod
        self._minScr = minScr
        self._nBox = numBoxes
        # augmentations
        self._aug_spineFlip = False
        self._aug_adjOne = False

    # functions for modifying behavior with augmentations
    def addSpineFlip(self):
        self._aug_spineFlip = True

    def addAdjustOne(self):
        self._aug_adjOne = True

    def scoreImgDetails(self, img):
        """also returns the y,x coords and score for each bridge"""
        detailL = []
        score = self._scoreImgHelp(img,detailL)
        return score,detailL

    def scoreImg(self, img):
        """the public-facing interface for the useful function.
        It applies all specified augmentations"""
        # I won't do anything with the individual box details
        return self._scoreImgHelp(img,[])

    def _scoreImgHelp(self, img, boxDetailL, initialCall=True):
        """the useful function.  "initialCall" allows this function
        to call itself recursively with an augmented
        image as input.  boxDetailL will be filled with tuples of
        box positions & scores: (y,x,score)
        """
        # for the details
        imgH,imgW = img.shape[:2]
        # get the boxes
        boxL = self._getOkBoxL(img)
        # if no boxes were found (shouldn't happen),
        # default null score is zero
        if len(boxL) == 0:
            score = 0.0
        else:
            # score each box
            bScoreL = []
            for b in boxL:
                # for the details output
                boxY = (b.yMin() + b.yMax()) / 2.0 / imgH
                boxX = (b.xMin() + b.xMax()) / 2.0 / imgW
                # legacy code from when I was augmenting each box
                # (current behavior: single item in augBoxL
                augBoxL = self._getAugBoxes(b, img)
                boxImgL = []
                for ab in augBoxL:
                    # extract the box sub-image (one vertebral bridge)
                    bImg = img[ab.yMin() : ab.yMax(), ab.xMin() : ab.xMax(), :]
                    boxImgL.append(bImg)
                # classes are numeric (correspond to amount of DISH-like growth)
                bClassResL = list(map(self._classMod.getClasses, boxImgL))
                bScore = np.mean(list(map(self._getScrFromClRes, bClassResL)))
                bScoreL.append(bScore)
                boxDetailL.append( (boxY,boxX,bScore) )
            # the final score is just the sum of all the bridge scores
            # across the spine
            score = sum(bScoreL)
        # creates a flipped version of the image to score, then
        # averages the two.  uses this function recursively,
        # applying initialCall==False to set recursion limit.

        if initialCall and self._aug_spineFlip:
            flipImg = np.flip(np.copy(img), 1)
            flipDetailL = []
            flipScore = self._scoreImgHelp(flipImg, flipDetailL, False)
            # I need to flip the details' positions on the x-axis
            for ypf,xpf,fsc in flipDetailL:
                boxDetailL.append( (ypf,1.0-xpf,fsc) )
            score = np.mean([score, flipScore])
        return score

    def _getAugBoxes(self, box, img):
        """This function allowed me to do box-specific
        data augmentation by shiftin the boxes around.
        That functionality was explored during the
        development phase but abandoned prior to
        deployment.  But I want to leave this here
        so that I don't have to re-engineer the rest
        of the logic (dealing with a box vs list of boxes).
        """
        imgH, imgW = img.shape[:2]
        boxL = [box]
        return boxL

    def _getScrFromClRes(self, classRes):
        """converts the bridge class names (strings) to
        their corresponding numbers.
        categories are "brN", where N = 0, 1, 2, or 3"""
        score = int(classRes.best()[-1])
        # if specified, adjusts the scores of 1 to a decimal
        # value in the range of [0-1], depending on the distribution
        # of zero versus non-zero scores.  this hueristic reduced
        # the contribution of noise from 1-scoring bridges (IMO the
        # hardest to classify versus 0, and that was also reflected
        # in classification model performance).
        if self._aug_adjOne and score == 1:
            pLess = classRes.score("br0")
            pMore = np.mean(list(map(classRes.score, ["br1", "br2", "br3"])))
            if pMore + pLess > 0:
                score = pMore / (pLess + pMore)
        return score

    def _getOkBoxL(self, img):
        """calls the obj-detect model and gets the most-
        confidently-identified boxes, using the instance-
        defined max number of boxes.
        """
        boxL = self._boxMod.getBoxes(img)
        boxL = list(filter(lambda b: b.score() >= self._minScr, boxL))
        if len(boxL) > self._nBox:
            # the n is the tiebreaker
            boxL = [(boxL[n].score(), n, boxL[n]) for n in range(len(boxL))]
            boxL.sort(reverse=True)
            boxL = boxL[: self._nBox]
            boxL = [b for (s, n, b) in boxL]
        return boxL


class ImageDirLister:
    """allows iteration through images in a
    directory, using the ImageLister interface
    """

    def __init__(self, hostDir, append=".png"):
        # check that the host dir exists
        if not (os.path.isdir(hostDir)):
            raise ValueError("host dir doesn't exist")
        self._hostD = os.path.abspath(hostDir)
        self._append = append

    def getImgFiles(self):
        imgFL = os.listdir(self._hostD)
        imgFL.sort()
        aLen = len(self._append)
        imgFL = list(filter(lambda i: i[-aLen:] == self._append, imgFL))
        imgFL = list(map(lambda i: os.path.join(self._hostD, i), imgFL))
        return imgFL


class ImageFileLister:
    """allows iteration through images files that are
    listed in a text file, using the ImageLister interface
    """

    def __init__(self, fileOfFiles):
        # check that the host dir exists
        if not (os.path.isfile(fileOfFiles)):
            raise ValueError("file-of-files doesn't exist")
        self._fofName = fileOfFiles

    def getImgFiles(self):
        with open(self._fofName) as f:
            imgFL = f.readlines()
        imgFL = list(map(lambda i: i.rstrip(), imgFL))
        return imgFL


class ImageDirScorer:
    """applies scores to a set of images and records
    the results to a file (or stdout)
    """

    def __init__(self, scorer, fileLister):
        self._scorer = scorer
        self._fileLister = fileLister

    def scoreImages(self, outfileName, outfileDetails=""):
        """outfileName is for the standard scores.
        outfileDetails is for extra by-box details"""
        imgFL = self._fileLister.getImgFiles()
        print("Analyzing " + str(len(imgFL)) + " images.")
        # if I'm writing to stdout, the output will be
        # the progress marker, so no need for dots
        if outfileName == "stdout":
            outf = sys.stdout
            outfDetails = sys.stdout
            progress = NullDotWriter()
        else:
            outf = open(outfileName, "w")
            progress = DotWriter(5, 50, 250)
        writeDetails = False
        if outfileDetails:
            writeDetails = True
            if outfileDetails == "stdout":
                outfDetails = sys.stdout
            else:
                outfDetails = open(outfileDetails,'w')
        count = 0
        for imgF in imgFL:
            progress.tick()
            if not (os.path.isfile(imgF)):
                raise ValueError("Image file not found: " + imgF)
            if len(imgF.split(".")) < 2:
                aLen = 0
            else:
                aLen = len(imgF.split(".")[-1]) + 1
            imgName = os.path.basename(imgF)[:-aLen]
            img = cv2.imread(imgF)
            if not(writeDetails):
                score = self._scorer.scoreImg(img)
            else:
                score,detailL = self._scorer.scoreImgDetails(img)
            outf.write(imgName + "\t" + str(score) + "\n")
            outf.flush()
            if writeDetails:
                detailL.sort()
                for n in range(len(detailL)):
                    ypos,xpos,scr = detailL[n]
                    deetStr = '\t'.join(list(map(str,[imgName,n+1,scr,ypos,xpos])))
                    outfDetails.write(deetStr + "\n")
                    outfDetails.flush()
        if outf != sys.stdout:
            outf.close()
        if writeDetails and outfDetails != sys.stdout:
            outfDetails.close()


class DotWriter:
    """progress tracker for UI that prints a dot
    after scoring the specified number of images,
    with intermediate bar markings & dot-per-line option
    """

    def __init__(self, perDot, perBar, perLine):
        self._pDot = perDot
        self._pBar = perBar
        self._pLine = perLine
        self._count = 0

    def tick(self):
        self._count += 1
        if self._count % self._pBar == 0:
            sys.stdout.write("|")
        elif self._count % self._pDot == 0:
            sys.stdout.write(".")
        if self._count % self._pLine == 0:
            sys.stdout.write("\n")
        sys.stdout.flush()


class NullDotWriter:
    """null progress tracker for UI"""

    def __init__(self):
        pass

    def tick(self):
        pass


class PerformanceAnalyzer:
    """a special class to run analysis on pre-annotated
    images and perform statistics on new vs old results.
    implemented within this script to accelerate model
    development & prototype scoring functions
    """

    def __init__(self, annotFile):
        self._imgfToScore = {}
        with open(annotFile) as f:
            line = f.readline()
            while line:
                if line[0] != ">":
                    imgF = line.strip()
                    self._imgfToScore[imgF] = 0.0
                else:
                    if len(line) == 1:
                        cols = [""]
                    cols = line[1:].rstrip().split("\t")
                    # the categories will be in the last column, either "brN" or "Br_N"
                    # where N is 0, 1, 2, or 3;
                    # "FV_" provides the option to give a float value
                    if cols[-1].find("FV_") == 0:
                        scr = float(cols[-1].split("_")[1])
                    else:
                        scr = int(cols[-1][-1])
                    self._imgfToScore[imgF] += scr
                line = f.readline()

    def scoreImages(self, scorer):
        print("Analyzing " + str(len(self._imgfToScore)) + " images.")
        annotL, modelL = [], []
        progress = DotWriter(5, 50, 250)
        for imgF in self._imgfToScore.keys():
            if not (os.path.isfile(imgF)):
                raise ValueError("Image file not found: " + imgF)
            progress.tick()
            annotL.append(self._imgfToScore[imgF])
            img = cv2.imread(imgF)
            score = scorer.scoreImg(img)
            modelL.append(score)
        sys.stdout.write("\n")
        print(str(scipy.stats.linregress(annotL, modelL)))


class TfClassifier:
    """applies the specified TF image-classification model"""

    def __init__(self, existingModelFile, categoryFile):
        self._modFile = existingModelFile
        self._catFile = categoryFile
        proto_as_ascii_lines = tf.gfile.GFile(categoryFile).readlines()
        self._labels = list(map(lambda i: i.rstrip(), proto_as_ascii_lines))
        # ## Load a (frozen) Tensorflow model into memory.
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._modFile, "rb") as fid:
                serialized_graph = fid.read()
                print(self._modFile)
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        self._sess = tf.Session(graph=self._detection_graph)

    def getClasses(self, image, spCl=None):
        # get the image tensor so I can re-size the image appropriately
        image_tensor = self._detection_graph.get_tensor_by_name("Placeholder:0")
        h, w = image.shape[:2]
        if h * w == 0:
            image = np.zeros(image_tensor.shape[1:])
        image_resized = cv2.resize(image, dsize=tuple(image_tensor.shape[1:3]))
        image_np_expanded = np.expand_dims(image_resized, axis=0)
        image_np_expanded = image_np_expanded.astype(np.float32)
        image_np_expanded /= 255
        answer_tensor = self._detection_graph.get_tensor_by_name("final_result:0")
        # Actual detection.
        (answer_tensor) = self._sess.run(
            [answer_tensor], feed_dict={image_tensor: image_np_expanded}
        )
        results = np.squeeze(answer_tensor)
        results = [(results[n], self._labels[n]) for n in range(len(self._labels))]
        return TfClassResult(results)

    def labels(self):
        return self._labels


class TfClassResult:
    """wraps a classification result
    into a convenient interface
    """

    # results: a list of score,label tuples
    def __init__(self, results):
        self._rD = {}
        for s, lb in results:
            self._rD[lb] = s
        self._lbmx = max(results)[1]

    def best(self):
        return self._lbmx

    def score(self, lb):
        return self._rD[lb]

    def labels(self):
        return self._rD.keys()


# separate out the box-drawing
class TfObjectDetector:
    """applies the specified TF object-detection model to images"""

    def __init__(self, existingModelFile, categoryFile):
        self._modFile = existingModelFile
        self._catFile = categoryFile
        # this graph
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._modFile, "rb") as fid:
                serialized_graph = fid.read()
                print(self._modFile)
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        with open(self._catFile) as f:
            catText = f.read()
        self._category_index = {}
        for entry in catText.split("item {")[1:]:
            idNum = int(entry.split("id:")[1].split("\n")[0].strip())
            idName = entry.split("name:")[1].split("\n")[0].strip()[1:-1]
            self._category_index[idNum] = {"id": idNum, "name": idName}
        self._sess = tf.Session(graph=self._detection_graph)
        # for my own convenience
        self._numToName = {}
        for d in self._category_index.values():
            self._numToName[d["id"]] = d["name"]

    def getClassIds(self):
        outD = {}
        for d in self._category_index.values():
            outD[d["name"]] = d["id"]
        return outD

    def getBoxes(self, image):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self._detection_graph.get_tensor_by_name("image_tensor:0")
        # Each box represents a part of the image where a particular object was detected.
        boxes = self._detection_graph.get_tensor_by_name("detection_boxes:0")
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self._detection_graph.get_tensor_by_name("detection_scores:0")
        classes = self._detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = self._detection_graph.get_tensor_by_name("num_detections:0")
        # Actual detection.
        (boxes, scores, classes, num_detections) = self._sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded},
        )
        h, w, ch = image.shape
        bL, scL, numB = boxes[0], scores[0], num_detections[0]
        classL = classes[0]
        boxL = []
        for n in range(int(numB)):
            yA, yB = int(bL[n][0] * h), int(bL[n][2] * h)
            xA, xB = int(bL[n][1] * w), int(bL[n][3] * w)
            clName = self._numToName[classL[n]]
            boxL.append(Box(xA, yA, xB, yB, scL[n], clName))
        return boxL


class Box:
    """A box, defined by two corners at points (x0,y0) and
    (x1,y1) on the plane of the DEXA image.  Units for
    point positions are 0-indexed pixels.  Score is the
    confidence score given to that box by the Object
    Detector model.  clName gives the label for the
    detected object's class.  Not used here since there
    is only one class of objects being detected, but
    useful for debugging so I kept it around.
    """

    def __init__(self, x0, y0, x1, y1, score, clName):
        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x1, y1
        self._score = score
        self._clName = clName

    # recover coords with min/max values
    def xMin(self):
        return min([self._x0, self._x1])

    def yMin(self):
        return min([self._y0, self._y1])

    def xMax(self):
        return max([self._x0, self._x1])

    def yMax(self):
        return max([self._y0, self._y1])

    def score(self):
        return self._score

    def name(self):
        return self._clName

    def exists(self):
        return self._x0 != self._x1 and self._y0 != self._y1

    # to allow for modifications
    def copy(self):
        return Box(self._x0, self._y0, self._x1, self._y1, self._score, self._clName)

    def translate(self, xTrans, yTrans):
        """slides the box along each axis (pixels)"""
        self._x0, self._x1 = self._x0 + xTrans, self._x1 + xTrans
        self._y0, self._y1 = self._y0 + yTrans, self._y1 + yTrans

    def constrain(self, imgW, imgH):
        """limits the box to the confines of the image"""
        if self.xMin() < 0:
            if self.xMax() < 0:
                self._x0, self._x1 = 0, 0
            else:
                self._x0, self._x1 = 0, self.xMax()
        if self.yMin() < 0:
            if self.yMax() < 0:
                self._y0, self._y1 = 0, 0
            else:
                self._y0, self._y1 = 0, self.yMax()
        if self.xMax() > imgW:
            if self.xMin() > imgW:
                self._x0, self._x1 = imgW, imgW
            else:
                self._x0, self._x1 = self.xMin(), imgW
        if self.yMax() > imgH:
            if self.yMin() > imgH:
                self._y0, self._y1 = imgH, imgH
            else:
                self._y0, self._y1 = self.yMin(), imgH


# constants defining source files and application
# variables for the ML models
WORKDIR = os.path.abspath(os.getcwd())

BOX_MODEL_FILE = WORKDIR + "/models/bridgeDetectorModel.pb"
BOX_MODEL_LABEL = WORKDIR + "/models/bridgeDetectorLabels.pbtxt"
BOX_NUMBER = "14"
BOX_NUMBER = "14"
BOX_MIN_SCORE = "0"
CLASS_MODEL_FILE = WORKDIR + "/models/bridgeScoreModel.pb"
CLASS_MODEL_LABEL = WORKDIR + "/models/bridgeScoreLabels.txt"


def main():
    # start the app
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_dir",
        help="input directory of images to be scored (or .txt file listing images)",
    )
    ap.add_argument("-o", "--output_file", help='output file of box locations ("stdout" is an option)')
    ap.add_argument(
        "-a",
        "--annot_file",
        help="a file of annotated images for performance comparison",
    )
    # data augmentation
    ap.add_argument(
        "--aug_flip",
        help="score each image twice, with a horizontal flip",
        action="store_true",
    )
    ap.add_argument(
        "--aug_one",
        help="downgrades scores of 1 by replacing with the ratio of scores <1 vs >1",
        action="store_true",
    )
    # extra output
    ap.add_argument(
        "--details",
        help="""an extra output file with bridge-by-bridge scoring details
             ("stdout" is an option), cols are [ID,n,score,ypos,xpos], where
             positions are fractions of the image height/width.""",
        default = "",
    )
    args = vars(ap.parse_args())

    # set things up
    boxMod = TfObjectDetector(BOX_MODEL_FILE, BOX_MODEL_LABEL)
    classMod = TfClassifier(CLASS_MODEL_FILE, CLASS_MODEL_LABEL)
    minScr = float(BOX_MIN_SCORE)
    numBoxes = int(BOX_NUMBER)
    scorer = DishScorer(boxMod, classMod, numBoxes, minScr)
    if args["aug_flip"]:
        scorer.addSpineFlip()
    if args["aug_one"]:
        scorer.addAdjustOne()

    # score new images
    if args["input_dir"]:
        if os.path.isdir(args["input_dir"]):
            imgLister = ImageDirLister(args["input_dir"])
        elif os.path.isfile(args["input_dir"]):
            imgLister = ImageFileLister(args["input_dir"])
        else:
            raise ValueError("input is nether a directory nor a file")
        imgMang = ImageDirScorer(scorer, imgLister)
        if args["output_file"]:
            outfName = args["output_file"]
        else:
            outfName = "stdout"

        writeDetails = False
        if args["details"]:
            writeDetails = True
            outfDetailsName = args["details"]

        if writeDetails:
            imgMang.scoreImages(outfName,outfileDetails=outfDetailsName)
        else:
            imgMang.scoreImages(outfName)

    # stats on pre-annotated images; ability to do both allows
    # a quick performance check to be added to every run of the
    # script, if you want for QC
    if args["annot_file"]:
        perfMang = PerformanceAnalyzer(args["annot_file"])
        perfMang.scoreImages(scorer)


if __name__ == "__main__":
    main()
