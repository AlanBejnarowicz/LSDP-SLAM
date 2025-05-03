import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


class FeatureExtractor:
    """
    Work in progress...
    """

    def __init__(self, img_dir: str) -> None:
        """
        Initializes the feature extractor with the image directory.

        :param img_dir: Directory containing images to extract features from.
        """

        if not os.path.exists(img_dir):
            raise ValueError('Image directory does not exist', img_dir)

        self.img_dir = img_dir

        self.img_name = os.listdir(self.img_dir)
        if len(self.img_name) < 2:
            raise ValueError('Not enough images in the directory', img_dir)
        self.img_name.sort()

        self.sift = cv.SIFT_create()
        self.MIN_MATCH_COUNT = 10


    def SIFT_extract_features(self) -> None:
        """
        Work in progress...
        """

        img1 = cv.imread(os.path.join(self.img_dir, self.img_name[0]), cv.IMREAD_GRAYSCALE) # queryImage
        img2 = cv.imread(os.path.join(self.img_dir, self.img_name[1]), cv.IMREAD_GRAYSCALE) # trainImage

        # find the keypoints and descriptors with SIF T
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
        
            h,w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts,M)

            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        else:
            print('Not enough matches are found - {}/{}'.format(len(good), self.MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask, # draw only inliers
                           flags=2)

        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray')
        plt.show()


def main() -> None:
    """
    Main function to run the feature extractor.
    """

    # Directory containing images
    img_dir = 'input'

    # Create an instance of FeatureExtractor
    feature_extractor = FeatureExtractor(img_dir)

    # Extract features using SIFT
    feature_extractor.SIFT_extract_features()


# Entry point of the script
if __name__ == "__main__":
    main()
