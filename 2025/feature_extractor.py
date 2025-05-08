import numpy as np
import cv2 as cv
from show_image import show_image
import os
import xml.etree.ElementTree as ET


class FeatureExtractor:
    """
    Work in progress...
    """

    def __init__(self, img_dir: str, calibration_file: str) -> None:
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

        # Read camera settings from XML file
        if not os.path.exists(calibration_file):
            raise ValueError('Calibration file does not exist', calibration_file)

        # Parse the XML file using ElementTree
        tree = ET.parse(calibration_file)
        root = tree.getroot()

        # Extract parameters from the XML structure
        width = int(root.find('width').text)
        height = int(root.find('height').text)
        f = float(root.find('f').text)
        cx = float(root.find('cx').text)
        cy = float(root.find('cy').text)
        k1 = float(root.find('k1').text)
        k2 = float(root.find('k2').text)
        k3 = float(root.find('k3').text)
        p1 = float(root.find('p1').text)
        p2 = float(root.find('p2').text)

        # Camera matrix
        self.camera_matrix = np.array([
            [f, 0, width/2 + cx],
            [0, f, height/2 + cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        # Print the camera matrix and distortion coefficients
        print(f'Camera Matrix:\n{self.camera_matrix}\n')
        print(f'Distortion Coefficients:\n{self.dist_coeffs}\n')

        self.essential_matrix = None
        self.rotation_matrix = None
        self.translation_vector = None


    def SIFT_extract_features(self) -> None:
        """
        Work in progress...
        """

        img1 = cv.imread(os.path.join(self.img_dir, self.img_name[0]), cv.IMREAD_GRAYSCALE) # queryImage
        img2 = cv.imread(os.path.join(self.img_dir, self.img_name[1]), cv.IMREAD_GRAYSCALE) # trainImage

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Store all the good matches as per Lowe's ratio test.
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
            dst = cv.perspectiveTransform(pts, M)

            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
            print('Not enough matches are found - {}/{}'.format(len(good), self.MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0), # Draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask, # Draw only inliers
                           flags=2)

        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        show_image('Good Matches', img3)

        # Estimate the esential matrix
        self.essential_matrix, mask = cv.findEssentialMat(src_pts, dst_pts, self.camera_matrix,
                                                     method=cv.RANSAC, prob=0.999, threshold=1.0)
        print(f'Essential Matrix:\n{self.essential_matrix}\n')

        # Recover pose (rotation matrix and translation vector) from the essential matrix
        points, R, t, mask = cv.recoverPose(self.essential_matrix, src_pts, dst_pts, self.camera_matrix)

        # Store the rotation matrix and translation vector
        self.rotation_matrix = R
        self.translation_vector = t

        # Print the results
        print(f'Number of inlier points: {points}')
        print(f'Rotation Matrix:\n{self.rotation_matrix}\n')
        print(f'Translation Vector:\n{self.translation_vector}\n')

        # The scale of translation is ambiguous in monocular vision
        print('Note: The translation vector direction is accurate, but the scale is ambiguous.')


def main() -> None:
    """
    Main function to run the feature extractor.
    """

    # Directory containing images
    img_dir = 'input'
    calibration_file = 'phantom4pro-calibration.xml'

    # Create an instance of FeatureExtractor
    feature_extractor = FeatureExtractor(img_dir, calibration_file)

    # Extract features using SIFT
    feature_extractor.SIFT_extract_features()


# Entry point of the script
if __name__ == "__main__":
    main()
