from Datatypes import *
from Frame import Frame
import numpy as np
import cv2

class ImagePair():
    """
    Class for working with image pairs.
    """
    def __init__(self, frame1: Frame, frame2: Frame, matcher, camera_matrix):
        self.frame1 = frame1
        self.frame2 = frame2
        self.matcher = matcher
        self.camera_matrix = camera_matrix


    def match_features(self):
        temp = self.matcher.match(
                self.frame1.descriptors, 
                self.frame2.descriptors)
        # Make a list with the following values
        # - feature 1 id
        # - feature 2 id
        # - image coordinate 1
        # - image coordinate 2
        # - match distance
        self.raw_matches: list[Match] = [
                Match(self.frame1.features[match.queryIdx].feature_id, 
                    self.frame2.features[match.trainIdx].feature_id,
                    self.frame1.features[match.queryIdx].keypoint.pt, 
                    self.frame2.features[match.trainIdx].keypoint.pt, 
                    self.frame1.features[match.queryIdx].descriptor, 
                    self.frame2.features[match.trainIdx].descriptor,
                    match.distance, np.random.random((3))) 
                for idx, match
                in enumerate(temp)]


        # Calculate distances between features and epipolar lines

        # Compute the fundamental matrix from the camera matrix
        F, _ = cv2.findFundamentalMat(
            np.array([match.keypoint1 for match in self.raw_matches]),
            np.array([match.keypoint2 for match in self.raw_matches]),
            cv2.FM_8POINT
        )

        # Calculate distances from points to epipolar lines
        epipolar_distances = []
        for match in self.raw_matches:
            # Convert points to homogeneous coordinates
            point1 = np.array([[match.keypoint1[0]], [match.keypoint1[1]], [1.0]])
            point2 = np.array([[match.keypoint2[0]], [match.keypoint2[1]], [1.0]])
            
            # Calculate epipolar line in the second image for point1
            line2 = F @ point1
            # Calculate distance from point2 to the epipolar line
            numerator = abs(line2.T @ point2)
            denominator = np.sqrt(line2[0]**2 + line2[1]**2)
            distance = numerator / denominator
            epipolar_distances.append(float(distance))

        # Calculate summary statistics
        epipolar_distances = np.array(epipolar_distances)
        epi_mean = np.mean(epipolar_distances)
        epi_std = np.std(epipolar_distances)
        epi_min = np.min(epipolar_distances)
        epi_max = np.max(epipolar_distances)
        epi_median = np.median(epipolar_distances)

        print(f"Epipolar constraint statistics:")
        print(f"\tMean distance: {epi_mean:.4f} pixels")
        print(f"\tStandard deviation: {epi_std:.4f} pixels")
        print(f"\tMinimal distance: {epi_min:.4f} pixels")
        print(f"\tMaximal distance: {epi_max:.4f} pixels")
        print(f"\tMedian distance: {epi_median:.4f} pixels")


        # Perform a very crude filtering of the matches
        self.filtered_matches: list[Match] = [match
                for match
                in self.raw_matches
                if match.distance < 1130]


    def visualize_matches(self, matches):
        h, w, _ = self.frame1.image.shape
        # Place the images next to each other.
        vis = np.concatenate((self.frame1.image, self.frame2.image), axis=1)

        # Draw the matches
        for match in matches:
            start_coord = (int(match.keypoint1[0]), int(match.keypoint1[1]))
            end_coord = (int(match.keypoint2[0] + w), int(match.keypoint2[1]))
            thickness = 1
            color = list(match.color * 256)
            vis = cv2.line(vis, start_coord, end_coord, color, thickness)

        return vis


    def determine_essential_matrix(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        confidence = 0.99
        ransacReprojecThreshold = 1
        self.essential_matrix, mask = cv2.findEssentialMat(
                points_in_frame_1,
                points_in_frame_2, 
                self.camera_matrix, 
                cv2.FM_RANSAC, 
                confidence,
                ransacReprojecThreshold)

        inlier_matches = [match 
                for match, inlier in zip(matches, mask.ravel() == 1)
                if inlier]

        return inlier_matches


    def get_image_points(self, matches):
        points_in_frame_1 = np.array(
                [match.keypoint1 for match in matches], dtype=np.float64)
        points_in_frame_2 = np.array(
                [match.keypoint2 for match in matches], dtype=np.float64)
        
        # Ensure points are in the correct shape for OpenCV functions
        if points_in_frame_1.shape[1] != 2:
            points_in_frame_1 = points_in_frame_1.reshape(-1, 2)
        if points_in_frame_2.shape[1] != 2:
            points_in_frame_2 = points_in_frame_2.reshape(-1, 2)

        # # Display images with keypoints
        # img1_with_keypoints = cv2.drawKeypoints(self.frame1.image.copy(), 
        #                        [cv2.KeyPoint(x, y, 7) for x, y in points_in_frame_1], 
        #                          None, (0, 255, 0), 4)
        # img2_with_keypoints = cv2.drawKeypoints(self.frame2.image.copy(),
        #                          [cv2.KeyPoint(x, y, 7) for x, y in points_in_frame_2], 
        #                             None, (0, 0, 255), 4)
        # cv2.imshow("Image 1 Keypoints", img1_with_keypoints)
        # cv2.imshow("Image 2 Keypoints", img2_with_keypoints)
        # cv2.waitKey(0)

        return points_in_frame_1, points_in_frame_2


    def estimate_camera_movement(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        retval, self.R, self.t, mask = cv2.recoverPose(
                self.essential_matrix, 
                points_in_frame_1, 
                points_in_frame_2, 
                self.camera_matrix)
        self.relative_pose = np.eye(4)
        self.relative_pose[:3, :3] = self.R
        self.relative_pose[:3, 3] = self.t.T[0]

        print("relative movement in image pair")
        print(self.relative_pose)


    def reconstruct_3d_points(self, matches, 
            first_projection_matrix = None, 
            second_projection_matrix = None):
        identify_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        estimated_transform = np.hstack((self.R.T, -self.R.T @ self.t))

        self.null_projection_matrix = self.camera_matrix @ identify_transform
        self.projection_matrix = self.camera_matrix @ estimated_transform

        if first_projection_matrix is not None:
            self.null_projection_matrix = self.camera_matrix @ first_projection_matrix
        if second_projection_matrix is not None:
            self.projection_matrix = self.camera_matrix @ second_projection_matrix

        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        self.points3d_reconstr = cv2.triangulatePoints(
                self.projection_matrix, 
                self.null_projection_matrix,
                points_in_frame_1.T, 
                points_in_frame_2.T) 

        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

        self.matches_with_3d_information = [
                Match3D(match.featureid1, match.featureid2, 
                    match.keypoint1, match.keypoint2, 
                    match.descriptor1, match.descriptor2, 
                    match.distance, match.color,
                    (self.points3d_reconstr[0, idx],
                        self.points3d_reconstr[1, idx],
                        self.points3d_reconstr[2, idx]))
                for idx, match 
                in enumerate(matches)]
        
        #print("Reconstructed points")
        #print(self.points3d_reconstr.transpose().shape)
        #print(self.points3d_reconstr.transpose())