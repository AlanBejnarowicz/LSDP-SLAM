import frame_grabber as fg
import coord_grabber as cg
import feature_extractor as fe


def main() -> None:
    """
    Main function to run the frame grabber and feature extractor.
    """

    video_source = 'DJI_0199.MOV'
    img_dir = 'input'
    csv_log_dir = 'DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv'
    calibration_file = 'phantom4pro-calibration.xml'

    # Create an instance of FrameGrabber
    # frame_grabber = fg.FrameGrabber(video_source, img_dir)

    # Grab frames from the video source
    # frame_grabber.grab_frames(divider=25, start=1200, end=-1)

    # Create an instance of CoordGrabber
    coord_grabber = cg.CoordGrabber(csv_log_dir)

    # Read GPS log
    gps_data = coord_grabber.read_gps_log()

    # Convert GPS to UTM
    utm_data = coord_grabber.convert_gps_to_utm()

    # Plot UTM data
    coord_grabber.plot_utm_data()

    # Create an instance of FeatureExtractor
    feature_extractor = fe.FeatureExtractor(img_dir, calibration_file)

    # Extract features using SIFT
    feature_extractor.SIFT_extract_features()


    # check the translation vector
    point1 = utm_data[1200][-3:]
    point2 = utm_data[1225][-3:]

    import numpy as np
    point1_array = np.array(point1) 
    point2_array = np.array(point2)
    print(f"Point 1 Array: {point1_array}")
    print(f"Point 2 Array: {point2_array}")
    print(point1_array + feature_extractor.translation_vector.squeeze())



# Entry point of the script
if __name__ == "__main__":
    main()
