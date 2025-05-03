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

    # Create an instance of FrameGrabber
    frame_grabber = fg.FrameGrabber(video_source, img_dir)

    # Grab frames from the video source
    frame_grabber.grab_frames(divider=25, start=1200, end=-1)

    # Create an instance of CoordGrabber
    coord_grabber = cg.CoordGrabber(csv_log_dir)

    # Read GPS log
    gps_data = coord_grabber.read_gps_log()

    # Convert GPS to UTM
    utm_data = coord_grabber.convert_gps_to_utm()

    # Plot UTM data
    coord_grabber.plot_utm_data()

    # Create an instance of FeatureExtractor
    feature_extractor = fe.FeatureExtractor(img_dir)

    # Extract features using SIFT
    feature_extractor.SIFT_extract_features()


# Entry point of the script
if __name__ == "__main__":
    main()
