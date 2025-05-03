import cv2 as cv
import os
import shutil


class FrameGrabber:
    """
    A class to grab frames from a video source and save them to a specified directory.
    """

    def __init__(self, video_source=0, output_dir='input') -> None:
        """
        Initializes the framegrabber with the video source and output directory.

        :param video_source: Video source (default is 0 for webcam).
        :param output_dir: Directory to save the frames (default is 'input').
        """

        self.video_source = video_source
        self.vid = cv.VideoCapture(self.video_source)

        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', video_source)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)


    def __del__(self) -> None:
        """
        Releases the video capture object when the framegrabber is deleted.
        """

        if self.vid.isOpened():
            self.vid.release()


    def grab_frames(self, divider=1, start=0, end=-1) -> None:
        """
        Grabs frames from the video source and saves them to the output directory.

        :param divider: Number of frames to skip (default is 1, meaning no frames are skipped).
        :param start: Starting index of frames to save (default is 0).
        :param end: Ending index of frames to save (default is -1, meaning all frames).
        """

        count = 0
        while True:
            ret, frame = self.vid.read()
            if not ret:
                break

            # cv.imshow('Frame', frame) # Uncomment the following line to display the video frames

            if count % divider == 0 and count >= start and (end == -1 or count <= end):
                cv.imwrite(os.path.join('input', f'frame_{count}.jpg'), frame)

            count += 1

            if cv.waitKey(-1) & 0xFF == ord('q'):
                break

        self.vid.release()
        cv.destroyAllWindows()


def main() -> None:
    """
    Main function to run the FrameGrabber.
    """

    video_source = 'DJI_0199.MOV' # Change this to your video source
    output_dir = 'input' # Change this to your desired output directory

    fg = FrameGrabber(video_source, output_dir)
    fg.grab_frames(divider=25, start=1200, end=-1)


# Entry point of the script
if __name__ == "__main__":
    main()
