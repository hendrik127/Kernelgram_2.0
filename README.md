# Kernelgram 2.0

Kernelgram 2.0 is an image processing application that provides various filters and features for image manipulation. It is built using OpenCV and Python. It also uses a pretrained YOLOv3 model for image detection (COCO).

## Project Structure

The project has the following structure:

- `app.py`: The main application file.
- `filters.py`: Contains the initial image filter functions.
- `filters2.py`: Contains the updated image filter functions.
- `haarcascade_frontalface_default.xml`: Haar cascade file for face detection.
- `models`: Directory containing pre-trained models.
  - `yolov3.cfg`: YOLOv3 configuration file.
  - `yolov3.txt`: Text file containing class names for YOLOv3.
  - `yolov3.weights`: Pre-trained weights for YOLOv3.
- `requirements.txt`: List of Python dependencies for the project.

## Usage

To use Kernelgram 2.0, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the `app.py` script to launch the application.
3. Use the provided filters and features to manipulate images.
4. Close the application when finished.


## Notes

- Make sure the pre-trained model files (`yolov3.cfg`, `yolov3.txt`, and `yolov3.weights`) are present in the `models` directory before running the application.

Enjoy using Kernelgram 2.0!

