# Kernelgram 2.0

Kernelgram 2.0 is an image processing application that provides various filters and features for image manipulation. It is built using OpenCV and Python. It also uses a pretrained YOLOv3 model for image detection (COCO).

## Project Structure

The project has the following structure:

- `app.py`: The main application file.
- `filters2.py`: Contains the updated image filter functions.
- `haarcascade_frontalface_default.xml`: Haar cascade file for face detection.
- `models`: Directory containing pre-trained models.
  - `yolov3.cfg`: YOLOv3 configuration file.
  - `yolov3.txt`: Text file containing class names for YOLOv3.
  - `yolov3.weights`: Pre-trained weights for YOLOv3.
- `requirements.txt`: List of Python dependencies for the project.

## Usage

To use Kernelgram 2.0, follow these steps:

1. Install the required dependencies by running `pip3 install -r requirements.txt`.
2. Download the zipped models folder from [here](https://drive.google.com/file/d/1wTQ_H_fOFhB68GltHcGJL5b1lfwOYmDV/view?usp=share_link).
3. Add it to the project directory.
4. Run the `app.py` script to launch the application.
5. Have fun!


## Notes
- This app was updated for python 3.11.
- Pictures get saved in your homedir/Pictures/ folder
  
## Troubleshooting

Some info that may help in troubleshooting.
# Tesseract OCR

Windows:
Visit the Tesseract OCR downloads page at https://github.com/UB-Mannheim/tesseract/wiki.
Download the latest stable version of Tesseract OCR for Windows.
Run the installer and follow the installation instructions.
During the installation process, make sure to select the option to add Tesseract to your system's PATH environment variable.

macOS:
Open Terminal.
Install Tesseract using Homebrew by running the command: brew install tesseract.

Linux (Ubuntu/Debian):
Open Terminal.
Run the following command to install Tesseract: sudo apt-get install tesseract-ocr.

# Qt
Windows:
Visit the Qt website (https://www.qt.io/download) and download the Qt 6 installer for Windows.
Run the downloaded installer and follow the installation wizard.
During the installation, select the components you want to install. Make sure to include the Qt libraries and development tools.
Choose the installation path and complete the installation process.

macOS:
brew install qt@6

Linux:
sudo apt-get install qt6-default

Enjoy using Kernelgram 2.0!

