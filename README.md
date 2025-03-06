# Arabic Speech Command Recognition with Noise Cancellation

This project is designed to recognize Arabic speech commands with noise cancellation using a trained SVM model. The system can recognize commands such as "zero", "yes", "left", "ok", "open", "start", "stop", and "down".
 
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


# Arabic Speech Command Recognition with Noise Cancellation

This project is designed to recognize Arabic speech commands with noise cancellation using a trained SVM model. The system can recognize commands such as "zero", "yes", "left", "ok", "open", "start", "stop", and "down".

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone the repository:
   
   git clone https://github.com/Bilalboubeka/Arabic_Speech_Command.git
   
  

## Usage

To start recognizing speech commands,
- select your microphone device: 1 
- press "y" to Record another command
- press "n" to exit


python your_script_name.py

## Features

- **Noise Cancellation**: Basic noise cancellation using spectral gating.
- **Feature Extraction**: Extracts MFCC features from audio recordings.
- **Command Recognition**: Recognizes specific Arabic speech commands with a trained SVM model.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Acknowledgments

Special thanks to [Librosa](https://librosa.org/) and [Sounddevice](https://python-sounddevice.readthedocs.io/) for their excellent libraries.
