# DiSi

This project is a Python-based application for audio analysis using Fast Fourier Transform (FFT). It visualizes audio data through three interactive plots: an FFT plot, a maximum power plot, and a maximum frequency plot. The application supports both normal and advanced playback modes, with features like play/pause, time range selection, interactive plot annotations, and a menu bar for file selection and mode toggling.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)

## Overview
The Real-Time FFT Audio Analyzer is a graphical user interface (GUI) application built with PyQt5 and PyQtGraph. It loads a WAV audio file (defaulting to `test.wav`) and provides real-time visualization of its frequency content. The application is designed for audio analysis, allowing users to observe the frequency spectrum, the power of the dominant frequencies over time, and the dominant frequencies themselves. It includes a menu bar for file selection and mode toggling, interactive features like hover labels, clickable annotations, and keyboard shortcuts for enhanced usability.

## Features
- **Menu Bar**:
  - **File Menu**:
    - "Open WAV File" (`Ctrl+O`): Select a WAV file to load (defaults to `test.wav` if no file is selected).
  - **Tools Menu**:
    - "Advanced Mode" (`A`): Toggle between normal and advanced modes (checkable menu item).
- **Three Interactive Plots**:
  - **FFT Plot**: Displays the real-time frequency spectrum (0–22.5 kHz) with magnitude in dB (-100 to 0 dB). Hover to see frequency and magnitude at the cursor.
  - **Max Power Plot**: Shows the power (in dB) of the dominant frequency over time. Hover to see time and power.
  - **Max Freq Plot**: Displays the dominant frequency (0–22.5 kHz) over time. Hover to see time and frequency.
- **Playback Modes**:
  - **Normal Mode**: Play the entire audio file with a slider to seek through the audio.
  - **Advanced Mode**: Play a specific time range by setting start and end times (in seconds).
- **Play/Pause Control**:
  - Toggle playback with the Play/Pause button or the spacebar.
  - Button text updates to "Play" or "Pause" based on the playback state.
- **Interactive Annotations**:
  - Left-click on the Max Freq plot to add a red vertical line on the FFT plot (at the selected frequency) and on the Max Power plot (at the selected time).
  - Press the "Delete" key to remove all lines.
- **Keyboard Shortcuts**:
  - **Spacebar**: Toggle play/pause (works in both normal and advanced modes).
  - **A Key**: Toggle between normal and advanced modes.
  - **Delete Key**: Remove all annotation lines from the FFT and Max Power plots.
  - **Ctrl+O**: Open a WAV file via the File menu.
- **Hover Information**:
  - Each plot has a hover label showing relevant data (frequency/magnitude, time/power, time/frequency) at the cursor position.
- **Time Display**:
  - A time label shows the current playback position in the format `MM:SS.CC` (minutes, seconds, centiseconds).

## Installation
1. **Clone the Repository**
2. **Install `requirements.txt`**
```bash
pip install -r requirements.txt
```

