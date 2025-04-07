import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QAction, QMenuBar
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from queue import Queue, Empty

class RealTimeFFT(QMainWindow):
    def __init__(self, audio_file='test.wav'):
        super().__init__()
        self.setWindowTitle("DiSi")
        self.resize(800, 900)  # Increased height for third graph

        # Update interval in milliseconds (tweakable)
        self.update_interval_ms = 30

        # Create menu bar
        self.menu_bar = self.menuBar()
        self.setup_menus()

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # FFT plot and its hover label
        self.fft_container = QVBoxLayout()
        self.fft_plot_widget = pg.PlotWidget()
        self.fft_plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot_widget.setLabel('left', 'Magnitude (dB)')
        self.fft_plot_widget.showGrid(x=True, y=True)
        self.fft_plot_widget.setYRange(-100, 0)
        self.fft_plot_widget.setXRange(0, 22500)
        self.fft_plot = self.fft_plot_widget.plot([], [], pen='y')
        self.fft_hover_label = QLabel("Freq: N/A, Mag: N/A dB")
        self.fft_container.addWidget(self.fft_plot_widget)
        self.fft_container.addWidget(self.fft_hover_label)
        self.layout.addLayout(self.fft_container)

        # Max power plot (third graph) and its hover label
        self.max_power_container = QVBoxLayout()
        self.max_power_plot_widget = pg.PlotWidget()
        self.max_power_plot_widget.setLabel('bottom', 'Time (s)')
        self.max_power_plot_widget.setLabel('left', 'Power (dB)')
        self.max_power_plot_widget.showGrid(x=True, y=True)
        self.max_power_plot_widget.setYRange(-100, 0)
        self.max_power_plot = self.max_power_plot_widget.plot([], [], pen='b')
        self.max_power_hover_label = QLabel("Time: N/A, Power: N/A dB")
        self.max_power_container.addWidget(self.max_power_plot_widget)
        self.max_power_container.addWidget(self.max_power_hover_label)
        self.layout.addLayout(self.max_power_container)

        # Max frequency plot and its hover label
        self.max_freq_container = QVBoxLayout()
        self.max_freq_plot_widget = pg.PlotWidget()
        self.max_freq_plot_widget.setLabel('bottom', 'Time (s)')
        self.max_freq_plot_widget.setLabel('left', 'Max Freq (Hz)')
        self.max_freq_plot_widget.showGrid(x=True, y=True)
        self.max_freq_plot_widget.setYRange(0, 22500)
        self.max_freq_plot = self.max_freq_plot_widget.plot([], [], pen='g')
        self.max_freq_hover_label = QLabel("Time: N/A, Freq: N/A Hz")
        self.max_freq_container.addWidget(self.max_freq_plot_widget)
        self.max_freq_container.addWidget(self.max_freq_hover_label)
        self.layout.addLayout(self.max_freq_container)

        # Control layout (will toggle between normal and advanced mode)
        self.controls_layout = QHBoxLayout()

        # Normal mode controls (slider)
        self.normal_controls_widget = QWidget()
        self.normal_controls = QHBoxLayout(self.normal_controls_widget)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.time_label = QLabel("0:00.00")
        self.normal_controls.addWidget(self.play_pause_button)
        self.normal_controls.addWidget(self.slider)
        self.normal_controls.addWidget(self.time_label)

        # Advanced mode controls (time range inputs)
        self.advanced_controls_widget = QWidget()
        self.advanced_controls = QHBoxLayout(self.advanced_controls_widget)
        self.advanced_play_pause_button = QPushButton("Play")
        self.advanced_play_pause_button.clicked.connect(self.toggle_play_pause_advanced)
        self.start_time_input = QLineEdit("0.00")
        self.start_time_input.setPlaceholderText("Start Time (s)")
        self.start_time_input.setFixedWidth(100)
        self.end_time_input = QLineEdit("0.00")
        self.end_time_input.setPlaceholderText("End Time (s)")
        self.end_time_input.setFixedWidth(100)
        self.set_range_button = QPushButton("Set Range")
        self.set_range_button.clicked.connect(self.set_playback_range)
        self.advanced_time_label = QLabel("0:00.00")
        self.advanced_controls.addWidget(self.advanced_play_pause_button)
        self.advanced_controls.addWidget(QLabel("Start:"))
        self.advanced_controls.addWidget(self.start_time_input)
        self.advanced_controls.addWidget(QLabel("End:"))
        self.advanced_controls.addWidget(self.end_time_input)
        self.advanced_controls.addWidget(self.set_range_button)
        self.advanced_controls.addWidget(self.advanced_time_label)

        # Initially show normal controls
        self.controls_layout.addWidget(self.normal_controls_widget)
        self.layout.addLayout(self.controls_layout)
        self.advanced_mode = False

        # Connect slider for normal mode
        self.slider.valueChanged.connect(self.update_time_label_and_fft)
        self.slider.sliderReleased.connect(self.seek_audio)

        # Initialize variables
        self.audio_file = audio_file
        self.queue = Queue()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Initialize playback-related variables
        self.start_idx = 0
        self.is_playing = False
        self.playback_start_idx = 0
        self.playback_end_idx = 0  # Will be set after loading audio

        # Lists to store max frequency and power data over time
        self.max_freqs = []
        self.max_powers = []  # List for power values
        self.timestamps = []

        # Store current FFT data for hover functionality
        self.current_freqs = []
        self.current_magnitude_db = []

        # Lists to store vertical lines, their labels, and highlighted X-axis ticks on all plots
        self.fft_lines = []  # (line, label, tick)
        self.max_power_lines = []  # (line, label, tick)
        self.max_freq_lines = []  # (line, label, tick)

        # Store original X-axis tick styles to restore them
        self.fft_original_ticks = None
        self.max_power_original_ticks = None
        self.max_freq_original_ticks = None

        # Load audio file after initializing all variables
        self.load_audio_file(self.audio_file)

        # Store original X-axis tick styles after loading the file
        self.fft_original_ticks = (self.fft_plot_widget.getAxis('bottom').style.get('tickFont'), 
                                   self.fft_plot_widget.getAxis('bottom').tickPen())
        self.max_power_original_ticks = (self.max_power_plot_widget.getAxis('bottom').style.get('tickFont'), 
                                         self.max_power_plot_widget.getAxis('bottom').tickPen())
        self.max_freq_original_ticks = (self.max_freq_plot_widget.getAxis('bottom').style.get('tickFont'), 
                                        self.max_freq_plot_widget.getAxis('bottom').tickPen())

        # Enable hover and click functionality
        self.fft_plot_widget.scene().sigMouseMoved.connect(self.on_fft_mouse_moved)
        self.max_power_plot_widget.scene().sigMouseMoved.connect(self.on_max_power_mouse_moved)
        self.max_power_plot_widget.scene().sigMouseClicked.connect(self.on_max_power_mouse_clicked)
        self.max_freq_plot_widget.scene().sigMouseMoved.connect(self.on_max_freq_mouse_moved)
        self.max_freq_plot_widget.scene().sigMouseClicked.connect(self.on_max_freq_mouse_clicked)

        # Enable key press detection
        self.setFocusPolicy(Qt.StrongFocus)

    def setup_menus(self):
        """Set up the menu bar with File and Tools menus."""
        file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Open WAV File", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        tools_menu = self.menu_bar.addMenu("Tools")
        self.advanced_action = QAction("Advanced Mode", self, checkable=True)
        self.advanced_action.setShortcut("A")
        self.advanced_action.triggered.connect(self.toggle_advanced_mode)
        tools_menu.addAction(self.advanced_action)

        reset_action = QAction("Reset Views", self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self.reset_views)
        tools_menu.addAction(reset_action)

    def open_file_dialog(self):
        """Open a file dialog to select a WAV file and reload the audio."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if file_path:
            self.audio_file = file_path
            self.load_audio_file(file_path)

    def load_audio_file(self, file_path):
        """Load the audio file and update the plots and controls."""
        if self.is_playing:
            if self.advanced_mode:
                self.toggle_play_pause_advanced()
            else:
                self.toggle_play_pause()

        self.max_freqs = []
        self.max_powers = []
        self.timestamps = []
        for line, label, tick in self.fft_lines:
            self.fft_plot_widget.removeItem(line)
            self.fft_plot_widget.removeItem(label)
        for line, label, tick in self.max_power_lines:
            self.max_power_plot_widget.removeItem(line)
            self.max_power_plot_widget.removeItem(label)
        for line, label, tick in self.max_freq_lines:
            self.max_freq_plot_widget.removeItem(line)
            self.max_freq_plot_widget.removeItem(label)
        self.fft_lines.clear()
        self.max_power_lines.clear()
        self.max_freq_lines.clear()
        self.fft_plot.clear()
        self.max_power_plot.clear()
        self.max_freq_plot.clear()

        if self.fft_original_ticks:
            font, pen = self.fft_original_ticks
            self.fft_plot_widget.getAxis('bottom').setTickFont(font)
            self.fft_plot_widget.getAxis('bottom').setTickPen(pen)
        if self.max_power_original_ticks:
            font, pen = self.max_power_original_ticks
            self.max_power_plot_widget.getAxis('bottom').setTickFont(font)
            self.max_power_plot_widget.getAxis('bottom').setTickPen(pen)
        if self.max_freq_original_ticks:
            font, pen = self.max_freq_original_ticks
            self.max_freq_plot_widget.getAxis('bottom').setTickFont(font)
            self.max_freq_plot_widget.getAxis('bottom').setTickPen(pen)

        try:
            self.sample_rate, self.data = wavfile.read(file_path)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return

        if len(self.data.shape) == 2:  # If stereo, take one channel
            self.data = self.data[:, 0]
        self.data = self.data / np.max(np.abs(self.data))  # Normalize

        self.total_duration = len(self.data) / self.sample_rate
        self.slider.setRange(0, int(self.total_duration * 100))  # Centiseconds
        self.end_time_input.setText(f"{self.total_duration:.2f}")

        self.max_power_plot_widget.setXRange(0, self.total_duration)
        self.max_freq_plot_widget.setXRange(0, self.total_duration)

        self.start_idx = 0
        self.is_playing = False
        self.playback_start_idx = 0
        self.playback_end_idx = len(self.data)

        self.current_freqs = []
        self.current_magnitude_db = []

        self.compute_max_frequencies()
        self.update_time_label_and_fft(0)

    def compute_max_frequencies(self):
        """Compute max frequencies and their corresponding power for the entire audio file at startup."""
        chunk_size = 1024
        self.max_freqs = []
        self.max_powers = []
        self.timestamps = []
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data[i:i + chunk_size]
            if len(chunk) < 1:
                continue
            nperseg = min(1024, len(chunk))
            noverlap = nperseg // 2 if nperseg > 1 else 0
            freqs, psd = welch(chunk, fs=self.sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
            mask = (freqs >= 0) & (freqs <= 22500)
            positive_freqs = freqs[mask]
            magnitude_db = 10 * np.log10(psd[mask] + 1e-10)
            max_idx = np.argmax(magnitude_db)
            max_freq = positive_freqs[max_idx]
            max_power = magnitude_db[max_idx]
            timestamp = i / self.sample_rate
            self.timestamps.append(timestamp)
            self.max_freqs.append(max_freq)
            self.max_powers.append(max_power)
        self.max_freq_plot.setData(self.timestamps, self.max_freqs)
        self.max_power_plot.setData(self.timestamps, self.max_powers)

    def on_fft_mouse_moved(self, pos):
        """Handle mouse movement over the FFT plot."""
        if self.fft_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.fft_plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if len(self.current_freqs) > 0 and len(self.current_magnitude_db) > 0:
                idx = np.argmin(np.abs(np.array(self.current_freqs) - x))
                closest_freq = self.current_freqs[idx]
                closest_mag = self.current_magnitude_db[idx]
                self.fft_hover_label.setText(f"Freq: {closest_freq:.2f} Hz, Mag: {closest_mag:.2f} dB")
            else:
                self.fft_hover_label.setText("Freq: N/A, Mag: N/A dB")

    def on_max_power_mouse_moved(self, pos):
        """Handle mouse movement over the Max Power plot."""
        if self.max_power_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.max_power_plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if self.timestamps and self.max_powers:
                idx = np.argmin(np.abs(np.array(self.timestamps) - x))
                closest_time = self.timestamps[idx]
                closest_power = self.max_powers[idx]
                self.max_power_hover_label.setText(f"Time: {closest_time:.2f} s, Power: {closest_power:.2f} dB")
            else:
                self.max_power_hover_label.setText("Time: N/A, Power: N/A dB")

    def on_max_freq_mouse_moved(self, pos):
        """Handle mouse movement over the Max Freq plot."""
        if self.max_freq_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.max_freq_plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if self.timestamps and self.max_freqs:
                idx = np.argmin(np.abs(np.array(self.timestamps) - x))
                closest_time = self.timestamps[idx]
                closest_freq = self.max_freqs[idx]
                self.max_freq_hover_label.setText(f"Time: {closest_time:.2f} s, Freq: {closest_freq:.2f} Hz")
            else:
                self.max_freq_hover_label.setText("Time: N/A, Freq: N/A Hz")

    def on_max_power_mouse_clicked(self, event):
        """Handle mouse click on the Max Power plot."""
        if event.button() == Qt.LeftButton and not event.double():
            pos = event.scenePos()
            if self.max_power_plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.max_power_plot_widget.getViewBox().mapSceneToView(pos)
                x = mouse_point.x()
                if self.timestamps and self.max_powers and self.max_freqs:
                    idx = np.argmin(np.abs(np.array(self.timestamps) - x))
                    selected_time = self.timestamps[idx]
                    selected_power = self.max_powers[idx]
                    selected_freq = self.max_freqs[idx]
                    self.add_line_to_plots(selected_time, selected_freq, selected_power)

    def on_max_freq_mouse_clicked(self, event):
        """Handle mouse click on the Max Freq plot."""
        if event.button() == Qt.LeftButton and not event.double():
            pos = event.scenePos()
            if self.max_freq_plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.max_freq_plot_widget.getViewBox().mapSceneToView(pos)
                x = mouse_point.x()
                if self.timestamps and self.max_freqs and self.max_powers:
                    idx = np.argmin(np.abs(np.array(self.timestamps) - x))
                    selected_time = self.timestamps[idx]
                    selected_freq = self.max_freqs[idx]
                    selected_power = self.max_powers[idx]
                    self.add_line_to_plots(selected_time, selected_freq, selected_power)

    def add_line_to_plots(self, selected_time, selected_freq, selected_power):
        """Add a vertical line to all plots with Y-axis value labels, without modifying X-axis ticks."""
        # FFT plot (frequency on X-axis, show magnitude at selected frequency)
        fft_line = pg.InfiniteLine(pos=selected_freq, angle=90, pen='r')
        self.fft_plot_widget.addItem(fft_line)
        fft_idx = np.argmin(np.abs(np.array(self.current_freqs) - selected_freq))
        fft_mag = self.current_magnitude_db[fft_idx] if len(self.current_magnitude_db) > fft_idx else 0
        fft_label = pg.TextItem(f"{fft_mag:.2f} dB", anchor=(0, 1), color='r')
        fft_label.setPos(selected_freq, fft_mag)
        self.fft_plot_widget.addItem(fft_label)
        # Store without tick value since we’re not highlighting
        self.fft_lines.append((fft_line, fft_label, None))

        # Max Power plot (time on X-axis, show power)
        power_line = pg.InfiniteLine(pos=selected_time, angle=90, pen='r')
        self.max_power_plot_widget.addItem(power_line)
        power_label = pg.TextItem(f"{selected_power:.2f} dB", anchor=(0, 1), color='r')
        power_label.setPos(selected_time, selected_power)
        self.max_power_plot_widget.addItem(power_label)
        self.max_power_lines.append((power_line, power_label, None))

        # Max Freq plot (time on X-axis, show frequency)
        freq_line = pg.InfiniteLine(pos=selected_time, angle=90, pen='r')
        self.max_freq_plot_widget.addItem(freq_line)
        freq_label = pg.TextItem(f"{selected_freq:.2f} Hz", anchor=(0, 1), color='r')
        freq_label.setPos(selected_time, selected_freq)
        self.max_freq_plot_widget.addItem(freq_label)
        self.max_freq_lines.append((freq_line, freq_label, None))

    def reset_views(self):
        """Reset the views of all plots to their initial ranges."""
        self.fft_plot_widget.setXRange(0, 22500)
        self.fft_plot_widget.setYRange(-100, 0)
        self.max_power_plot_widget.setXRange(0, self.total_duration)
        self.max_power_plot_widget.setYRange(-100, 0)
        self.max_freq_plot_widget.setXRange(0, self.total_duration)
        self.max_freq_plot_widget.setYRange(0, 22500)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_A:
            self.toggle_advanced_mode()
        elif event.key() == Qt.Key_Space:
            if self.advanced_mode:
                self.toggle_play_pause_advanced()
            else:
                self.toggle_play_pause()
        elif event.key() == Qt.Key_R:
            self.reset_views()
        elif event.key() == Qt.Key_Delete:
            for line, label, _ in self.fft_lines:  # Ignore tick value
                self.fft_plot_widget.removeItem(line)
                self.fft_plot_widget.removeItem(label)
            font, pen = self.fft_original_ticks
            self.fft_plot_widget.getAxis('bottom').setTickFont(font)
            self.fft_plot_widget.getAxis('bottom').setTickPen(pen)
            self.fft_lines.clear()

            for line, label, _ in self.max_power_lines:
                self.max_power_plot_widget.removeItem(line)
                self.max_power_plot_widget.removeItem(label)
            font, pen = self.max_power_original_ticks
            self.max_power_plot_widget.getAxis('bottom').setTickFont(font)
            self.max_power_plot_widget.getAxis('bottom').setTickPen(pen)
            self.max_power_lines.clear()

            for line, label, _ in self.max_freq_lines:
                self.max_freq_plot_widget.removeItem(line)
                self.max_freq_plot_widget.removeItem(label)
            font, pen = self.max_freq_original_ticks
            self.max_freq_plot_widget.getAxis('bottom').setTickFont(font)
            self.max_freq_plot_widget.getAxis('bottom').setTickPen(pen)
            self.max_freq_lines.clear()

    def toggle_advanced_mode(self):
        """Toggle between normal and advanced mode."""
        self.advanced_mode = not self.advanced_mode
        self.advanced_action.setChecked(self.advanced_mode)
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                self.controls_layout.removeWidget(widget)
                widget.setParent(None)
        if self.advanced_mode:
            self.controls_layout.addWidget(self.advanced_controls_widget)
            print("Switched to Advanced Mode")
        else:
            self.playback_start_idx = 0
            self.playback_end_idx = len(self.data)
            self.start_idx = 0
            self.update_time_label_and_fft(0)
            self.controls_layout.addWidget(self.normal_controls_widget)
            print("Switched to Normal Mode")

    def set_playback_range(self):
        """Set the playback range based on user input in Advanced Mode."""
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())
            if start_time < 0 or end_time > self.total_duration or start_time >= end_time:
                print("Invalid time range. Start must be >= 0, End must be <= duration, and Start < End.")
                return
            self.playback_start_idx = int(start_time * self.sample_rate)
            self.playback_end_idx = int(end_time * self.sample_rate)
            self.start_idx = self.playback_start_idx
            self.update_time_label_and_fft(int(start_time * 100))
            print(f"Playback range set: {start_time:.2f}s to {end_time:.2f}s")
        except ValueError:
            print("Invalid input. Please enter valid numbers for start and end times.")

    def compute_fft_at_position(self, start_idx):
        """Compute FFT for the chunk at the given start index."""
        chunk_size = 1024
        chunk = self.data[start_idx:start_idx + chunk_size]
        if len(chunk) < 1:
            return [], []
        nperseg = min(1024, len(chunk))
        noverlap = nperseg // 2 if nperseg > 1 else 0
        freqs, psd = welch(chunk, fs=self.sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
        mask = (freqs >= 0) & (freqs <= 22500)
        positive_freqs = freqs[mask]
        magnitude_db = 10 * np.log10(psd[mask] + 1e-10)
        return positive_freqs, magnitude_db

    def start_audio_stream(self, start_idx=None, end_idx=None):
        """Start audio stream, optionally within a specified range."""
        if start_idx is not None:
            self.start_idx = start_idx
        if end_idx is None:
            end_idx = len(self.data)

        def audio_callback(outdata, frames, time, status):
            if status:
                print(status)
            if self.start_idx >= end_idx:
                outdata[:] = 0
                self.start_idx = end_idx
                self.is_playing = False
                raise sd.CallbackStop()
            chunk = self.data[self.start_idx:self.start_idx + frames]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk[:, None]
                outdata[len(chunk):] = 0
                self.start_idx += len(chunk)
                self.is_playing = False
                raise sd.CallbackStop()
            else:
                outdata[:] = chunk[:, None]
            self.start_idx += frames

            nperseg = min(1024, len(chunk))
            noverlap = nperseg // 2 if nperseg > 1 else 0
            freqs, psd = welch(chunk, fs=self.sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
            mask = (freqs >= 0) & (freqs <= 22500)
            positive_freqs = freqs[mask]
            magnitude_db = 10 * np.log10(psd[mask] + 1e-10)
            self.queue.put((positive_freqs, magnitude_db))

        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, callback=audio_callback)
        self.stream.start()

    def toggle_play_pause(self):
        """Toggle between play and pause states in Normal Mode."""
        if not self.is_playing:
            self.is_playing = True
            self.play_pause_button.setText("Pause")
            self.start_audio_stream()
            self.timer.start(self.update_interval_ms)
        else:
            self.is_playing = False
            self.play_pause_button.setText("Play")
            self.timer.stop()
            self.stream.stop()
            self.stream.close()

    def toggle_play_pause_advanced(self):
        """Toggle between play and pause states in Advanced Mode."""
        if not self.is_playing:
            self.is_playing = True
            self.advanced_play_pause_button.setText("Pause")
            self.start_audio_stream(self.playback_start_idx, self.playback_end_idx)
            self.timer.start(self.update_interval_ms)
        else:
            self.is_playing = False
            self.advanced_play_pause_button.setText("Play")
            self.timer.stop()
            self.stream.stop()
            self.stream.close()

    def seek_audio(self):
        """Seek to a position in Normal Mode using the slider."""
        value = self.slider.value()
        self.start_idx = int((value / 100) * self.sample_rate)
        self.update_time_label_and_fft(value)
        if self.is_playing:
            self.toggle_play_pause()
            self.toggle_play_pause()

    def update_time_label_and_fft(self, value=None):
        """Update the time label and FFT plot."""
        if value is None:
            current_time = self.start_idx / self.sample_rate
        else:
            current_time = value / 100

        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        centiseconds = int((current_time * 100) % 100)
        time_text = f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
        if self.advanced_mode:
            self.advanced_time_label.setText(time_text)
        else:
            self.time_label.setText(time_text)

        self.start_idx = int(current_time * self.sample_rate)
        positive_freqs, magnitude_db = self.compute_fft_at_position(self.start_idx)
        self.fft_plot.setData(positive_freqs, magnitude_db)
        self.current_freqs = positive_freqs
        self.current_magnitude_db = magnitude_db

    def update_plot(self):
        """Update the plot during playback."""
        try:
            positive_freqs, magnitude_db = self.queue.get_nowait()
            self.fft_plot.setData(positive_freqs, magnitude_db)
            self.current_freqs = positive_freqs
            self.current_magnitude_db = magnitude_db
            current_time_cs = int((self.start_idx / self.sample_rate) * 100)
            if not self.advanced_mode:
                self.slider.setValue(current_time_cs)
            self.update_time_label_and_fft(current_time_cs)
            if not self.is_playing:
                self.timer.stop()
                self.stream.stop()
                self.stream.close()
        except Empty:
            pass

    def closeEvent(self, event):
        self.timer.stop()
        if self.is_playing:
            self.stream.stop()
            self.stream.close()
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    from PyQt5.QtGui import QFont
    app = QApplication(sys.argv)
    window = RealTimeFFT()
    window.show()
    sys.exit(app.exec_())