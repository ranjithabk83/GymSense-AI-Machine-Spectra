/**
 * GYMSENSE AI - Sensor Acquisition & Signal Processing Engine
 * Student: Ranjitha B K | USN: 1SB24AI041
 * Branch: AIML | Event: MACHINE SPECTRA 1.0
 * 
 * Manages DeviceMotionEvent, DeviceOrientationEvent, 3s stillness calibration,
 * rolling window buffering, and realistic desktop test simulation.
 */

class SensorManager {
  constructor() {
    this.hasAccelerometer = false;
    this.hasGyroscope = false;
    this.isListening = false;
    this.isCalibrated = false;
    this.isSimulating = false;

    // Current real-time sensor reading
    this.currentReading = {
      accel_x: 0,
      accel_y: 0,
      accel_z: 9.81,
      accel_mag: 9.81,
      gyro_alpha: 0,
      gyro_beta: 0,
      gyro_gamma: 0,
      gyro_mag: 0,
      timestamp: Date.now()
    };

    // Rolling sensor window buffer (50 samples @ 20Hz = 2.5 seconds)
    this.windowSize = 50;
    this.sensorBuffer = [];

    // Calibration baselines
    this.baseline = {
      offset_x: 0,
      offset_y: 0,
      offset_z: 9.81,
      noise_variance: 0.05
    };

    // Listeners for telemetry callbacks
    this.listeners = [];
    this.simInterval = null;
    this.simActivity = 'REST';
    this.simPhase = 0;
  }

  onReading(callback) {
    this.listeners.push(callback);
  }

  async requestPermissions() {
    try {
      // iOS 13+ DeviceMotionEvent permission request
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        const motionPermission = await DeviceMotionEvent.requestPermission();
        if (motionPermission !== 'granted') {
          throw new Error('Device motion permission was denied.');
        }
      }

      // iOS 13+ DeviceOrientationEvent permission request
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        const orientationPermission = await DeviceOrientationEvent.requestPermission();
        if (orientationPermission !== 'granted') {
          console.warn('Orientation permission not granted.');
        }
      }

      this.attachEventListeners();
      return { success: true };
    } catch (err) {
      console.warn('Physical sensor permission error:', err);
      // Fallback: If on desktop or non-sensor device, enable simulator seamlessly
      return { success: false, error: err.message || 'Sensors not supported on this device.' };
    }
  }

  attachEventListeners() {
    if (this.isListening) return;
    this.isListening = true;

    // 1. Device Motion (Accelerometer)
    window.addEventListener('devicemotion', (event) => {
      this.hasAccelerometer = true;
      const acc = event.accelerationIncludingGravity || event.acceleration;
      if (acc) {
        this.currentReading.accel_x = acc.x || 0;
        this.currentReading.accel_y = acc.y || 0;
        this.currentReading.accel_z = acc.z || 9.81;
        this.currentReading.accel_mag = Math.sqrt(
          Math.pow(this.currentReading.accel_x, 2) +
          Math.pow(this.currentReading.accel_y, 2) +
          Math.pow(this.currentReading.accel_z, 2)
        );
      }

      // Gyroscope rotation rate if provided in motion event
      if (event.rotationRate) {
        this.hasGyroscope = true;
        this.currentReading.gyro_alpha = event.rotationRate.alpha || 0;
        this.currentReading.gyro_beta = event.rotationRate.beta || 0;
        this.currentReading.gyro_gamma = event.rotationRate.gamma || 0;
        this.currentReading.gyro_mag = Math.sqrt(
          Math.pow(this.currentReading.gyro_alpha, 2) +
          Math.pow(this.currentReading.gyro_beta, 2) +
          Math.pow(this.currentReading.gyro_gamma, 2)
        );
      }

      this.currentReading.timestamp = Date.now();
      this.pushToBuffer({ ...this.currentReading });
    });

    // 2. Device Orientation (fallback Gyroscope angles)
    window.addEventListener('deviceorientation', (event) => {
      if (!this.hasGyroscope) {
        this.hasGyroscope = true;
        this.currentReading.gyro_alpha = event.alpha || 0;
        this.currentReading.gyro_beta = event.beta || 0;
        this.currentReading.gyro_gamma = event.gamma || 0;
        this.currentReading.gyro_mag = Math.sqrt(
          Math.pow(this.currentReading.gyro_alpha, 2) +
          Math.pow(this.currentReading.gyro_beta, 2) +
          Math.pow(this.currentReading.gyro_gamma, 2)
        );
      }
    });
  }

  pushToBuffer(sample) {
    this.sensorBuffer.push(sample);
    if (this.sensorBuffer.length > this.windowSize) {
      this.sensorBuffer.shift();
    }

    // Broadcast reading to subscribers (e.g. waveform chart)
    for (const cb of this.listeners) {
      cb(sample);
    }
  }

  getRecentWindow() {
    return [...this.sensorBuffer];
  }

  /* ==========================================================
     3-SECOND STILLNESS CALIBRATION
     ========================================================== */
  async calibrate(onTick) {
    return new Promise((resolve) => {
      let secondsLeft = 3;
      if (onTick) onTick(secondsLeft);

      const calibrationSamples = [];
      const sampleInterval = setInterval(() => {
        calibrationSamples.push({ ...this.currentReading });
      }, 50);

      const countdownInterval = setInterval(() => {
        secondsLeft -= 1;
        if (onTick) onTick(secondsLeft);

        if (secondsLeft <= 0) {
          clearInterval(countdownInterval);
          clearInterval(sampleInterval);

          // Compute baseline statistics
          if (calibrationSamples.length > 0) {
            const axList = calibrationSamples.map(s => s.accel_x);
            const ayList = calibrationSamples.map(s => s.accel_y);
            const azList = calibrationSamples.map(s => s.accel_z);

            const meanX = axList.reduce((a, b) => a + b, 0) / axList.length;
            const meanY = ayList.reduce((a, b) => a + b, 0) / ayList.length;
            const meanZ = azList.reduce((a, b) => a + b, 0) / azList.length;

            const varX = axList.reduce((a, b) => a + Math.pow(b - meanX, 2), 0) / axList.length;
            const varY = ayList.reduce((a, b) => a + Math.pow(b - meanY, 2), 0) / ayList.length;
            const varZ = azList.reduce((a, b) => a + Math.pow(b - meanZ, 2), 0) / azList.length;

            this.baseline.offset_x = meanX;
            this.baseline.offset_y = meanY;
            this.baseline.offset_z = meanZ;
            this.baseline.noise_variance = varX + varY + varZ;
          }

          this.isCalibrated = true;
          this.hasAccelerometer = true;
          this.hasGyroscope = true;
          resolve(this.baseline);
        }
      }, 1000);
    });
  }

  /* ==========================================================
     DESKTOP & JUDGE DEMONSTRATION SIMULATOR
     ========================================================== */
  startSimulation(activity = 'REST') {
    this.isSimulating = true;
    this.simActivity = activity.toUpperCase();
    this.hasAccelerometer = true;
    this.hasGyroscope = true;
    this.isCalibrated = true;

    if (this.simInterval) clearInterval(this.simInterval);

    const dt = 0.05; // 20 Hz
    this.simInterval = setInterval(() => {
      const g = 9.80665;
      const noiseA = () => (Math.random() - 0.5) * 0.4;
      const noiseG = () => (Math.random() - 0.5) * 3.0;

      this.simPhase += dt * 2 * Math.PI * 0.45;
      const p = this.simPhase;

      let ax = 0, ay = 0, az = g, ga = 0, gb = 0, gg = 0;

      switch (this.simActivity) {
        case 'REST':
          ax = noiseA() * 0.3;
          ay = noiseA() * 0.3;
          az = g + noiseA() * 0.3;
          ga = noiseG() * 0.2;
          gb = noiseG() * 0.2;
          gg = noiseG() * 0.2;
          break;

        case 'WALKING':
          ax = 1.2 * Math.sin(p * 2) + noiseA();
          ay = g + 2.6 * Math.sin(p * 4) + noiseA();
          az = 2.0 * Math.cos(p * 4) + noiseA();
          ga = 25 * Math.sin(p * 2) + noiseG();
          gb = 45 * Math.sin(p * 4) + noiseG();
          gg = 35 * Math.cos(p * 2) + noiseG();
          break;

        case 'RUNNING':
          ax = 2.8 * Math.sin(p * 3) + noiseA();
          ay = g + 8.5 * Math.sin(p * 6) + noiseA();
          az = 5.0 * Math.cos(p * 6) + noiseA();
          ga = 60 * Math.sin(p * 3) + noiseG();
          gb = 110 * Math.sin(p * 6) + noiseG();
          gg = 80 * Math.cos(p * 3) + noiseG();
          break;

        case 'BICEP CURL':
          ax = 0.7 * Math.sin(p) + noiseA();
          ay = g * Math.cos(0.9 * Math.sin(p)) + 2.2 * Math.sin(p) + noiseA();
          az = g * Math.sin(0.9 * Math.sin(p)) + 3.5 * Math.cos(p) + noiseA();
          ga = 15 * Math.sin(p) + noiseG();
          gb = 95 * Math.cos(p) + noiseG();
          gg = 20 * Math.sin(p + 0.4) + noiseG();
          break;

        case 'HAMMER CURL':
          ax = 1.8 * Math.sin(p) + noiseA();
          ay = g * Math.cos(0.85 * Math.sin(p)) + 2.5 * Math.sin(p) + noiseA();
          az = g * Math.sin(0.4 * Math.sin(p)) + 2.0 * Math.cos(p) + noiseA();
          ga = 45 * Math.sin(p) + noiseG();
          gb = 30 * Math.cos(p) + noiseG();
          gg = 80 * Math.cos(p) + noiseG();
          break;

        case 'SQUAT':
          ax = 0.6 * Math.sin(p) + noiseA();
          ay = g - 3.8 * Math.sin(p) + noiseA();
          az = 2.5 * Math.cos(p) + noiseA();
          ga = 10 * Math.sin(p) + noiseG();
          gb = 55 * Math.sin(p) + noiseG();
          gg = 12 * Math.cos(p) + noiseG();
          break;

        case 'LUNGE':
          ax = 1.2 * Math.sin(p) + noiseA();
          ay = g - 3.2 * Math.sin(p) + noiseA();
          az = 2.2 * Math.sin(p) + noiseA();
          ga = 25 * Math.sin(p) + noiseG();
          gb = 48 * Math.sin(p) + noiseG();
          gg = 32 * Math.cos(p) + noiseG();
          break;

        case 'JUMPING JACK':
          ax = 6.5 * Math.sin(p * 2.5) + noiseA();
          ay = g + 7.8 * Math.sin(p * 2.5) + noiseA();
          az = 3.0 * Math.cos(p * 2.5) + noiseA();
          ga = 75 * Math.cos(p * 2.5) + noiseG();
          gb = 40 * Math.sin(p * 2.5) + noiseG();
          gg = 120 * Math.sin(p * 2.5) + noiseG();
          break;

        case 'SHOULDER PRESS':
          ax = 0.8 * Math.sin(p) + noiseA();
          ay = g + 4.2 * Math.sin(p) + noiseA();
          az = 1.4 * Math.cos(p) + noiseA();
          ga = 18 * Math.sin(p) + noiseG();
          gb = 32 * Math.cos(p) + noiseG();
          gg = 22 * Math.sin(p) + noiseG();
          break;

        case 'FRONT RAISE':
          ax = 0.5 * Math.sin(p) + noiseA();
          ay = g * Math.cos(0.9 * Math.sin(p)) + 2.8 * Math.sin(p) + noiseA();
          az = g * Math.sin(0.9 * Math.sin(p)) + 4.2 * Math.cos(p) + noiseA();
          ga = 14 * Math.sin(p) + noiseG();
          gb = 85 * Math.cos(p) + noiseG();
          gg = 15 * Math.sin(p) + noiseG();
          break;

        case 'LATERAL RAISE':
          ax = 4.8 * Math.sin(p) + noiseA();
          ay = g * Math.cos(0.8 * Math.sin(p)) + 2.5 * Math.sin(p) + noiseA();
          az = 0.9 * Math.cos(p) + noiseA();
          ga = 20 * Math.sin(p) + noiseG();
          gb = 18 * Math.sin(p) + noiseG();
          gg = 82 * Math.cos(p) + noiseG();
          break;

        case 'WORKOUT':
        default:
          ax = 2.0 * Math.sin(p * 1.5) + noiseA();
          ay = g + 3.0 * Math.cos(p * 1.5) + noiseA();
          az = 2.5 * Math.sin(p * 1.5) + noiseA();
          ga = 35 * Math.sin(p * 1.5) + noiseG();
          gb = 40 * Math.cos(p * 1.5) + noiseG();
          gg = 30 * Math.sin(p * 1.5) + noiseG();
          break;
      }

      this.currentReading = {
        accel_x: parseFloat(ax.toFixed(3)),
        accel_y: parseFloat(ay.toFixed(3)),
        accel_z: parseFloat(az.toFixed(3)),
        accel_mag: parseFloat(Math.sqrt(ax*ax + ay*ay + az*az).toFixed(3)),
        gyro_alpha: parseFloat(ga.toFixed(2)),
        gyro_beta: parseFloat(gb.toFixed(2)),
        gyro_gamma: parseFloat(gg.toFixed(2)),
        gyro_mag: parseFloat(Math.sqrt(ga*ga + gb*gb + gg*gg).toFixed(2)),
        timestamp: Date.now()
      };

      this.pushToBuffer({ ...this.currentReading });
    }, 50);
  }

  setSimulatedActivity(activity) {
    this.simActivity = activity.toUpperCase();
  }

  stopSimulation() {
    this.isSimulating = false;
    if (this.simInterval) {
      clearInterval(this.simInterval);
      this.simInterval = null;
    }
  }
}

window.sensorManager = new SensorManager();
