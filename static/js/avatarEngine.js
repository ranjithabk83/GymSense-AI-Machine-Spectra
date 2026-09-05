/**
 * GYMSENSE AI - 60 FPS Skeletal Avatar Animation Engine
 * Student: Ranjitha B K | USN: 1SB24AI041
 * Branch: AIML | Event: MACHINE SPECTRA 1.0
 * 
 * Provides 100% consistent Female & Male 3D-styled animated avatars
 * across 12 dynamic exercise kinematic states.
 */

class AvatarEngine {
  constructor(canvasId, defaultAvatar = 'female') {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.gender = defaultAvatar; // 'female' | 'male'
    this.activity = 'REST';      // Current exercise state
    this.phase = 0;              // Continuous animation phase angle
    this.speed = 1.0;            // Motion cadence multiplier
    this.isRunning = false;
    this.animationFrameId = null;
    this.lastTimestamp = 0;
    this.pulseEffect = 0;        // Rep celebration pulse

    // Resize handling for crisp Retina / HiDPI screens
    if (this.canvas) {
      this.handleResize();
      window.addEventListener('resize', () => this.handleResize());
    }
  }

  setGender(gender) {
    this.gender = gender === 'male' ? 'male' : 'female';
  }

  setActivity(activity) {
    const formatted = activity.toUpperCase().trim();
    if (this.activity !== formatted) {
      this.activity = formatted;
      // Adjust cadence speed per activity
      if (['RUNNING', 'JUMPING JACK'].includes(formatted)) {
        this.speed = 2.2;
      } else if (['WALKING'].includes(formatted)) {
        this.speed = 1.4;
      } else if (['REST'].includes(formatted)) {
        this.speed = 0.5;
      } else {
        this.speed = 1.0; // Lifting exercises (~2.5s cycle)
      }
    }
  }

  triggerRepEffect() {
    this.pulseEffect = 1.0;
  }

  handleResize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.scale(dpr, dpr);
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastTimestamp = performance.now();
    const renderLoop = (timestamp) => {
      if (!this.isRunning) return;
      const dt = (timestamp - this.lastTimestamp) / 1000;
      this.lastTimestamp = timestamp;

      this.update(dt);
      this.render();

      this.animationFrameId = requestAnimationFrame(renderLoop);
    };
    this.animationFrameId = requestAnimationFrame(renderLoop);
  }

  stop() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  update(dt) {
    // Increment cyclical motion phase
    const cycleFreq = 0.8 * this.speed; // Hz
    this.phase += 2 * Math.PI * cycleFreq * dt;
    if (this.phase > 2000 * Math.PI) this.phase %= (2 * Math.PI);

    // Decay rep pulse effect
    if (this.pulseEffect > 0) {
      this.pulseEffect = Math.max(0, this.pulseEffect - dt * 2.0);
    }
  }

  render() {
    if (!this.ctx || !this.displayWidth || !this.displayHeight) return;
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;

    ctx.clearRect(0, 0, w, h);

    const centerX = w / 2;
    const groundY = h * 0.88;
    const p = this.phase;

    // Draw Floor Shadow & Ambient Glow
    this.drawFloorGlow(ctx, centerX, groundY, p);

    // Calculate Kinematics based on active exercise
    const pose = this.calculateKinematics(this.activity, p);

    // Render Avatar Character (Female or Male)
    ctx.save();
    ctx.translate(centerX + pose.rootX, groundY - pose.rootY);

    if (this.gender === 'female') {
      this.drawFemaleAvatar(ctx, pose, p);
    } else {
      this.drawMaleAvatar(ctx, pose, p);
    }

    ctx.restore();

    // Draw Rep Pulse Ripple
    if (this.pulseEffect > 0) {
      this.drawRepRipple(ctx, centerX, groundY - 110);
    }
  }

  drawFloorGlow(ctx, cx, cy, p) {
    const shadowWidth = 90 + Math.sin(p) * 6;
    const shadowHeight = 18 + Math.cos(p) * 2;

    // Radial floor shadow
    const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, shadowWidth);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.6)');
    grad.addColorStop(0.4, 'rgba(255, 46, 147, 0.18)');
    grad.addColorStop(1, 'transparent');

    ctx.save();
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, shadowWidth, shadowHeight, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  drawRepRipple(ctx, cx, cy) {
    const alpha = this.pulseEffect;
    const radius = 60 + (1 - alpha) * 90;

    ctx.save();
    ctx.strokeStyle = `rgba(255, 46, 147, ${alpha * 0.8})`;
    ctx.lineWidth = 3 * alpha;
    ctx.shadowColor = '#ff2e93';
    ctx.shadowBlur = 16;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = `rgba(0, 242, 254, ${alpha * 0.6})`;
    ctx.lineWidth = 2 * alpha;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  calculateKinematics(activity, p) {
    const sinP = Math.sin(p);
    const cosP = Math.cos(p);
    const halfP = p * 0.5;

    // Default neutral standing pose
    let pose = {
      rootX: 0,
      rootY: 10,
      bodyScale: 1.0,
      torsoTilt: 0,
      headBob: Math.sin(p * 2) * 2,
      // Left Arm
      leftUpperArmAngle: 0.15,
      leftForearmAngle: 0.2,
      // Right Arm
      rightUpperArmAngle: -0.15,
      rightForearmAngle: -0.2,
      // Left Leg
      leftHipAngle: 0.05,
      leftKneeAngle: 0.05,
      // Right Leg
      rightHipAngle: -0.05,
      rightKneeAngle: 0.05,
      // Equipment
      holdingDumbbells: false,
      dumbbellAngle: 0,
      isHammerGrip: false
    };

    switch (activity) {
      case 'REST':
        pose.rootY = 10 + Math.sin(p * 0.8) * 2;
        pose.torsoTilt = Math.sin(p * 0.5) * 0.02;
        pose.leftUpperArmAngle = 0.18 + Math.sin(p * 0.8) * 0.03;
        pose.leftForearmAngle = 0.25;
        pose.rightUpperArmAngle = -0.18 - Math.sin(p * 0.8) * 0.03;
        pose.rightForearmAngle = -0.25;
        break;

      case 'WALKING':
        pose.rootY = 10 + Math.abs(Math.sin(p)) * 8;
        pose.rootX = Math.sin(halfP) * 3;
        pose.torsoTilt = 0.06;
        // Alternating leg swings
        pose.leftHipAngle = Math.sin(p) * 0.45;
        pose.leftKneeAngle = Math.max(0, -Math.sin(p) * 0.5);
        pose.rightHipAngle = -Math.sin(p) * 0.45;
        pose.rightKneeAngle = Math.max(0, Math.sin(p) * 0.5);
        // Opposite arm swings
        pose.leftUpperArmAngle = -Math.sin(p) * 0.45;
        pose.leftForearmAngle = 0.4;
        pose.rightUpperArmAngle = Math.sin(p) * 0.45;
        pose.rightForearmAngle = -0.4;
        break;

      case 'RUNNING':
        pose.rootY = 14 + Math.abs(Math.sin(p)) * 16;
        pose.torsoTilt = 0.18;
        pose.leftHipAngle = Math.sin(p) * 0.85;
        pose.leftKneeAngle = Math.max(0.1, -Math.sin(p) * 1.1);
        pose.rightHipAngle = -Math.sin(p) * 0.85;
        pose.rightKneeAngle = Math.max(0.1, Math.sin(p) * 1.1);
        pose.leftUpperArmAngle = -Math.sin(p) * 0.9;
        pose.leftForearmAngle = 1.3;
        pose.rightUpperArmAngle = Math.sin(p) * 0.9;
        pose.rightForearmAngle = -1.3;
        break;

      case 'BICEP CURL':
        pose.holdingDumbbells = true;
        pose.rootY = 10;
        // Smooth curl interpolation (0 to 1)
        const curlT = (Math.sin(p) + 1) / 2; // 0 (down) -> 1 (up)
        pose.leftUpperArmAngle = 0.08;
        pose.rightUpperArmAngle = -0.08;
        // Forearm curls up towards shoulder
        pose.leftForearmAngle = 0.2 + curlT * 2.2;
        pose.rightForearmAngle = -0.2 - curlT * 2.2;
        pose.dumbbellAngle = -curlT * 0.3;
        break;

      case 'HAMMER CURL':
        pose.holdingDumbbells = true;
        pose.isHammerGrip = true;
        pose.rootY = 10;
        const hammerT = (Math.sin(p) + 1) / 2;
        pose.leftUpperArmAngle = 0.05;
        pose.rightUpperArmAngle = -0.05;
        pose.leftForearmAngle = 0.25 + hammerT * 2.1;
        pose.rightForearmAngle = -0.25 - hammerT * 2.1;
        break;

      case 'SQUAT':
        pose.holdingDumbbells = false;
        const squatT = (Math.sin(p) + 1) / 2; // 0 = standing, 1 = deep squat
        pose.rootY = 10 - squatT * 38;
        pose.torsoTilt = squatT * 0.32;
        // Knees bend outwards/downwards
        pose.leftHipAngle = -squatT * 0.9;
        pose.leftKneeAngle = squatT * 1.8;
        pose.rightHipAngle = squatT * 0.9;
        pose.rightKneeAngle = squatT * 1.8;
        // Arms reach forward for balance
        pose.leftUpperArmAngle = -0.3 - squatT * 0.8;
        pose.leftForearmAngle = 0.6;
        pose.rightUpperArmAngle = 0.3 + squatT * 0.8;
        pose.rightForearmAngle = -0.6;
        break;

      case 'LUNGE':
        pose.holdingDumbbells = true;
        const lungeT = (Math.sin(p) + 1) / 2;
        pose.rootY = 10 - lungeT * 30;
        pose.torsoTilt = 0.05;
        pose.leftHipAngle = 0.6 * lungeT;
        pose.leftKneeAngle = 1.4 * lungeT;
        pose.rightHipAngle = -0.7 * lungeT;
        pose.rightKneeAngle = 1.3 * lungeT;
        pose.leftUpperArmAngle = 0.1;
        pose.rightUpperArmAngle = -0.1;
        break;

      case 'JUMPING JACK':
        const jackT = (Math.sin(p) + 1) / 2; // 0 = together, 1 = wide
        pose.rootY = 10 + Math.abs(Math.sin(p)) * 14;
        // Legs splay wide
        pose.leftHipAngle = jackT * 0.45;
        pose.rightHipAngle = -jackT * 0.45;
        // Arms arc all the way overhead
        pose.leftUpperArmAngle = jackT * 2.6;
        pose.leftForearmAngle = 0.2;
        pose.rightUpperArmAngle = -jackT * 2.6;
        pose.rightForearmAngle = -0.2;
        break;

      case 'SHOULDER PRESS':
        pose.holdingDumbbells = true;
        const pressT = (Math.sin(p) + 1) / 2; // 0 = shoulder height, 1 = fully pressed overhead
        pose.rootY = 10 + pressT * 4;
        // Upper arm stays abducted
        pose.leftUpperArmAngle = 1.3 + pressT * 1.4;
        pose.leftForearmAngle = 0.8 - pressT * 0.6;
        pose.rightUpperArmAngle = -1.3 - pressT * 1.4;
        pose.rightForearmAngle = -0.8 + pressT * 0.6;
        break;

      case 'FRONT RAISE':
        pose.holdingDumbbells = true;
        const frontT = (Math.sin(p) + 1) / 2;
        pose.rootY = 10;
        pose.leftUpperArmAngle = -frontT * 1.45;
        pose.leftForearmAngle = 0.1;
        pose.rightUpperArmAngle = -frontT * 1.45;
        pose.rightForearmAngle = 0.1;
        break;

      case 'LATERAL RAISE':
        pose.holdingDumbbells = true;
        const latT = (Math.sin(p) + 1) / 2;
        pose.rootY = 10;
        // Arms raise out to sides like wings
        pose.leftUpperArmAngle = latT * 1.45;
        pose.leftForearmAngle = 0.15;
        pose.rightUpperArmAngle = -latT * 1.45;
        pose.rightForearmAngle = -0.15;
        break;

      case 'WORKOUT':
      default:
        // Active energetic training stance / boxer rhythm
        pose.rootY = 10 + Math.abs(Math.sin(p * 2)) * 6;
        pose.rootX = Math.sin(p) * 4;
        pose.torsoTilt = 0.08;
        pose.leftUpperArmAngle = 0.7 + Math.sin(p) * 0.3;
        pose.leftForearmAngle = 1.4;
        pose.rightUpperArmAngle = -0.6 - Math.cos(p) * 0.3;
        pose.rightForearmAngle = -1.4;
        pose.leftHipAngle = 0.15;
        pose.rightHipAngle = -0.2;
        break;
    }

    return pose;
  }

  /* ==========================================================
     FEMALE AVATAR RENDERING ENGINE (Consistent 3D Aesthetics)
     ========================================================== */
  drawFemaleAvatar(ctx, pose, p) {
    const skinTone = '#fcd5b5';
    const skinShadow = '#eeb896';
    const hairColor = '#3a2016';
    const hairHighlight = '#5a3424';
    const outfitPink = '#ff2e93';
    const jacketMint = '#2dd4bf';
    const jacketWhite = '#f8fafc';
    const sneakerWhite = '#ffffff';

    // 1. Torso & Pelvis Anchor Points
    const hipY = -95;
    const neckY = -165;
    const headY = -195;

    // 2. Legs (Left & Right)
    this.drawLeg(ctx, -12, hipY, pose.leftHipAngle, pose.leftKneeAngle, '#1e293b', sneakerWhite, true);
    this.drawLeg(ctx, 12, hipY, pose.rightHipAngle, pose.rightKneeAngle, '#1e293b', sneakerWhite, false);

    // 3. Torso / Sports Outfit
    ctx.save();
    ctx.rotate(pose.torsoTilt);

    // Torso Base (Pink Fitness Top)
    const torsoGrad = ctx.createLinearGradient(0, neckY, 0, hipY);
    torsoGrad.addColorStop(0, outfitPink);
    torsoGrad.addColorStop(0.7, '#d91b7d');
    torsoGrad.addColorStop(1, '#1e293b'); // High waist tights

    ctx.fillStyle = torsoGrad;
    ctx.beginPath();
    ctx.moveTo(-18, neckY + 15);
    ctx.quadraticCurveTo(-22, (neckY + hipY) / 2, -15, hipY);
    ctx.lineTo(15, hipY);
    ctx.quadraticCurveTo(22, (neckY + hipY) / 2, 18, neckY + 15);
    ctx.closePath();
    ctx.fill();

    // Sporty Mint Jacket Trim / Accents
    ctx.fillStyle = jacketMint;
    ctx.beginPath();
    ctx.ellipse(-14, neckY + 28, 6, 20, -0.2, 0, Math.PI * 2);
    ctx.ellipse(14, neckY + 28, 6, 20, 0.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = jacketWhite;
    ctx.fillRect(-2, neckY + 15, 4, 40);

    // 4. Arms (Left & Right)
    this.drawFemaleArm(ctx, -20, neckY + 20, pose.leftUpperArmAngle, pose.leftForearmAngle, pose, true, jacketMint, skinTone);
    this.drawFemaleArm(ctx, 20, neckY + 20, pose.rightUpperArmAngle, pose.rightForearmAngle, pose, false, jacketMint, skinTone);

    // 5. Head, Face & Long Brown Hair
    ctx.save();
    ctx.translate(0, headY + pose.headBob);

    // Long Back Hair
    ctx.fillStyle = hairColor;
    ctx.beginPath();
    ctx.moveTo(-18, -10);
    ctx.quadraticCurveTo(-28, 25, -20, 55);
    ctx.lineTo(20, 55);
    ctx.quadraticCurveTo(28, 25, 18, -10);
    ctx.closePath();
    ctx.fill();

    // Neck
    ctx.fillStyle = skinShadow;
    ctx.fillRect(-6, 12, 12, 14);

    // Face Oval
    const faceGrad = ctx.createRadialGradient(-3, -2, 2, 0, 0, 20);
    faceGrad.addColorStop(0, skinTone);
    faceGrad.addColorStop(1, skinShadow);
    ctx.fillStyle = faceGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, 16, 20, 0, 0, Math.PI * 2);
    ctx.fill();

    // Front Hair Bangs & Ponytail
    ctx.fillStyle = hairHighlight;
    ctx.beginPath();
    ctx.arc(0, -6, 18, Math.PI, Math.PI * 2);
    ctx.quadraticCurveTo(0, 4, 18, -2);
    ctx.fill();

    // Expressive Eyes & Smile
    ctx.fillStyle = '#1e1b4b';
    ctx.beginPath();
    ctx.ellipse(-6, -2, 2.5, 3.5, 0, 0, Math.PI * 2);
    ctx.ellipse(6, -2, 2.5, 3.5, 0, 0, Math.PI * 2);
    ctx.fill();

    // Eye Sparkles
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(-5.2, -3.2, 1, 0, Math.PI * 2);
    ctx.arc(6.8, -3.2, 1, 0, Math.PI * 2);
    ctx.fill();

    // Rosy Cheeks
    ctx.fillStyle = 'rgba(255, 46, 147, 0.35)';
    ctx.beginPath();
    ctx.arc(-9, 5, 4, 0, Math.PI * 2);
    ctx.arc(9, 5, 4, 0, Math.PI * 2);
    ctx.fill();

    // Confident Smile
    ctx.strokeStyle = '#d91b7d';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(0, 4, 5, 0.2, Math.PI - 0.2);
    ctx.stroke();

    ctx.restore();
    ctx.restore();
  }

  /* ==========================================================
     MALE AVATAR RENDERING ENGINE (Consistent 3D Aesthetics)
     ========================================================== */
  drawMaleAvatar(ctx, pose, p) {
    const skinTone = '#f8c8a0';
    const skinShadow = '#e09e72';
    const hairColor = '#2d1810';
    const topBlack = '#0f172a';
    const shortsBlack = '#1e293b';
    const shoeBlack = '#090d16';

    const hipY = -95;
    const neckY = -170;
    const headY = -200;

    // 1. Legs & Shorts
    this.drawLeg(ctx, -14, hipY, pose.leftHipAngle, pose.leftKneeAngle, shortsBlack, shoeBlack, true, 8);
    this.drawLeg(ctx, 14, hipY, pose.rightHipAngle, pose.rightKneeAngle, shortsBlack, shoeBlack, false, 8);

    // 2. Torso (Black Sleeveless Gym Top)
    ctx.save();
    ctx.rotate(pose.torsoTilt);

    const torsoGrad = ctx.createLinearGradient(0, neckY, 0, hipY);
    torsoGrad.addColorStop(0, topBlack);
    torsoGrad.addColorStop(0.8, '#1e293b');
    torsoGrad.addColorStop(1, shortsBlack);

    ctx.fillStyle = torsoGrad;
    ctx.beginPath();
    ctx.moveTo(-24, neckY + 12);
    ctx.lineTo(-18, hipY);
    ctx.lineTo(18, hipY);
    ctx.lineTo(24, neckY + 12);
    ctx.closePath();
    ctx.fill();

    // Neon Accent Line on Gym Tank
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-16, neckY + 25);
    ctx.lineTo(-12, hipY - 10);
    ctx.stroke();

    // 3. Athletic Muscular Arms
    this.drawMaleArm(ctx, -24, neckY + 16, pose.leftUpperArmAngle, pose.leftForearmAngle, pose, true, skinTone);
    this.drawMaleArm(ctx, 24, neckY + 16, pose.rightUpperArmAngle, pose.rightForearmAngle, pose, false, skinTone);

    // 4. Head, Spiky Hair & Face
    ctx.save();
    ctx.translate(0, headY + pose.headBob);

    // Neck
    ctx.fillStyle = skinShadow;
    ctx.fillRect(-7, 12, 14, 14);

    // Strong Jaw Face
    ctx.fillStyle = skinTone;
    ctx.beginPath();
    ctx.moveTo(-14, -6);
    ctx.lineTo(-13, 8);
    ctx.lineTo(0, 20);
    ctx.lineTo(13, 8);
    ctx.lineTo(14, -6);
    ctx.closePath();
    ctx.fill();

    // Styled Spiky Brown Hair
    ctx.fillStyle = hairColor;
    ctx.beginPath();
    ctx.moveTo(-18, -4);
    ctx.lineTo(-22, -18);
    ctx.lineTo(-14, -15);
    ctx.lineTo(-10, -26);
    ctx.lineTo(0, -22);
    ctx.lineTo(10, -28);
    ctx.lineTo(14, -16);
    ctx.lineTo(22, -18);
    ctx.lineTo(18, -4);
    ctx.closePath();
    ctx.fill();

    // Athletic Eyes & Smile
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.ellipse(-5, 0, 2.5, 3, 0, 0, Math.PI * 2);
    ctx.ellipse(5, 0, 2.5, 3, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 7, 5, 0.1, Math.PI - 0.1);
    ctx.stroke();

    ctx.restore();
    ctx.restore();
  }

  /* --- Helper: Draw Arm with Dumbbell Equipment --- */
  drawFemaleArm(ctx, sx, sy, upperAngle, foreAngle, pose, isLeft, sleeveColor, skinColor) {
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(upperAngle);

    // Upper Arm (Sporty Sleeve)
    ctx.fillStyle = sleeveColor;
    ctx.beginPath();
    ctx.roundRect(-5, 0, 10, 34, 5);
    ctx.fill();

    // Forearm
    ctx.translate(0, 32);
    ctx.rotate(foreAngle);

    ctx.fillStyle = skinColor;
    ctx.beginPath();
    ctx.roundRect(-4, 0, 8, 30, 4);
    ctx.fill();

    // Hand / Equipment
    ctx.translate(0, 28);
    if (pose.holdingDumbbells) {
      this.drawDumbbell(ctx, pose.dumbbellAngle, pose.isHammerGrip);
    } else {
      ctx.fillStyle = skinColor;
      ctx.beginPath();
      ctx.arc(0, 2, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  drawMaleArm(ctx, sx, sy, upperAngle, foreAngle, pose, isLeft, skinColor) {
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(upperAngle);

    // Muscular Upper Arm
    ctx.fillStyle = skinColor;
    ctx.beginPath();
    ctx.roundRect(-6, 0, 12, 36, 6);
    ctx.fill();

    // Forearm
    ctx.translate(0, 34);
    ctx.rotate(foreAngle);

    ctx.fillStyle = skinColor;
    ctx.beginPath();
    ctx.roundRect(-5, 0, 10, 32, 5);
    ctx.fill();

    // Hand / Dumbbell
    ctx.translate(0, 30);
    if (pose.holdingDumbbells) {
      this.drawDumbbell(ctx, pose.dumbbellAngle, pose.isHammerGrip);
    } else {
      ctx.fillStyle = skinColor;
      ctx.beginPath();
      ctx.arc(0, 2, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  drawLeg(ctx, sx, sy, hipAngle, kneeAngle, pantsColor, shoeColor, isLeft, thickness = 7) {
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(hipAngle);

    // Thigh
    ctx.fillStyle = pantsColor;
    ctx.beginPath();
    ctx.roundRect(-thickness, 0, thickness * 2, 45, thickness);
    ctx.fill();

    // Shin & Knee
    ctx.translate(0, 42);
    ctx.rotate(kneeAngle);

    ctx.fillStyle = pantsColor;
    ctx.beginPath();
    ctx.roundRect(-thickness + 1, 0, (thickness - 1) * 2, 45, thickness - 1);
    ctx.fill();

    // Sneaker
    ctx.translate(0, 42);
    ctx.fillStyle = shoeColor;
    ctx.beginPath();
    ctx.ellipse(isLeft ? -4 : 4, 3, 10, 6, 0, 0, Math.PI * 2);
    ctx.fill();

    // Sneaker Sole Highlight
    ctx.fillStyle = '#ff2e93';
    ctx.fillRect(isLeft ? -12 : -6, 6, 18, 3);

    ctx.restore();
  }

  drawDumbbell(ctx, angle = 0, isHammer = false) {
    ctx.save();
    ctx.rotate(angle);

    const barLength = isHammer ? 28 : 34;
    const plateRadius = 10;

    // Metallic Center Handle
    ctx.fillStyle = '#94a3b8';
    ctx.fillRect(-barLength / 2, -2.5, barLength, 5);

    // Chrome/Pink Weighted Plates on Ends
    const plateGrad = ctx.createLinearGradient(0, -plateRadius, 0, plateRadius);
    plateGrad.addColorStop(0, '#ff2e93');
    plateGrad.addColorStop(0.5, '#475569');
    plateGrad.addColorStop(1, '#0f172a');

    ctx.fillStyle = plateGrad;
    // Left Plate
    ctx.beginPath();
    ctx.roundRect(-barLength / 2 - 4, -plateRadius, 6, plateRadius * 2, 2);
    ctx.fill();
    // Right Plate
    ctx.beginPath();
    ctx.roundRect(barLength / 2 - 2, -plateRadius, 6, plateRadius * 2, 2);
    ctx.fill();

    ctx.restore();
  }
}

window.AvatarEngine = AvatarEngine;
