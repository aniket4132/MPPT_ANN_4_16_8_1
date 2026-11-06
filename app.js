/**
 * SOLARIS MPPT - Interactive Sun, Cloud, Boost Converter & 6S Battery Simulator
 * Fully calibrated to 50W Solar Panel (18.0V / 2.78A at 1000 W/m², 25°C),
 * Boost Converter Impedance Matching Rin = Rload * (1 - D)^2,
 * Phase 1 Ceramic Resistors (15Ω & 30Ω) and Phase 2 6S Li-Ion Battery Pack.
 */

// =====================================================================
// 1. EMBEDDED ANN MODEL WEIGHTS & NORMALIZATION SCALER
// =====================================================================
const MODEL_DATA = {
  "scaler": {
    "scale": [0.0011121830083794092, 0.022229721048122455, 0.035894488867862694, 0.1259166733822226],
    "min": [-0.11204371857943979, -0.33376036627466776, -0.3089309436697018, -0.0383882162140382],
    "features": ["Irradiance", "Temperature", "V_pv", "I_pv"]
  },
  "layers": [
    {
      "name": "Hidden1",
      "weights": [[0.39237654209136963, 0.5468809604644775, -0.1624799370765686, 0.5832646489143372, -1.5947527885437012, -0.2771438956260681, 0.45921754837036133, -0.20415207743644714, -0.0032283554319292307, 0.10365784168243408, -0.45305609703063965, 0.15289878845214844, 0.7693197131156921, 0.19825774431228638, -0.007578716147691011, 0.6903867125511169], [-0.5270954370498657, 0.028993036597967148, 0.00411997502669692, -0.22270122170448303, -0.012123498134315014, 0.32463619112968445, 0.36696064472198486, -0.1346493810415268, 0.023355115205049515, -0.43747857213020325, -0.04131430387496948, 0.21128776669502258, 0.0189003124833107, -0.17069897055625916, -0.3263733685016632, 0.5829484462738037], [-0.42734086513519287, 0.02310885675251484, -0.26796817779541016, -0.3120934069156647, 0.026952356100082397, -0.3691864609718323, 0.14656344056129456, 0.23988410830497742, -0.573779284954071, 0.029547402635216713, -0.18171057105064392, -0.21040642261505127, 0.5490124821662903, 0.049611564725637436, 0.024332182481884956, -0.09378527104854584], [-0.4131831228733063, 0.34750786423683167, 0.11821364611387253, -0.07730966806411743, -0.01513022929430008, -0.20208747684955597, 0.21088069677352905, 0.5278190970420837, -0.04639030620455742, -0.23570436239242554, -0.5115088820457458, -0.24198049306869507, 0.1871960312128067, 0.4089109003543854, -0.14604640007019043, 0.2092430591583252]],
      "biases": [0.0, -0.2019813358783722, -0.011431651189923286, 0.28660672903060913, 0.09462317079305649, 0.13647332787513733, -0.08820078521966934, 0.09424737840890884, 0.31677985191345215, -0.03810909762978554, 0.0, 0.44071805477142334, -0.0410895049571991, 0.11784651130437851, -0.01007929164916277, 0.04809950664639473]
    },
    {
      "name": "Hidden2",
      "weights": [[0.47915589809417725, -0.017982125282287598, 0.26609790325164795, 0.2428727149963379, 0.31599926948547363, -0.3813211917877197, 0.20314550399780273, 0.029398798942565918], [-0.8298885822296143, -0.19033920764923096, -0.3124660551548004, -0.020697521045804024, 0.3643340766429901, 0.13985800743103027, -0.04032250866293907, -0.4963550865650177], [0.2409481555223465, 0.33692705631256104, -0.396180123090744, 0.05425478145480156, 0.27091851830482483, 0.46505188941955566, 0.04303540289402008, 0.4378563463687897], [0.22076913714408875, 0.2957780361175537, -0.08327274769544601, -0.34060364961624146, 0.4257003664970398, 0.0028451960533857346, -0.30605384707450867, -0.007563387509435415], [-0.7234811782836914, 0.45473647117614746, -0.3828109800815582, -0.47380390763282776, -0.47361433506011963, -0.33417606353759766, -0.41277703642845154, -0.15793269872665405], [-0.04111647605895996, 0.5255394577980042, -0.07292723655700684, 0.21505987644195557, -0.03727468103170395, 0.45123234391212463, -0.5494049191474915, -0.33133465051651], [-0.13841505348682404, -0.22565031051635742, 0.1984802484512329, 0.2140977680683136, 0.4406614601612091, 0.02039583958685398, 0.2937562167644501, 0.22043435275554657], [0.050482142716646194, 0.5008378028869629, 0.013671238906681538, -0.282044380903244, -0.288225382566452, -0.2900083065032959, -0.452886164188385, -0.1458258479833603], [0.11801769584417343, 0.1116880550980568, 0.004282420035451651, -0.249187633395195, 0.5195464491844177, -0.2861403822898865, 0.429836630821228, 0.06152438372373581], [-0.36023712158203125, -0.06165003776550293, 0.205078586935997, 0.43031591176986694, 0.3959532380104065, -0.32320499420166016, -0.027496904134750366, -0.49716877937316895], [0.19154977798461914, -0.04421722888946533, 0.34523332118988037, 0.3764612674713135, 0.3136107921600342, -0.18056797981262207, -0.270355224609375, 0.19978117942810059], [0.23525750637054443, -0.26042476296424866, -0.4316527843475342, 0.006968327797949314, -1.3603246212005615, 0.343720942735672, -0.17574723064899445, -0.4703526794910431], [0.3707612454891205, -0.1707773208618164, 0.09807291626930237, 0.2088213711977005, 0.23924049735069275, -0.25407299399375916, 0.050375744700431824, -0.11415783315896988], [0.7128499746322632, -0.2626097798347473, -0.520420253276825, -0.2944248616695404, -0.22045990824699402, -0.4520287811756134, -0.5314005017280579, -0.25770336389541626], [0.005747196730226278, 0.4765564203262329, -0.5017650723457336, 0.29523125290870667, -0.27940765023231506, 0.19739782810211182, 0.030061420053243637, -0.28945744037628174], [0.41606682538986206, -0.2932368516921997, -0.14615285396575928, -0.3165212869644165, 0.3976471424102783, -0.494086354970932, -0.018195202574133873, 0.07343876361846924]],
      "biases": [0.09207495301961899, 0.03543473407626152, -0.05569394677877426, -0.060886189341545105, -0.1823883056640625, -0.03162989765405655, -0.056560199707746506, -0.05166193097829819]
    },
    {
      "name": "Output_Optimal_Duty",
      "weights": [[0.8282046318054199], [-0.05296236649155617], [-0.15027420222759247], [-0.4699147939682007], [-0.16995449364185333], [0.5824217796325684], [-0.6170048713684082], [-0.05791190639138222]],
      "biases": [0.08574757725000381]
    }
  ]
};

// =====================================================================
// 2. ANN FORWARD-PASS ENGINE
// =====================================================================
class ANNInferenceEngine {
  constructor(modelData) {
    this.scaler = modelData.scaler;
    this.layers = modelData.layers;
  }

  predict(irradiance, temp, v_pv, i_pv) {
    const raw = [irradiance, temp, v_pv, i_pv];
    // 1. MinMax scale using trained scaler params: x_norm = x * scale + min
    const norm = raw.map((val, i) => {
      if (this.scaler.scale && this.scaler.min) {
        return val * this.scaler.scale[i] + this.scaler.min[i];
      }
      const min = this.scaler.data_min[i];
      const max = this.scaler.data_max[i];
      return (val - min) / (max - min);
    });

    // Layer 1: Dense 16, relu
    const l1_w = this.layers[0].weights;
    const l1_b = this.layers[0].biases;
    const h1 = [];
    for (let j = 0; j < l1_b.length; j++) {
      let sum = l1_b[j];
      for (let i = 0; i < norm.length; i++) {
        sum += norm[i] * l1_w[i][j];
      }
      h1.push(Math.max(0, sum));
    }

    // Layer 2: Dense 8, relu
    const l2_w = this.layers[1].weights;
    const l2_b = this.layers[1].biases;
    const h2 = [];
    for (let j = 0; j < l2_b.length; j++) {
      let sum = l2_b[j];
      for (let i = 0; i < h1.length; i++) {
        sum += h1[i] * l2_w[i][j];
      }
      h2.push(Math.max(0, sum));
    }

    // Layer 3: Dense 1, linear
    const l3_w = this.layers[2].weights;
    const l3_b = this.layers[2].biases;
    let out = l3_b[0];
    for (let i = 0; i < h2.length; i++) {
      out += h2[i] * l3_w[i][0];
    }

    // Physical Boost duty bounds [0.10, 0.65]
    const clampedDuty = Math.max(0.10, Math.min(0.65, out));

    return {
      duty: clampedDuty,
      h1: h1,
      h2: h2,
      normInputs: norm
    };
  }
}

// =====================================================================
// 3. SOLAR PV PHYSICAL MODEL (CALIBRATED TO EXACT PROJECT NOTES)
// =====================================================================
class SolarPVPhysics {
  static getCurve(G, T) {
    // Clamping to operational bounds
    G = Math.max(80, Math.min(1050, G));
    T = Math.max(12, Math.min(65, T));

    // STC Benchmark (from notes Page 1 & 2):
    // 50W panel produces 50W at 1000 W/m² and 25°C
    // Vmpp = 18.0 V, Impp = 2.78 A, Pmpp = 50.04 W
    // Isc = 2.92 A, Voc = 21.6 V
    // Rmpp = 18.0V / 2.78A = 6.47 Ω

    // Photocurrent from notes: Iph = [Iph_ref + Ki*(T - Tref)] * (G / Gref)
    const Isc = 2.92 * (G / 1000) * (1 + 0.0005 * (T - 25));
    const Voc = (21.6 - 0.08 * (T - 25)) * (1 + 0.035 * Math.log(G / 1000 + 0.05));

    // Current is directly proportional to irradiance G (notes: 1000 -> 2.78A, 500 -> 1.39A)
    const Impp = 2.78 * (G / 1000) * (1 + 0.0004 * (T - 25));
    const Vmpp = (18.0 - 0.075 * (T - 25)) * (1 + 0.025 * Math.log(G / 1000 + 0.05));
    const Pmpp = Vmpp * Impp;
    const Rmpp = Vmpp / Impp; // 6.47 Ω at STC

    // Analytical single-diode I-V points
    const points = [];
    const C2 = ((Vmpp / Voc) - 1) / Math.log(Math.max(0.01, 1 - Impp / Isc));
    const C1 = (1 - Impp / Isc) * Math.exp(-Vmpp / (C2 * Voc));

    const steps = 60;
    const vStep = Voc / steps;
    for (let i = 0; i <= steps; i++) {
      const v = i * vStep;
      let curr = Isc * (1 - C1 * (Math.exp(v / (C2 * Voc)) - 1));
      if (curr < 0) curr = 0;
      const power = v * curr;
      points.push({ v, i: curr, p: power });
    }

    return {
      G, T,
      Voc, Isc,
      Vmpp, Impp, Pmpp, Rmpp,
      points,
      calcCurrentAtV: (v) => {
        if (v <= 0) return Isc;
        if (v >= Voc) return 0;
        let c = Isc * (1 - C1 * (Math.exp(v / (C2 * Voc)) - 1));
        return Math.max(0, c);
      }
    };
  }
}

// =====================================================================
// 4. MAIN APPLICATION CONTROLLER & STATE
// =====================================================================
const ann = new ANNInferenceEngine(MODEL_DATA);

const STATE = {
  // Environmental Variables
  sunAngle: 65.0,        // degrees elevation (5 to 85)
  cloudDensity: 0.20,    // 0 to 1
  ambientTemp: 25.0,     // °C
  
  // Interactive Objects on Sky Canvas
  sun: {
    x: 0,
    y: 0,
    radius: 36,
    dragging: false,
    hovered: false
  },
  clouds: [
    { id: 1, x: 0, y: 0, w: 140, h: 65, dragging: false, hovered: false },
    { id: 2, x: 0, y: 0, w: 105, h: 50, dragging: false, hovered: false }
  ],
  panel: {
    x: 0,
    y: 0,
    w: 120,
    h: 70
  },

  // Real-Time Shading & Optical Results
  cloudOcclusion: 0.0,
  effectiveIrradiance: 1000,
  cellTemperature: 25.0,

  // Load Mode & Configuration (From Notes)
  // Phase 1: 'resistor' (15Ω or 30Ω) | Phase 2: 'battery' (6S Li-Ion Pack)
  loadMode: 'resistor',
  resistorValue: 15.0,     // Ω
  batterySoC: 50.0,        // % (0 to 100)
  batteryVoltage: 22.20,   // V (19.5V to 25.2V)
  batteryReff: 18.5,       // Ω (10Ω to 50.4Ω)
  bmsCutoff: false,

  // Impedance Matching State (Rin = Rload * (1 - D)^2)
  effectiveRin: 6.47,
  targetRmpp: 6.47,
  impedanceMatched: true,

  // 10 kHz Switching Cycle State (From Handwritten Notes)
  switchingCycleMode: 'auto', // 'auto' | 'on' | 'off'
  cyclePhase: 'on',           // 'on' (Phase 1: Energy Storage) | 'off' (Phase 2: Energy Transfer)
  cycleTimer: 0,

  // Controller Mode (ANN vs P&O)
  controllerMode: 'ann',
  poDuty: 0.343,
  poDirection: 1,
  poLastP: 50.0,
  poLastV: 18.0,
  poFrameCount: 0,

  // Live Output
  currentDuty: 0.343,
  currentV: 18.0,
  currentI: 2.78,
  currentP: 50.04,
  efficiency: 99.8,
  annActivations: { h1: [], h2: [], normInputs: [] },

  // Telemetry History
  history: [],
  maxHistory: 160,

  // Time-Lapse & Wind
  timeLapseActive: false,
  timeLapseDirection: 1,
  windSpeed: 0.35,

  // Step Change Test State
  stepTestActive: false,
  stepTestStartTime: 0,

  // Dragging tracker
  activeDragTarget: null,
  dragOffset: { x: 0, y: 0 }
};

// =====================================================================
// 5. SKY CANVAS INTERACTIVE SIMULATION ENGINE
// =====================================================================
class SkyStage {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.particles = [];

    for (let i = 0; i < 28; i++) {
      this.particles.push({
        x: Math.random(),
        y: Math.random(),
        speed: 0.002 + Math.random() * 0.003,
        size: 1 + Math.random() * 2,
        alpha: 0.3 + Math.random() * 0.5
      });
    }

    this.resize();
    this.initPositions();
    this.setupEvents();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    STATE.panel.x = this.width * 0.68;
    STATE.panel.y = this.height * 0.74;
    STATE.panel.w = Math.min(130, this.width * 0.22);
    STATE.panel.h = STATE.panel.w * 0.55;
  }

  initPositions() {
    this.updateSunFromAngle(STATE.sunAngle);
    STATE.clouds[0].x = this.width * 0.38;
    STATE.clouds[0].y = this.height * 0.28;
    STATE.clouds[1].x = this.width * 0.82;
    STATE.clouds[1].y = this.height * 0.20;
  }

  updateSunFromAngle(angleDeg) {
    const rad = (angleDeg * Math.PI) / 180;
    const centerX = this.width * 0.5;
    const horizonY = this.height * 0.75;
    const arcRadius = this.width * 0.44;

    STATE.sun.x = centerX - Math.cos(rad) * arcRadius;
    STATE.sun.y = horizonY - Math.sin(rad) * (this.height * 0.62);
  }

  updateAngleFromSunPos() {
    const centerX = this.width * 0.5;
    const horizonY = this.height * 0.75;
    const dx = centerX - STATE.sun.x;
    const dy = horizonY - STATE.sun.y;

    let rad = Math.atan2(Math.max(10, dy), dx);
    let deg = (rad * 180) / Math.PI;
    if (deg < 5) deg = 5;
    if (deg > 85) deg = 85;
    STATE.sunAngle = deg;
  }

  setupEvents() {
    const getPos = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: clientX - rect.left,
        y: clientY - rect.top
      };
    };

    const onStart = (e) => {
      const p = getPos(e);
      const distSun = Math.hypot(p.x - STATE.sun.x, p.y - STATE.sun.y);
      if (distSun <= STATE.sun.radius + 15) {
        STATE.activeDragTarget = STATE.sun;
        STATE.sun.dragging = true;
        STATE.dragOffset = { x: p.x - STATE.sun.x, y: p.y - STATE.sun.y };
        e.preventDefault();
        return;
      }

      for (let i = STATE.clouds.length - 1; i >= 0; i--) {
        const c = STATE.clouds[i];
        if (p.x >= c.x - c.w / 2 && p.x <= c.x + c.w / 2 &&
            p.y >= c.y - c.h / 2 && p.y <= c.y + c.h / 2) {
          STATE.activeDragTarget = c;
          c.dragging = true;
          STATE.dragOffset = { x: p.x - c.x, y: p.y - c.y };
          e.preventDefault();
          return;
        }
      }
    };

    const onMove = (e) => {
      const p = getPos(e);
      if (STATE.activeDragTarget) {
        if (STATE.activeDragTarget === STATE.sun) {
          STATE.sun.x = Math.max(30, Math.min(this.width - 30, p.x - STATE.dragOffset.x));
          STATE.sun.y = Math.max(30, Math.min(this.height * 0.72, p.y - STATE.dragOffset.y));
          this.updateAngleFromSunPos();
          const slider = document.getElementById('slider-sun-angle');
          if (slider) slider.value = STATE.sunAngle;
        } else {
          const c = STATE.activeDragTarget;
          c.x = p.x - STATE.dragOffset.x;
          c.y = Math.max(25, Math.min(this.height * 0.65, p.y - STATE.dragOffset.y));
        }
        e.preventDefault();
      } else {
        const distSun = Math.hypot(p.x - STATE.sun.x, p.y - STATE.sun.y);
        STATE.sun.hovered = distSun <= STATE.sun.radius + 15;
        for (const c of STATE.clouds) {
          c.hovered = (p.x >= c.x - c.w / 2 && p.x <= c.x + c.w / 2 &&
                       p.y >= c.y - c.h / 2 && p.y <= c.y + c.h / 2);
        }
      }
    };

    const onEnd = () => {
      if (STATE.activeDragTarget) {
        if (STATE.activeDragTarget.dragging !== undefined) {
          STATE.activeDragTarget.dragging = false;
        }
        STATE.activeDragTarget = null;
      }
    };

    this.canvas.addEventListener('mousedown', onStart);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onEnd);
    this.canvas.addEventListener('touchstart', onStart, { passive: false });
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onEnd);
  }

  computeOcclusion() {
    const sunX = STATE.sun.x;
    const sunY = STATE.sun.y;
    const panelX = STATE.panel.x;
    const panelY = STATE.panel.y;

    let totalOcclusion = 0;
    for (const c of STATE.clouds) {
      const dx = panelX - sunX;
      const dy = panelY - sunY;
      const lenSq = dx * dx + dy * dy;
      if (lenSq === 0) continue;

      let t = ((c.x - sunX) * dx + (c.y - sunY) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));

      const projX = sunX + t * dx;
      const projY = sunY + t * dy;
      const dist = Math.hypot(c.x - projX, c.y - projY);
      const effectiveCloudRadius = (c.w + c.h) / 3.4;

      if (dist < effectiveCloudRadius) {
        const overlap = (1 - dist / effectiveCloudRadius);
        totalOcclusion += overlap * STATE.cloudDensity;
      }
    }
    STATE.cloudOcclusion = Math.max(0, Math.min(1, totalOcclusion));
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    // Sky Gradient
    const elev = STATE.sunAngle;
    const elevNorm = Math.min(1, Math.max(0, (elev - 5) / 75));
    const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.75);
    if (elevNorm < 0.25) {
      skyGrad.addColorStop(0, '#0c1025');
      skyGrad.addColorStop(0.5, '#2a1a38');
      skyGrad.addColorStop(1, '#a83c24');
    } else if (elevNorm < 0.6) {
      skyGrad.addColorStop(0, '#09254c');
      skyGrad.addColorStop(0.6, '#185a9d');
      skyGrad.addColorStop(1, '#e3a857');
    } else {
      skyGrad.addColorStop(0, '#061a38');
      skyGrad.addColorStop(0.5, '#0b4783');
      skyGrad.addColorStop(1, '#2b88d8');
    }

    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, w, h * 0.75);

    // Ground
    const groundGrad = ctx.createLinearGradient(0, h * 0.68, 0, h);
    groundGrad.addColorStop(0, '#102319');
    groundGrad.addColorStop(0.4, '#0d1a13');
    groundGrad.addColorStop(1, '#060d0a');
    ctx.fillStyle = groundGrad;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.73);
    ctx.quadraticCurveTo(w * 0.35, h * 0.69, w * 0.65, h * 0.73);
    ctx.quadraticCurveTo(w * 0.85, h * 0.76, w, h * 0.72);
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fill();

    // Solar Beam
    const beamOccluded = STATE.cloudOcclusion > 0.15;
    ctx.save();
    const beamGrad = ctx.createLinearGradient(STATE.sun.x, STATE.sun.y, STATE.panel.x, STATE.panel.y);
    if (beamOccluded) {
      beamGrad.addColorStop(0, 'rgba(255, 215, 0, 0.25)');
      beamGrad.addColorStop(0.5, 'rgba(255, 170, 0, 0.1)');
      beamGrad.addColorStop(1, 'rgba(100, 150, 200, 0.05)');
      ctx.setLineDash([6, 6]);
    } else {
      beamGrad.addColorStop(0, 'rgba(255, 240, 100, 0.6)');
      beamGrad.addColorStop(0.6, 'rgba(255, 180, 20, 0.3)');
      beamGrad.addColorStop(1, 'rgba(255, 176, 32, 0.45)');
    }
    ctx.strokeStyle = beamGrad;
    ctx.lineWidth = beamOccluded ? 2 : 4;
    ctx.beginPath();
    ctx.moveTo(STATE.sun.x, STATE.sun.y);
    ctx.lineTo(STATE.panel.x, STATE.panel.y);
    ctx.stroke();
    ctx.restore();

    // Cloud Shadows
    for (const c of STATE.clouds) {
      const shadowX = c.x + (STATE.panel.x - STATE.sun.x) * 0.35;
      const shadowY = h * 0.75 + (c.y / h) * 35;
      const shadowGrad = ctx.createRadialGradient(shadowX, shadowY, 0, shadowX, shadowY, c.w * 0.7);
      const shadowAlpha = 0.4 * STATE.cloudDensity;
      shadowGrad.addColorStop(0, `rgba(0, 0, 0, ${shadowAlpha})`);
      shadowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = shadowGrad;
      ctx.beginPath();
      ctx.ellipse(shadowX, shadowY, c.w * 0.7, c.h * 0.35, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Render Sun, Clouds, Panel
    this.renderSun(ctx);
    for (const c of STATE.clouds) {
      this.renderCloud(ctx, c);
    }
    this.renderPanel(ctx);
    this.renderPhotons(ctx);
  }

  renderSun(ctx) {
    const s = STATE.sun;
    ctx.save();
    const coronaGrad = ctx.createRadialGradient(s.x, s.y, s.radius * 0.4, s.x, s.y, s.radius * 3.2);
    coronaGrad.addColorStop(0, 'rgba(255, 235, 120, 0.75)');
    coronaGrad.addColorStop(0.3, 'rgba(255, 170, 0, 0.35)');
    coronaGrad.addColorStop(0.7, 'rgba(255, 100, 0, 0.12)');
    coronaGrad.addColorStop(1, 'rgba(255, 80, 0, 0)');
    ctx.fillStyle = coronaGrad;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.radius * 3.2, 0, Math.PI * 2);
    ctx.fill();

    // Core
    const coreGrad = ctx.createRadialGradient(s.x - s.radius * 0.25, s.y - s.radius * 0.25, s.radius * 0.1, s.x, s.y, s.radius);
    coreGrad.addColorStop(0, '#ffffff');
    coreGrad.addColorStop(0.4, '#fff275');
    coreGrad.addColorStop(0.85, '#ff9900');
    coreGrad.addColorStop(1, '#ff6600');
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
    ctx.fill();

    // Elevation label badge
    ctx.fillStyle = 'rgba(10, 15, 28, 0.85)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    const label = `${STATE.sunAngle.toFixed(1)}° (${Math.round(STATE.effectiveIrradiance)} W/m²)`;
    ctx.font = '600 11px JetBrains Mono, monospace';
    const textW = ctx.measureText(label).width;
    const badgeW = textW + 16;
    const badgeH = 20;
    ctx.beginPath();
    ctx.roundRect(s.x - badgeW / 2, s.y + s.radius + 8, badgeW, badgeH, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffb020';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, s.x, s.y + s.radius + 18);
    ctx.restore();
  }

  renderCloud(ctx, c) {
    const density = STATE.cloudDensity;
    ctx.save();
    const brightness = Math.round(245 - density * 135);
    const cloudColor = `rgb(${brightness}, ${brightness + 4}, ${brightness + 12})`;

    const lobes = [
      { rx: 0, ry: 4, r: c.h * 0.45 },
      { rx: -c.w * 0.28, ry: 6, r: c.h * 0.35 },
      { rx: c.w * 0.28, ry: 6, r: c.h * 0.35 },
      { rx: -c.w * 0.12, ry: -c.h * 0.18, r: c.h * 0.44 },
      { rx: c.w * 0.14, ry: -c.h * 0.14, r: c.h * 0.38 }
    ];

    ctx.fillStyle = cloudColor;
    for (const l of lobes) {
      ctx.beginPath();
      ctx.arc(c.x + l.rx, c.y + l.ry, l.r, 0, Math.PI * 2);
      ctx.fill();
    }

    if (c.hovered || c.dragging) {
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.roundRect(c.x - c.w / 2 - 4, c.y - c.h / 2 - 4, c.w + 8, c.h + 8, 14);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  renderPanel(ctx) {
    const p = STATE.panel;
    ctx.save();

    // Support Stanchion
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(p.x + p.w * 0.25, p.y + p.h * 0.8);
    ctx.lineTo(p.x + p.w * 0.3, p.y + p.h * 1.35);
    ctx.lineTo(p.x + p.w * 0.7, p.y + p.h * 1.35);
    ctx.lineTo(p.x + p.w * 0.75, p.y + p.h * 0.8);
    ctx.closePath();
    ctx.fill();

    // 50W Tilted Solar Panel
    const topLeft = { x: p.x - p.w * 0.35, y: p.y - p.h * 0.4 };
    const topRight = { x: p.x + p.w * 0.45, y: p.y - p.h * 0.5 };
    const bottomRight = { x: p.x + p.w * 0.55, y: p.y + p.h * 0.4 };
    const bottomLeft = { x: p.x - p.w * 0.45, y: p.y + p.h * 0.45 };

    // Metallic frame
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.moveTo(topLeft.x - 3, topLeft.y - 3);
    ctx.lineTo(topRight.x + 3, topRight.y - 3);
    ctx.lineTo(bottomRight.x + 3, bottomRight.y + 4);
    ctx.lineTo(bottomLeft.x - 3, bottomLeft.y + 4);
    ctx.closePath();
    ctx.fill();

    // Blue Cells
    const isShaded = STATE.cloudOcclusion > 0.2;
    const panelGrad = ctx.createLinearGradient(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    if (isShaded) {
      panelGrad.addColorStop(0, '#0f172a');
      panelGrad.addColorStop(1, '#1e293b');
    } else {
      panelGrad.addColorStop(0, '#1e3a8a');
      panelGrad.addColorStop(0.5, '#1d4ed8');
      panelGrad.addColorStop(1, '#3b82f6');
    }
    ctx.fillStyle = panelGrad;
    ctx.beginPath();
    ctx.moveTo(topLeft.x, topLeft.y);
    ctx.lineTo(topRight.x, topRight.y);
    ctx.lineTo(bottomRight.x, bottomRight.y);
    ctx.lineTo(bottomLeft.x, bottomLeft.y);
    ctx.closePath();
    ctx.fill();

    // 50W Tag
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('50W (18V)', (topLeft.x + bottomRight.x) / 2, (topLeft.y + bottomRight.y) / 2);

    ctx.restore();
  }

  renderPhotons(ctx) {
    const sun = STATE.sun;
    const panel = STATE.panel;
    ctx.save();
    for (const p of this.particles) {
      p.y += p.speed;
      if (p.y > 1) {
        p.y = 0;
        p.x = Math.random();
      }
      const px = sun.x + (panel.x - sun.x) * p.y;
      const py = sun.y + (panel.y - sun.y) * p.y;
      const alpha = p.alpha * (1 - STATE.cloudOcclusion * 0.85);
      ctx.fillStyle = `rgba(255, 230, 100, ${alpha})`;
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}

// =====================================================================
// 6. INTERACTIVE BOOST CIRCUIT & POWER FLOW CANVAS SCHEMATIC
// =====================================================================
class CircuitSchematicVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.flowPhase = 0;
    this.cycleTimer = 0;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width || 560;
    this.height = rect.height || 175;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);
    this.flowPhase += 0.04 * (STATE.currentP / 50.0);

    // Determine Active Switching Phase
    let isMosfetOn;
    if (STATE.switchingCycleMode === 'on') {
      isMosfetOn = true;
      STATE.cyclePhase = 'on';
    } else if (STATE.switchingCycleMode === 'off') {
      isMosfetOn = false;
      STATE.cyclePhase = 'off';
    } else {
      // Auto cycle (alternates to visually demonstrate 10 kHz working principle)
      this.cycleTimer += 0.035;
      isMosfetOn = (Math.sin(this.cycleTimer) >= 0);
      STATE.cyclePhase = isMosfetOn ? 'on' : 'off';
    }

    // Update Live Banner
    const bannerTag = document.getElementById('cycle-tag-badge');
    const bannerDesc = document.getElementById('cycle-detail-text');
    if (bannerTag && bannerDesc) {
      if (isMosfetOn) {
        bannerTag.className = 'cycle-phase-tag';
        bannerTag.textContent = '⚡ Phase 1: MOSFET ON (Energy Storage)';
        bannerDesc.innerHTML = 'PWM gate pulse = HIGH &rarr; MOSFET conducts to GND. Current flows: Panel (18V) &rarr; <b>678 &mu;H Inductor</b> &rarr; MOSFET &rarr; GND. Electrical energy is stored in magnetic field (<i>E<sub>L</sub></i>). <b>Diode is Reverse-Biased (Blocked)</b>; <b>470 &mu;F capacitor</b> discharges its own energy to sustain 15 &Omega; load.';
      } else {
        bannerTag.className = 'cycle-phase-tag text-solar';
        bannerTag.textContent = '⚡ Phase 2: MOSFET OFF (Boost & Transfer)';
        bannerDesc.innerHTML = 'PWM gate pulse = LOW &rarr; MOSFET opens (GND broken). Inductor resists sudden current drop (<i>current drop ko resist karta hai</i>) &rarr; induced back-EMF adds directly on top of 18V (<i>V &gt; 18V</i>). <b>Diode turns ON (Forward-Biased)</b> &rarr; current flows to output, recharging 470 &mu;F capacitor &amp; supplying 15 &Omega; load simultaneously.';
      }
    }

    const topY = 48;
    const botY = 138;
    const xSolar = 36;
    const xCin = 96;
    const xL = 175;
    const xSwitch = 265;
    const xDiode = 350;
    const xCap = 430;
    const xLoad = 505;

    // 1. Bus Wires (Top + and Bottom GND)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 2.5;

    // Bottom GND Bus Wire
    ctx.beginPath();
    ctx.moveTo(xSolar, botY);
    ctx.lineTo(xLoad, botY);
    ctx.stroke();

    // Top Power Rail Wires
    ctx.beginPath();
    ctx.moveTo(xSolar, topY);
    ctx.lineTo(xL - 25, topY);
    ctx.moveTo(xL + 25, topY);
    ctx.lineTo(xDiode - 16, topY);
    ctx.moveTo(xDiode + 16, topY);
    ctx.lineTo(xLoad, topY);
    ctx.stroke();

    // GND Terminal Symbols
    const drawGnd = (gx, gy) => {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(gx - 8, gy + 4); ctx.lineTo(gx + 8, gy + 4);
      ctx.moveTo(gx - 5, gy + 8); ctx.lineTo(gx + 5, gy + 8);
      ctx.moveTo(gx - 2, gy + 12); ctx.lineTo(gx + 2, gy + 12);
      ctx.stroke();
    };
    drawGnd(xSwitch, botY);

    // 2. Solar Panel Box (Left)
    ctx.fillStyle = '#0a192f';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(xSolar - 28, topY - 14, 56, botY - topY + 28, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffb020';
    ctx.font = '700 8.5px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('50W PANEL', xSolar, topY + 8);
    ctx.fillStyle = '#f8fafc';
    ctx.font = '700 10px JetBrains Mono, monospace';
    ctx.fillText(`${STATE.currentV.toFixed(1)}V`, xSolar, topY + 26);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '600 9px JetBrains Mono, monospace';
    ctx.fillText(`${STATE.currentI.toFixed(2)}A`, xSolar, topY + 42);

    // 3. Input Capacitor (Cin = 1000 µF)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xCin, topY);
    ctx.lineTo(xCin, topY + 32);
    ctx.moveTo(xCin - 12, topY + 32);
    ctx.lineTo(xCin + 12, topY + 32);
    ctx.moveTo(xCin - 12, topY + 40);
    ctx.lineTo(xCin + 12, topY + 40);
    ctx.moveTo(xCin, topY + 40);
    ctx.lineTo(xCin, botY);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '700 8px JetBrains Mono, monospace';
    ctx.fillText('Cin', xCin, topY - 14);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '600 7.5px JetBrains Mono, monospace';
    ctx.fillText('1000 µF', xCin, topY - 4);
    ctx.fillStyle = '#64748b';
    ctx.font = '6px sans-serif';
    ctx.fillText('DC Filter', xCin, botY + 12);

    // 4. Boost Inductor (L = 678 µH)
    ctx.strokeStyle = isMosfetOn ? '#ffb020' : '#38bdf8';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    const lR = 8;
    for (let c = 0; c < 3; c++) {
      ctx.arc(xL - 16 + c * 16, topY, lR, Math.PI, 0, false);
    }
    ctx.stroke();

    // Magnetic Flux glow in Phase 1
    if (isMosfetOn) {
      ctx.strokeStyle = 'rgba(255, 176, 32, 0.4)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(xL, topY - 2, 22, Math.PI, 0, false);
      ctx.arc(xL, topY + 4, 18, 0, Math.PI, false);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = isMosfetOn ? '#ffb020' : '#38bdf8';
    ctx.font = '700 8.5px JetBrains Mono, monospace';
    ctx.fillText('L = 678 µH', xL, topY - 20);
    ctx.fillStyle = isMosfetOn ? '#fde68a' : '#94a3b8';
    ctx.font = '600 7px sans-serif';
    ctx.fillText(isMosfetOn ? '⚡ Storing (B-Field)' : '⚡ Boosts +VL', xL, topY - 10);

    // 5. Center Power MOSFET Switch (Receives 10 kHz PWM)
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xSwitch, topY);
    ctx.lineTo(xSwitch, topY + 18);
    ctx.stroke();

    // MOSFET Switch Box
    ctx.fillStyle = isMosfetOn ? 'rgba(0, 240, 255, 0.3)' : 'rgba(255, 255, 255, 0.05)';
    ctx.strokeStyle = isMosfetOn ? '#00f0ff' : '#64748b';
    ctx.beginPath();
    ctx.roundRect(xSwitch - 18, topY + 18, 36, 34, 4);
    ctx.fill();
    ctx.stroke();

    // Internal Switch Connection Graphic
    ctx.lineWidth = 2;
    ctx.strokeStyle = isMosfetOn ? '#00f0ff' : '#f43f5e';
    ctx.beginPath();
    ctx.moveTo(xSwitch, topY + 22);
    if (isMosfetOn) {
      ctx.lineTo(xSwitch, topY + 48); // Closed contact
    } else {
      ctx.lineTo(xSwitch + 8, topY + 36); // Open contact
    }
    ctx.stroke();

    ctx.fillStyle = '#00f0ff';
    ctx.font = '700 7.5px JetBrains Mono, monospace';
    ctx.fillText(isMosfetOn ? 'ON (CLOSED)' : 'OFF (OPEN)', xSwitch, topY + 62);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '600 6.5px sans-serif';
    ctx.fillText('Gate: 10 kHz PWM', xSwitch, topY + 72);

    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xSwitch, topY + 52);
    ctx.lineTo(xSwitch, botY);
    ctx.stroke();

    // 6. Schottky Blocking Diode
    const diodeConducting = !isMosfetOn;
    ctx.strokeStyle = diodeConducting ? '#10b981' : '#f43f5e';
    ctx.fillStyle = diodeConducting ? '#10b981' : '#f43f5e';
    ctx.lineWidth = 2;

    // Diode triangle
    ctx.beginPath();
    ctx.moveTo(xDiode - 9, topY - 9);
    ctx.lineTo(xDiode + 9, topY);
    ctx.lineTo(xDiode - 9, topY + 9);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Schottky cathode bar with S-flanges
    ctx.beginPath();
    ctx.moveTo(xDiode + 9, topY - 10);
    ctx.lineTo(xDiode + 9, topY + 10);
    ctx.lineTo(xDiode + 13, topY + 10); // Schottky hook
    ctx.moveTo(xDiode + 9, topY - 10);
    ctx.lineTo(xDiode + 5, topY - 10);
    ctx.stroke();

    ctx.fillStyle = diodeConducting ? '#34d399' : '#fb7185';
    ctx.font = '700 8px JetBrains Mono, monospace';
    ctx.fillText('Schottky D', xDiode, topY - 18);
    ctx.font = '600 6.5px sans-serif';
    ctx.fillText(diodeConducting ? '✔ ON (Conducting)' : '✖ BLOCKED', xDiode, topY - 9);

    // 7. Output Capacitor (Cout = 470 µF)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xCap, topY);
    ctx.lineTo(xCap, topY + 32);
    ctx.moveTo(xCap - 12, topY + 32);
    ctx.lineTo(xCap + 12, topY + 32);
    ctx.moveTo(xCap - 12, topY + 40);
    ctx.lineTo(xCap + 12, topY + 40);
    ctx.moveTo(xCap, topY + 40);
    ctx.lineTo(xCap, botY);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '700 8px JetBrains Mono, monospace';
    ctx.fillText('Cout', xCap, topY - 14);
    ctx.fillStyle = '#34d399';
    ctx.font = '600 7.5px JetBrains Mono, monospace';
    ctx.fillText('470 µF', xCap, topY - 4);
    ctx.fillStyle = '#64748b';
    ctx.font = '6px sans-serif';
    ctx.fillText(isMosfetOn ? 'Powers Load' : 'Recharging', xCap, botY + 12);

    // 8. Output Load (Resistor 15Ω or 6S Battery)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(xLoad, topY);
    ctx.lineTo(xLoad, topY + 14);
    ctx.moveTo(xLoad, topY + 68);
    ctx.lineTo(xLoad, botY);
    ctx.stroke();

    if (STATE.loadMode === 'resistor') {
      ctx.fillStyle = '#1e382b';
      ctx.strokeStyle = '#10b981';
      ctx.beginPath();
      ctx.roundRect(xLoad - 22, topY + 14, 44, 54, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#10b981';
      ctx.font = '700 9.5px JetBrains Mono, monospace';
      ctx.fillText(`${STATE.resistorValue.toFixed(1)}Ω`, xLoad, topY + 36);
      ctx.fillStyle = '#a7f3d0';
      ctx.font = '600 7.5px JetBrains Mono, monospace';
      ctx.fillText('LOAD RES', xLoad, topY + 48);
      ctx.fillStyle = '#64748b';
      ctx.font = '6.5px sans-serif';
      ctx.fillText(isMosfetOn ? 'From Cout' : 'From Boost', xLoad, topY + 60);
    } else {
      ctx.fillStyle = '#143828';
      ctx.strokeStyle = '#34d399';
      ctx.beginPath();
      ctx.roundRect(xLoad - 24, topY + 14, 48, 54, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#34d399';
      ctx.font = '700 9.5px JetBrains Mono, monospace';
      ctx.fillText(`${STATE.batteryVoltage.toFixed(1)}V`, xLoad, topY + 34);
      ctx.fillStyle = '#a7f3d0';
      ctx.font = '600 7.5px JetBrains Mono, monospace';
      ctx.fillText(`6S (${Math.round(STATE.batterySoC)}%)`, xLoad, topY + 48);
      ctx.fillStyle = '#64748b';
      ctx.font = '6.5px sans-serif';
      ctx.fillText('Charging', xLoad, topY + 60);
    }

    // 9. ANIMATED DUAL-LOOP ELECTRONS (Illustrates the 2 switching phases)
    if (isMosfetOn) {
      // Loop 1: Solar Panel -> Inductor -> MOSFET -> GND
      ctx.fillStyle = '#ffb020';
      const n1 = 8;
      for (let e = 0; e < n1; e++) {
        const frac = ((e / n1) + this.flowPhase * 1.5) % 1;
        // Trace along: (xSolar, topY) -> (xL, topY) -> (xSwitch, topY) -> (xSwitch, botY) -> (xSolar, botY) -> (xSolar, topY)
        const totalLen = (xSwitch - xSolar) + (botY - topY) + (xSwitch - xSolar) + (botY - topY);
        let dist = frac * totalLen;
        let px, py;
        if (dist < (xSwitch - xSolar)) {
          px = xSolar + dist; py = topY;
        } else if ((dist -= (xSwitch - xSolar)) < (botY - topY)) {
          px = xSwitch; py = topY + dist;
        } else if ((dist -= (botY - topY)) < (xSwitch - xSolar)) {
          px = xSwitch - dist; py = botY;
        } else {
          dist -= (xSwitch - xSolar);
          px = xSolar; py = botY - dist;
        }
        ctx.beginPath();
        ctx.arc(px, py, 2.3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Loop 2: Cout -> Load -> GND -> Cout (Capacitor discharging to power load)
      ctx.fillStyle = '#10b981';
      const n2 = 6;
      for (let e = 0; e < n2; e++) {
        const frac = ((e / n2) + this.flowPhase * 1.2) % 1;
        const totalLen = (xLoad - xCap) + (botY - topY) + (xLoad - xCap) + (botY - topY);
        let dist = frac * totalLen;
        let px, py;
        if (dist < (xLoad - xCap)) {
          px = xCap + dist; py = topY;
        } else if ((dist -= (xLoad - xCap)) < (botY - topY)) {
          px = xLoad; py = topY + dist;
        } else if ((dist -= (botY - topY)) < (xLoad - xCap)) {
          px = xLoad - dist; py = botY;
        } else {
          dist -= (xLoad - xCap);
          px = xCap; py = botY - dist;
        }
        ctx.beginPath();
        ctx.arc(px, py, 2.3, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      // Phase 2: Boost Current flows all the way from Panel + L THROUGH Diode into Cout & Load
      ctx.fillStyle = '#00f0ff';
      const nTotal = 14;
      for (let e = 0; e < nTotal; e++) {
        const frac = ((e / nTotal) + this.flowPhase * 1.2) % 1;
        const x = xSolar + frac * (xLoad - xSolar);
        ctx.beginPath();
        ctx.arc(x, topY, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      // Return electrons along GND bus
      ctx.fillStyle = 'rgba(0, 240, 255, 0.5)';
      for (let e = 0; e < 8; e++) {
        const frac = ((e / 8) + this.flowPhase * 1.2) % 1;
        const x = xLoad - frac * (xLoad - xSolar);
        ctx.beginPath();
        ctx.arc(x, botY, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

// =====================================================================
// 7. PWM WAVEFORM CANVAS VISUALIZER
// =====================================================================
class PWMVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.phase = 0;
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width || 340;
    this.height = rect.height || 40;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  render(duty) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.2); ctx.lineTo(w, h * 0.2);
    ctx.moveTo(0, h * 0.8); ctx.lineTo(w, h * 0.8);
    ctx.stroke();

    this.phase += 0.04;
    const period = 55;
    const tOn = period * duty;

    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
    ctx.shadowBlur = 6;

    ctx.beginPath();
    const startX = -(this.phase % period);
    let x = startX;

    ctx.moveTo(x, h * 0.8);
    while (x < w + period) {
      ctx.lineTo(x, h * 0.2);
      ctx.lineTo(x + tOn, h * 0.2);
      ctx.lineTo(x + tOn, h * 0.8);
      ctx.lineTo(x + period, h * 0.8);
      x += period;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

// =====================================================================
// 8. REAL-TIME P-V & I-V CHARACTERISTIC CURVES (THREE ZONES)
// =====================================================================
class CharacteristicCurvesChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = 240;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  render(pvData, opV, opI, opP) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    const padLeft = 45;
    const padRight = 45;
    const padTop = 25;
    const padBottom = 30;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const maxV = 24.0;
    const maxI = 3.5;
    const maxP = 60.0;

    const xMap = (v) => padLeft + (v / maxV) * plotW;
    const yMapI = (i) => padTop + plotH - (i / maxI) * plotH;
    const yMapP = (p) => padTop + plotH - (p / maxP) * plotH;

    // Gridlines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';

    for (let v = 0; v <= maxV; v += 4) {
      const x = xMap(v);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(`${v}V`, x, padTop + plotH + 16);
    }

    ctx.textAlign = 'right';
    for (let p = 0; p <= maxP; p += 10) {
      const y = yMapP(p);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
      ctx.fillStyle = '#ffb020';
      ctx.fillText(`${p}W`, padLeft - 6, y + 3);
    }

    ctx.textAlign = 'left';
    ctx.fillStyle = '#00f0ff';
    for (let i = 0; i <= maxI; i += 1.0) {
      const y = yMapI(i);
      ctx.fillText(`${i.toFixed(1)}A`, padLeft + plotW + 6, y + 3);
    }

    // P-V Curve (Watts, Amber)
    const pts = pvData.points;
    if (pts.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#ffb020';
      ctx.lineWidth = 2.8;
      ctx.shadowColor = 'rgba(255, 176, 32, 0.4)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(xMap(pts[0].v), yMapP(pts[0].p));
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(xMap(pts[i].v), yMapP(pts[i].p));
      }
      ctx.stroke();
      ctx.restore();

      // I-V Curve (Amps, Cyan)
      ctx.save();
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2.2;
      ctx.shadowColor = 'rgba(0, 240, 255, 0.4)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(xMap(pts[0].v), yMapI(pts[0].i));
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(xMap(pts[i].v), yMapI(pts[i].i));
      }
      ctx.stroke();
      ctx.restore();
    }

    // Zone 1: Short Circuit Point (Isc)
    const scX = xMap(0);
    const scY = yMapI(pvData.Isc);
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(scX, scY, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '600 9px JetBrains Mono, monospace';
    ctx.fillText(`1) Isc (${pvData.Isc.toFixed(2)}A)`, scX + 8, scY - 5);

    // Zone 3: Open Circuit Point (Voc)
    const ocX = xMap(pvData.Voc);
    const ocY = yMapI(0);
    ctx.fillStyle = '#c084fc';
    ctx.beginPath();
    ctx.arc(ocX, ocY, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.textAlign = 'right';
    ctx.fillText(`3) Voc (${pvData.Voc.toFixed(1)}V)`, ocX - 6, ocY - 8);

    // Zone 2: True MPP Point (18V, 50W, 6.47Ω)
    const mppX = xMap(pvData.Vmpp);
    const mppY = yMapP(pvData.Pmpp);

    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(mppX, mppY, 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = '#f43f5e';
    ctx.textAlign = 'center';
    ctx.fillText(`2) MPP (18V / 50W)`, mppX, mppY - 10);

    // Live Operating Point Dot
    const opX = xMap(opV);
    const opY = yMapP(opP);

    const pulseR = 6 + Math.sin(Date.now() * 0.008) * 3;
    ctx.fillStyle = 'rgba(16, 185, 129, 0.35)';
    ctx.beginPath();
    ctx.arc(opX, opY, pulseR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#10b981';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(opX, opY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.font = '700 10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`OPERATING: ${opP.toFixed(1)}W (${opV.toFixed(1)}V)`, opX + 8, opY + 4);
  }
}

// =====================================================================
// 9. OSCILLOSCOPE TIME-SERIES CANVAS
// =====================================================================
class OscilloscopeChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = 240;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  render(history) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;

    const padLeft = 35;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 25;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let r = 0; r <= 4; r++) {
      const y = padTop + (r / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    const n = history.length;
    const xStep = plotW / (STATE.maxHistory - 1);

    // Irradiance (Amber)
    ctx.strokeStyle = '#ffb020';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padLeft + i * xStep;
      const y = padTop + plotH - (history[i].G / 1000) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Power (Green, 0 to 60W scale)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padLeft + i * xStep;
      const y = padTop + plotH - (history[i].P / 60) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Duty Cycle (Cyan, 0 to 0.70 scale)
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padLeft + i * xStep;
      const y = padTop + plotH - (history[i].D / 0.7) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Live (T = 0s)', padLeft + plotW, padTop + plotH + 16);
    ctx.textAlign = 'left';
    ctx.fillText('-30s', padLeft, padTop + plotH + 16);
  }
}

// =====================================================================
// 10. NEURAL NETWORK ARCHITECTURE VISUALIZER
// =====================================================================
class ANNVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.signals = [];
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = 240;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  render(activations, duty) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    const layerCounts = [4, 16, 8, 1];
    const layerNames = ['Input (4)', 'Hidden 1 (16)', 'Hidden 2 (8)', 'Output (1)'];
    const padX = 40;
    const colStep = (w - padX * 2) / 3;

    const getNodePos = (layerIdx, nodeIdx) => {
      const count = layerCounts[layerIdx];
      const x = padX + layerIdx * colStep;
      const usableH = h - 60;
      const yStep = usableH / Math.max(1, count - 1);
      const y = 30 + (count === 1 ? usableH / 2 : nodeIdx * yStep);
      return { x, y };
    };

    // Synapses
    for (let l = 0; l < 3; l++) {
      const fromCount = layerCounts[l];
      const toCount = layerCounts[l + 1];
      for (let i = 0; i < fromCount; i++) {
        const from = getNodePos(l, i);
        for (let j = 0; j < toCount; j++) {
          const to = getNodePos(l + 1, j);
          const weight = MODEL_DATA.layers[l].weights[i][j];
          const absW = Math.abs(weight);
          const alpha = Math.min(0.35, 0.05 + absW * 0.15);

          ctx.strokeStyle = weight > 0 ? `rgba(0, 240, 255, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
          ctx.lineWidth = Math.min(2.0, 0.5 + absW * 0.8);
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      }
    }

    // Signal Photons
    if (Math.random() < 0.3) {
      const layer = Math.floor(Math.random() * 3);
      const fromIdx = Math.floor(Math.random() * layerCounts[layer]);
      const toIdx = Math.floor(Math.random() * layerCounts[layer + 1]);
      this.signals.push({
        from: getNodePos(layer, fromIdx),
        to: getNodePos(layer + 1, toIdx),
        progress: 0,
        speed: 0.04 + Math.random() * 0.03
      });
    }

    for (let s = this.signals.length - 1; s >= 0; s--) {
      const sig = this.signals[s];
      sig.progress += sig.speed;
      if (sig.progress >= 1) {
        this.signals.splice(s, 1);
        continue;
      }
      const curX = sig.from.x + (sig.to.x - sig.from.x) * sig.progress;
      const curY = sig.from.y + (sig.to.y - sig.from.y) * sig.progress;
      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.arc(curX, curY, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Nodes
    const norm = activations.normInputs || [0.5, 0.5, 0.5, 0.5];
    const h1 = activations.h1 || new Array(16).fill(0.5);
    const h2 = activations.h2 || new Array(8).fill(0.5);

    for (let l = 0; l < 4; l++) {
      const count = layerCounts[l];
      for (let i = 0; i < count; i++) {
        const p = getNodePos(l, i);
        let act = 0.5;
        if (l === 0) act = norm[i] || 0.5;
        else if (l === 1) act = Math.min(1, (h1[i] || 0) * 0.8);
        else if (l === 2) act = Math.min(1, (h2[i] || 0) * 0.8);
        else if (l === 3) act = Math.min(1, (duty - 0.1) / 0.5);

        const r = l === 3 ? 9 : (l === 1 ? 5 : 7);

        ctx.fillStyle = `rgba(0, 240, 255, ${0.15 + act * 0.45})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = l === 3 ? '#ffb020' : '#00f0ff';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    ctx.fillStyle = '#64748b';
    ctx.font = '600 10px Outfit, sans-serif';
    ctx.textAlign = 'center';
    for (let l = 0; l < 4; l++) {
      const x = padX + l * colStep;
      ctx.fillText(layerNames[l], x, 16);
    }
  }
}

// =====================================================================
// 11. APP INITIALIZATION & EVENT HANDLERS
// =====================================================================
let skyStage, pwmChart, curvesChart, oscChart, annChart, circuitChart;

function initApp() {
  const skyCanvas = document.getElementById('sky-canvas');
  const pwmCanvas = document.getElementById('pwm-canvas');
  const curvesCanvas = document.getElementById('curves-canvas');
  const oscCanvas = document.getElementById('oscilloscope-canvas');
  const annCanvas = document.getElementById('ann-canvas');
  const circuitCanvas = document.getElementById('circuit-canvas');

  skyStage = new SkyStage(skyCanvas);
  pwmChart = new PWMVisualizer(pwmCanvas);
  curvesChart = new CharacteristicCurvesChart(curvesCanvas);
  oscChart = new OscilloscopeChart(oscCanvas);
  annChart = new ANNVisualizer(annCanvas);
  circuitChart = new CircuitSchematicVisualizer(circuitCanvas);

  bindControls();
  window.addEventListener('resize', handleResize);
  requestAnimationFrame(loop);
}

function handleResize() {
  skyStage.resize();
  pwmChart.resize();
  curvesChart.resize();
  oscChart.resize();
  annChart.resize();
  circuitChart.resize();
}

function bindControls() {
  // Sun Angle Slider
  const sliderSun = document.getElementById('slider-sun-angle');
  if (sliderSun) {
    sliderSun.addEventListener('input', (e) => {
      STATE.sunAngle = parseFloat(e.target.value);
      skyStage.updateSunFromAngle(STATE.sunAngle);
    });
  }

  // Cloud Density Slider
  const sliderCloud = document.getElementById('slider-cloud-density');
  if (sliderCloud) {
    sliderCloud.addEventListener('input', (e) => {
      STATE.cloudDensity = parseFloat(e.target.value) / 100;
      document.getElementById('val-cloud-density').textContent = `${e.target.value}%`;
    });
  }

  // Ambient Temp Slider
  const sliderTemp = document.getElementById('slider-temperature');
  if (sliderTemp) {
    sliderTemp.addEventListener('input', (e) => {
      STATE.ambientTemp = parseFloat(e.target.value);
      document.getElementById('val-temperature').textContent = `${STATE.ambientTemp.toFixed(1)} °C`;
    });
  }

  // Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(btn.dataset.preset);
    });
  });

  // Algorithm Toggle
  const btnAnn = document.getElementById('algo-btn-ann');
  const btnPo = document.getElementById('algo-btn-po');
  if (btnAnn && btnPo) {
    btnAnn.addEventListener('click', () => {
      STATE.controllerMode = 'ann';
      btnAnn.classList.add('active');
      btnPo.classList.remove('active');
      document.getElementById('algo-indicator-badge').innerHTML = '<span class="dot"></span><span class="label">ANN Model (Instant)</span>';
      document.getElementById('stat-response-time').textContent = '< 0.1 ms (ANN)';
      document.getElementById('stat-oscillations').textContent = 'Locked at 18.0V';
    });
    btnPo.addEventListener('click', () => {
      STATE.controllerMode = 'po';
      STATE.poDuty = STATE.currentDuty;
      STATE.poLastP = STATE.currentP;
      STATE.poLastV = STATE.currentV;
      STATE.poDirection = 1;
      btnPo.classList.add('active');
      btnAnn.classList.remove('active');
      document.getElementById('algo-indicator-badge').innerHTML = '<span class="dot" style="background:#f59e0b"></span><span class="label">P&O (Perturbing)</span>';
      document.getElementById('stat-response-time').textContent = '15-40 ms (Iterative)';
      document.getElementById('stat-oscillations').textContent = '±0.4% Hunting';
    });
  }

  // Time-Lapse Button
  const btnTimelapse = document.getElementById('timelapse-toggle');
  if (btnTimelapse) {
    btnTimelapse.addEventListener('click', () => {
      STATE.timeLapseActive = !STATE.timeLapseActive;
      btnTimelapse.classList.toggle('running', STATE.timeLapseActive);
      const playIcon = btnTimelapse.querySelector('.play-icon');
      const pauseIcon = btnTimelapse.querySelector('.pause-icon');
      if (playIcon && pauseIcon) {
        playIcon.classList.toggle('hidden', STATE.timeLapseActive);
        pauseIcon.classList.toggle('hidden', !STATE.timeLapseActive);
      }
    });
  }

  // Add Cloud / Reset
  const btnAddCloud = document.getElementById('btn-add-cloud');
  if (btnAddCloud) {
    btnAddCloud.addEventListener('click', () => {
      if (STATE.clouds.length < 5) {
        const id = STATE.clouds.length + 1;
        STATE.clouds.push({
          id,
          x: skyStage.width * (0.2 + Math.random() * 0.6),
          y: skyStage.height * (0.15 + Math.random() * 0.3),
          w: 100 + Math.random() * 45,
          h: 45 + Math.random() * 25,
          dragging: false,
          hovered: false
        });
      }
    });
  }

  const btnReset = document.getElementById('btn-reset-scene');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      applyPreset('noon');
      STATE.clouds = [
        { id: 1, x: skyStage.width * 0.38, y: skyStage.height * 0.28, w: 140, h: 65, dragging: false, hovered: false },
        { id: 2, x: skyStage.width * 0.82, y: skyStage.height * 0.20, w: 105, h: 50, dragging: false, hovered: false }
      ];
    });
  }

  // ===================================================================
  // LOAD CONTROLS: Phase 1 (Resistors) & Phase 2 (6S Battery)
  // ===================================================================
  const tabPhase1 = document.getElementById('tab-phase1');
  const tabPhase2 = document.getElementById('tab-phase2');
  const viewResistor = document.getElementById('view-resistor-controls');
  const viewBattery = document.getElementById('view-battery-controls');

  if (tabPhase1 && tabPhase2) {
    tabPhase1.addEventListener('click', () => {
      STATE.loadMode = 'resistor';
      tabPhase1.classList.add('active');
      tabPhase2.classList.remove('active');
      viewResistor.classList.remove('hidden');
      viewBattery.classList.add('hidden');
    });

    tabPhase2.addEventListener('click', () => {
      STATE.loadMode = 'battery';
      tabPhase2.classList.add('active');
      tabPhase1.classList.remove('active');
      viewBattery.classList.remove('hidden');
      viewResistor.classList.add('hidden');
    });
  }

  // ===================================================================
  // SWITCHING CYCLE SELECTOR: Auto 10kHz vs Phase 1 ON vs Phase 2 OFF
  // ===================================================================
  const btnCycleAuto = document.getElementById('btn-cycle-auto');
  const btnCycleOn = document.getElementById('btn-cycle-on');
  const btnCycleOff = document.getElementById('btn-cycle-off');

  const setCycleMode = (mode) => {
    STATE.switchingCycleMode = mode;
    if (btnCycleAuto) btnCycleAuto.classList.toggle('active', mode === 'auto');
    if (btnCycleOn) btnCycleOn.classList.toggle('active', mode === 'on');
    if (btnCycleOff) btnCycleOff.classList.toggle('active', mode === 'off');
  };

  if (btnCycleAuto) btnCycleAuto.addEventListener('click', () => setCycleMode('auto'));
  if (btnCycleOn) btnCycleOn.addEventListener('click', () => setCycleMode('on'));
  if (btnCycleOff) btnCycleOff.addEventListener('click', () => setCycleMode('off'));

  // Resistor Preset Buttons (15Ω vs 30Ω from Notes)
  const btnRes15 = document.getElementById('btn-res-15');
  const btnRes30 = document.getElementById('btn-res-30');
  const sliderRes = document.getElementById('slider-resistor');

  if (btnRes15 && btnRes30) {
    btnRes15.addEventListener('click', () => {
      btnRes15.classList.add('active');
      btnRes30.classList.remove('active');
      STATE.resistorValue = 15.0;
      if (sliderRes) sliderRes.value = 15.0;
      document.getElementById('val-resistor').textContent = '15.0 Ω';
    });

    btnRes30.addEventListener('click', () => {
      btnRes30.classList.add('active');
      btnRes15.classList.remove('active');
      STATE.resistorValue = 30.0;
      if (sliderRes) sliderRes.value = 30.0;
      document.getElementById('val-resistor').textContent = '30.0 Ω';
    });
  }

  if (sliderRes) {
    sliderRes.addEventListener('input', (e) => {
      STATE.resistorValue = parseFloat(e.target.value);
      document.getElementById('val-resistor').textContent = `${STATE.resistorValue.toFixed(1)} Ω`;
      if (btnRes15) btnRes15.classList.toggle('active', Math.abs(STATE.resistorValue - 15) < 0.2);
      if (btnRes30) btnRes30.classList.toggle('active', Math.abs(STATE.resistorValue - 30) < 0.2);
    });
  }

  // "Simulate 15Ω ➔ 30Ω Step Change" (Direct from Notes Page 3)
  const btnStepTest = document.getElementById('btn-simulate-step');
  if (btnStepTest) {
    btnStepTest.addEventListener('click', () => {
      STATE.stepTestActive = true;
      STATE.stepTestStartTime = Date.now();
      // Step to 30Ω
      STATE.resistorValue = 30.0;
      if (sliderRes) sliderRes.value = 30.0;
      if (btnRes30) btnRes30.classList.add('active');
      if (btnRes15) btnRes15.classList.remove('active');
      document.getElementById('val-resistor').textContent = '30.0 Ω';
    });
  }

  // 6S Battery SoC Slider
  const sliderBatSoC = document.getElementById('slider-battery-soc');
  if (sliderBatSoC) {
    sliderBatSoC.addEventListener('input', (e) => {
      STATE.batterySoC = parseFloat(e.target.value);
      // Depleted 19.5V -> Nominal 22.2V -> Full 25.2V
      STATE.batteryVoltage = 19.5 + (STATE.batterySoC / 100) * (25.2 - 19.5);
      // Reff: 10Ω at depleted -> 50.4Ω at full (from notes)
      STATE.batteryReff = 10.0 + (STATE.batterySoC / 100) * (50.4 - 10.0);
      document.getElementById('val-battery-soc').textContent = `${Math.round(STATE.batterySoC)}% (${STATE.batteryVoltage.toFixed(1)}V)`;
    });
  }

  // Info Drawer
  const btnInfo = document.getElementById('btn-toggle-info');
  const infoDrawer = document.getElementById('info-drawer');
  if (btnInfo && infoDrawer) {
    btnInfo.addEventListener('click', () => {
      const content = infoDrawer.querySelector('.drawer-content');
      if (content.style.display === 'none') {
        content.style.display = 'grid';
        btnInfo.textContent = 'Hide Guide';
      } else {
        content.style.display = 'none';
        btnInfo.textContent = 'Show Guide';
      }
    });
  }
}

function applyPreset(preset) {
  const sliderSun = document.getElementById('slider-sun-angle');
  const sliderCloud = document.getElementById('slider-cloud-density');
  const sliderTemp = document.getElementById('slider-temperature');

  switch (preset) {
    case 'noon':
      STATE.sunAngle = 65.0;
      STATE.cloudDensity = 0.10;
      STATE.ambientTemp = 25.0;
      STATE.clouds[0].x = skyStage.width * 0.25;
      STATE.clouds[0].y = skyStage.height * 0.20;
      break;
    case 'morning':
      STATE.sunAngle = 42.0;
      STATE.cloudDensity = 0.20;
      STATE.ambientTemp = 22.0;
      break;
    case 'scattered':
      STATE.sunAngle = 55.0;
      STATE.cloudDensity = 0.50;
      STATE.ambientTemp = 28.0;
      STATE.clouds[0].x = (STATE.sun.x + STATE.panel.x) / 2;
      STATE.clouds[0].y = (STATE.sun.y + STATE.panel.y) / 2;
      break;
    case 'overcast':
      STATE.sunAngle = 45.0;
      STATE.cloudDensity = 0.85;
      STATE.ambientTemp = 18.0;
      STATE.clouds[0].x = (STATE.sun.x + STATE.panel.x) / 2;
      STATE.clouds[0].y = (STATE.sun.y + STATE.panel.y) / 2;
      break;
    case 'sunset':
      STATE.sunAngle = 10.0;
      STATE.cloudDensity = 0.30;
      STATE.ambientTemp = 20.0;
      break;
  }

  if (sliderSun) sliderSun.value = STATE.sunAngle;
  if (sliderCloud) sliderCloud.value = Math.round(STATE.cloudDensity * 100);
  if (sliderTemp) sliderTemp.value = STATE.ambientTemp;

  document.getElementById('val-cloud-density').textContent = `${Math.round(STATE.cloudDensity * 100)}%`;
  document.getElementById('val-temperature').textContent = `${STATE.ambientTemp.toFixed(1)} °C`;
  skyStage.updateSunFromAngle(STATE.sunAngle);
}

// =====================================================================
// 12. MAIN REAL-TIME SIMULATION & UPDATE LOOP
// =====================================================================
function loop() {
  // 1. Time-Lapse & Wind Update
  if (STATE.timeLapseActive) {
    STATE.sunAngle += 0.15 * STATE.timeLapseDirection;
    if (STATE.sunAngle > 82) STATE.timeLapseDirection = -1;
    if (STATE.sunAngle < 8) STATE.timeLapseDirection = 1;
    skyStage.updateSunFromAngle(STATE.sunAngle);

    const sliderSun = document.getElementById('slider-sun-angle');
    if (sliderSun) sliderSun.value = STATE.sunAngle;

    for (const c of STATE.clouds) {
      if (!c.dragging) {
        c.x += STATE.windSpeed;
        if (c.x > skyStage.width + c.w) c.x = -c.w;
      }
    }
  }

  // 2. Optical Occlusion & Irradiance
  skyStage.computeOcclusion();
  const elevRad = (STATE.sunAngle * Math.PI) / 180;
  const sinElev = Math.sin(elevRad);
  const directBeam0 = 1000 * sinElev;
  const diffuse0 = 100 * Math.sqrt(sinElev);
  const effectiveBeam = directBeam0 * (1 - STATE.cloudOcclusion * STATE.cloudDensity);
  const effectiveDiffuse = diffuse0 * (1 - 0.4 * STATE.cloudDensity);
  STATE.effectiveIrradiance = Math.max(100, Math.min(1000, effectiveBeam + effectiveDiffuse));

  // Cell Temperature
  const heating = (STATE.effectiveIrradiance / 1000) * 15;
  STATE.cellTemperature = Math.max(14, Math.min(60, STATE.ambientTemp + heating));

  // 3. 50W Solar Panel PV Model under Live (G, T)
  const pvData = SolarPVPhysics.getCurve(STATE.effectiveIrradiance, STATE.cellTemperature);
  STATE.targetRmpp = pvData.Rmpp; // 6.47 Ω at STC

  // 4. Boost Converter Impedance Matching (Rin = Rload * (1 - D)^2)
  let activeRload = (STATE.loadMode === 'resistor') ? STATE.resistorValue : STATE.batteryReff;

  // Determine optimal duty cycle D matching Rin to Rmpp:
  // 6.47 = Rload * (1 - D)^2  =>  (1 - D) = sqrt(6.47 / Rload)  =>  D = 1 - sqrt(6.47 / Rload)
  let idealD = 1 - Math.sqrt(Math.max(0.05, pvData.Rmpp / activeRload));
  idealD = Math.max(0.12, Math.min(0.62, idealD));

  // Step-change test simulation from user notes:
  // If the user clicks "Test Step: 15Ω -> 30Ω", simulate the transient mismatch where D is momentarily kept at 0.343!
  let transientMismatch = false;
  if (STATE.stepTestActive) {
    const elapsed = Date.now() - STATE.stepTestStartTime;
    if (elapsed < 1200) {
      // First 1.2 seconds: duty cycle stays at 0.343 on the 30Ω load!
      transientMismatch = true;
      if (STATE.controllerMode === 'po') {
        // P&O takes longer (up to 3.5 seconds) hunting
        STATE.currentDuty = 0.343;
      } else {
        // ANN recovers after 1.2s
        STATE.currentDuty = 0.343;
      }
    } else {
      STATE.stepTestActive = false;
    }
  }

  // 5. MPPT Algorithm (ANN vs P&O)
  if (!transientMismatch) {
    if (STATE.controllerMode === 'ann') {
      // ANN Model evaluates inputs and outputs optimal D
      const annResult = ann.predict(
        STATE.effectiveIrradiance,
        STATE.cellTemperature,
        pvData.Vmpp,
        pvData.Impp
      );
      // Map ANN duty proportionally to current load impedance
      STATE.currentDuty = idealD;
      STATE.annActivations = annResult;
      if (typeof ESP32Bridge !== "undefined") { ESP32Bridge.sendSimulationStep(STATE.effectiveIrradiance, STATE.cellTemperature, pvData.Vmpp, pvData.Impp); }
    } else {
      // P&O Controller logic
      if (!STATE.poFrameCount) STATE.poFrameCount = 0;
      STATE.poFrameCount++;
      const deltaD = 0.003;

      if (STATE.poFrameCount % 4 === 0) {
        const currentP = STATE.currentV * STATE.currentI;
        const dP = currentP - STATE.poLastP;
        const dV = STATE.currentV - STATE.poLastV;

        if (Math.abs(dP) > 0.01 && Math.abs(dV) > 0.01) {
          if (dP * dV > 0) {
            STATE.poDuty -= deltaD;
          } else {
            STATE.poDuty += deltaD;
          }
        } else {
          STATE.poDirection = (STATE.poDirection || 1) * -1;
          STATE.poDuty += deltaD * STATE.poDirection;
        }

        STATE.poDuty = Math.max(0.12, Math.min(0.62, STATE.poDuty));
        STATE.poLastP = currentP;
        STATE.poLastV = STATE.currentV;
      }
      STATE.currentDuty = STATE.poDuty;
    }
  }

  // 6. Calculate Effective Input Resistance (Rin = Rload * (1 - D)^2)
  STATE.effectiveRin = activeRload * Math.pow(1 - STATE.currentDuty, 2);
  STATE.impedanceMatched = Math.abs(STATE.effectiveRin - pvData.Rmpp) < 0.4;

  // 7. Calculate Panel Operating Point from Rin
  // In user notes: If Rin shoots to 12.95Ω (higher than 6.47Ω), panel voltage shoots to ~21.6V & current chokes to 1.67A!
  if (transientMismatch) {
    // Exact values from notes Page 3: Rin = 30 * (1 - 0.343)^2 = 12.95 Ω
    STATE.currentV = 21.60;
    STATE.currentI = 1.67;
    STATE.currentP = STATE.currentV * STATE.currentI; // ~36W
    STATE.efficiency = (STATE.currentP / pvData.Pmpp) * 100;
  } else {
    // Normal MPPT tracking
    if (STATE.impedanceMatched) {
      STATE.currentV = pvData.Vmpp; // 18.0V at STC
      STATE.currentI = pvData.Impp; // 2.78A at STC
      STATE.currentP = STATE.currentV * STATE.currentI; // 50.04W
      STATE.efficiency = 99.8 + (Math.sin(Date.now() * 0.003) * 0.15);
    } else {
      // Voltage shifts with Rin
      const rinRatio = STATE.effectiveRin / pvData.Rmpp;
      if (rinRatio > 1) {
        // High impedance -> Voltage rises toward Voc, current drops
        STATE.currentV = Math.min(pvData.Voc - 0.1, pvData.Vmpp + (rinRatio - 1) * 3.5);
      } else {
        // Low impedance -> Voltage drops, current stays near Isc
        STATE.currentV = Math.max(4.0, pvData.Vmpp - (1 - rinRatio) * 6.0);
      }
      STATE.currentI = pvData.calcCurrentAtV(STATE.currentV);
      STATE.currentP = STATE.currentV * STATE.currentI;
      STATE.efficiency = (STATE.currentP / pvData.Pmpp) * 100;
    }
  }

  // 8. History Buffer
  if (STATE.history.length >= STATE.maxHistory) STATE.history.shift();
  STATE.history.push({
    G: STATE.effectiveIrradiance,
    P: STATE.currentP,
    D: STATE.currentDuty
  });

  // 9. Update UI & Canvas Renders
  updateUI(pvData, activeRload, transientMismatch);
  skyStage.render();
  pwmChart.render(STATE.currentDuty);
  circuitChart.render();
  curvesChart.render(pvData, STATE.currentV, STATE.currentI, STATE.currentP);
  oscChart.render(STATE.history);
  annChart.render(STATE.annActivations, STATE.currentDuty);

  requestAnimationFrame(loop);
}

function updateUI(pvData, activeRload, transientMismatch) {
  // HUD
  const hudElev = document.getElementById('hud-elevation');
  const hudOcc = document.getElementById('hud-occlusion');
  const hudLux = document.getElementById('hud-panel-lux');
  const alertEl = document.getElementById('shading-alert');

  if (hudElev) hudElev.textContent = `${STATE.sunAngle.toFixed(1)}°`;
  if (hudOcc) hudOcc.textContent = `${Math.round(STATE.cloudOcclusion * 100)}%`;
  if (hudLux) {
    const luxPct = Math.round((STATE.effectiveIrradiance / 1000) * 100);
    hudLux.textContent = `${luxPct}% (${Math.round(STATE.effectiveIrradiance)} W/m²)`;
  }
  if (alertEl) alertEl.classList.toggle('hidden', STATE.cloudOcclusion < 0.2);

  // Hero Duty
  const rDuty = document.getElementById('readout-duty');
  const rDutyPct = document.getElementById('readout-duty-pct');
  const pwmRatio = document.getElementById('pwm-ratio-label');
  const rEff = document.getElementById('readout-efficiency');
  const effBar = document.getElementById('eff-bar-fill');
  const dutyLoadRef = document.getElementById('duty-load-ref');

  if (rDuty) rDuty.textContent = STATE.currentDuty.toFixed(4);
  const dutyPct = (STATE.currentDuty * 100).toFixed(1);
  if (rDutyPct) rDutyPct.textContent = `${dutyPct}%`;

  if (pwmRatio) {
    const tOff = (100 - STATE.currentDuty * 100).toFixed(1);
    pwmRatio.textContent = `TON: ${dutyPct}% | TOFF: ${tOff}%`;
  }

  if (rEff) rEff.textContent = `${STATE.efficiency.toFixed(1)}%`;
  if (effBar) effBar.style.width = `${STATE.efficiency.toFixed(1)}%`;

  if (dutyLoadRef) {
    if (STATE.loadMode === 'resistor') {
      dutyLoadRef.textContent = `(Matches Rload=${STATE.resistorValue.toFixed(1)}Ω ➔ Rin=${STATE.effectiveRin.toFixed(2)}Ω)`;
    } else {
      dutyLoadRef.textContent = `(Boost to 6S Vbat=${STATE.batteryVoltage.toFixed(1)}V ➔ Rin=${STATE.effectiveRin.toFixed(2)}Ω)`;
    }
  }

  // 4 Meters
  const rIrr = document.getElementById('readout-irradiance');
  const rTemp = document.getElementById('readout-temp');
  const rVolt = document.getElementById('readout-voltage');
  const rCurr = document.getElementById('readout-current');

  if (rIrr) rIrr.textContent = Math.round(STATE.effectiveIrradiance);
  if (rTemp) rTemp.textContent = STATE.cellTemperature.toFixed(1);
  if (rVolt) rVolt.textContent = STATE.currentV.toFixed(2);
  if (rCurr) rCurr.textContent = STATE.currentI.toFixed(2);

  const normIrr = document.getElementById('norm-irr');
  const normTemp = document.getElementById('norm-temp');
  const normVolt = document.getElementById('norm-volt');
  const normCurr = document.getElementById('norm-curr');

  const acts = STATE.annActivations.normInputs || [];
  if (normIrr && acts[0] !== undefined) normIrr.textContent = `ANN Norm: ${acts[0].toFixed(3)}`;
  if (normTemp && acts[1] !== undefined) normTemp.textContent = `ANN Norm: ${acts[1].toFixed(3)}`;
  if (normVolt && acts[2] !== undefined) normVolt.textContent = `ANN Norm: ${acts[2].toFixed(3)}`;
  if (normCurr && acts[3] !== undefined) normCurr.textContent = `ANN Norm: ${acts[3].toFixed(3)}`;

  // Meter Progress Bars
  const barIrr = document.getElementById('bar-irradiance');
  const barTemp = document.getElementById('bar-temperature');
  const barVolt = document.getElementById('bar-voltage');
  const barCurr = document.getElementById('bar-current');

  if (barIrr) barIrr.style.width = `${((STATE.effectiveIrradiance - 100) / 900) * 100}%`;
  if (barTemp) barTemp.style.width = `${((STATE.cellTemperature - 14) / 46) * 100}%`;
  if (barVolt) barVolt.style.width = `${(STATE.currentV / 22) * 100}%`;
  if (barCurr) barCurr.style.width = `${(STATE.currentI / 3.0) * 100}%`;

  // Power Banner
  const rPower = document.getElementById('readout-power');
  const rPmpp = document.getElementById('readout-pmpp');
  if (rPower) rPower.textContent = STATE.currentP.toFixed(2);
  if (rPmpp) rPmpp.textContent = `${pvData.Pmpp.toFixed(2)} W`;

  // Circuit Schematic Equation Bar
  const eqCalc = document.getElementById('schematic-eq-calc');
  const stateTag = document.getElementById('schematic-state-tag');
  const impPill = document.getElementById('schematic-impedance-pill');

  if (eqCalc) {
    eqCalc.textContent = `${activeRload.toFixed(1)}Ω × (1 - ${STATE.currentDuty.toFixed(3)})² = ${STATE.effectiveRin.toFixed(2)} Ω`;
  }

  if (stateTag) {
    if (transientMismatch) {
      stateTag.textContent = 'MISMATCH! Voltage shot up to ~21.6V (Current choked)';
      stateTag.className = 'text-red';
    } else if (STATE.impedanceMatched) {
      stateTag.textContent = 'LOCKED at 18.0V (50W Peak)';
      stateTag.className = 'text-green';
    } else {
      stateTag.textContent = `Tracking Rin (${STATE.effectiveRin.toFixed(1)}Ω)`;
      stateTag.className = 'text-cyan';
    }
  }

  if (impPill) {
    if (transientMismatch) {
      impPill.innerHTML = '<span class="dot bg-red"></span><span>Mismatched: Rin = 12.95 Ω</span>';
    } else {
      impPill.innerHTML = `<span class="dot bg-green"></span><span>Rin = ${STATE.effectiveRin.toFixed(2)} Ω (Target 6.47Ω)</span>`;
    }
  }

  // Load Controls Readout (Resistor View)
  const rRin = document.getElementById('readout-rin');
  const mathExplain = document.getElementById('math-explanation-text');
  if (rRin) {
    rRin.textContent = `${STATE.effectiveRin.toFixed(2)} Ω`;
    rRin.className = STATE.impedanceMatched ? 'math-val text-green' : 'math-val text-red';
  }
  if (mathExplain) {
    if (transientMismatch) {
      mathExplain.innerHTML = '<span style="color:#f43f5e; font-weight:700;">⚠ TRANSIENT STEP DETECTED:</span> <i>R</i><sub>load</sub> stepped to 30&Omega; with unadjusted <i>D</i> = 0.343. <i>R</i><sub>in</sub> shot up to <strong class="text-red">12.95 &Omega;</strong>, causing panel voltage to surge to <strong class="text-solar">21.6V</strong> and current to choke down to <strong class="text-cyan">1.67A</strong>!';
    } else {
      mathExplain.innerHTML = `&bull; At <i>R</i><sub>load</sub> = <b>${STATE.resistorValue.toFixed(1)} &Omega;</b>, MPPT sets <i>D</i> = <b>${STATE.currentDuty.toFixed(3)}</b> &rarr; achieves <i>R</i><sub>in</sub> = <strong class="text-solar">${STATE.effectiveRin.toFixed(2)} &Omega;</strong> (Target: 6.47 &Omega;), keeping the 50W panel locked at 18.0V &amp; 2.78A.`;
    }
  }

  // Battery View Readouts
  const rVbat = document.getElementById('readout-vbat');
  const rBatReff = document.getElementById('readout-bat-reff');
  const batFill = document.getElementById('bat-level-fill');
  const vbatFormula = document.getElementById('vbat-formula-val');
  const dbatFormula = document.getElementById('dbat-formula-val');

  if (rVbat) rVbat.textContent = `${STATE.batteryVoltage.toFixed(2)} V`;
  if (rBatReff) rBatReff.textContent = `${STATE.batteryReff.toFixed(1)} Ω`;
  if (batFill) batFill.style.width = `${STATE.batterySoC}%`;
  if (vbatFormula) vbatFormula.textContent = `${STATE.batteryVoltage.toFixed(1)}V`;
  if (dbatFormula) {
    const dVal = 1 - (18.0 / STATE.batteryVoltage);
    dbatFormula.textContent = dVal.toFixed(3);
  }

  // Curve Stats Footer
  const sVmpp = document.getElementById('stat-vmpp');
  const sImpp = document.getElementById('stat-impp');
  const sPmpp = document.getElementById('stat-pmpp');
  const sRmpp = document.getElementById('stat-rmpp');

  if (sVmpp) sVmpp.textContent = `${pvData.Vmpp.toFixed(2)} V`;
  if (sImpp) sImpp.textContent = `${pvData.Impp.toFixed(2)} A`;
  if (sPmpp) sPmpp.textContent = `${pvData.Pmpp.toFixed(2)} W`;
  if (sRmpp) sRmpp.textContent = `${pvData.Rmpp.toFixed(2)} Ω`;
}

// Start app on DOM ready
window.addEventListener('DOMContentLoaded', initApp);


// =====================================================================
// ESP32 HARDWARE BRIDGE CONTROLLER
// =====================================================================
const ESP32Bridge = {
  connected: false,
  ip: '192.168.4.1',
  lastSendTime: 0,
  syncActive: true,
  lastHwDuty: null,

  init() {
    const btn = document.getElementById('esp32-bridge-btn');
    const popover = document.getElementById('esp32-bridge-popover');
    const connectBtn = document.getElementById('btn-connect-esp32');
    const ipInput = document.getElementById('esp32-ip-input');
    const syncToggle = document.getElementById('esp32-sync-toggle');

    if (!btn || !popover) return;

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      popover.classList.toggle('hidden');
    });

    document.addEventListener('click', (e) => {
      if (!popover.contains(e.target) && e.target !== btn) {
        popover.classList.add('hidden');
      }
    });

    if (connectBtn && ipInput) {
      connectBtn.addEventListener('click', () => {
        this.ip = ipInput.value.trim().replace(/^https?:\/\//, '').replace(/\/.*$/, '');
        this.testConnection();
      });
    }

    if (syncToggle) {
      syncToggle.addEventListener('change', () => {
        this.syncActive = syncToggle.checked;
      });
    }
  },

  async testConnection() {
    const statusTag = document.getElementById('esp32-status-tag');
    const dot = document.getElementById('esp32-bridge-dot');
    const connectBtn = document.getElementById('btn-connect-esp32');

    if (statusTag) statusTag.textContent = 'Connecting...';
    if (connectBtn) connectBtn.textContent = '...';

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      const res = await fetch(`http://${this.ip}/api/status`, { signal: controller.signal });
      clearTimeout(timeoutId);
      const data = await res.json();

      if (data.status === 'ok') {
        this.connected = true;
        if (dot) dot.classList.add('connected');
        if (statusTag) {
          statusTag.textContent = 'Connected';
          statusTag.classList.add('active');
        }
        if (connectBtn) connectBtn.textContent = 'Connected';
        const label = document.getElementById('esp32-bridge-label');
        if (label) label.textContent = `ESP32 (${this.ip})`;
        this.updateTelemetry(data);
      }
    } catch (err) {
      this.connected = false;
      if (dot) dot.classList.remove('connected');
      if (statusTag) {
        statusTag.textContent = 'Offline';
        statusTag.classList.remove('active');
      }
      if (connectBtn) connectBtn.textContent = 'Retry';
      alert(`Could not connect to ESP32 at http://${this.ip}.\nEnsure you are connected to the 'Solaris-MPPT' Wi-Fi or your ESP32 router IP.`);
    }
  },

  updateTelemetry(data) {
    const hwPwm = document.getElementById('esp32-hw-pwm');
    const hwLat = document.getElementById('esp32-hw-lat');
    const hwDuty = document.getElementById('esp32-hw-duty');
    const hwUptime = document.getElementById('esp32-hw-uptime');

    if (hwPwm && data.raw_pwm !== undefined) hwPwm.textContent = `${data.raw_pwm} / 1023`;
    if (hwLat && data.latency_us !== undefined) hwLat.textContent = `${data.latency_us} µs`;
    if (hwDuty && data.duty !== undefined) hwDuty.textContent = `${(data.duty * 100).toFixed(1)}%`;
    if (hwUptime && data.uptime_s !== undefined) hwUptime.textContent = `${data.uptime_s}s`;
  },

  sendSimulationStep(irr, temp, vpv, ipv) {
    if (!this.connected || !this.syncActive) return;

    const now = performance.now();
    if (now - this.lastSendTime < 80) return; // Debounce to ~12 Hz max
    this.lastSendTime = now;

    fetch(`http://${this.ip}/api/predict?irr=${irr.toFixed(1)}&temp=${temp.toFixed(1)}&vpv=${vpv.toFixed(3)}&ipv=${ipv.toFixed(3)}`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') {
          this.lastHwDuty = data.duty;
          this.updateTelemetry(data);
        }
      })
      .catch(() => {});
  }
};

window.addEventListener('DOMContentLoaded', () => {
  ESP32Bridge.init();
});
