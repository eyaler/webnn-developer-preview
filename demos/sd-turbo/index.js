// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with webnn in onnxruntime-web.
//

import { setupORT, showCompatibleChromiumVersion } from '../../assets/js/common_utils.js';

const log = (i) => {
  console.log(i);
  if(getMode()) {
    document.getElementById("status").innerText += `\n[${getTime()}] ${i}`;
  } else {
    document.getElementById("status").innerText += `\n${i}`;
  }
};

const logError = (i) => {
  console.error(i);
  if(getMode()) {
    document.getElementById("status").innerText += `\n[${getTime()}] ${i}`;
  } else {
    document.getElementById("status").innerText += `\n${i}`;
  }
};

/*
 * get configuration from url
 */
function getConfig() {
  const query = window.location.search.substring(1);
  var config = {
    model: "https://huggingface.co/eyaler/sd-turbo-webnn/resolve/main",
    mode: "none",
    provider: "webnn",
    devicetype: "gpu",
    threads: "1",
    images: "1",
    ort: "test"
  };

  config.threads = parseInt(config.threads);
  config.images = parseInt(config.images);
  return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape) {
  function randn() {
    // Use the Box-Muller transform
    let u = Math.random();
    let v = Math.random();
    let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return z;
  }
  let size = 1;
  shape.forEach((element) => {
    size *= element;
  });

  let data = new Float32Array(size);
  // Loop over the shape dimensions
  for (let i = 0; i < size; i++) {
    data[i] = randn();
  }
  return data;
}

let textEncoderFetchProgress = 0;
let unetFetchProgress = 0;
let vaeDecoderFetchProgress = 0;
let vaeEncoderFetchProgress = 0;
let textEncoderCompileProgress = 0;
let unetCompileProgress = 0;
let vaeEncoderCompileProgress = 0;
let vaeDecoderCompileProgress = 0;

// Get model via Origin Private File System
async function getModelOPFS(name, url, updateModel) {
  const root = await navigator.storage.getDirectory();
  let fileHandle;

  async function updateFile() {
    console.log('updating', name)
    const response = await fetch(url);
    const buffer = await readResponse(name, response);
    fileHandle = await root.getFileHandle(name, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(buffer);
    await writable.close();
    return buffer;
  }

  if (updateModel) {
    return await updateFile();
  }

  try {
    fileHandle = await root.getFileHandle(name);
    const blob = await fileHandle.getFile();
    let buffer = await blob.arrayBuffer();
    if (buffer) {

        if (name == "sd_turbo_text_encoder") {
          textEncoderFetchProgress = 20.0;
        } else if (name == "sd_turbo_unet") {
          unetFetchProgress = 65.0;
        } else if (name == "sd_turbo_vae_decoder") {
          vaeDecoderFetchProgress = 2.0;
        } else if (name == "sd_turbo_vae_encoder") {
          vaeEncoderFetchProgress = 1.0;
        }


      updateProgress();
      updateLoadWave(progress.toFixed(2));
      return buffer;
    }
  } catch (e) {
    return await updateFile();
  }
}

async function readResponse(name, response) {
  const contentLength = response.headers.get("Content-Length");
  let total = parseInt(contentLength ?? "0");
  let buffer = new Uint8Array(total);
  let loaded = 0;

  const reader = response.body.getReader();
  async function read() {
    const { done, value } = await reader.read();
    if (done) return;

    let newLoaded = loaded + value.length;
    let fetchProgress = (newLoaded / contentLength) * 100;

      if (name == "sd_turbo_text_encoder") {
        textEncoderFetchProgress = 0.2 * fetchProgress;
      } else if (name == "sd_turbo_unet") {
        unetFetchProgress = 0.65 * fetchProgress;
      } else if (name == "sd_turbo_vae_decoder") {
        vaeDecoderFetchProgress = 0.02 * fetchProgress;
      } else if (name == "sd_turbo_vae_encoder") {
        vaeEncoderFetchProgress = 0.01 * fetchProgress;
      }



    updateProgress();
    updateLoadWave(progress.toFixed(2));

    if (newLoaded > total) {
      total = newLoaded;
      let newBuffer = new Uint8Array(total);
      newBuffer.set(buffer);
      buffer = newBuffer;
    }
    buffer.set(value, loaded);
    loaded = newLoaded;
    return read();
  }

  await read();
  return buffer;
}

const getMode = () => {
  return (getQueryValue("mode") === "normal") ? false : true;
};


const updateProgress = () => {
    progress =
    textEncoderFetchProgress +
    unetFetchProgress +
    vaeDecoderFetchProgress + vaeEncoderFetchProgress +
    textEncoderCompileProgress +
    unetCompileProgress +
    vaeDecoderCompileProgress + vaeEncoderCompileProgress;
    }

let load_finished
/*
 * load models used in the pipeline
 */
async function load_models(models) {
  load.disabled = true;
  refetch.disabled = true

  img_div_0.setAttribute("class", "frame loadwave");
  buttons.setAttribute("class", "button-group key loading");

  log("[Load] ONNX Runtime Execution Provider: " + config.provider);
  log("[Load] ONNX Runtime EP device type: " + config.devicetype);
  updateLoadWave(0.0);

  for (const [name, model] of Object.entries(models)) {
    let modelNameInLog = "";
    try {
      let start = performance.now();
      let modelUrl;
      if (name == "text_encoder") {
        modelNameInLog = "Text Encoder";
        modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
      } else if (name == "unet") {
        modelNameInLog = "UNet";
        modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
      } else if (name == "vae_decoder") {
        modelNameInLog = "VAE Decoder";
        modelUrl = `${config.model}/${name}/model.onnx`;
      } else if (name == "vae_encoder") {
        modelNameInLog = "VAE Encoder";
        modelUrl = `${config.model}/${name}/model.onnx`;
      }
      log(`[Load] Loading model ${modelNameInLog} · ${model.size}`);
      let modelBuffer = await getModelOPFS(`sd_turbo_${name}`, modelUrl, refetch.checked);
      let modelFetchTime = (performance.now() - start).toFixed(2);
      if (name == "text_encoder") {
        textEncoderFetch.innerHTML = modelFetchTime;
      } else if (name == "unet") {
        unetFetch.innerHTML = modelFetchTime;
      } else if (name == "vae_decoder") {
        vaeFetch.innerHTML = modelFetchTime;
      } else if (name == "vae_encoder") {
        vaeEncoderFetch.innerHTML = modelFetchTime;
      }
      log(`[Load] ${modelNameInLog} loaded · ${modelFetchTime}ms`);
      log(`[Session Create] Beginning ${modelNameInLog}`);

      start = performance.now();
      const sess_opt = { ...opt, ...model.opt };
      models[name].sess = await ort.InferenceSession.create(
        modelBuffer,
        sess_opt
      );
      let createTime = (performance.now() - start).toFixed(2);


        if (name == "text_encoder") {
          textEncoderCreate.innerHTML = createTime;
          textEncoderCompileProgress = 2;
          updateProgress();
          updateLoadWave(progress.toFixed(2));
        } else if (name == "unet") {
          unetCreate.innerHTML = createTime;
          unetCompileProgress = 9;
          updateProgress();
          updateLoadWave(progress.toFixed(2));
        } else if (name == "vae_decoder") {
          vaeCreate.innerHTML = createTime;
          vaeDecoderCompileProgress = 1;
          updateProgress();
          updateLoadWave(progress.toFixed(2));
        } else if (name == "vae_encoder") {
          vaeEncoderCreate.innerHTML = createTime;
          vaeEncoderCompileProgress = 1;
          updateProgress();
          updateLoadWave(progress.toFixed(2));
        }


      if (getMode()) {
        log(
          `[Session Create] Create ${modelNameInLog} completed · ${createTime}ms`
        );
      } else {
        log(`[Session Create] Create ${modelNameInLog} completed`);
      }
    } catch (e) {
      log(`[Load] ${modelNameInLog} failed, ${e.stack}`);
    }
  }

  updateLoadWave(100.0);
  log("[Session Create] Ready to generate images");
  let image_area = document.querySelectorAll("#image_area>div");
  image_area.forEach((i) => {
    i.setAttribute("class", "frame done");
  });
  buttons.setAttribute("class", "button-group key loaded");
  generate.disabled = false;
  document
    .querySelector("#user-input")
    .setAttribute("class", "form-control enabled");
        document
  load_finished = true
}

const config = getConfig();

let models = {
  unet: {
    // original model from dw, then wm dump new one from local graph optimization.
    url: "unet/model_layernorm.onnx",
    size: "1.61GB",
    opt: { graphOptimizationLevel: "disabled", }, // avoid wasm heap issue (need Wasm memory 64)
  },
  text_encoder: {
    // orignal model from gu, wm convert the output to fp16.
    url: "text_encoder/model_layernorm.onnx",
    size: "649MB",
    opt: { graphOptimizationLevel: "disabled", },
    // opt: { freeDimensionOverrides: { batch_size: 1, sequence_length: 77 } },
  },
  vae_decoder: {
    // use gu's model has precision lose in webnn caused by instanceNorm op,
    // covert the model to run instanceNorm in fp32 (insert cast nodes).
    url: "vae_decoder/model.onnx",
    size: "94.5MB",
    // opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
    opt: {
      freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 },
    },
  },
  vae_encoder: {
    url: "vae_encoder/model.onnx",
    size: "68.4MB",
    opt: {
      freeDimensionOverrides: { batch_size: 1, num_channels: 3, height: 512, width: 512 },
    },
  },
};

let progress = 0;
let inferenceProgress = 0;

let tokenizer;
//const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

/* to get sigmas:
import torch
num_timesteps = 1000
beta_start = 0.00085  // https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json
beta_end = 0.012  // https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json
betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2  // for 'scaled linear' schedule, https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json
alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
sigmas = (((1 - alphas_cumprod) / alphas_cumprod) ** 0.5).flip(0)
*/

const sigmas = [14.6146, 14.5263, 14.4386, 14.3515, 14.2651, 14.1793, 14.0942, 14.0097,
        13.9259, 13.8426, 13.7600, 13.6779, 13.5965, 13.5157, 13.4355, 13.3558,
        13.2768, 13.1983, 13.1204, 13.0431, 12.9663, 12.8901, 12.8145, 12.7394,
        12.6648, 12.5908, 12.5174, 12.4444, 12.3720, 12.3001, 12.2288, 12.1579,
        12.0876, 12.0177, 11.9484, 11.8796, 11.8113, 11.7434, 11.6761, 11.6092,
        11.5428, 11.4769, 11.4114, 11.3464, 11.2819, 11.2178, 11.1542, 11.0910,
        11.0283, 10.9661, 10.9042, 10.8428, 10.7819, 10.7214, 10.6613, 10.6016,
        10.5423, 10.4835, 10.4250, 10.3670, 10.3094, 10.2522, 10.1954, 10.1390,
        10.0829, 10.0273,  9.9720,  9.9172,  9.8627,  9.8086,  9.7548,  9.7015,
         9.6485,  9.5959,  9.5436,  9.4917,  9.4401,  9.3889,  9.3381,  9.2876,
         9.2374,  9.1876,  9.1381,  9.0890,  9.0402,  8.9917,  8.9436,  8.8958,
         8.8483,  8.8011,  8.7543,  8.7078,  8.6615,  8.6156,  8.5700,  8.5247,
         8.4798,  8.4351,  8.3907,  8.3466,  8.3028,  8.2593,  8.2161,  8.1732,
         8.1305,  8.0882,  8.0461,  8.0043,  7.9628,  7.9216,  7.8806,  7.8399,
         7.7995,  7.7593,  7.7194,  7.6798,  7.6404,  7.6013,  7.5624,  7.5238,
         7.4854,  7.4473,  7.4095,  7.3718,  7.3345,  7.2974,  7.2605,  7.2238,
         7.1874,  7.1512,  7.1153,  7.0796,  7.0441,  7.0089,  6.9739,  6.9391,
         6.9045,  6.8701,  6.8360,  6.8021,  6.7684,  6.7349,  6.7016,  6.6686,
         6.6357,  6.6031,  6.5707,  6.5384,  6.5064,  6.4746,  6.4430,  6.4115,
         6.3803,  6.3493,  6.3185,  6.2878,  6.2574,  6.2271,  6.1970,  6.1672,
         6.1375,  6.1080,  6.0786,  6.0495,  6.0205,  5.9917,  5.9631,  5.9347,
         5.9065,  5.8784,  5.8505,  5.8227,  5.7952,  5.7678,  5.7405,  5.7135,
         5.6866,  5.6599,  5.6333,  5.6069,  5.5806,  5.5546,  5.5286,  5.5029,
         5.4773,  5.4518,  5.4265,  5.4013,  5.3763,  5.3515,  5.3268,  5.3022,
         5.2778,  5.2536,  5.2295,  5.2055,  5.1817,  5.1580,  5.1344,  5.1110,
         5.0878,  5.0646,  5.0416,  5.0188,  4.9961,  4.9735,  4.9510,  4.9287,
         4.9065,  4.8845,  4.8625,  4.8407,  4.8191,  4.7975,  4.7761,  4.7548,
         4.7336,  4.7126,  4.6917,  4.6709,  4.6502,  4.6296,  4.6092,  4.5888,
         4.5686,  4.5485,  4.5286,  4.5087,  4.4890,  4.4693,  4.4498,  4.4304,
         4.4111,  4.3919,  4.3728,  4.3538,  4.3350,  4.3162,  4.2975,  4.2790,
         4.2606,  4.2422,  4.2240,  4.2059,  4.1878,  4.1699,  4.1521,  4.1343,
         4.1167,  4.0992,  4.0817,  4.0644,  4.0472,  4.0300,  4.0130,  3.9960,
         3.9791,  3.9624,  3.9457,  3.9291,  3.9126,  3.8962,  3.8799,  3.8637,
         3.8475,  3.8315,  3.8155,  3.7997,  3.7839,  3.7682,  3.7526,  3.7370,
         3.7216,  3.7062,  3.6910,  3.6758,  3.6606,  3.6456,  3.6307,  3.6158,
         3.6010,  3.5863,  3.5716,  3.5571,  3.5426,  3.5282,  3.5139,  3.4996,
         3.4855,  3.4714,  3.4573,  3.4434,  3.4295,  3.4157,  3.4020,  3.3883,
         3.3747,  3.3612,  3.3478,  3.3344,  3.3211,  3.3078,  3.2947,  3.2816,
         3.2686,  3.2556,  3.2427,  3.2299,  3.2171,  3.2044,  3.1918,  3.1792,
         3.1667,  3.1543,  3.1419,  3.1296,  3.1173,  3.1051,  3.0930,  3.0809,
         3.0689,  3.0570,  3.0451,  3.0333,  3.0215,  3.0098,  2.9982,  2.9866,
         2.9751,  2.9636,  2.9522,  2.9408,  2.9295,  2.9183,  2.9071,  2.8960,
         2.8849,  2.8739,  2.8629,  2.8520,  2.8412,  2.8304,  2.8196,  2.8089,
         2.7983,  2.7877,  2.7771,  2.7666,  2.7562,  2.7458,  2.7355,  2.7252,
         2.7149,  2.7048,  2.6946,  2.6845,  2.6745,  2.6645,  2.6545,  2.6446,
         2.6348,  2.6250,  2.6152,  2.6055,  2.5959,  2.5862,  2.5767,  2.5671,
         2.5577,  2.5482,  2.5388,  2.5295,  2.5202,  2.5109,  2.5017,  2.4925,
         2.4834,  2.4743,  2.4653,  2.4563,  2.4473,  2.4384,  2.4295,  2.4207,
         2.4119,  2.4031,  2.3944,  2.3857,  2.3771,  2.3685,  2.3599,  2.3514,
         2.3429,  2.3345,  2.3261,  2.3177,  2.3094,  2.3011,  2.2929,  2.2846,
         2.2765,  2.2683,  2.2602,  2.2522,  2.2441,  2.2361,  2.2282,  2.2202,
         2.2124,  2.2045,  2.1967,  2.1889,  2.1811,  2.1734,  2.1658,  2.1581,
         2.1505,  2.1429,  2.1354,  2.1278,  2.1204,  2.1129,  2.1055,  2.0981,
         2.0908,  2.0834,  2.0762,  2.0689,  2.0617,  2.0545,  2.0473,  2.0402,
         2.0331,  2.0260,  2.0190,  2.0120,  2.0050,  1.9980,  1.9911,  1.9842,
         1.9773,  1.9705,  1.9637,  1.9569,  1.9502,  1.9434,  1.9367,  1.9301,
         1.9234,  1.9168,  1.9103,  1.9037,  1.8972,  1.8907,  1.8842,  1.8777,
         1.8713,  1.8649,  1.8586,  1.8522,  1.8459,  1.8396,  1.8333,  1.8271,
         1.8209,  1.8147,  1.8085,  1.8024,  1.7963,  1.7902,  1.7841,  1.7781,
         1.7721,  1.7661,  1.7601,  1.7542,  1.7482,  1.7423,  1.7365,  1.7306,
         1.7248,  1.7190,  1.7132,  1.7074,  1.7017,  1.6960,  1.6903,  1.6846,
         1.6790,  1.6734,  1.6678,  1.6622,  1.6566,  1.6511,  1.6456,  1.6401,
         1.6346,  1.6291,  1.6237,  1.6183,  1.6129,  1.6075,  1.6022,  1.5968,
         1.5915,  1.5862,  1.5810,  1.5757,  1.5705,  1.5653,  1.5601,  1.5549,
         1.5497,  1.5446,  1.5395,  1.5344,  1.5293,  1.5243,  1.5192,  1.5142,
         1.5092,  1.5042,  1.4992,  1.4943,  1.4894,  1.4844,  1.4796,  1.4747,
         1.4698,  1.4650,  1.4601,  1.4553,  1.4506,  1.4458,  1.4410,  1.4363,
         1.4316,  1.4269,  1.4222,  1.4175,  1.4128,  1.4082,  1.4036,  1.3990,
         1.3944,  1.3898,  1.3852,  1.3807,  1.3762,  1.3717,  1.3672,  1.3627,
         1.3582,  1.3538,  1.3493,  1.3449,  1.3405,  1.3361,  1.3318,  1.3274,
         1.3231,  1.3187,  1.3144,  1.3101,  1.3058,  1.3016,  1.2973,  1.2931,
         1.2888,  1.2846,  1.2804,  1.2762,  1.2721,  1.2679,  1.2638,  1.2596,
         1.2555,  1.2514,  1.2473,  1.2432,  1.2392,  1.2351,  1.2311,  1.2271,
         1.2231,  1.2191,  1.2151,  1.2111,  1.2071,  1.2032,  1.1993,  1.1953,
         1.1914,  1.1875,  1.1836,  1.1798,  1.1759,  1.1721,  1.1682,  1.1644,
         1.1606,  1.1568,  1.1530,  1.1492,  1.1454,  1.1417,  1.1379,  1.1342,
         1.1305,  1.1268,  1.1231,  1.1194,  1.1157,  1.1121,  1.1084,  1.1048,
         1.1011,  1.0975,  1.0939,  1.0903,  1.0867,  1.0832,  1.0796,  1.0760,
         1.0725,  1.0690,  1.0654,  1.0619,  1.0584,  1.0549,  1.0514,  1.0480,
         1.0445,  1.0410,  1.0376,  1.0342,  1.0307,  1.0273,  1.0239,  1.0205,
         1.0171,  1.0138,  1.0104,  1.0070,  1.0037,  1.0004,  0.9970,  0.9937,
         0.9904,  0.9871,  0.9838,  0.9805,  0.9773,  0.9740,  0.9707,  0.9675,
         0.9643,  0.9610,  0.9578,  0.9546,  0.9514,  0.9482,  0.9450,  0.9418,
         0.9387,  0.9355,  0.9324,  0.9292,  0.9261,  0.9230,  0.9198,  0.9167,
         0.9136,  0.9105,  0.9074,  0.9044,  0.9013,  0.8982,  0.8952,  0.8921,
         0.8891,  0.8861,  0.8830,  0.8800,  0.8770,  0.8740,  0.8710,  0.8680,
         0.8650,  0.8621,  0.8591,  0.8562,  0.8532,  0.8503,  0.8473,  0.8444,
         0.8415,  0.8386,  0.8357,  0.8328,  0.8299,  0.8270,  0.8241,  0.8212,
         0.8184,  0.8155,  0.8126,  0.8098,  0.8070,  0.8041,  0.8013,  0.7985,
         0.7957,  0.7929,  0.7901,  0.7873,  0.7845,  0.7817,  0.7789,  0.7762,
         0.7734,  0.7706,  0.7679,  0.7651,  0.7624,  0.7597,  0.7569,  0.7542,
         0.7515,  0.7488,  0.7461,  0.7434,  0.7407,  0.7380,  0.7353,  0.7327,
         0.7300,  0.7273,  0.7247,  0.7220,  0.7194,  0.7167,  0.7141,  0.7115,
         0.7088,  0.7062,  0.7036,  0.7010,  0.6984,  0.6958,  0.6932,  0.6906,
         0.6880,  0.6855,  0.6829,  0.6803,  0.6777,  0.6752,  0.6726,  0.6701,
         0.6675,  0.6650,  0.6625,  0.6599,  0.6574,  0.6549,  0.6524,  0.6499,
         0.6473,  0.6448,  0.6423,  0.6399,  0.6374,  0.6349,  0.6324,  0.6299,
         0.6274,  0.6250,  0.6225,  0.6201,  0.6176,  0.6151,  0.6127,  0.6103,
         0.6078,  0.6054,  0.6029,  0.6005,  0.5981,  0.5957,  0.5933,  0.5908,
         0.5884,  0.5860,  0.5836,  0.5812,  0.5788,  0.5764,  0.5741,  0.5717,
         0.5693,  0.5669,  0.5645,  0.5622,  0.5598,  0.5574,  0.5551,  0.5527,
         0.5504,  0.5480,  0.5457,  0.5433,  0.5410,  0.5386,  0.5363,  0.5340,
         0.5316,  0.5293,  0.5270,  0.5247,  0.5223,  0.5200,  0.5177,  0.5154,
         0.5131,  0.5108,  0.5085,  0.5062,  0.5039,  0.5016,  0.4993,  0.4970,
         0.4947,  0.4924,  0.4901,  0.4878,  0.4856,  0.4833,  0.4810,  0.4787,
         0.4765,  0.4742,  0.4719,  0.4696,  0.4674,  0.4651,  0.4629,  0.4606,
         0.4583,  0.4561,  0.4538,  0.4516,  0.4493,  0.4471,  0.4448,  0.4426,
         0.4403,  0.4381,  0.4358,  0.4336,  0.4313,  0.4291,  0.4268,  0.4246,
         0.4224,  0.4201,  0.4179,  0.4156,  0.4134,  0.4112,  0.4089,  0.4067,
         0.4045,  0.4022,  0.4000,  0.3977,  0.3955,  0.3933,  0.3910,  0.3888,
         0.3866,  0.3843,  0.3821,  0.3799,  0.3776,  0.3754,  0.3731,  0.3709,
         0.3687,  0.3664,  0.3642,  0.3619,  0.3597,  0.3574,  0.3552,  0.3529,
         0.3507,  0.3484,  0.3462,  0.3439,  0.3417,  0.3394,  0.3372,  0.3349,
         0.3326,  0.3304,  0.3281,  0.3258,  0.3235,  0.3213,  0.3190,  0.3167,
         0.3144,  0.3121,  0.3098,  0.3075,  0.3052,  0.3029,  0.3006,  0.2983,
         0.2959,  0.2936,  0.2913,  0.2889,  0.2866,  0.2843,  0.2819,  0.2795,
         0.2772,  0.2748,  0.2724,  0.2700,  0.2677,  0.2653,  0.2628,  0.2604,
         0.2580,  0.2556,  0.2531,  0.2507,  0.2482,  0.2458,  0.2433,  0.2408,
         0.2383,  0.2358,  0.2332,  0.2307,  0.2281,  0.2256,  0.2230,  0.2204,
         0.2178,  0.2152,  0.2125,  0.2099,  0.2072,  0.2045,  0.2018,  0.1991,
         0.1963,  0.1935,  0.1907,  0.1879,  0.1850,  0.1822,  0.1793,  0.1763,
         0.1734,  0.1704,  0.1673,  0.1642,  0.1611,  0.1580,  0.1548,  0.1515,
         0.1482,  0.1449,  0.1415,  0.1380,  0.1345,  0.1308,  0.1271,  0.1234,
         0.1195,  0.1155,  0.1114,  0.1072,  0.1028,  0.0983,  0.0936,  0.0886,
         0.0834,  0.0779,  0.0720,  0.0656,  0.0586,  0.0507,  0.0413,  0.0292]

const opt = {
  executionProviders: [config.provider],
  enableMemPattern: false,
  enableCpuMemArena: false,
  extra: {
    session: {
      disable_prepacking: "1",
      use_device_allocator_for_initializers: "1",
      use_ort_model_bytes_directly: "1",
      use_ort_model_bytes_for_initializers: "1",
    },
  },
  logSeverityLevel: 0,
};

/*
 * scale the latents
 */
function scale_model_inputs(t, sigma) {
  const d_i = t.data;
  const d_o = new Float32Array(d_i.length);

  const divi = (sigma ** 2 + 1) ** 0.5;
  for (let i = 0; i < d_i.length; i++) {
    d_o[i] = d_i[i] / divi;
  }
  return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
function step(model_output, sample, sigma, gamma, scaling_factor) {
  const d_o = new Float32Array(model_output.data.length);
  const prev_sample = new ort.Tensor(d_o, model_output.dims);
  const sigma_hat = sigma * (gamma + 1);

  for (let i = 0; i < model_output.data.length; i++) {
    const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
    const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
    const dt = 0 - sigma_hat;
    d_o[i] = (sample.data[i] + derivative * dt) / scaling_factor;
  }
  return prev_sample;
}

function draw_out_image(t) {
  const imageData = t.toImageData({ tensorLayout: "NHWC", format: "RGB" });
  const canvas = document.getElementById(`img_canvas_safety`);
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function resize_image(image_nr, targetWidth, targetHeight) {
  // let canvas = document.createElement('canvas');
  // Use img_canvas_test to ensure the input
  const canvas = document.getElementById(`img_canvas_test`);
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  let ctx = canvas.getContext("2d");
  let canvas_source = document.getElementById(`img_canvas_${image_nr}`);
  ctx.drawImage(
    canvas_source,
    0,
    0,
    canvas_source.width,
    canvas_source.height,
    0,
    0,
    targetWidth,
    targetHeight
  );
  let imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);

  return imageData;
}

function normalizeImageData(imageData) {
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];
  const { data, width, height } = imageData;
  const numPixels = width * height;

  let array = new Float32Array(numPixels * 4).fill(0);

  for (let i = 0; i < numPixels; i++) {
    const offset = i * 4;
    for (let c = 0; c < 3; c++) {
      const normalizedValue = (data[offset + c] / 255 - mean[c]) / std[c];
      // data[offset + c] = Math.round(normalizedValue * 255);
      array[offset + c] = normalizedValue * 255;
    }
  }

  // return imageData;
  return { data: array, width: width, height: height };
}

function get_tensor_from_image(imageData, format, norm=2, offset=1) {
  const { data, width, height } = imageData;
  const numPixels = width * height;
  const channels = 3;
  let rearrangedData = new Float32Array(numPixels * channels);
  let destOffset = 0;

  for (let i = 0; i < numPixels; i++) {
    const srcOffset = i * 4;
    const r = (data[srcOffset] / 255 * norm - offset);
    const g = (data[srcOffset + 1] / 255 * norm - offset);
    const b = (data[srcOffset + 2] / 255 * norm - offset);

    if (format === "NCHW") {
      rearrangedData[destOffset] = r;
      rearrangedData[destOffset + numPixels] = g;
      rearrangedData[destOffset + 2 * numPixels] = b;
      destOffset++;
    } else if (format === "NHWC") {
      rearrangedData[destOffset] = r;
      rearrangedData[destOffset + 1] = g;
      rearrangedData[destOffset + 2] = b;
      destOffset += channels;
    } else {
      throw new Error("Invalid format specified.");
    }
  }

  const tensorShape =
    format === "NCHW"
      ? [1, channels, height, width]
      : [1, height, width, channels];
  let tensor = new ort.Tensor(
    "float16",
    convertToUint16Array(rearrangedData),
    tensorShape
  );

  return tensor;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
 */
function draw_image(t, image_nr) {
  let pix = t.data;
  for (var i = 0; i < pix.length; i++) {
    let x = pix[i];
    x = x / 2 + 0.5;
    if (x < 0) x = 0;
    if (x > 1) x = 1;
    pix[i] = x;
  }
  const imageData = t.toImageData({ tensorLayout: "NCHW", format: "RGB" });
  const canvas = document.getElementById(`img_canvas_${image_nr}`);
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

let last_prompt, last_hidden_state, generating, noise, have_output, output, cartoon

async function generate_image(load=true) {
  if (!load_finished && !load.disabled && load) {
    await load_models(models)
  }
  else if (generating || !load_finished) {
    return
  }
  generating = true
  if (input_image_mode.value != 'output' && input_image_mode.value != 'cartoon')
    generate.disabled = true

  try {
    let start

    log(`[Session Run] Beginning`);

    const prompt = document.querySelector("#user-input");
    if (prompt.value != last_prompt) {
        const { input_ids } = await tokenizer(prompt.value, {
          padding: true,
          max_length: 77,
          truncation: true,
          return_tensor: false,
        });

        // text_encoder
        start = performance.now();
        let result = await models.text_encoder.sess.run({
          input_ids: new ort.Tensor("int32", input_ids, [1, input_ids.length]),
        });
        last_hidden_state = result.last_hidden_state
        let sessionRunTimeTextEncode = (performance.now() - start).toFixed(2);
        textEncoderRun1.innerHTML = sessionRunTimeTextEncode;

        if (getMode()) {
          log(
            `[Session Run] Text encode execution time: ${sessionRunTimeTextEncode}ms`
          );
        } else {
          log(`[Session Run] Text encode completed`);
        }
        last_prompt = prompt.value
    }
    const j = 0
      let startTotal = performance.now();
      const latent_shape = [1, 4, 64, 64];

    const start_step = input_image_mode.value == 'text' ? 0 : (sigmas.length - 1) * Math.max(0, Math.min(image_strength.valueAsNumber | 0, 1))
    const sigma = sigmas[start_step]
    const timestep = sigmas.length - 1 - start_step

    if (!noise || input_image_mode.value == 'text' || input_image_mode.value == 'image' || input_image_mode.value == 'output' || input_image_mode.value == 'cartoon' && !have_output)
        noise = new ort.Tensor(
            randn_latents(latent_shape),
            latent_shape
        );

  // vae_encoder
  let latent
  if (input_image_mode.value != 'text') {
    start = performance.now();
    let input_sample
    if (input_image_mode.value == 'cartoon' && have_output)
        input_sample = cartoon
    else if (input_image_mode.value == 'output' && have_output)
        input_sample = output
    else {
        const ctx = img_canvas_input.getContext('2d', {willReadFrequently: true})
        const image_data = ctx.getImageData(0, 0, 512, 512)
        input_sample = get_tensor_from_image(image_data, 'NCHW', 2, 1)
        have_output = true
    }
    const {latent_sample} = await models.vae_encoder.sess.run({
            sample: input_sample,
        });
    latent = convertToFloat32Array(latent_sample.data).map((value, index) => value*vae_scaling_factor + noise.data[index]*sigma)
    let vaeEncoderRunTime = (performance.now() - start).toFixed(2);
    document.getElementById(`vaeEncoderRun${j + 1}`).innerHTML = vaeEncoderRunTime;

    if (getMode()) {
      log(
        `[Session Run][Image ${
          j + 1
        }] VAE encode execution time: ${vaeEncoderRunTime}ms`
      );
    } else {
      log(`[Session Run][Image ${j + 1}] VAE encode completed`);
    }
  } else {
    have_output = true
    latent = noise.data.map(n => n*sigma)
  }

    latent = new ort.Tensor("float32", latent, latent_shape);
    const scaled = scale_model_inputs(latent, sigma);
    const latent16 = new ort.Tensor(
      "float16",
      convertToUint16Array(scaled.data),
      scaled.dims
    )

  // unet
  start = performance.now();
  let feed = {
    sample: latent16,
    timestep: new ort.Tensor("float16", new Uint16Array([toHalf(timestep)]), [
      1,
    ]),
    encoder_hidden_states: last_hidden_state,
  };
  let { out_sample } = await models.unet.sess.run(feed);
  let unetRunTime = (performance.now() - start).toFixed(2);
  document.getElementById(`unetRun${j + 1}`).innerHTML = unetRunTime;

  if (getMode()) {
    log(
      `[Session Run][Image ${j + 1}] UNet execution time: ${unetRunTime}ms`
    );
  } else {
    log(`[Session Run][Image ${j + 1}] UNet completed`);
  }

  // scheduler
  const new_latents = step(
    new ort.Tensor(
      "float32",
      convertToFloat32Array(out_sample.data),
      out_sample.dims
    ),
    latent, sigma, gamma, vae_scaling_factor
  );

  // vae_decoder
  start = performance.now();
  const { sample } = await models.vae_decoder.sess.run({
    latent_sample: new_latents,
  });
  let vaeRunTime = (performance.now() - start).toFixed(2);
  document.getElementById(`vaeRun${j + 1}`).innerHTML = vaeRunTime;

  if (getMode()) {
    log(
      `[Session Run][Image ${
        j + 1
      }] VAE decode execution time: ${vaeRunTime}ms`
    );
  } else {
    log(`[Session Run][Image ${j + 1}] VAE decode completed`);
  }
  output = new ort.Tensor(
            "float16",
            convertToUint16Array(sample.data, true),
            sample.dims
          )
  draw_image(sample, j);
  cartoon = new ort.Tensor(
            "float16",
            convertToUint16Array(sample.data),
            sample.dims
          )
  let totalRunTime = (
    performance.now() -
    startTotal
  ).toFixed(2);
  if (getMode()) {
    log(`[Total] Image ${j + 1} execution time: ${totalRunTime}ms`);
  }
  document.getElementById(`runTotal${j + 1}`).innerHTML = totalRunTime;

  // let out_image = new ort.Tensor("float32", convertToFloat32Array(out_images.data), out_images.dims);
  // draw_out_image(out_image);

    // this is a gpu-buffer we own, so we need to dispose it
    // last_hidden_state.dispose();
    log("[Info] Images generation completed");
  } catch (e) {
    log("[Error] " + e.stack);
  }
  if (input_image_mode.value == 'text' || input_image_mode.value == 'image' || input_image_mode.value == 'output' || input_image_mode.value == 'cartoon')
    generate.disabled = false
  generating = false
}

async function hasFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter.features.has("shader-f16");
  } catch (e) {
    return false;
  }
}

// ref: http://stackoverflow.com/questions/32633585/how-do-you-convert-to-half-floats-in-javascript
const toHalf = (function () {
  var floatView = new Float32Array(1);
  var int32View = new Int32Array(floatView.buffer);

  /* This method is faster than the OpenEXR implementation (very often
   * used, eg. in Ogre), with the additional benefit of rounding, inspired
   * by James Tursa?s half-precision code. */
  return function toHalf(val) {
    floatView[0] = val;
    var x = int32View[0];

    var bits = (x >> 16) & 0x8000; /* Get the sign */
    var m = (x >> 12) & 0x07ff; /* Keep one extra bit for rounding */
    var e = (x >> 23) & 0xff; /* Using int is faster here */

    /* If zero, or denormal, or exponent underflows too much for a denormal
     * half, return signed zero. */
    if (e < 103) {
      return bits;
    }

    /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
    if (e > 142) {
      bits |= 0x7c00;
      /* If exponent was 0xff and one mantissa bit was set, it means NaN,
       * not Inf, so make sure we set one mantissa bit too. */
      bits |= (e == 255 ? 0 : 1) && x & 0x007fffff;
      return bits;
    }

    /* If exponent underflows but not too much, return a denormal */
    if (e < 113) {
      m |= 0x0800;
      /* Extra rounding may overflow and set mantissa to 0 and exponent
       * to 1, which is OK. */
      bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
      return bits;
    }

    bits |= ((e - 112) << 10) | (m >> 1);
    /* Extra rounding. An overflow will set mantissa to 0 and increment
     * the exponent, which is OK. */
    bits += m & 1;
    return bits;
  };
})();

// This function converts a Float16 stored as the bits of a Uint16 into a Javascript Number.
// Adapted from: https://gist.github.com/martinkallman/5049614
// input is a Uint16 (eg, new Uint16Array([value])[0])

function float16ToNumber(input) {
  // Create a 32 bit DataView to store the input
  const arr = new ArrayBuffer(4);
  const dv = new DataView(arr);

  // Set the Float16 into the last 16 bits of the dataview
  // So our dataView is [00xx]
  dv.setUint16(2, input, false);

  // Get all 32 bits as a 32 bit integer
  // (JS bitwise operations are performed on 32 bit signed integers)
  const asInt32 = dv.getInt32(0, false);

  // All bits aside from the sign
  let rest = asInt32 & 0x7fff;
  // Sign bit
  let sign = asInt32 & 0x8000;
  // Exponent bits
  const exponent = asInt32 & 0x7c00;

  // Shift the non-sign bits into place for a 32 bit Float
  rest <<= 13;
  // Shift the sign bit into place for a 32 bit Float
  sign <<= 16;

  // Adjust bias
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
  rest += 0x38000000;
  // Denormals-as-zero
  rest = exponent === 0 ? 0 : rest;
  // Re-insert sign bit
  rest |= sign;

  // Set the adjusted float32 (stored as int32) back into the dataview
  dv.setInt32(0, rest, false);

  // Get it back out as a float32 (which js will convert to a Number)
  const asFloat32 = dv.getFloat32(0, false);

  return asFloat32;
}

// convert Uint16Array to Float32Array
function convertToFloat32Array(fp16_array) {
  const fp32_array = new Float32Array(fp16_array.length);
  for (let i = 0; i < fp32_array.length; i++) {
    fp32_array[i] = float16ToNumber(fp16_array[i]);
  }
  return fp32_array;
}

// convert Float32Array to Uint16Array
function convertToUint16Array(fp32_array, clip) {
  const fp16_array = new Uint16Array(fp32_array.length);
  for (let i = 0; i < fp16_array.length; i++) {
    let x = fp32_array[i]
    if (clip) {
        if (x < -1) x = -1
        else if (x > 1) x = 1
    }
    fp16_array[i] = toHalf(x);
  }
  return fp16_array;
}

const padNumber = (num, fill) => {
  let len = ("" + num).length;
  return Array(fill > len ? fill - len + 1 || 0 : 0).join(0) + num;
};

const getTime = () => {
  let date = new Date(),
    hour = padNumber(date.getHours(), 2),
    min = padNumber(date.getMinutes(), 2),
    sec = padNumber(date.getSeconds(), 2);
  return `${hour}:${min}:${sec}`;
};

const checkWebNN = async () => {
  let status = document.querySelector("#webnnstatus");
  let info = document.querySelector("#info");
  let webnnStatus = await webNnStatus();

  if (webnnStatus.webnn) {
    status.setAttribute("class", "green");
    info.innerHTML = "WebNN supported";
    load.disabled = false;
  } else {
    if (webnnStatus.error) {
      status.setAttribute("class", "red");
      info.innerHTML = `WebNN not supported: ${webnnStatus.error} <a id="webnn_na" href="../../install.html" title="WebNN Setup Guide">Set up WebNN</a>`;
      logError(`[Error] ${webnnStatus.error}`);
    } else {
      status.setAttribute("class", "red");
      info.innerHTML = "WebNN not supported";
      logError(`[Error] WebNN not supported`);
    }
  }

  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    status.innerHTML = "";
  }
};

let context
const webNnStatus = async () => {
  let result = {};
  try {
    context = await navigator.ml.createContext();
    if (context) {
      try {
        const builder = new MLGraphBuilder(context);
        if (builder) {
          result.webnn = true;
          return result;
        } else {
          result.webnn = false;
          return result;
        }
      } catch (e) {
        result.webnn = false;
        result.error = e.message;
        return result;
      }
    } else {
      result.webnn = false;
      return result;
    }
  } catch (ex) {
    result.webnn = false;
    result.error = ex.message;
    return result;
  }
};

const getQueryValue = (name) => {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name);
};

let textEncoderFetch = null;
let textEncoderCreate = null;
let textEncoderRun1 = null;
let unetFetch = null;
let unetCreate = null;
let unetRun1 = null;
let vaeRun1 = null;
let vaeEncoderRun1 = null;
let vaeFetch = null;
let vaeCreate = null;
let vaeEncoderFetch = null;
let vaeEncoderCreate = null;
let runTotal1 = null;
let dev = null;
let dataElement = null;
let loadwave = null;
let loadwaveData = null;

const updateLoadWave = (value) => {
  loadwave = document.querySelectorAll(".loadwave");
  loadwaveData = document.querySelectorAll(".loadwave-data strong");

  if (loadwave && loadwaveData) {
    loadwave.forEach((l) => {
      l.style.setProperty(`--loadwave-value`, value);
    });
    loadwaveData.forEach((data) => {
      data.innerHTML = value;
    });

    if (value === 100) {
      loadwave.forEach((l) => {
        l.dataset.value = value;
      });
    }
  }
};

const ui = async () => {
  const prompt = document.querySelector("#user-input");
  const title = document.querySelector("#title");
  const dev = document.querySelector("#dev");
  const dataElement = document.querySelector("#data");

  if (!getMode()) {
    dev.setAttribute("class", "mt-1");
    dataElement.setAttribute("class", "hide");
  }

  await setupORT('sd-turbo', 'dev');
  showCompatibleChromiumVersion('sd-turbo');

  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    title.innerHTML = "WebGPU";
  }
  await checkWebNN();

  // const img_div_ids = ["#img_div_0", "#img_div_1", "#img_div_2", "#img_div_3"];
  // [img_div_0, img_div_1, img_div_2, img_div_3] = img_div_ids.map(id => document.querySelector(id));

  const elementIds = [
    "#textEncoderFetch",
    "#textEncoderCreate",
    "#textEncoderRun1",
    "#vaeEncoderFetch",
    "#vaeEncoderCreate",
    "#vaeEncoderRun1",
    "#unetRun1",
    "#runTotal1",
    "#unetFetch",
    "#unetCreate",
    "#vaeFetch",
    "#vaeCreate",
    "#vaeRun1",
  ];

  [
    textEncoderFetch,
    textEncoderCreate,
    textEncoderRun1,
    vaeEncoderFetch,
    vaeEncoderCreate,
    vaeEncoderRun1,
    unetRun1,
    runTotal1,
    unetFetch,
    unetCreate,
    vaeFetch,
    vaeCreate,
    vaeRun1,
  ] = elementIds.map((id) => document.querySelector(id));

  switch (config.provider) {
    case "webgpu":
      if (!("gpu" in navigator)) {
        throw new Error("webgpu is NOT supported");
      }
      opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
      break;
    case "webnn":
      let webnnStatus = await webNnStatus();
      if (webnnStatus.webnn) {
        opt.executionProviders = [
          {
            name: "webnn",
            deviceType: config.devicetype
          },
        ];
      }
      break;
  }

  prompt.value = "Benjamin Netanyahu holding a white pigeon";
  // Event listener for Ctrl + Enter or CMD + Enter
  prompt.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key === "Enter")
        generate_image()
  });
  generate.addEventListener("click", function (e) {
    if (input_image_mode.value == 'output' || input_image_mode.value == 'cartoon')
        have_output = false
    generate_image()
  });
  image_strength.addEventListener("keydown", function (e) {
    if (e.key === "Enter")
        generate_image()
  });

  load.addEventListener("click", () => {
    if (config.provider === "webgpu") {
      hasFp16().then((fp16) => {
        if (fp16) {
          generate_image();
        } else {
          log(`[Error] Your GPU or Browser doesn't support webgpu/f16`);
        }
      });
    } else {
      generate_image();
    }
  });

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  //ort.env.logLevel = "verbose";
  //ort.env.debug = true;

  const path = "eyaler/sd-turbo-webnn/resolve/main/tokenizer";
  tokenizer = await AutoTokenizer.from_pretrained(path);
  tokenizer.pad_token_id = 0;

  await draw_input_image('input.jpg')
};

document.addEventListener("DOMContentLoaded", ui, false);


async function draw_input_image(src) {
  if (input_image_mode.value != 'output' && input_image_mode.value != 'cartoon')
    input_image_mode.value = 'image'
  const img = new Image()
  img.src = src
  await img.decode()
  const ctx = img_canvas_input.getContext('2d', {willReadFrequently: true})
  const minDim = Math.min(img.width, img.height)
  ctx.drawImage(img, (img.width-minDim)/2, (img.height-minDim)/2, minDim, minDim, 0, 0, 512, 512)
  have_output = false
}

const pickerOpts = {
  types: [
    {
      description: "Images",
      accept: {
        "image/*": [".png", ".gif", ".jpeg", ".jpg", ".webp"],
      },
    },
  ],
  excludeAcceptAllOption: true,
  multiple: false,
}

async function upload_image() {
    const [fileHandle] = await window.showOpenFilePicker(pickerOpts)
    await draw_input_image(URL.createObjectURL(await fileHandle.getFile()))
}

async function drop(ev) {
    ev.preventDefault()
    img_canvas_input.style.outline = 'none'
    if (ev.dataTransfer.items && ev.dataTransfer.items[0].kind == 'file')
        await draw_input_image(URL.createObjectURL(ev.dataTransfer.items[0].getAsFile()))
}

img_canvas_input.width = 512
img_canvas_input.height = 512
img_canvas_input.addEventListener('click', upload_image)
data_input.addEventListener('click', upload_image)
img_canvas_input.addEventListener('dragover', ev => ev.preventDefault())
img_canvas_input.addEventListener('dragenter', () => img_canvas_input.style.outline = '10px dashed green')
img_canvas_input.addEventListener('dragleave', () => img_canvas_input.style.outline = 'none')
img_canvas_input.addEventListener('dragend', () => img_canvas_input.style.outline = 'none')
img_canvas_input.addEventListener('drop', ev => drop(ev))

let stream
input_image_mode.addEventListener('change', async () => {
    if (input_image_mode.value == 'camera') {
        const video = document.createElement('video')
        try {
            stream = await navigator.mediaDevices.getUserMedia({video: true})
        }
        catch {
            input_image_mode.value = 'image'
            return
        }
        stream.addEventListener('inactive', () => input_image_mode.value = 'image')
        video.srcObject = stream
        video.play()
        const ctx = img_canvas_input.getContext('2d', {willReadFrequently: true})
        ctx.save()
        ctx.translate(512, 0)
        ctx.scale(-1, 1)
        const updateCanvas = () => {
            if (input_image_mode.value == 'camera') {
                const minDim = Math.min(video.videoWidth, video.videoHeight)
                ctx.drawImage(video, (video.videoWidth-minDim)/2, (video.videoHeight-minDim)/2, minDim, minDim, 0, 0, 512, 512)
                generate_image(false)
                video.requestVideoFrameCallback(updateCanvas)
            }
        }
        video.requestVideoFrameCallback(updateCanvas);
    } else {
        if (!generating)
            generate.disabled = false
        const ctx = img_canvas_input.getContext('2d', {willReadFrequently: true})
        ctx.restore()
        if (stream)
            stream.getTracks().forEach(track => track.stop())
        if (input_image_mode.value == 'output' || input_image_mode.value == 'cartoon') {
            const updateInput = async () => {
                if (input_image_mode.value == 'output' || input_image_mode.value == 'cartoon') {
                    generate_image(false)
                    setTimeout(updateInput)
                }
            }
            updateInput()
        }
    }
})
