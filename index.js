
const vw = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
const vh = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
let arrowImage = new Image();
arrowImage.src = 'Assets/arro.png';
const start = 10;
const width = 0.208 * vw;
const height = 0.017 * vh;

let drawLayer = (layer, x, y, width, height) => {
    ctx.beginPath();
    ctx.fillStyle = "#c9ee59";
    ctx.fillRect(x, y, width, height);
    ctx.rect(x, y, width, height);
    ctx.strokeStyle = "#73d019";
    ctx.stroke();
    ctx.font = "10px serif";
    ctx.fillStyle = "#000"
    if (layer.attributes != null && 'activation' in layer.attributes) {
        ctx.fillText("Layer: " + layer.name, x + width / 2 - 60, y + 10);
        ctx.fillText("Activation: " + layer.attributes.activation, x + width / 2 + 5, y + 10);
    }
    else {
        ctx.fillText(layer.name, x + width / 2 - 20, y + 10);
    }
}

let drawArrow = (x, y) => {
    ctx.drawImage(arrowImage, x, y, 10, 10);
}

let drawModelArchitecture = (layers) => {
    let x = start;
    let y = start;
    layers.forEach(layer => {
        drawLayer(layer, x, y, width, height);
        drawArrow(width / 2 + 10, y + height + 2);
        y += (2 * height + 1);
    });
    drawLayer({
        name: 'Finish',
        attributes: {}
    }, x, y, width, height);
}

const dense = document.getElementById('dense');
const conv2d = document.getElementById('conv2d');
const maxPool2d = document.getElementById('maxPooling2d');
const canvas = document.getElementById('canvas');
const compile = document.getElementById('compile');
const analysis = document.getElementById('analysis');
const compileModel = document.getElementById('train-model');
const makeModel = document.getElementById('make-model');
const analyzeModel = document.getElementById('evaluate-model');
const pictorial = document.getElementsByClassName('pictorial');
const ctx = canvas.getContext('2d');

try {
    dense.addEventListener("click", () => {
        const submenu = document.getElementById('sub-menu-dense');
        submenu.classList.toggle('hidden');
    });
} catch (e) {
    console.log(e);
}

try {
    compile.addEventListener("click", () => {
        makeModel.classList.add('hidden');
        compileModel.classList.remove('hidden');
        pictorial[0].classList.add('hidden');
    });
} catch (e) {
    console.log(e);
}

try {
    analysis.addEventListener("click", () => {
        compileModel.classList.add('hidden');
        analyzeModel.classList.remove('hidden');
    })
} catch (e) {
    console.log(e);
}

try {
    maxPool2d.addEventListener("click", () => {
        const submenu = document.getElementById('sub-menu-maxPooling2d');
        submenu.classList.toggle('hidden');
    })
} catch (e) {
    console.log(e);
}

try {
    conv2d.addEventListener("click", () => {
        const submenu = document.getElementById('sub-menu-conv2d');
        submenu.classList.toggle('hidden');
    })
} catch (e) {
    console.log(e);
}