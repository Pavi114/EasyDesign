/**
 * Class LAYER:
 * A class to store information about a layer.
 * Currently supported layers: Dense, Conv2d, Flatten, MaxPooling2D
 * Prototypes:
 *            name: Name of the layer,
 *            attributes: Key-value pairs of layer parameters like input shape, 
 *                        activation, kernel function.
 *            addLayer(): Returns the tfjs layer according to the name and attributes chosen
 */

class Layer {
    constructor(name, attributes) {
        this.name = name;
        this.attributes = attributes;
    }

    addLayer() {
        if (this.name == 'dense') {
            let attr = {
                units: Number(this.attributes.outputShape),
                activation: this.attributes.activation
            };
            if ('inputShape' in this.attributes) {
                attr.inputShape = this.attributes.inputShape;
            }
            return tf.layers.dense(attr);
        }
        else if (this.name == 'conv2d') {
            let attr = {
                activation: this.attributes.activation,
                kernelSize: this.attributes.kernelSize
            };
            if ('inputShape' in this.attributes) {
                attr.inputShape = this.attributes.inputShape;
            }
            if ('filter' in this.attributes) {
                attr.filters = this.attributes.filter;
            }
            if ('strides' in this.attributes) {
                attr.strides = this.attributes.strides;
            }
            return tf.layers.conv2d(attr);
        }
        else if (this.name == 'maxPooling2d') {
            let attr = {};
            if ('poolSize' in this.attributes) {
                attr.poolSize = [this.attributes.poolSize, this.attributes.poolSize];
            }
            if ('strides' in this.attributes) {
                attr.strides = [this.attributes.strides, this.attributes.strides];
            }
            return tf.layers.maxPooling2d(attr);
        }
        else if (this.name == 'flatten') {
            return tf.layers.flatten();
        }
    }
}

/**
 * Class MODEL:
 * A class to store information about the architecture of the model. 
 * Prototypes:
 *              modelType: Denotes the type of model (sequential, model) though 
 *                              currently only sequential is supported.
 *              layers: Array of Layer objects denoting the layers of the model.
 *              model: A tfjs model object.
 *              createLayer(): A function to create and add a new Layer to the architecture.
 *                             Input =:   
 *                                      name => name of layer, 
 *                                      attr => attributes of the layer
 *                             Output =: None
 *              getOptimizer(): A function to return the tfjs optimizer object according to the optimizer chosen.
 *                              Input =: 
 *                                       optimizer => Name of the optimizer (adam, sgd, rmsprop).
 *                              Output =: 
 *                                       tfjs optimizer object
 *              getLoss(): A function to return the tfjs loss object according to the loss function chosen.
 *                         Input =:
 *                                  loss => Name of the loss (mse, categoricalCrossentropy).
 *                         Output =: 
 *                                  tfjs loss object
 *              trainModel(): A function to train the model given the inputs and outputs.
 *                              Input =:
 *                                      inputs => Training input to be given to the model (in tensors).
 *                                      labels => Actual Training Outputs of the dataset (in tensors).
 *                                      params => Object containing information about training 
 *                                                  (epochs, batch size, loss, optimizer, accuracy).
 *              evaluateModel(): A function to evaluate the model. 
 *                               Inputs =: 
 *                                          input => Test Input (in tensors)
 *                                          output => Test Output (in tensors)
 *                               Output =: 
 *                                          Return accuracy of Model.
 */

class Model {
    constructor(layers) {
        this.modelType = 'sequential';
        this.layers = [];
        this.model = tf.sequential();
    }

    createLayer = (name, attr) => {
        let layer = new Layer(name, attr);
        this.layers.push(layer);
        this.model.add(layer.addLayer());
    }

    getOptimizer = (optimizer) => {
        if (optimizer == 'adam')
            return tf.train.adam();
        else if (optimizer == 'sgd')
            return tf.train.sgd();
        else if (optimizer == 'rmsprop')
            return tf.train.rmsprop();
    }

    getLoss = (loss) => {
        if (loss == 'meanSquaredError')
            return tf.metrics.meanSquaredError
        else if (loss == 'categoricalCrossentropy')
            return tf.metrics.categoricalCrossentropy;
    }

    async trainModel(inputs, labels, params) {

        const loss = this.getLoss(params.loss);
        const optimizer = this.getOptimizer(params.optimizer);
        const batchSize = params.batchSize;
        const epochs = params.epochs;

        /**
         * Just for sample
         */

        if (isSample) {
            const TRAIN_DATA_SIZE = 5500;
            const data = new MnistData();
            await data.load();
            const [trainXs, trainYs] = tf.tidy(() => {
                const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
                return [
                    d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
                    d.labels
                ];
            });

            inputs = trainXs;
            labels = trainYs;
        }

        this.model.compile({
            optimizer: optimizer,
            loss: loss,
            metrics: params.metric
        });

        tfvis.visor().open();
        return await this.model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss'],
                { height: 200, callbacks: ['onEpochEnd'] }
            )
        });
    }

    async evaluateModel(input, output) {

        if (isSample) {
            const IMAGE_WIDTH = 28;
            const IMAGE_HEIGHT = 28;
            const testDataSize = 500;
            const data = new MnistData();
            await data.load();
            const testData = data.nextTestBatch(testDataSize);
            input = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
            output = testData.labels.argMax([-1]);
        }

        let predictData = this.model.predict(input).argMax([-1]);

        if (isSample) {
            const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
            const classAccuracy = await tfvis.metrics.perClassAccuracy(output, predictData);
            const container = { name: 'Accuracy', tab: 'Evaluation' };
            tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
        }

        let trueData = isSample ? output : output.argMax([-1]);
        let eval_test = await tfvis.metrics.accuracy(trueData, predictData);
        input.dispose();
        output.dispose();
        return eval_test;
    }
}

/**
 * getAttributes(): A function to get the attributes of a layer from the user. 
 *                  Input =: 
 *                           layerName => name of layer
 *                  Output =: 
 *                           Object containing layer params (activation, units, kernel, filters, strides) 
 * 
 * getTrainParams(): A function to get the attributes of training the model from the user.
 *                   Input =:
 *                           None
 *                   Output =:
 *                           Object containing training params (epoch, loss, batch size, metrics, optimizer)
 */

let getAttributes = (layerName) => {
    let attr = [];
    let inputShape = null;
    let activation = null;
    let outputShape = null;
    let filter = null;
    let strides = null;
    let kernel = null;
    let poolSize = null;
    if (['dense', 'conv2d'].includes(layerName)) {
        inputShape = document.getElementById(('input-shape-' + layerName));
        activation = document.getElementById(('activation-' + layerName));
    }
    if (layerName == 'conv2d') {
        filter = document.getElementById(('filter-' + layerName));
        kernel = document.getElementById('kernel-size-' + layerName);
    }
    if (['maxPooling2d', 'conv2d'].includes(layerName)) {
        strides = document.getElementById('stride-' + layerName);
    }
    if (layerName == 'dense')
        outputShape = document.getElementById(('output-shape-' + layerName));
    if (layerName == 'maxPooling2d') {
        poolSize = document.getElementById('pool-size-' + layerName);
    }
    if (model.layers.length != 0 && inputShape != null && inputShape.value != "") {
        alert("Invalid Input Shape");
        return;
    }
    if (model.layers.length == 0 && inputShape != null && (inputShape.value == "" || inputShape.value < 0)) {
        alert("Invalid Input Shape");
        return;
    }
    if (outputShape != null && (outputShape.value == "" || outputShape.value <= 0)) {
        alert("Invalid Output Shape");
        return;
    }
    if (kernel != null && kernel.value == "") {
        alert("Enter kernel size");
        return;
    }
    if (kernel != null && kernel.value <= 0) {
        alert("Invalid kernel size");
        return;
    }
    if (filter != null && filter.value <= 0 || strides != null && strides.value <= 0) {
        alert("Enter positive values");
        return;
    }
    if (poolSize != null && poolSize.value <= 0) {
        alert("Enter positive pool size");
        return;
    }
    if (model.layers.length == 0) {
        let iShape = [];
        let regex = /^[0-9,]+$/;
        if (!regex.test(inputShape.value)) {
            alert("Invalid Input Shape");
            return;
        }
        iShape = inputShape.value.split(',').map(num => Number(num));
        attr.inputShape = iShape;
        inputShape.value = "";
    }
    if (activation != null && activation[activation.selectedIndex].value != "Activation function") {
        attr.activation = activation[activation.selectedIndex].value;
        activation.selectedIndex = 0;
    }
    if (layerName == 'conv2d') {
        attr.kernelSize = Number(kernel.value);
        kernel.value = "";
        if (filter.value != "") {
            attr.filter = Number(filter.value);
            filter.value = "";
        }
        if (strides.value != "") {
            attr.strides = Number(strides.value);
            strides.value = "";
        }
    }
    if (layerName == 'dense') {
        attr.outputShape = Number(outputShape.value);
        outputShape.value = "";
    }
    if (layerName == 'maxPooling2d') {
        attr.poolSize = Number(poolSize.value);
        poolSize.value = "";
        attr.strides = Number(strides.value);
        strides.value = "";
    }
    return attr;
}

let getTrainParams = () => {
    let params = [];
    const epochs = document.getElementById('epochs');
    const batchSize = document.getElementById('batch-size');
    const optimizer = document.getElementById('optimizer');
    const metrics = document.querySelectorAll('input[type=checkbox]:checked');
    const loss = document.getElementById('loss');
    if (epochs.value == "" || epochs.value <= 0) {
        alert("Invalid epochs count");
        return;
    }
    if (batchSize.value == "" || batchSize.value % 2 != 0) {
        alert("invalid batch size");
        return;
    }
    params.epochs = Number(epochs.value);
    params.batchSize = Number(batchSize.value);
    params.optimizer = optimizer[optimizer.selectedIndex].value;
    params.loss = loss[loss.selectedIndex].value;
    params.metrics = [];
    metrics.forEach(metric => {
        params.metrics.push(metric.value);
    });
    return params;
}

let model = new Model([]);
let isSample = false;

const addDenseButton = document.getElementById('add-dense');
const addConv2dButton = document.getElementById('add-conv2d');
const addMaxPool2D = document.getElementById('add-maxPooling2d');
const flatten = document.getElementById('flatten');
const trainModel = document.getElementById('train');
const evaluateModel = document.getElementById('evaluate');
const summary = document.getElementById('summary');
const saveModel = document.getElementById('save-model');
const summaryList = summary.getElementsByTagName('li');
const summaryModel = document.getElementById('model-summary');
const sample = document.getElementById('sample');

try {
    addDenseButton.addEventListener("click", (event) => {
        let attr = [];
        attr = getAttributes('dense');
        if (attr != null) {
            model.createLayer('dense', attr);
            drawModelArchitecture(model.layers);
        }
    });
}
catch (e) {
    console.log(e);
}

try {
    addConv2dButton.addEventListener("click", () => {
        let attr = [];
        attr = getAttributes('conv2d');
        if (attr != null) {
            model.createLayer('conv2d', attr);
            drawModelArchitecture(model.layers);

        }
    })
} catch (e) {
    console.log(e);
}

try {
    addMaxPool2D.addEventListener("click", () => {
        if (model.layers.length == 0) {
            alert("No Input Available");
            return;
        }
        else {
            let attr = [];
            attr = getAttributes('maxPooling2d');
            if (attr != null) {
                model.createLayer('maxPooling2d', attr);
                drawModelArchitecture(model.layers);
            }
        }
    })
} catch (e) {
    console.log(e);
}

try {
    flatten.addEventListener("click", () => {
        if (model.layers.length == 0) {
            alert("No Input Available");
            return;
        }
        else {
            model.createLayer('flatten', null);
            drawModelArchitecture(model.layers);
        }
    })
} catch (e) {
    console.log(e);
}

try {
    trainModel.addEventListener("click", async () => {
        if (model.layers.length <= 0) {
            alert("No layers available to compile");
            return;
        }
        let params = getTrainParams();

        let inputData = document.getElementById('input-train-data').files[0];
        let outputData = document.getElementById('output-train-data').files[0];
        let data = await new Response(inputData).text()
        let input = JSON.parse(data).map(obj => Object.values(obj));
        data = await new Response(outputData).text()
        let output = JSON.parse(data).map(obj => Object.values(obj));
        input = tf.tensor2d(input, [input.length, 4]);
        output = tf.tensor2d(output, [output.length, 1]);

        if (params == null) {
            return;
        }

        await model.trainModel(input, output, params);
    });

} catch (e) {
    console.log(e);
}

try {
    evaluateModel.addEventListener("click", async () => {

        let testInput = null;
        let testOutput = null;
        if (!isSample) {
            let inputData = document.getElementById('input-test-data').files[0];
            let outputData = document.getElementById('output-test-data').files[0];
            let data = await new Response(inputData).text()
            testInput = JSON.parse(data).map(obj => Object.values(obj));
            data = await new Response(outputData).text()
            testOutput = JSON.parse(data).map(obj => Object.values(obj));
            testInput = tf.tensor2d(testInput, [testInput.length, 4]);
            testOutput = tf.tensor2d(testOutput, [testOutput.length, 1]);
        }

        const accuracy = await model.evaluateModel(testInput, testOutput);
        const epochs = document.getElementById('epochs');
        summary.classList.remove('hidden');
        summaryList[0].innerHTML += epochs.value;
        summaryList[1].innerHTML += accuracy;
    })
} catch (e) {
    console.log(e);
}

try {
    saveModel.addEventListener("click", async () => {

        return await model.model.save('downloads://model');
    })
} catch (e) {
    console.log(e);
}

try {
    summaryModel.addEventListener("click", () => {
        tfvis.visor().toggle();
        const surface = { name: 'Layer Summary', tab: 'Model Inspection' };
        tfvis.show.modelSummary(surface, model.model);
    })
} catch (e) {
    console.log(e);
}

try {
    sample.addEventListener("click", async () => {
        isSample = true;
        let params = getTrainParams();
        let input = null;
        let output = null;
        await model.trainModel(input, output, params);
    })
} catch (e) {
    console.log(e);
}