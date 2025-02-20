const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

let model;

// Load Trained Model
(async function () {
    model = await tf.loadLayersModel('file://./model/model.json');
    console.log("Model Loaded!");
})();

app.post('/predict', async (req, res) => {
    const { height } = req.body;
    const input = tf.tensor2d([height], [1, 1]);
    const prediction = model.predict(input);
    const weight = (await prediction.data())[0];

    res.json({ predicted_weight: weight.toFixed(2) });
});

app.listen(3000, () => console.log('ML API running on port 3000'));
