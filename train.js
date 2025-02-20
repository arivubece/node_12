const tf = require('@tensorflow/tfjs-node'); // Correct package
const path = require('path');

// Training Data: Height (X) → Weight (y)
const heights = [150, 160, 170, 180, 190];
const weights = [50, 60, 70, 80, 90];

// Convert to Tensors
const X = tf.tensor2d(heights, [5, 1]);
const y = tf.tensor2d(weights, [5, 1]);

// Define Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

console.log("Model directory:", path.resolve(__dirname, 'model'));

// Train Model
model.fit(X, y, { epochs: 100 }).then(() => {
    console.log("Model trained!");

    // Save Model
    model.save('file://c:/node.js1/model').then(() => {
        console.log("Model saved!");
    }).catch(err => console.error("Save Error:", err));
}).catch(err => console.error("Training Error:", err));
