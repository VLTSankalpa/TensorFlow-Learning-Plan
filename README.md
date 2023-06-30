# TensorFlow Learning Plan

## TensorFlow API Hierarchy: AI TensorFlow contains multiple abstraction layers

TensorFlow is an open-source, high-performance library for numerical computation that uses directed graphs. TensorFlow graphs are portable between different devices. You can write your computation code in a high-level language like Python and execute it quickly at run time. You can build a DAG in Python, store it in a saved model, and restore it in a C++ program for low latency predictions.

## TensorFlow Toolkit

- "A tensor is like a multi-dimensional spreadsheet filled with data."
- "In TensorFlow, these multi-dimensional spreadsheets (tensors) move through a flowchart (graph) of operations - that's why it's called TensorFlow."
- "The flowchart of operations (Directed Acyclic Graph) is a blueprint of your model's calculations, not tied to any specific programming language."
- "You can make your model's blueprint (DAG) in Python, save it, and then use it in a faster language like C++ for quicker results."
- "With TensorFlow, you can write your model in Python and have it run on various kinds of hardware - your computer's processor (CPU), graphics card (GPU), or even specialized AI chips (TPUs)."
- "TensorFlow's engine is smart - it can optimize your model to get the best performance from whatever hardware it's running on."

## How to Master TensorFlow: Your Learning Plan

1. **Computer Vision with TensorFlow:**

   a. **Theory:** Study Convolutional Neural Networks (CNN) and its variants, which are commonly used in computer vision tasks and datasets. Understand how CNNs process image data.

   b. **Practice:** Implement image classification, object detection, image segmentation, new image content generation, etc. using TensorFlow. Start with simple datasets like MNIST, then move to more complex ones like CIFAR-10, ImageNet, etc.

2. **Natural Language Processing (NLP) with TensorFlow:**

   a. **Theory:** Learn the basics of NLP, text preprocessing techniques, and popular models like RNN, LSTM, GRU, and Transformer-based models. Study the architecture of the Language Model (LM) and how these models learn to generate text.

   b. **Practice:** Implement these techniques and models in TensorFlow. Start with small projects like sentiment analysis, text classification,  new text content generation, summarisation, and question answering and then move to build your language models for unique domain requirements.

3. **Building Custom Time Series Forecasting Models Using Neural Networks:**

   a. **Theory:** Learn about different architectures used for time series forecasting, particularly LSTM (Long Short Term Memory) and Transformer models.

   b. **Practice:** Implement these models in TensorFlow for multivariate time series forecasting. Be sure to understand how to incorporate static and dynamic variables.

4. **TensorFlow Functional API:**

   a. **Theory:** Understand the difference between TensorFlow's Sequential and Functional API. Learn how the Functional API is more flexible and how it allows you to build more complex models with shared layers, multiple inputs or outputs.

   b. **Practice:** Implement neural networks using TensorFlow's Functional API. Experiment with shared layers, branched architectures, and multiple inputs or outputs.

5. **TensorFlow on Different Hardware and Distributed Training:**

   a. **Theory:** Understand how TensorFlow can leverage different types of hardware including CPUs, GPUs, and TPUs (Tensor Processing Units). Learn about distributed training, and how it allows you to scale up the training process across multiple devices and machines.

   b. **Practice:** Implement distributed training in TensorFlow. Experiment with different configurations, and compare the performance on different hardware.

6. **TensorFlow Extended (TFX) and ML Pipelines:**

   a. **Theory:** Learn about TFX, an end-to-end platform for deploying production ML pipelines. Understand the different components like Data Validation, Transform, Trainer, Evaluator, etc.

   b. **Practice:** Implement a simple ML pipeline using TFX and TensorFlow. Then, try to automate it for end-to-end ML workflows.

7. **TensorFlow Model Hosting and Monitoring:**

   a. **Theory:** Learn about serving models in TensorFlow, versioning, and how to use TensorFlow with Docker and Kubernetes. Study model monitoring and understand concepts like data skew and model drift.

   b. **Practice:** Deploy a model using TensorFlow Serving. Then, implement model monitoring to detect data skew and model drift, and create a system for automatic retraining.

8. **Data Preprocessing with TensorFlow and Keras:**

   a. **Theory:** Understand the built-in data preprocessing capabilities of TensorFlow and Keras.

   b. **Practice:** Replace your manual preprocessing steps with TensorFlow and Keras data preprocessing layers. Measure the performance improvements.

9. **Generative AI with TensorFlow:**

   a. **Theory:** Learn about Generative Models like GANs (Generative Adversarial Networks), VAEs (Variational Autoencoders), and their applications. Understand how they can generate new data that's similar to the data they were trained on.

   b. **Practice:** Implement GANs and VAEs in TensorFlow to generate images, music, or any other type of data. Experiment with different architectures and loss functions.

10. **Restricted Boltzmann Machines (RBM) for Recommendation Systems:**

    a. **Theory:** Understand the theory behind RBMs, a type of generative artificial neural network that can learn a probability distribution over its set of inputs. Also, delve into how recommendation systems work. They are a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.

    b. **Practice:** Implement an RBM in TensorFlow for a recommendation system. A practical application of this would be in suggesting products to online shoppers, recommending songs on Spotify, or movies on Netflix.

11. **Autoencoders for Anomaly Detection:**

    a. **Theory:** Start by understanding the theory behind autoencoders, a type of neural network used for learning efficient codings of input data. Also, understand the concept of anomaly detection, which is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

    b. **Practice:** Implement an autoencoder in TensorFlow for anomaly detection. A common application of this would be in detecting fraudulent transactions, system intrusions, or in medical anomaly detection.

12. **TensorFlow Mobile and Edge Deployment:**

    a. **Theory:** Learn about TensorFlow Lite, which allows you to run TensorFlow models on mobile and edge devices with low latency. Understand the conversion process from a TensorFlow model to TensorFlow Lite.

    b. **Practice:** Convert a TensorFlow model to TensorFlow Lite format, and deploy it on a mobile or edge device. Test the performance and make necessary optimizations.

## Getting Started: Setup your TensorFlow Development Environment (Cloud)

- **Google Colab**: Completely free and accessible with a Gmail account, Google Colab offers a GPU-accelerated environment. However, due to shared hardware usage, performance may vary and be slower during peak times. It also has timeouts.
- **Google Cloud Free Tier**: Google Cloud provides a $300 credit for three months, including access to Vertex AI Workbench and a Tesla T4 16GB GPU for free. A credit or debit card with international transaction capability is required, but rest assured that you will not be charged.
- **Local Laptop/Desktop with NVIDIA GPU**: If you have a local machine equipped with an NVIDIA GPU, you can set up TensorFlow to utilize it. For Mac M1, you'll need to install Conda Forge and set up the TensorFlow environment. For an Intel-based PC, installation of NVIDIA drivers and TensorFlow GPU Docker is required.
- **Local Laptop/Desktop with CPU**: For smaller, less computation-intensive deep learning models, a CPU setup can suffice.

## Getting Started: Setup your TensorFlow Development Environment (Local)

If you have a local machine equipped with an Apple M1 Metal GPU or NVIDIA GPU, you can set up TensorFlow to utilize it.

- For Mac M1, you'll need to install Conda Forge and set up the TensorFlow environment. [Instructional Video](https://www.youtube.com/watch?v=5DgWvU0p2bk)
- For an Intel-based PC, installation of NVIDIA drivers and TensorFlow GPU Docker is required.
    ```
    docker pull tensorflow/tensorflow:latest-gpu-jupyter
    docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
    ```

## Your 1st Neural Network: TensorFlow Hello World

```python
# Import required libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
# Print the version of TensorFlow
print(tf.__version__)
# Build a simple Sequential model - a linear model with one layer
# Dense layer with one neuron (units=1) and input of shape 1 (input_shape=[1])
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compile the model - using stochastic gradient descent (sgd) as optimizer and mean squared error as loss function
model.compile(optimizer='sgd', loss='mean_squared_error')
# Declare model inputs and outputs for training
# xs is the array of input data, ys is the array of corresponding output data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# Train the model for 500 epochs (an epoch is a complete pass through the dataset)
model.fit(xs, ys, epochs=500)
# Use the model to make a prediction - here we're predicting the output for input 10.0
print(model.predict([10.0]))
```

## Colab Notebook Files

[1. Your 1st Neural Network: TensorFlow Hello World](https://colab.research.google.com/drive/1NJJMPdd_sHYG3D84_Jo3AtzqLDGJL35C?usp=sharing)

[2. Fashion MNIST: Simple Neural Network](https://colab.research.google.com/drive/1PYppXSmepg_qQOyZEi6dB6bxNcwGTEQv?usp=sharing)

[3. Callbacks](https://colab.research.google.com/drive/1M23VC7b5vCa9D1f0jvUqN1z8VeAX2C3w?usp=sharing)

[4. Fashion MNIST: How CNNs process image data](https://colab.research.google.com/drive/1beui00TQkDnd9XhENcW61CPBAfiP4FeO?usp=sharing)

[5. Kaggle Cats vs. Dogs dataset](https://colab.research.google.com/drive/1jFunzJyPCP1XUrxEcyV9cRPk3J6IoEu8?usp=sharing)

[6. Data Augmentation](https://colab.research.google.com/drive/1CcDEKmAhm2KKOOsAYIrxz88VRHkIotAo?usp=sharing)

[7. Transfer Learning in computer vision](https://colab.research.google.com/drive/1_KJmUyjeVjHWERo7A8WJfogMUuInGcU8?usp=sharing)




