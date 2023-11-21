# 🌞 Solar Panels Dust Detection

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

__Solar panels__ have become integral components across various industries, such as residential, agricultural, manufacturing, healthcare, and retail. Over their extended use, these panels often accumulate dust, influenced by climate, surrounding vegetation, limited maintenance, and bird droppings. This accumulation hampers their performance as the dust layer obstructs sunlight, the core energy source for solar cells. Consequently, this leads to a noticeable reduction in the panels' efficiency and lifespan, while also escalating operational costs and resulting in less efficient utilization of resources. 

<img src = "https://github.com/suhasmaddali/Images/blob/main/Solar%20Panel%20GitHub%20Images.jpg" />

## Challenges

Utilizing solar panels for electricity generation presents certain challenges, chief among them being their inherently low efficiency. Nonetheless, they prove highly effective in regions with plentiful sunlight. A significant factor that further diminishes their efficiency is the accumulation of dust on their surfaces. This layer of dust acts as a barrier, hindering the panels' ability to absorb solar radiation effectively, thus compromising their overall performance in harnessing solar energy.

## Deep Learning and Data Science

* The advancement of deep learning models, particularly those specializing in image recognition, offers a promising solution to the issue of dust accumulation on solar panels.
* Utilizing Convolutional Neural Networks (CNNs) for image processing allows for the extraction of valuable insights, enabling predictions about the cleanliness or dustiness of solar panels. 
* The past decade has witnessed a surge in the development of transformers, notably in vision-related tasks (termed vision transformers), enhancing image analysis capabilities.
* Implementing learning strategies like transfer learning can further enhance performance while reducing the need for extensive training.
* Consequently, these technological advancements pave the way for more effective systems to detect dust on solar panels, thereby increasing efficiency and lowering operational and maintenance costs.

## Exploratory Data Analysis (EDA)

* Conducting exploratory data analysis on the image dataset can uncover unique patterns, facilitating targeted feature engineering to boost model accuracy.
* Initial examination of the solar panel images reveals a wide variety of inconsistent representations of dust accumulation. 
* Hence, it becomes crucial to gather a more uniform and representative dataset specifically focused on dusty solar panels.
* Given the relatively small size of the dataset, there's an elevated risk of overfitting, particularly when employing complex models via transfer learning. 
* Adhering to these steps is essential to maximize the effectiveness and value of our product in detecting dust on solar panels.

## Feature Engineering

* The input data, comprising image arrays, is initially unnormalized.
* To address this, we implement a normalization step, adjusting the pixel values to have a maximum of 1 and a minimum of 0.
* We will employ Feature Engineering techniques like semantic segmentation to enhance the efficacy of our chosen deep learning image recognition model.

## Metrics

In this scenario, with only two classes (clean or dusty), the task is framed as a binary classification problem. Our approach involves utilizing binary cross-entropy loss as the primary criterion for optimizing the weights of deep neural networks, continuing this iterative process until the desired model performance is achieved. The following are the key metrics that will be employed to monitor and assess the performance of various models.

* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1-Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__ROC-AUC Curves__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

## Deep Learning Models

Our strategy involves experimenting with a variety of Convolutional Neural Networks (CNNs), each configured differently, in conjunction with diverse transfer learning techniques. This approach aims to identify the most effective model for deployment. The following list details the specific models that will be evaluated in this process.

<table>
  <tr>
    <td>
      <ul>
        <li>VGG16</li>
        <li>VGG19</li>
        <li>InceptionNet</li>
        <li>MobileNet</li>
        <li>Xception</li>
        <li>MobileNetV2</li>
        <li>ResNet50</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>CNN (Configuration 1)</li>
        <li>CNN (Configuration 2)</li>
        <li>CNN (Configuration 3)</li>
        <li>CNN (Configuration 4)</li>
        <li>VGG16 (Feature Extractor) + ML Models</li>
        <li>MobileNet (Feature Extractor) + ML Models</li>
        <li>InceptionNet (Feature Extractor) + ML Models</li>
        <li>CNN (Feature Extractor) + ML Models</li>
      </ul>
    </td>
  </tr>
</table>

## Outcomes

* Since the optimal model demonstrates strong performance on the solar panel dataset, it can be deployed for real-time applications in a lightweight manner, suitable for mobile and IoT devices.
* Utilizing frameworks like Tensorflow Extended (TFX), the deployment, debugging, and maintenance of a broad spectrum of models become streamlined when transitioning into production, contributing to an efficient machine learning lifecycle.

## Future Scope

* Since the optimal model demonstrates strong performance on the solar panel dataset, it can be deployed for real-time applications in a lightweight manner, suitable for mobile and IoT devices.
* Utilizing frameworks like Tensorflow Extended (TFX), the deployment, debugging, and maintenance of a broad spectrum of models become streamlined when transitioning into production, contributing to an efficient machine learning lifecycle.

## 💻 Training with NVIDIA's RTX 2080 GPU

* GPU-accelerated deep learning frameworks provide the versatility needed to design and train deep neural networks effectively.
* Leveraging cuDNN and Nvidia's graphics drivers enabled rapid training of models utilizing GPU cores instead of CPU cores, markedly enhancing efficiency.
* This approach resulted in a substantial acceleration in the training and development of convolutional neural networks (CNNs), optimizing the overall process.

## Thanks