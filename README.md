🧠 Clustering CNN Features for Breast Cancer Prediction Using Swarm Intelligence

Author: Gokul Ganesan

Repo: Clustering-CNN-Features-For-Breast-Cancer-Prediction

⸻

## 📌 Overview

This project explores the integration of Swarm Intelligence algorithms into Computer Vision, applying them to the clustering of breast cancer histopathological images. We extract features using a fine-tuned VGG16 CNN and compare traditional KMeans clustering against a biologically-inspired Flocking Algorithm (e.g., Clusterflock / Flock by Leader).

The goal: Unsupervised learning for classification, which reduces the risks of human labeling errors in medical diagnoses.

⸻

❓ Research Question

How can Swarm Intelligence improve the unsupervised classification of breast cancer images based on CNN-extracted features?

⸻

🚀 Motivation

•	Swarm intelligence is biologically inspired and novel in computer vision applications.

•	It has proven effective in tasks like gene clustering and document similarity (e.g., TF-IDF spaces).

•	Applying it to high-dimensional CNN feature spaces provides insight into non-linear clustering capabilities.

•	Real-world significance: Enhancing medical imaging diagnostics without human bias.

⸻

🗃️ Dataset

BreakHis – Breast Cancer Histopathological Image Dataset
Source: Kaggle

•	7,909 total images

•	Magnifications: 40x, 100x, 200x, 400x (ignored for this project)

•	Classification:

•	2 main categories: Benign and Malignant

•	4 subtypes within each → 8 subtypes total

⸻

🧠 Methodology

🔍 Feature Extraction

•	Model: VGG16 with ImageNet weights

•	Fine-tuned on the BreakHis dataset (Dense layer trained, base layers frozen)

•	Extracted feature vectors (512-D) from the global average pooling layer

•	Applied PCA (n=50) to reduce noise and dimensionality

🔗 Clustering Algorithms

1.	KMeans
	
 	•	Clustering into k=2 (benign vs malignant) and k=8 (subtypes)

2.	Swarm Intelligence (Flock by Leader / Clusterflock)
	
 	•	Each image = virtual “bird”
	
 	•	Assigns clusters (“flocks”) based on neighbor proximity
	
 	•	Key hyperparameters:
	
 	•	max_distance = 0.1
	
 	•	min_samples = 40
	
 	•	Bayesian search used for tuning


📊 Results

<img width="1285" height="690" alt="Screenshot 2025-07-21 at 8 37 51 PM" src="https://github.com/user-attachments/assets/b2cfc320-fb4d-4234-9571-aadb6ef41b88" />

<img width="1277" height="715" alt="Screenshot 2025-07-21 at 8 38 22 PM" src="https://github.com/user-attachments/assets/7f744459-09da-453c-a196-c7c7b3bb5076" />


•	Fine-tuning the CNN significantly improved clustering performance.
•	Swarm Intelligence outperformed traditional clustering in both binary and multi-class classification.
•	Demonstrated strong handling of non-linear and high-dimensional feature relationships.

⸻

📈 Visualizations
	•	PCA (2D) and t-SNE plots were used to illustrate cluster separations.
	•	Swarm Intelligence clusters displayed non-linear boundaries, unlike KMeans.

⸻

⚙️ Technical Details
	•	Frameworks: TensorFlow / Keras, Scikit-learn, Matplotlib, NumPy
	•	Training:
	•	Initial 5 epochs on Dense layer (freeze base)
	•	Additional 10 epochs full-model training
	•	Dropout 0.5 added to avoid overfitting

⸻

📚 Resources
	•	Clusterflock: A flocking algorithm
	•	Flock by Leader: Biologically Inspired Clustering
	•	BreakHis Dataset

⸻

✅ Conclusion

Swarm Intelligence offers a powerful, flexible method for clustering CNN feature vectors—especially in medical imaging, where unsupervised learning reduces reliance on possibly flawed labels. This project demonstrates its superiority over traditional methods like KMeans in capturing complex, high-dimensional similarity structures.
