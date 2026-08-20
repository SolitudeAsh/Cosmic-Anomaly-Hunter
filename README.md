🌌 **Cosmic Anomaly Hunter**

**Project Summary Report**

**1. Project Overview**

**Cosmic Anomaly Hunter** is an interactive AI‑driven tool designed to assist astronomers in identifying interesting or unusual structures in large astronomical image archives. The system combines **supervised classification** (galaxy morphology), **unsupervised anomaly detection**, and **explainability** (Grad‑CAM) into a user‑friendly Streamlit web application.

The central research question is:

*Can AI help astronomers identify interesting or unusual structures in large astronomical image archives that could otherwise be overlooked?*

The project demonstrates a complete end‑to‑end pipeline from raw data ingestion to interactive visual exploration.

**2. Data Acquisition & Preparation**

We used the **Galaxy Zoo 2 (GZ2)** dataset, which provides volunteer classifications for hundreds of thousands of SDSS galaxies.

**Data files:**

- zoo2MainSpecz.csv – Contains morphological vote fractions (e.g., t01\_smooth\_or\_features\_a01\_smooth\_fraction, t04\_spiral\_a08\_spiral\_fraction, t08\_odd\_feature\_a24\_merger\_fraction).
- gz2\_filename\_mapping.csv – Maps SDSS dr7objid to image asset\_id (used for file naming).
- **Image files** – Supplied as .jpg images named by asset\_id (e.g., 1.jpg, 2.jpg, …).

**Data preparation steps:**

1. **Label assignment** – We defined three classes:
   
   - **0** – Elliptical / Smooth (if smooth\_fraction > 0.5).
   - **1** – Spiral (if spiral\_fraction > 0.5).
   - **2** – Merger / Other (if merger\_fraction > 0.5 or fallback).

2) **Merging** – Joined the classification data with the filename mapping on dr7objid.
3) **Image existence check** – Verified that all matched images were present; missing entries were dropped.
4) **Final metadata** – Saved as metadata.csv with columns: filename, dr7objid, asset\_id, label.

**Dataset Summary:**

- **Final dataset size:** \~50,000 images (exact count depends on the downloaded subset)
- **Class distribution:**
  
  - **0 (Elliptical):** \~40%
  - **1 (Spiral):** \~50%
  - **2 (Merger/Other):** \~10%

**3. AI / Data Science Components**

**3.1 Classifier (Identify Mode)**

- **Architecture:** ResNet50 pre‑trained on ImageNet, fine‑tuned for 3‑class classification.
- **Training:** Adam optimizer, learning rate 1×10−4, batch size 32, 3 epochs.
- **Performance:** Achieved **\~95% validation accuracy** after 3 epochs.
- **Output:** Predicted class (Elliptical/Spiral/Merger) and confidence score.

**3.2 Anomaly Detection (Discover Mode)**

- **Approach:** Convolutional Autoencoder trained **only on normal galaxies** (labels 0 and 1).
- **Architecture:** 5‑layer encoder / 5‑layer decoder with transposed convolutions.
- **Anomaly score:** Reconstruction Mean Squared Error (MSE) between input and reconstructed image.
  
  - High MSE = unusual object (potential merger, lens, or artifact).
- **Training:** 20 epochs, MSE loss, Adam optimizer.
- **Result:** The autoencoder reconstructs normal galaxies well, but poorly reconstructs mergers/others, thus flagging them as anomalies.

**3.3 Explainability (Explore Mode)**

- **Method:** Grad‑CAM (Gradient‑weighted Class Activation Mapping).
- **Implementation:** Hooked the last convolutional layer of ResNet50 to obtain gradients and activations; generated a heatmap overlay highlighting the regions most influential for the predicted class.
- **Benefit:** Provides visual interpretability, allowing users to see *why* the model made a certain prediction.

**3.4 Dimensionality Reduction (Planned)**

- **Technique:** t‑SNE on feature embeddings from the classifier’s penultimate layer.
- **Purpose:** To create a 2D map where similar galaxies cluster together, aiding in visual exploration and discovery of outliers.

**4. System Architecture & Workflow**

The end‑to‑end pipeline is implemented in Python and deployed as a **Streamlit web application**.

**Workflow:**

1. **User uploads** an astronomical image (or the app loads a sample from the dataset).
2. **Image preprocessing** – Resize to 224×224, normalize with ImageNet statistics.
3. **Classifier forward pass** – Obtains class probabilities and confidence.
4. **Autoencoder reconstruction** – Computes MSE anomaly score.
5. **Grad‑CAM computation** – Generates heatmap for the predicted class.
6. **Display** – Shows original image, heatmap overlay, prediction, confidence, and anomaly score side‑by‑side.

**Technology stack:**

- **Deep Learning:** PyTorch, torchvision.
- **Data handling:** Pandas, NumPy, PIL.
- **Visualization:** Matplotlib, Streamlit.
- **Explainability:** Custom Grad‑CAM implementation with OpenCV.

**5. Application Features**

The Streamlit app provides:

- **Upload** – Drag‑and‑drop or browse for JPG/PNG/JPEG images.
- **Sample fallback** – If no image is uploaded, the first image from the metadata is used.
- **Side‑by‑side view** – Original image on the left, Grad‑CAM heatmap overlay on the right.
- **Predictions** – Displays the predicted class (e.g., “Elliptical”) with confidence percentage.
- **Anomaly Score** – A numerical score; higher values indicate more unusual structures.

**Example output:**

- Prediction: **Elliptical (confidence: 99.12%)**
- Anomaly Score: **3492.19** (higher = more unusual)

**6. Results & Preliminary Evaluation**

- The classifier achieves high accuracy on validation data, indicating that the fine‑tuned ResNet50 can reliably distinguish between elliptical, spiral, and merger galaxies.
- The autoencoder successfully assigns higher reconstruction errors to galaxies labelled as “Merger/Other” compared to normal galaxies, validating its anomaly‑detection capability.
- Grad‑CAM heatmaps qualitatively align with human intuition (e.g., focusing on spiral arms for spiral galaxies, or on the central bulge for ellipticals).

**7. Discussion & Strengths**

- **Combines supervised and unsupervised learning** – Demonstrates versatility in applying both paradigms to a real‑world problem.
- **Explainable AI** – Grad‑CAM opens the black box, building trust with end‑users.
- **Interactive interface** – Makes the tool accessible to non‑experts (astronomers, educators, citizen scientists).
- **Scalability** – The pipeline can be extended to larger datasets (e.g., full GZ2, Hubble, JWST).
- **Portfolio appeal** – Visually striking, scientifically relevant, and technically comprehensive.

**Challenges encountered:**

- Image file naming discrepancies – resolved by using asset\_id from the mapping file.
- Streamlit deprecation warnings – updated use\_column\_width to use\_container\_width.
- Grad‑CAM hook warnings – non‑full backward hooks, but functionality works.

**8. Future Work & Enhancements**

- **Bounding box detection** – Incorporate an object detector (e.g., YOLO) to locate multiple galaxies in a single field.
- **Multi‑band imagery** – Support FITS files and colour composites from SDSS/HST.
- **Gravitational lens candidate search** – Fine‑tune the anomaly detector specifically for arc‑like features.
- **Active learning** – Integrate human feedback to improve the anomaly ranking.
- **Deployment** – Host the app on a cloud platform (e.g., Streamlit Cloud, Hugging Face Spaces) for public use.
- **Embedding map** – Add a t‑SNE visualization to browse the entire dataset and spot outliers interactively.

**9. Conclusion**

The **Cosmic Anomaly Hunter** successfully demonstrates how modern AI techniques can assist astronomers in sifting through massive image archives. By combining classification, anomaly detection, and explainability in an interactive dashboard, the tool not only performs accurate predictions but also provides visual justifications and flags potentially interesting candidates for human follow‑up.

This project stands as a strong portfolio piece, showcasing skills in:

- Data engineering (cleaning, merging, verifying).
- Deep learning (transfer learning, autoencoders).
- Explainable AI (Grad‑CAM).
- Full‑stack development (Streamlit, PyTorch).

The project tagline – *“Let AI search the universe for what humans might have missed”* – captures the essence of its scientific and exploratory value.

**10. Project Repository & Files – Detailed Breakdown**

All source code, data preparation scripts, model training, and the app are organised in the following structure:

Plaintext

Cosmic hunter/

├── Data/

│   ├── zoo2MainSpecz.csv              # Raw GZ2 vote fractions

│   ├── gz2\_filename\_mapping.csv       # Raw ID‑to‑filename mapping

│   ├── metadata.csv                   # Curated master file (generated)

│   └── images\_gz2/images/             # All .jpg galaxy images

├── models/

│   ├── classifier\_resnet50.pth        # Trained classifier weights

│   └── autoencoder.pth                # Trained autoencoder weights

├── src/

│   ├── prepare\_data.py                # Data preparation & merging

│   ├── data\_loader.py                 # PyTorch Dataset & transforms

│   ├── classifier.py                  # Trains the ResNet50 classifier

│   ├── autoencoder.py                 # Trains the anomaly‑detection AE

│   ├── explain.py                     # Grad‑CAM implementation

│   ├── embedding.py                   # (Planned) t‑SNE embedding

│   └── app.py                         # Streamlit interactive frontend

└── requirements.txt                   # Python dependencies

**10.1 Data Files (CSVs)**

**File**

**Role**

**Consumed By**

**zoo2MainSpecz.csv**

Raw classification data from Galaxy Zoo 2. Contains volunteer vote fractions for morphological features, e.g. t01\_smooth\_or\_features\_a01\_smooth\_fraction(smoothness), t04\_spiral\_a08\_spiral\_fraction (spiral structure), and t08\_odd\_feature\_a24\_merger\_fraction (merger signature).

prepare\_data.py

**gz2\_filename\_mapping.csv**

Raw mapping file. Links the SDSS object ID (dr7objid) to a numerical asset\_id(used for the actual image filenames, e.g. 1.jpg, 2.jpg).

prepare\_data.py

**metadata.csv**

**Generated master file** (output of prepare\_data.py). Contains only the images that were successfully found on disk. Columns: filename (e.g. 1.jpg), dr7objid (SDSS ID), asset\_id (numeric ID), and label (0 = Elliptical, 1 = Spiral, 2 = Merger/Other). This is the **single source of truth** used by all ML scripts and the app.

data\_loader.py (and indirectly all training/inference scripts)

**10.2 Python Scripts – Detailed Working**

**prepare\_data.py – Data Preparation**

- **Inputs**: zoo2MainSpecz.csv, gz2\_filename\_mapping.csv, and the raw image folder.
- **Function**:
  
  1. Reads the raw classification votes.
  2. Applies a **label assignment rule**:
     
     - If spiral\_fraction > 0.5 → label **1** (Spiral)
     - Else if smooth\_fraction > 0.5 → label **0** (Elliptical)
     - Else if merger\_fraction > 0.5 → label **2** (Merger)
     - Fallback → label **2** (Other / uncertain)

* 

3. Merges the classification data with the filename mapping on dr7objid.
4. Constructs the actual image filename from asset\_id (e.g., 1.jpg).
5. **Verifies image existence** – drops entries whose image files are missing.
6. **Saves** the curated, validated subset as metadata.csv.

- **Output**: metadata.csv.

**data\_loader.py – Dataset & Transforms**

- **Inputs**: metadata.csv path and image directory path.
- **Function**:
  
  1. Defines the GalaxyDataset class (inherits from torch.utils.data.Dataset).
  2. Reads metadata.csv during initialisation.
  3. In \_\_getitem\_\_, it loads the image from disk using the filename column, converts it to RGB, applies the specified transforms, and returns (image\_tensor, label).
  4. Provides the get\_transforms() function that standardises images to 224×224, converts them to PyTorch tensors, and normalises them using ImageNet statistics (mean & std).
- **Role**: Serves as the **data bridge** between the CSV metadata and the neural network models.

**classifier.py – Supervised Training (Identify Mode)**

- **Inputs**: metadata.csv and the image folder (via data\_loader.py).
- **Function**:
  
  1. Splits the dataset into **80% training** and **20% validation**.
  2. Loads a **ResNet50** pre‑trained on ImageNet.
  3. Replaces the final fully‑connected layer with a new one for **3 classes**.
  4. Trains the network using **CrossEntropyLoss** and the **Adam** optimizer for 3 epochs.
  5. Evaluates validation accuracy after each epoch.
  6. **Saves** the trained model weights to models/classifier\_resnet50.pth.
- **Output**: classifier\_resnet50.pth (model weights).

**autoencoder.py – Unsupervised Anomaly Detection (Discover Mode)**

- **Inputs**: metadata.csv and the image folder (via data\_loader.py).
- **Function**:
  
  1. **Filters** the dataset to keep **only normal galaxies** (labels 0 and 1). Mergers (label 2) are excluded from training.
  2. Defines a **convolutional autoencoder** (5‑layer encoder, 5‑layer decoder with transposed convolutions).
  3. Trains the autoencoder using **Mean Squared Error (MSE) loss** – it learns to reconstruct only normal galaxies.
  4. **Saves** the trained autoencoder weights to models/autoencoder.pth.
- **Output**: autoencoder.pth (model weights).
- **Anomaly logic**: During inference, the reconstruction error (MSE) is used as the anomaly score. High error = unusual structure (because the model never learned to reconstruct it well).

**explain.py – Grad‑CAM Explainability (Explore Mode)**

- **Inputs**: The trained classifier (classifier\_resnet50.pth loaded into a ResNet50 model) and an input image tensor.
- **Function**:
  
  1. Hooks into the **last convolutional layer** of ResNet50 (via forward/backward hooks).
  2. Performs a forward pass to get the class prediction, then a backward pass to get gradients for the predicted class.
  3. Computes a **Grad‑CAM** heatmap by globally pooling the gradients and weighting the convolutional feature maps.
  4. Upsamples the heatmap to 224×224, applies ReLU, and normalises it to a [0,1] range.
- **Output**: A 2D NumPy array representing the heatmap, which is overlaid on the original image in the app to show which pixels most influenced the prediction.

**embedding.py – Dimensionality Reduction (Planned Enhancement)**

- **Status**: Currently a stub; fully developed for future integration.
- **Planned function**: Extract high‑dimensional feature vectors from the classifier’s penultimate layer, then apply **t‑SNE** or **UMAP** to project them into 2D. This will allow users to visually browse the entire dataset, spot natural clusters, and identify outlier galaxies on the embedding map.

**app.py – Interactive Web Interface**

- **Inputs**: The trained models (.pth files), metadata.csv, and user‑uploaded images.
- **Function**:
  
  1. **Caches** and loads the classifier and autoencoder using @st.cache\_resource for performance.
  2. Displays an **uploader** for JPG/PNG/JPEG images; if none is uploaded, it loads a sample image from metadata.csv.
  3. Preprocesses the image using the same transforms from data\_loader.py.
  4. Runs **classifier inference** to obtain the predicted class and confidence.
  5. Runs **autoencoder inference** to compute the reconstruction error (anomaly score).
  6. Calls grad\_cam() from explain.py to generate the heatmap.
  7. Uses **Streamlit columns** to display the original image and the Grad‑CAM overlay side‑by‑side.
  8. Renders the prediction, confidence, and anomaly score as text below the images.
- **Role**: Orchestrates the entire pipeline and serves as the **front‑end** for interactive exploration.

**10.3 Model Files (.pth)**

**File**

**Role**

**classifier\_resnet50.pth**

Saved weights of the fine‑tuned ResNet50 classifier. Loaded by app.py to perform real‑time image classification.

**autoencoder.pth**

Saved weights of the trained convolutional autoencoder. Loaded by app.py to compute the anomaly score for any input image.

**10.4 Requirements & Execution**

**requirements.txt** – Python dependencies:

Plaintext

torch

torchvision

streamlit

pandas

numpy

scikit-learn

matplotlib

opencv-python-headless

pillow

tqdm

**Installation Command:**

Bash

pip install -r requirements.txt

**Running the Full Pipeline**

1. **Prepare the data:**
   
 ```Bash
   python src/prepare_data.py
 ```

2. **Train the classifier:**

```Bash
python src/classifier.py
```

3.  **Train the autoencoder:**

```Bash
python src/autoencoder.py
```

4. **Launch the app:**

```Bash
streamlit run src/app.py
```

