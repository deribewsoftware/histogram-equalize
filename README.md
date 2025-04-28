
# 📸 Image Histogram Equalization in Python

Welcome to the **Image Histogram Equalization** project!  
This simple Python project enhances the quality of grayscale images by improving their contrast through histogram equalization.

---

## ✨ Features
- Read and display a grayscale image.
- Calculate and visualize the histogram.
- Perform histogram equalization to enhance image contrast.
- Compare original vs. equalized images.
- Clean and simple implementation using:
  - **OpenCV**
  - **NumPy**
  - **Matplotlib**

---

## 🛠️ Technologies Used
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

---

## 📂 Project Structure
```
histogram/
│
├── demoImages/
│   └── badQuality.jpg      # Sample low-contrast image
│
├── app.py                  # Main Python script for histogram equalization
├── README.md               # Project documentation
└── requirements.txt        # List of required Python packages
```

---

## 🔥 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/histogram-equalization.git
   cd histogram-equalization
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```

4. **View the result:**
   - The original and equalized images will be displayed using Matplotlib.
   - Histograms will show contrast improvement.

---

## 🧠 How Histogram Equalization Works

Histogram equalization improves the contrast of an image by effectively spreading out the most frequent intensity values. This method enhances the visibility of details in images that are dark or washed out.

---

## 🖼️ Example Output

| Original Image | After Histogram Equalization |
| :------------: | :---------------------------: |
| ![Original](demoImages/badQuality.jpg) | ![Equalized](demoImages/equalized.jpg) |

*(Sample output — customize with your actual results.)*

---

## 📋 Requirements

- Python 3.7 or higher
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

(You can install them easily using the `requirements.txt`.)

Example `requirements.txt`:
```
opencv-python
numpy
matplotlib
```

---

## ✍️ Author
- **Your Name** — [GitHub Profile](https://github.com/your-username)

---


