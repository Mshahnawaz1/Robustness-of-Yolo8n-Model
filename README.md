# 🖼️ Image Degradation Robustness Analysis

This project analyzes how image degradation (Gaussian Noise and JPEG Compression) affects object or face detection performance.

  
## ⚙️ Setup

1. **Requirements:** Python ≥ 3.8
    
2. **Create & Activate Virtual Environment:**
    
  ``` bash
  python3 -m venv venv
  ```
  ``` bash
  venv\Scripts\activate (Windows)
  ```
 ``` bash
 source venv/bin/activate  # macOS/Linux
 ```
    
4. **Install Dependencies:**
    
    `pip install -r requirements.txt`
    

## 📂 Structure

`image-robustness-project/ ├── image/grp-photo.jpg ├── src/analysis.py ├── README.md ├── requirements.txt └── venv/`

## 💻 Run Analysis

`python3 src/analysis.py`

### Outputs

- **Visual Plot:** Original vs degraded images.
    
- **Metrics Plot:** Recall & IoU decay with degradation.
    

## 📊 Insights

- JPEG compression causes a **sharp drop** in Recall (blocking artifacts).
    
- Gaussian noise causes a **gradual decay** in performance.
    

## 🤝 Contributing

Fork → Create branch → Commit → Push → Pull Request.
