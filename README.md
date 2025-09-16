# 🏠 Property Price Prediction  

A Machine Learning project to predict property prices based on features like location, size, and number of rooms.  

---

## 🚀 Features  
- Predict property prices using ML models  
- Simple, Enhanced & Realtime app versions  
- Streamlit/Flask based interface  
- User-friendly web interface for property price prediction
- Input validation and interactive form
- Real-time price prediction with detailed insights

---

## 🌐 Streamlit Web App

The project includes a beautiful Streamlit web application that provides an intuitive interface for property price prediction.

### ✨ Features
- **Interactive Form**: Easy-to-use form with property details
- **Real-time Prediction**: Instant price prediction using trained ML model
- **Smart Insights**: Price per sq ft, category classification, and key factors
- **Input Validation**: Built-in validation to ensure accurate inputs
- **Visual Design**: Clean, modern interface with emojis and styling

### 🚀 Running the Streamlit App

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn flask flask-cors flask-limiter
   ```

2. **Navigate to Project Directory**:
   ```bash
   cd "Property Price Prediction"
   ```

3. **Generate Model (if not already exists)**:
   ```bash
   python main.py
   ```
   This will create the trained model files in the `models/` directory.

4. **Run the Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in Browser**:
   The app will automatically open in your browser at `http://localhost:8501`

### 📋 How to Use

1. **Fill Property Details**: Enter basic information like property type, square feet, bedrooms, bathrooms, and year built
2. **Set Location Features**: Choose location quality, distance to city center, and crime rate
3. **Select Amenities**: Check available amenities like parking, pool, garden, gym, and security
4. **Get Prediction**: Click "Predict Price" to get instant price estimation
5. **Review Results**: See the predicted price, price per sq ft, category, and key insights

### 🎯 Example Usage

For a 2000 sq ft apartment with:
- 3 bedrooms, 2 bathrooms
- Built in 2010 (14 years old)
- Medium location quality
- 15km from city center
- With parking

The app predicts: **$360,636** (Mid-Range category, $180/sq ft)

---

## 📂 Project Structure  
