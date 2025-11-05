# 🌸 Iris Species Predictor

A modern, interactive web application for predicting Iris flower species using machine learning and AI-powered species information.

## ✨ Features

- **ML Prediction**: Predict Iris species based on flower measurements using Decision Tree Classifier
- **AI Integration**: Get detailed species information using Gemini API with markdown formatting
- **Modern UI**: Beautiful, responsive interface with custom CSS, images, and smooth animations
- **Enhanced UX**: Professional layout with sidebar navigation, metrics, and visual feedback
- **Multi-page Navigation**: Prediction, About, and Model Info pages

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in your project root:

```bash
# Required: Your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Gemini model name (default: gemini-1.5-flash)
GEMINI_MODEL_NAME=gemini-1.5-flash
```

### 3. Run the Application

```bash
# Option 1: Run the main application (recommended)
streamlit run backend.py

# Option 2: Run alternative entry point
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Your Gemini API key |
| `GEMINI_MODEL_NAME` | No | `gemini-1.5-flash` | Gemini model to use |

### Setting Environment Variables

#### Option 1: .env file (Recommended)
Create a `.env` file in your project root with your configuration.

#### Option 2: Shell/Command Prompt
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"
```

#### Option 3: Streamlit command line
```bash
streamlit run app.py --server.environmentVariables GEMINI_API_KEY=your_key
```

## 📁 Project Structure

```
iris_ml_project/
├── app.py              # Alternative Streamlit application entry point
├── backend.py          # Main application with business logic, UI, and API calls
├── train_model.py      # Model training script
├── iris_model.pkl      # Trained ML model (Decision Tree)
├── label_encoder.pkl   # Label encoder for species
├── IRIS.csv           # Dataset (150 samples)
├── .env               # Environment variables (create from env_example.txt)
├── env_example.txt    # Environment variables template
└── README.md          # This file
```

## 🎯 Usage

1. **Navigate** to the Prediction page
2. **Adjust** the sliders for flower measurements
3. **Click** "Predict Species" to get the prediction
4. **View** detailed species information from Gemini AI
5. **Explore** other pages for more information

## 🔍 Pages

- **Prediction**: Main prediction interface with flower measurements input
- **About**: Information about the Iris dataset and project details
- **Model Info**: Technical details about the Decision Tree Classifier model

## 🛠️ Development

### Adding New Features

1. **Backend Logic**: Add methods to `IrisPredictor` class in `backend.py`
2. **UI Components**: Add methods to `UIManager` class in `backend.py`
3. **Styling**: Update CSS in `UIManager.load_custom_css()` method
4. **Navigation**: Update the page selection in `UIManager.render_sidebar()`

### Code Organization

- **`backend.py`**: Complete application with classes, UI, and main function
- **`app.py`**: Alternative entry point (imports from backend.py)
- **Environment Variables**: Loaded via python-dotenv from `.env` file

## 🔐 Security

- API keys are stored in environment variables (not hardcoded)
- No sensitive data in the codebase
- Proper error handling for missing configuration

## 📊 Model Information

- **Algorithm**: Decision Tree Classifier (scikit-learn)
- **Dataset**: 150 Iris samples (50 per species)
- **Features**: 4 numerical measurements (sepal/petal length & width)
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Accuracy**: ~97% on test data
- **Training**: Cross-validation used for evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.
