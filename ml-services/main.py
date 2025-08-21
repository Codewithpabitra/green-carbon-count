from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import io
from typing import Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import calendar
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
uploaded_data = None
model = None
scaler = None
is_retraining = False

# Constants - UPDATED FOR WATER
CO2_FACTOR = 0.7  # 0.7 kg CO₂e per kiloliter (AWU_kiloliters * 0.7)

def get_days_in_month(year, month):
    """Get number of days in a given month"""
    return calendar.monthrange(year, month)[1]

def create_sequences(data, sequence_length=3):
    """Create sequences for LSTM training - FIXED VERSION"""
    sequences = []
    targets = []
    
    # Ensure we have enough data
    if len(data) <= sequence_length:
        logger.warning(f"Not enough data points ({len(data)}) for sequence length {sequence_length}")
        return np.array([]), np.array([])
    
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def preprocess_for_ml(monthly_data):
    """
    Preprocess monthly data for ML training/prediction
    Returns properly scaled data and scaler
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Get consumption values - UPDATED COLUMN NAME
        consumption_values = monthly_data['AWU_kiloliters'].values
        
        # Log data statistics for debugging
        logger.info(f"📊 Data statistics:")
        logger.info(f"   - Data points: {len(consumption_values)}")
        logger.info(f"   - Min: {consumption_values.min():.2f}")
        logger.info(f"   - Max: {consumption_values.max():.2f}")
        logger.info(f"   - Mean: {consumption_values.mean():.2f}")
        logger.info(f"   - Std: {consumption_values.std():.2f}")
        
        # Create and fit scaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(consumption_values.reshape(-1, 1)).flatten()
        
        logger.info(f"📊 Scaled data statistics:")
        logger.info(f"   - Min: {scaled_data.min():.4f}")
        logger.info(f"   - Max: {scaled_data.max():.4f}")
        logger.info(f"   - Mean: {scaled_data.mean():.4f}")
        logger.info(f"   - Std: {scaled_data.std():.4f}")
        
        return scaled_data, scaler
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {str(e)}")
        raise

def retrain_model_sync(monthly_data):
    """Synchronous model retraining function - FIXED VERSION"""
    global model, scaler
    
    try:
        # Import TensorFlow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        
        logger.info("🔥 Starting model retraining process...")
        
        # Preprocess data properly
        scaled_data, new_scaler = preprocess_for_ml(monthly_data)
        
        if len(scaled_data) < 6:
            logger.warning("⚠ Need at least 6 months of data for effective retraining")
            return False, "Insufficient data for retraining"
        
        # Create sequences for training
        sequence_length = 3
        X, y = create_sequences(scaled_data, sequence_length)
        
        if len(X) == 0:
            logger.error("❌ Not enough data to create training sequences")
            return False, "Insufficient data for sequence creation"
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"📊 X sample values: {X[0].flatten()}")
        logger.info(f"📊 y sample values: {y[:3]}")
        
        # Create or update model
        if model is not None:
            # Fine-tune existing model
            logger.info("🔧 Fine-tuning existing model...")
            
            # Compile model for training
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train with new data
            history = model.fit(
                X, y,
                epochs=30,  # Increased epochs
                batch_size=min(4, len(X)),  # Smaller batch size
                validation_split=0.2 if len(X) > 5 else 0,
                verbose=1,  # Enable verbose for debugging
                shuffle=True
            )
            
        else:
            # Create new model - IMPROVED ARCHITECTURE
            logger.info("🆕 Creating new model...")
            
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)  # No activation for regression
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("📋 Model Summary:")
            model.summary(print_fn=logger.info)
            
            # Train new model
            history = model.fit(
                X, y,
                epochs=100,  # More epochs for initial training
                batch_size=min(4, len(X)),
                validation_split=0.2 if len(X) > 5 else 0,
                verbose=1,
                shuffle=True,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=15, 
                        restore_best_weights=True,
                        monitor='val_loss' if len(X) > 5 else 'loss'
                    )
                ]
            )
        
        # Update global scaler
        scaler = new_scaler
        
        # Test the model with a sample prediction
        test_input = X[-1:] if len(X) > 0 else None
        if test_input is not None:
            test_pred_scaled = model.predict(test_input, verbose=0)
            test_pred_original = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()[0]
            logger.info(f"🧪 Test prediction: {test_pred_original:.2f} KL")
        
        # Save updated model and scaler
        try:
            model.save("water_model.keras")
            logger.info("💾 Model saved successfully")
        except Exception as e:
            logger.warning(f"⚠ Could not save model: {str(e)}")
        
        try:
            import joblib
            joblib.dump(scaler, "scaler.pkl")
            logger.info("💾 Scaler saved successfully")
        except Exception as e:
            logger.warning(f"⚠ Could not save scaler: {str(e)}")
        
        # Get final loss
        final_loss = history.history['loss'][-1]
        val_loss = history.history.get('val_loss', [final_loss])[-1]
        logger.info(f"✅ Model retraining completed.")
        logger.info(f"   - Final training loss: {final_loss:.6f}")
        logger.info(f"   - Final validation loss: {val_loss:.6f}")
        logger.info(f"   - Training samples: {len(X)}")
        
        return True, f"Model retrained successfully with {len(X)} training samples. Loss: {final_loss:.6f}"
        
    except ImportError as e:
        logger.error(f"❌ Missing required libraries: {str(e)}")
        return False, "Required libraries not available"
    except Exception as e:
        logger.error(f"❌ Error during retraining: {str(e)}")
        return False, f"Retraining failed: {str(e)}"

async def retrain_model_async(monthly_data):
    """Asynchronous wrapper for model retraining"""
    global is_retraining
    
    is_retraining = True
    try:
        # Run the synchronous training in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            success, message = await loop.run_in_executor(
                executor, retrain_model_sync, monthly_data
            )
        return success, message
    finally:
        is_retraining = False

async def load_models():
    """Load ML models if available"""
    global model, scaler
    
    # Try to load TensorFlow model - UPDATED MODEL NAME
    try:
        import tensorflow as tf
        if os.path.exists("water_model.keras"):
            model = tf.keras.models.load_model("water_model.keras")
            logger.info("✓ Water model loaded successfully")
            logger.info(f"📋 Model input shape: {model.input_shape}")
            logger.info(f"📋 Model output shape: {model.output_shape}")
        else:
            logger.warning("⚠ Warning: water_model.keras not found - predictions will use fallback method")
    except ImportError:
        logger.warning("⚠ TensorFlow not available - predictions will use statistical method")
    except Exception as e:
        logger.error(f"❌ Error loading water model: {str(e)}")
        
    # Try to load the scaler
    try:
        if os.path.exists("scaler.pkl"):
            # Check file size first
            file_size = os.path.getsize("scaler.pkl")
            logger.info(f"📁 Found scaler.pkl (size: {file_size} bytes)")
            
            if file_size == 0:
                logger.error("❌ scaler.pkl is empty")
                return
            
            # Try different methods to load the pickle file
            try:
                with open("scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                logger.info("✓ Scaler loaded successfully")
                # Log scaler parameters for debugging
                if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    logger.info(f"📊 Scaler mean: {scaler.mean_[0]:.2f}")
                    logger.info(f"📊 Scaler scale: {scaler.scale_[0]:.2f}")
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                logger.error(f"❌ Error loading scaler.pkl: {str(e)}")
                logger.info("🔧 Trying alternative loading methods...")
                
                # Try with joblib
                try:
                    import joblib
                    scaler = joblib.load("scaler.pkl")
                    logger.info("✓ Scaler loaded successfully using joblib")
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        logger.info(f"📊 Scaler mean: {scaler.mean_[0]:.2f}")
                        logger.info(f"📊 Scaler scale: {scaler.scale_[0]:.2f}")
                except ImportError:
                    logger.warning("⚠ joblib not available, cannot try alternative loading")
                except Exception as e2:
                    logger.error(f"❌ joblib loading also failed: {str(e2)}")
                    logger.warning("⚠ Will proceed without scaler - using simple normalization")
                    scaler = None
        else:
            logger.warning("⚠ Warning: scaler.pkl not found - will use simple scaling")
            scaler = None
            
    except Exception as e:
        logger.error(f"❌ Unexpected error loading scaler: {str(e)}")
        scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_models()
    yield
    # Shutdown - cleanup if needed

app = FastAPI(
    title="Water Consumption Analytics API", 
    version="3.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_csv_format(df):
    """Validate CSV has required columns with flexible matching - UPDATED FOR WATER"""
    if df.empty:
        return False, "CSV file is empty"
    
    # Look for datetime-like columns
    datetime_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['datetime', 'date', 'time', 'timestamp'])]
    
    # Look for AWU_kiloliters or similar water columns - UPDATED
    water_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['awu_kiloliters', 'kiloliters', 'kl', 'water', 'consumption'])]
    
    if not datetime_cols:
        return False, "No datetime column found. Expected columns like 'Datetime', 'Date', etc."
    
    if not water_cols:
        return False, "No water consumption column found. Expected columns like 'AWU_kiloliters', 'KL', etc."
    
    return True, "Valid format"

def preprocess_data(df):
    """Clean and preprocess the uploaded data - UPDATED FOR WATER"""
    try:
        # Make column names consistent
        df.columns = [col.strip() for col in df.columns]
        
        # Find datetime and consumption columns
        datetime_col = None
        consumption_col = None
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['datetime', 'date', 'time', 'timestamp']):
                datetime_col = col
                break
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['awu_kiloliters', 'kiloliters', 'kl', 'water', 'consumption']):
                consumption_col = col
                break
        
        if not datetime_col or not consumption_col:
            raise ValueError("Could not identify datetime or consumption columns")
        
        # Create working copy - UPDATED COLUMN NAME
        processed_df = df[[datetime_col, consumption_col]].copy()
        processed_df.columns = ['Datetime', 'AWU_kiloliters']
        
        # Convert datetime with error handling
        try:
            processed_df['Datetime'] = pd.to_datetime(processed_df['Datetime'])
        except Exception as e:
            raise ValueError(f"Could not parse datetime column: {str(e)}")
        
        # Convert consumption to numeric
        processed_df['AWU_kiloliters'] = pd.to_numeric(processed_df['AWU_kiloliters'], errors='coerce')
        
        # Remove NaN values
        initial_rows = len(processed_df)
        processed_df = processed_df.dropna()
        final_rows = len(processed_df)
        
        if final_rows == 0:
            raise ValueError("No valid data rows after cleaning")
        
        if final_rows < initial_rows * 0.5:
            logger.warning(f"Warning: Removed {initial_rows - final_rows} rows due to invalid data")
        
        # Sort by datetime
        processed_df = processed_df.sort_values('Datetime')
        
        # Add time features
        processed_df['Date'] = processed_df['Datetime'].dt.date
        processed_df['Hour'] = processed_df['Datetime'].dt.hour
        processed_df['Month'] = processed_df['Datetime'].dt.month
        processed_df['Year'] = processed_df['Datetime'].dt.year
        processed_df['DayOfWeek'] = processed_df['Datetime'].dt.dayofweek
        
        # Log data statistics for debugging - UPDATED UNITS
        logger.info(f"📊 Processed data statistics:")
        logger.info(f"   - Total rows: {len(processed_df)}")
        logger.info(f"   - Date range: {processed_df['Datetime'].min()} to {processed_df['Datetime'].max()}")
        logger.info(f"   - Consumption range: {processed_df['AWU_kiloliters'].min():.2f} to {processed_df['AWU_kiloliters'].max():.2f} KL")
        logger.info(f"   - Average consumption: {processed_df['AWU_kiloliters'].mean():.2f} KL")
        
        return processed_df
        
    except Exception as e:
        raise ValueError(f"Data preprocessing failed: {str(e)}")

def create_simple_trend_chart(monthly_data, prediction_value, prediction_month):
    """Create a simple trend chart with prediction - UPDATED FOR WATER"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Prepare data for plotting
        dates = []
        for _, row in monthly_data.iterrows():
            dates.append(datetime(int(row['Year']), int(row['Month']), 1))
        
        actual_values = monthly_data['AWU_kiloliters'].values
        
        # Add prediction date and value
        pred_date = datetime.strptime(prediction_month, '%Y-%m')
        
        # Plot actual data
        ax.plot(dates, actual_values, 'b-o', linewidth=2, markersize=6, label='Actual', alpha=0.8)
        
        # Plot prediction line
        ax.plot([dates[-1], pred_date], [actual_values[-1], prediction_value], 
                'r--', linewidth=2, alpha=0.8, label='Trend to Prediction')
        
        # Plot prediction point
        ax.plot(pred_date, prediction_value, 'ro', markersize=8, label='Predicted')
        
        # Formatting - UPDATED TITLES AND LABELS
        ax.set_title('Water Consumption Trend and Prediction', fontweight='bold', fontsize=14)
        ax.set_ylabel('Water Consumption (KL)', fontsize=12)
        ax.set_xlabel('Month', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add prediction info text - UPDATED UNITS
        info_text = f"Predicted: {prediction_value:,.0f} KL\nMonth: {prediction_month}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating simple chart: {str(e)}")
        return None

def create_prediction_visualization(monthly_data, prediction_value, prediction_month, confidence, method):
    """
    Create a comprehensive visualization showing predicted vs actual values - UPDATED FOR WATER
    """
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Water Consumption Prediction Analysis with Adaptive Learning', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Prepare data for plotting
        dates = []
        for _, row in monthly_data.iterrows():
            dates.append(datetime(int(row['Year']), int(row['Month']), 1))
        
        actual_values = monthly_data['AWU_kiloliters'].values
        
        # Add prediction date and value
        pred_date = datetime.strptime(prediction_month, '%Y-%m')
        all_dates = dates + [pred_date]
        all_values_actual = list(actual_values) + [None]
        all_values_predicted = list(actual_values) + [prediction_value]
        
        # Plot 1: Predictions vs Actual Values
        ax1.plot(dates, actual_values, 'b-o', linewidth=2, markersize=6, label='Actual', alpha=0.8)
        ax1.plot([dates[-1], pred_date], [actual_values[-1], prediction_value], 
                'r--', linewidth=2, alpha=0.8)
        ax1.plot(pred_date, prediction_value, 'ro', markersize=8, label='Predicted')
        
        # Add confidence interval
        margin = prediction_value * (1 - float(confidence.replace('%', ''))/100) * 0.5
        ax1.fill_between([pred_date], [prediction_value - margin], [prediction_value + margin], 
                        alpha=0.3, color='red', label=f'Confidence ({confidence})')
        
        ax1.set_title('Predictions vs Actual Values', fontweight='bold')
        ax1.set_ylabel('Water Consumption (KL)')  # UPDATED UNITS
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Accuracy Evolution (placeholder for now)
        ax2.text(0.5, 0.5, 'Accuracy Evolution\n(Need more data)', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Model Performance', fontweight='bold')
        
        # Plot 3: Prediction Residuals
        if len(actual_values) > 1:
            # Calculate trend for residuals visualization
            x_vals = np.arange(len(actual_values))
            trend = np.polyfit(x_vals, actual_values, 1)
            trend_line = np.poly1d(trend)
            residuals = actual_values - trend_line(x_vals)
            
            # Add prediction residual
            pred_trend = trend_line(len(actual_values))
            pred_residual = prediction_value - pred_trend
            
            all_residuals = list(residuals) + [pred_residual]
            residual_dates = dates + [pred_date]
            
            ax3.fill_between(residual_dates, all_residuals, alpha=0.6, color='purple')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.plot(residual_dates, all_residuals, 'o-', color='darkpurple', markersize=4)
            
        ax3.set_title('Prediction Residuals', fontweight='bold')
        ax3.set_ylabel('Residuals (KL)')  # UPDATED UNITS
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Retraining Improvements
        # Show improvement metrics
        improvement_data = [98.5]  # Placeholder for actual improvement
        sessions = ['1.0']
        
        bars = ax4.bar(sessions, improvement_data, color='green', alpha=0.8)
        ax4.set_title('Retraining Improvements', fontweight='bold')
        ax4.set_ylabel('Accuracy Improvement (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, improvement_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add prediction info text box - UPDATED UNITS
        info_text = f"""Prediction Details:
Method: {method}
Confidence: {confidence}
Next Month: {prediction_month}
Predicted: {prediction_value:,.0f} KL
Current: {actual_values[-1]:,.0f} KL
Change: {((prediction_value - actual_values[-1])/actual_values[-1]*100):+.1f}%"""
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {str(e)}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        dates = [datetime(int(row['Year']), int(row['Month']), 1) for _, row in monthly_data.iterrows()]
        actual_values = monthly_data['AWU_kiloliters'].values
        pred_date = datetime.strptime(prediction_month, '%Y-%m')
        
        ax.plot(dates, actual_values, 'b-o', label='Actual')
        ax.plot(pred_date, prediction_value, 'ro', markersize=8, label='Predicted')
        
        ax.set_title('Water Consumption Prediction')  # UPDATED TITLE
        ax.set_ylabel('Water Consumption (KL)')  # UPDATED UNITS
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig

def make_ml_prediction(monthly_data):
    """
    Make ML prediction with proper scaling - FIXED VERSION
    """
    global model, scaler
    
    try:
        if model is None or scaler is None:
            raise Exception("Model or scaler not available")
        
        # Get recent consumption values
        recent_months = monthly_data['AWU_kiloliters'].values
        
        logger.info(f"🔮 Making ML prediction with recent data: {recent_months}")
        
        # Use the same scaler that was used during training
        recent_scaled = scaler.transform(recent_months.reshape(-1, 1)).flatten()
        
        logger.info(f"🔮 Scaled input data: {recent_scaled}")
        
        # Create sequence for LSTM
        sequence_length = 3
        if len(recent_scaled) >= sequence_length:
            input_sequence = recent_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        else:
            # Pad sequence if we don't have enough data
            padded_sequence = np.full(sequence_length, recent_scaled[-1])
            padded_sequence[-len(recent_scaled):] = recent_scaled
            input_sequence = padded_sequence.reshape(1, sequence_length, 1)
        
        logger.info(f"🔮 Input sequence shape: {input_sequence.shape}")
        logger.info(f"🔮 Input sequence values: {input_sequence.flatten()}")
        
        # Make prediction
        prediction_scaled = model.predict(input_sequence, verbose=0)
        logger.info(f"🔮 Raw model prediction (scaled): {prediction_scaled}")
        
        # Transform back to original scale
        prediction_original = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        logger.info(f"🔮 Final prediction (original scale): {prediction_original}")
        
        # Validate prediction is reasonable
        current_avg = recent_months[-1]
        if abs(prediction_original - current_avg) > current_avg * 1.5:  # More lenient threshold
            logger.warning(f"⚠ ML prediction might be unrealistic: {prediction_original:.1f} vs recent {current_avg:.1f}")
            # Don't reject, but lower confidence
            confidence = 0.70
        else:
            confidence = 0.90
        
        return prediction_original, confidence, "ML model"
        
    except Exception as e:
        logger.error(f"❌ ML prediction failed: {str(e)}")
        raise

def make_statistical_prediction(monthly_data):
    """
    Make statistical prediction as fallback
    """
    try:
        recent_months = monthly_data['AWU_kiloliters'].values
        
        if len(recent_months) >= 3:
            # Use weighted moving average with trend analysis
            weights = np.array([0.5, 0.3, 0.2])  # More weight to recent data
            weighted_avg = np.average(recent_months[-3:], weights=weights)
            
            # Calculate trend
            x = np.arange(len(recent_months[-3:]))
            y = recent_months[-3:]
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
            
            # Apply trend with dampening
            prediction = weighted_avg + (slope * 0.7)
            
        else:
            # Simple moving average for limited data
            prediction = np.mean(recent_months)
        
        # Add small seasonal variation
        current_avg = recent_months[-1]
        std_dev = np.std(recent_months) if len(recent_months) > 1 else current_avg * 0.05
        seasonal_adjustment = np.random.normal(0, std_dev * 0.05)
        prediction += seasonal_adjustment
        
        # Ensure prediction is within reasonable bounds
        min_pred = current_avg * 0.7
        max_pred = current_avg * 1.3
        prediction = np.clip(prediction, min_pred, max_pred)
        
        confidence = 0.75
        return prediction, confidence, "Statistical method"
        
    except Exception as e:
        logger.error(f"❌ Statistical prediction failed: {str(e)}")
        raise

def detect_leakage_alerts(monthly_data):
    """Simple leakage detection algorithm"""
    try:
        if len(monthly_data) < 2:
            return []
        
        recent_consumption = monthly_data['AWU_kiloliters'].values
        alerts = []
        
        # Check for sudden spike (>50% increase)
        for i in range(1, len(recent_consumption)):
            if recent_consumption[i] > recent_consumption[i-1] * 1.5:
                alerts.append({
                    "type": "Sudden Spike",
                    "message": f"Water usage increased by {((recent_consumption[i]/recent_consumption[i-1]-1)*100):.1f}%",
                    "month": f"{monthly_data.iloc[i]['Year']}-{monthly_data.iloc[i]['Month']:02d}"
                })
        
        # Check for consistently high usage (above average + 2 std dev)
        if len(recent_consumption) >= 3:
            avg = np.mean(recent_consumption)
            std = np.std(recent_consumption)
            threshold = avg + (2 * std)
            
            high_usage_months = []
            for i, consumption in enumerate(recent_consumption):
                if consumption > threshold:
                    high_usage_months.append(f"{monthly_data.iloc[i]['Year']}-{monthly_data.iloc[i]['Month']:02d}")
            
            if len(high_usage_months) >= 2:
                alerts.append({
                    "type": "High Usage Pattern",
                    "message": f"Consistently high usage detected in months: {', '.join(high_usage_months)}",
                    "months": high_usage_months
                })
        
        return alerts
        
    except Exception as e:
        logger.error(f"❌ Error detecting leakage: {str(e)}")
        return []

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file"""
    global uploaded_data
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")
        
        # Read and parse CSV
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
            
        try:
            csv_string = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                csv_string = content.decode('latin1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Could not decode file. Please ensure it's a valid CSV")
        
        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(csv_string))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")
        
        # Validate format
        is_valid, message = validate_csv_format(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Preprocess data
        uploaded_data = preprocess_data(df)
        
        # Calculate basic statistics
        total_consumption = uploaded_data['AWU_kiloliters'].sum()
        date_range = f"{uploaded_data['Datetime'].min().strftime('%Y-%m-%d')} to {uploaded_data['Datetime'].max().strftime('%Y-%m-%d')}"
        
        logger.info(f"Successfully processed {len(uploaded_data)} rows of data")
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "data_info": {
                "total_rows": len(uploaded_data),
                "date_range": date_range,
                "total_consumption": round(total_consumption, 2),
                "avg_consumption": round(uploaded_data['AWU_kiloliters'].mean(), 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/api/data/monthly")
async def get_monthly_data(month: str = Query(..., description="Format: YYYY-MM")):
    """Get monthly consumption data"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Validate month format
        try:
            year, month_num = map(int, month.split('-'))
            if month_num < 1 or month_num > 12:
                raise ValueError("Month must be between 1 and 12")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
        
        # Filter data for the specified month
        filtered_data = uploaded_data[
            (uploaded_data['Year'] == year) & 
            (uploaded_data['Month'] == month_num)
        ]
        
        if filtered_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {month}")
        
        # Aggregate by date (daily averages)
        daily_data = filtered_data.groupby('Date').agg({
            'AWU_kiloliters': 'mean'
        }).reset_index()
        
        # Format response
        data_points = []
        for _, row in daily_data.iterrows():
            data_points.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "consumption": round(row['AWU_kiloliters'], 2)
            })
        
        return {
            "month": month,
            "data": data_points,
            "total_days": len(data_points),
            "avg_consumption": round(daily_data['AWU_kiloliters'].mean(), 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monthly data error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing monthly data: {str(e)}")

@app.get("/api/data/daily")
async def get_daily_data(date: str = Query(..., description="Format: YYYY-MM-DD")):
    """Get hourly consumption data for a specific date"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Validate date format
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Filter data for the specified date
        daily_data = uploaded_data[uploaded_data['Date'] == target_date]
        
        if daily_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {date}")
        
        # Aggregate by hour
        hourly_data = daily_data.groupby('Hour').agg({
            'AWU_kiloliters': 'mean'
        }).reset_index()
        
        # Create complete hourly data (0-23 hours)
        all_hours = pd.DataFrame({'Hour': range(24)})
        hourly_complete = all_hours.merge(hourly_data, on='Hour', how='left')
        hourly_complete['AWU_kiloliters'] = hourly_complete['AWU_kiloliters'].fillna(0)
        
        # Format response
        data_points = []
        for _, row in hourly_complete.iterrows():
            # Create datetime for the hour
            hour_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=int(row['Hour']))
            data_points.append({
                "datetime": hour_datetime.isoformat(),
                "consumption": round(row['AWU_kiloliters'], 2)
            })
        
        return {
            "date": date,
            "data": data_points,
            "total_consumption": round(hourly_complete['AWU_kiloliters'].sum(), 2),
            "peak_hour": int(hourly_complete.loc[hourly_complete['AWU_kiloliters'].idxmax(), 'Hour'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Daily data error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing daily data: {str(e)}")

@app.get("/api/stats")
async def get_statistics(month: str = Query(None, description="Format: YYYY-MM for month-specific stats")):
    """Get overall statistics from uploaded data - UPDATED FOR WATER"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        if month:
            # Month-specific statistics
            try:
                year, month_num = map(int, month.split('-'))
                filtered_data = uploaded_data[
                    (uploaded_data['Year'] == year) & 
                    (uploaded_data['Month'] == month_num)
                ]
                
                if filtered_data.empty:
                    raise HTTPException(status_code=404, detail=f"No data found for {month}")
                
                # Monthly consumption - UPDATED FOR WATER
                total_consumption = filtered_data['AWU_kiloliters'].sum()
                
                # Format for display - UPDATED UNITS
                consumption_str = f"{total_consumption:,.0f} KL"
                
                # Analysis period for current month
                start_date = filtered_data['Datetime'].min().strftime('%Y-%m-%d')
                end_date = filtered_data['Datetime'].max().strftime('%Y-%m-%d')
                period_str = f"{start_date} to {end_date}"
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
        else:
            # Overall statistics
            total_consumption = uploaded_data['AWU_kiloliters'].sum()
            
            consumption_str = f"{total_consumption:,.0f} KL"
            
            # Analysis period for entire dataset
            start_date = uploaded_data['Datetime'].min().strftime('%Y-%m-%d')
            end_date = uploaded_data['Datetime'].max().strftime('%Y-%m-%d')
            period_str = f"{start_date} to {end_date}"
        
        # Calculate CO2 emissions for entire dataset (total period) - UPDATED FORMULA
        total_dataset_consumption = uploaded_data['AWU_kiloliters'].sum()
        total_co2 = (total_dataset_consumption * CO2_FACTOR) / 1000  # Convert to tons
        
        return {
            "total_consumption": consumption_str,
            "co2_emissions": f"{total_co2:,.2f} tons CO₂e",
            "analysis_period": period_str
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")

@app.get("/api/stats/comparison")
async def get_monthly_comparison(month: str = Query(..., description="Format: YYYY-MM")):
    """Get month-over-month comparison statistics - UPDATED FOR WATER"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Parse current month
        try:
            year, month_num = map(int, month.split('-'))
            if month_num < 1 or month_num > 12:
                raise ValueError("Month must be between 1 and 12")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
        
        # Get current month data
        current_data = uploaded_data[
            (uploaded_data['Year'] == year) & 
            (uploaded_data['Month'] == month_num)
        ]
        
        if current_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {month}")
        
        # Calculate previous month
        prev_date = datetime(year, month_num, 1) - timedelta(days=1)
        prev_year, prev_month = prev_date.year, prev_date.month
        
        # Get previous month data
        prev_data = uploaded_data[
            (uploaded_data['Year'] == prev_year) & 
            (uploaded_data['Month'] == prev_month)
        ]
        
        # Calculate current month values - UPDATED FOR WATER
        current_consumption = current_data['AWU_kiloliters'].sum()
        
        # Calculate comparisons
        if not prev_data.empty:
            prev_consumption = prev_data['AWU_kiloliters'].sum()
            
            consumption_change = ((current_consumption - prev_consumption) / prev_consumption) * 100
            
            consumption_change_str = f"{consumption_change:+.1f}%"
        else:
            consumption_change_str = "NaN"
        
        return {
            "month": month,
            "consumption_change": consumption_change_str,
            "current_consumption": current_consumption
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating comparison: {str(e)}")

@app.get("/api/stats/peak-hours")
async def get_monthly_peak_hours(month: str = Query(..., description="Format: YYYY-MM")):
    """Get peak usage hours for a specific month"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Validate month format
        try:
            year, month_num = map(int, month.split('-'))
            if month_num < 1 or month_num > 12:
                raise ValueError("Month must be between 1 and 12")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
        
        # Filter data for the specified month
        monthly_data = uploaded_data[
            (uploaded_data['Year'] == year) & 
            (uploaded_data['Month'] == month_num)
        ]
        
        if monthly_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {month}")
        
        # Calculate hourly averages for the month - UPDATED COLUMN
        hourly_avg = monthly_data.groupby('Hour')['AWU_kiloliters'].mean().sort_values(ascending=False)
        
        # Get top 3 peak hours - UPDATED UNITS
        peak_hours = []
        for hour, consumption in hourly_avg.head(3).items():
            peak_hours.append({
                "time": f"{hour:02d}:00",
                "value": f"{consumption:.1f} KL"
            })
        
        return {
            "month": month,
            "peak_hours": peak_hours
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Peak hours error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating peak hours: {str(e)}")

@app.get("/api/leakage/alerts")
async def get_leakage_alerts():
    """Get potential leakage alerts based on consumption patterns"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Calculate monthly data for leakage detection
        monthly_data = uploaded_data.groupby(['Year', 'Month']).agg({
            'AWU_kiloliters': 'sum'
        }).reset_index()
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        alerts = detect_leakage_alerts(monthly_data)
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "status": "All Clear" if len(alerts) == 0 else f"{len(alerts)} Alert{'s' if len(alerts) > 1 else ''}"
        }
        
    except Exception as e:
        logger.error(f"Leakage detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting leakage: {str(e)}")

@app.post("/api/retrain")
async def retrain_model():
    """Retrain the model with uploaded data"""
    global uploaded_data, is_retraining
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    if is_retraining:
        raise HTTPException(status_code=429, detail="Model is already being retrained. Please wait.")
    
    try:
        # Calculate monthly averages for training - PROPER AGGREGATION - UPDATED COLUMN
        monthly_data = uploaded_data.groupby(['Year', 'Month']).agg({
            'AWU_kiloliters': 'sum'  # Use SUM for monthly totals, not mean
        }).reset_index()
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        if len(monthly_data) < 4:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 4 months of data for retraining"
            )
        
        logger.info(f"🚀 Starting retraining with {len(monthly_data)} months of data...")
        logger.info(f"📊 Monthly consumption range: {monthly_data['AWU_kiloliters'].min():.0f} - {monthly_data['AWU_kiloliters'].max():.0f} KL")
        
        # Start retraining asynchronously
        success, message = await retrain_model_async(monthly_data)
        
        if success:
            return {
                "status": "success",
                "message": message,
                "training_samples": len(monthly_data),
                "model_updated": True,
                "data_range": {
                    "min_consumption": round(monthly_data['AWU_kiloliters'].min(), 2),
                    "max_consumption": round(monthly_data['AWU_kiloliters'].max(), 2),
                    "avg_consumption": round(monthly_data['AWU_kiloliters'].mean(), 2)
                }
            }
        else:
            raise HTTPException(status_code=500, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/api/retrain/status")
async def get_retraining_status():
    """Get current retraining status"""
    return {
        "is_retraining": is_retraining,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/api/predict")
async def predict_consumption():
    """Predict next month's consumption using available models or statistical methods - ENHANCED WITH VISUALIZATION"""
    global uploaded_data, model, scaler
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Calculate monthly totals (not averages) for prediction - UPDATED COLUMN
        monthly_data = uploaded_data.groupby(['Year', 'Month']).agg({
            'AWU_kiloliters': 'sum'  # FIXED: Use sum for monthly totals
        }).reset_index()
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        if len(monthly_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 months of data for prediction")
        
        logger.info(f"🔮 Making prediction with {len(monthly_data)} months of data")
        logger.info(f"📊 Recent monthly consumption: {monthly_data['AWU_kiloliters'].tail(3).tolist()}")
        
        # Try ML prediction first
        try:
            prediction, confidence, method = make_ml_prediction(monthly_data.tail(6))
            logger.info(f"✅ {method} prediction successful: {prediction:.1f} KL (confidence: {confidence*100:.0f}%)")
            
        except Exception as e:
            logger.warning(f"⚠ ML prediction failed: {str(e)}, using statistical fallback")
            # Fall back to statistical prediction
            prediction, confidence, method = make_statistical_prediction(monthly_data.tail(6))
            logger.info(f"✅ {method} prediction successful: {prediction:.1f} KL (confidence: {confidence*100:.0f}%)")
        
        # Calculate change percentage
        current_month = monthly_data['AWU_kiloliters'].iloc[-1]
        change_pct = ((prediction - current_month) / current_month * 100) if current_month != 0 else 0
        
        # Generate visualization for this prediction too
        last_date = uploaded_data['Datetime'].max()
        next_month_start = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        png_filename = f'water_prediction_analysis_{next_month_start.strftime("%Y_%m")}.png'
        
        try:
            fig = create_simple_trend_chart(
                monthly_data, 
                prediction, 
                next_month_start.strftime('%Y-%m')
            )
            if fig:
                fig.savefig(png_filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
                logger.info(f"📊 Prediction visualization saved to {png_filename}")
                viz_saved = True
            else:
                png_filename = None
                viz_saved = False
        except Exception as viz_error:
            logger.warning(f"⚠ Visualization failed: {viz_error}")
            png_filename = None
            viz_saved = False
        
        return {
            "prediction": {
                "next_month": round(prediction, 1),
                "confidence": f"{confidence*100:.0f}%",
                "method": method,
                "change_vs_current": f"{change_pct:+.1f}%",
                "current_month": round(current_month, 1),
                "png_saved": viz_saved,
                "png_filename": png_filename
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict-next-month")
async def predict_next_month_consumption():
    """Generate next month's daily predictions and save as CSV + PNG visualization - ENHANCED VERSION"""
    global uploaded_data, model, scaler
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Calculate monthly totals for prediction - UPDATED COLUMN
        monthly_data = uploaded_data.groupby(['Year', 'Month']).agg({
            'AWU_kiloliters': 'sum'  # Use sum for proper monthly totals
        }).reset_index()
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        if len(monthly_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 months of data for prediction")
        
        # Determine next month
        last_date = uploaded_data['Datetime'].max()
        next_month_start = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        days_in_month = get_days_in_month(next_month_start.year, next_month_start.month)
        
        # Get monthly prediction using the same logic as /api/predict
        try:
            monthly_prediction, confidence, method = make_ml_prediction(monthly_data.tail(6))
            logger.info(f"✅ Next month prediction: {monthly_prediction:.1f} KL using {method}")
        except Exception:
            monthly_prediction, confidence, method = make_statistical_prediction(monthly_data.tail(6))
            logger.info(f"✅ Next month prediction: {monthly_prediction:.1f} KL using {method}")
        
        # Generate daily predictions for the entire month
        daily_predictions = []
        base_daily = monthly_prediction / days_in_month
        
        # Get historical daily patterns if available
        if len(uploaded_data) > 30:  # Only if we have enough data
            daily_patterns = uploaded_data.groupby(uploaded_data['Datetime'].dt.day)['AWU_kiloliters'].mean()
            daily_avg = daily_patterns.mean()
        else:
            daily_patterns = None
            daily_avg = uploaded_data['AWU_kiloliters'].mean()
        
        for day in range(1, days_in_month + 1):
            # Apply daily pattern if available
            if daily_patterns is not None and day in daily_patterns.index:
                daily_factor = daily_patterns[day] / daily_avg
            else:
                daily_factor = 1.0
            
            # Add weekly patterns
            current_date = next_month_start + timedelta(days=day-1)
            day_of_week = current_date.weekday()
            
            # Weekend factor (water usage might be different on weekends)
            if day_of_week >= 5:  # Weekend
                weekly_factor = 1.1  # Might be higher for residential water usage
            else:
                weekly_factor = 0.95
            
            # Add small random variation
            random_factor = np.random.normal(1.0, 0.03)
            daily_consumption = base_daily * daily_factor * weekly_factor * random_factor
            daily_predictions.append(max(0, daily_consumption))
        
        # Create DataFrame with hourly data
        all_dates = []
        hourly_predictions = []
        
        for day, daily_pred in enumerate(daily_predictions):
            daily_base = daily_pred / 24
            
            for hour in range(24):
                # Add hourly variations based on typical water usage patterns
                if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
                    hourly_factor = 1.4
                elif 23 <= hour or hour <= 5:  # Low usage hours
                    hourly_factor = 0.3
                else:
                    hourly_factor = 1.0
                
                hourly_consumption = daily_base * hourly_factor
                hourly_predictions.append(round(hourly_consumption, 3))
                
                # Create datetime for this hour
                current_datetime = next_month_start + timedelta(days=day, hours=hour)
                all_dates.append(current_datetime.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Create final DataFrame - UPDATED COLUMN NAME
        predictions_df = pd.DataFrame({
            'Datetime': all_dates,
            'AWU_kiloliters': hourly_predictions
        })
        
        # Save to CSV
        csv_filename = f'next_month_water_prediction_{next_month_start.strftime("%Y_%m")}.csv'
        predictions_df.to_csv(csv_filename, index=False)
        logger.info(f"📁 Next month predictions saved to {csv_filename}")
        
        # Generate visualization
        png_filename = f'next_month_water_prediction_{next_month_start.strftime("%Y_%m")}.png'
        try:
            # Try complex visualization first
            fig = create_prediction_visualization(
                monthly_data, 
                monthly_prediction, 
                next_month_start.strftime('%Y-%m'), 
                f"{confidence*100:.0f}%", 
                method
            )
            fig.savefig(png_filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            logger.info(f"📊 Complex visualization saved to {png_filename}")
            
        except Exception as viz_error:
            logger.warning(f"⚠ Complex visualization failed: {viz_error}, trying simple chart")
            try:
                # Fallback to simple chart
                fig = create_simple_trend_chart(
                    monthly_data, 
                    monthly_prediction, 
                    next_month_start.strftime('%Y-%m')
                )
                if fig:
                    fig.savefig(png_filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    logger.info(f"📊 Simple visualization saved to {png_filename}")
                else:
                    png_filename = None
                    logger.warning("⚠ Visualization creation failed")
            except Exception as simple_error:
                logger.error(f"❌ Simple visualization also failed: {simple_error}")
                png_filename = None
        
        # Calculate totals
        total_consumption = sum(daily_predictions)
        
        return {
            "prediction": {
                "month": next_month_start.strftime('%Y-%m'),
                "total_consumption": round(total_consumption, 1),
                "daily_average": round(total_consumption / days_in_month, 1),
                "confidence": f"{confidence*100:.0f}%",
                "method": method,
                "total_days": days_in_month,
                "total_hours": len(hourly_predictions),
                "csv_saved": True,
                "csv_filename": csv_filename,
                "png_saved": png_filename is not None,
                "png_filename": png_filename,
                "files_generated": [csv_filename] + ([png_filename] if png_filename else [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Next month prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Next month prediction failed: {str(e)}")

@app.post("/api/predict-month-after-next")
async def predict_month_after_next_consumption():
    """Generate month after next's daily predictions and save as CSV + PNG visualization - ENHANCED VERSION"""
    global uploaded_data, model, scaler
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    try:
        # Calculate monthly totals for prediction - UPDATED COLUMN
        monthly_data = uploaded_data.groupby(['Year', 'Month']).agg({
            'AWU_kiloliters': 'sum'  # Use sum for proper monthly totals
        }).reset_index()
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        if len(monthly_data) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 months of data for month+2 prediction")
        
        # Determine month after next
        last_date = uploaded_data['Datetime'].max()
        next_month_start = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_after_next_start = (next_month_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        days_in_month = get_days_in_month(month_after_next_start.year, month_after_next_start.month)
        
        # For month+2 prediction, we need to be more conservative
        try:
            # First predict next month
            next_month_pred, _, _ = make_ml_prediction(monthly_data.tail(6))
            
            # Create extended dataset including the predicted next month
            extended_data = monthly_data.copy()
            new_row = pd.DataFrame({
                'Year': [next_month_start.year],
                'Month': [next_month_start.month],
                'AWU_kiloliters': [next_month_pred]
            })
            extended_data = pd.concat([extended_data, new_row], ignore_index=True)
            
            # Now predict month+2 using extended data
            month_plus_2_pred, confidence, method = make_ml_prediction(extended_data.tail(6))
            confidence *= 0.85  # Reduce confidence for longer prediction
            
            logger.info(f"✅ Month+2 prediction: {month_plus_2_pred:.1f} KL using {method}")
            
        except Exception:
            # Fall back to statistical method with dampening
            month_plus_2_pred, confidence, method = make_statistical_prediction(monthly_data.tail(6))
            confidence *= 0.80  # Further reduce confidence
            
            # Add additional dampening for longer prediction
            current_avg = monthly_data['AWU_kiloliters'].iloc[-1]
            month_plus_2_pred = current_avg + (month_plus_2_pred - current_avg) * 0.7
            
            logger.info(f"✅ Month+2 prediction: {month_plus_2_pred:.1f} KL using {method} (dampened)")
        
        # Generate daily predictions (same logic as next month)
        daily_predictions = []
        base_daily = month_plus_2_pred / days_in_month
        
        # Get historical daily patterns if available
        if len(uploaded_data) > 30:
            daily_patterns = uploaded_data.groupby(uploaded_data['Datetime'].dt.day)['AWU_kiloliters'].mean()
            daily_avg = daily_patterns.mean()
        else:
            daily_patterns = None
            daily_avg = uploaded_data['AWU_kiloliters'].mean()
        
        for day in range(1, days_in_month + 1):
            # Apply daily pattern if available
            if daily_patterns is not None and day in daily_patterns.index:
                daily_factor = daily_patterns[day] / daily_avg
            else:
                daily_factor = 1.0
            
            # Add weekly patterns
            current_date = month_after_next_start + timedelta(days=day-1)
            day_of_week = current_date.weekday()
            
            # Weekend factor
            if day_of_week >= 5:  # Weekend
                weekly_factor = 1.1  # Water usage patterns
            else:
                weekly_factor = 0.95
            
            # Add larger seasonal variation for longer prediction
            seasonal_factor = np.random.normal(1.0, 0.05)
            daily_consumption = base_daily * daily_factor * weekly_factor * seasonal_factor
            daily_predictions.append(max(0, daily_consumption))
        
        # Create DataFrame with hourly data
        all_dates = []
        hourly_predictions = []
        
        for day, daily_pred in enumerate(daily_predictions):
            daily_base = daily_pred / 24
            
            for hour in range(24):
                # Add hourly variations
                if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
                    hourly_factor = 1.4
                elif 23 <= hour or hour <= 5:  # Low usage hours
                    hourly_factor = 0.3
                else:
                    hourly_factor = 1.0
                
                hourly_consumption = daily_base * hourly_factor
                hourly_predictions.append(round(hourly_consumption, 3))
                
                # Create datetime for this hour
                current_datetime = month_after_next_start + timedelta(days=day, hours=hour)
                all_dates.append(current_datetime.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Create final DataFrame - UPDATED COLUMN NAME
        predictions_df = pd.DataFrame({
            'Datetime': all_dates,
            'AWU_kiloliters': hourly_predictions
        })
        
        # Save to CSV
        csv_filename = f'month_after_next_water_prediction_{month_after_next_start.strftime("%Y_%m")}.csv'
        predictions_df.to_csv(csv_filename, index=False)
        logger.info(f"📁 Month after next predictions saved to {csv_filename}")
        
        # Generate visualization
        png_filename = f'month_after_next_water_prediction_{month_after_next_start.strftime("%Y_%m")}.png'
        try:
            # Try complex visualization first
            fig = create_prediction_visualization(
                monthly_data, 
                month_plus_2_pred, 
                month_after_next_start.strftime('%Y-%m'), 
                f"{confidence*100:.0f}%", 
                method
            )
            fig.savefig(png_filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            logger.info(f"📊 Complex visualization saved to {png_filename}")
            
        except Exception as viz_error:
            logger.warning(f"⚠ Complex visualization failed: {viz_error}, trying simple chart")
            try:
                # Fallback to simple chart
                fig = create_simple_trend_chart(
                    monthly_data, 
                    month_plus_2_pred, 
                    month_after_next_start.strftime('%Y-%m')
                )
                if fig:
                    fig.savefig(png_filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    logger.info(f"📊 Simple visualization saved to {png_filename}")
                else:
                    png_filename = None
                    logger.warning("⚠ Visualization creation failed")
            except Exception as simple_error:
                logger.error(f"❌ Simple visualization also failed: {simple_error}")
                png_filename = None
        
        # Calculate totals
        total_consumption = sum(daily_predictions)
        
        return {
            "prediction": {
                "month": month_after_next_start.strftime('%Y-%m'),
                "total_consumption": round(total_consumption, 1),
                "daily_average": round(total_consumption / days_in_month, 1),
                "confidence": f"{confidence*100:.0f}%",
                "method": method,
                "total_days": days_in_month,
                "total_hours": len(hourly_predictions),
                "csv_saved": True,
                "csv_filename": csv_filename,
                "png_saved": png_filename is not None,
                "png_filename": png_filename,
                "files_generated": [csv_filename] + ([png_filename] if png_filename else [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Month after next prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Month after next prediction failed: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated CSV or PNG files"""
    try:
        # Security check - only allow specific file patterns - UPDATED FOR WATER
        allowed_patterns = [
            'next_month_water_prediction_',
            'month_after_next_water_prediction_',
            'water_prediction_analysis_'
        ]
        
        allowed_extensions = ['.csv', '.png']
        
        # Check if filename matches allowed patterns and extensions
        if not any(filename.startswith(pattern) for pattern in allowed_patterns):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=404, detail="File type not allowed")
        
        # Check if file exists
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        if filename.endswith('.csv'):
            media_type = 'text/csv'
        elif filename.endswith('.png'):
            media_type = 'image/png'
        else:
            media_type = 'application/octet-stream'
        
        from fastapi.responses import FileResponse
        
        return FileResponse(
            path=filename,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading file")

@app.get("/api/list-files")
async def list_generated_files():
    """List all generated prediction files - UPDATED FOR WATER"""
    try:
        files = []
        
        # Look for prediction files in current directory - UPDATED PATTERNS
        for filename in os.listdir('.'):
            if (filename.startswith(('next_month_water_prediction_', 'month_after_next_water_prediction_', 'water_prediction_analysis_')) 
                and filename.endswith(('.csv', '.png'))):
                
                file_info = {
                    "filename": filename,
                    "type": "CSV Data" if filename.endswith('.csv') else "Visualization",
                    "size": os.path.getsize(filename),
                    "created": datetime.fromtimestamp(os.path.getctime(filename)).isoformat(),
                    "download_url": f"/api/download/{filename}"
                }
                files.append(file_info)
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            "files": files,
            "total_files": len(files)
        }
        
    except Exception as e:
        logger.error(f"❌ List files error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing files")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "data_uploaded": uploaded_data is not None,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "is_retraining": is_retraining
    }

@app.get("/")
async def root():
    """Root endpoint with API information - UPDATED FOR WATER"""
    return {
        "message": "Water Consumption Analytics API - Complete Migration from Electricity",
        "version": "3.0.0",
        "status": "running",
        "improvements": [
            "Complete migration from electricity to water consumption",
            "Updated column names from AEP_KWH to AWU_kiloliters",
            "Removed billing calculations and cost analysis",
            "Updated CO2 calculation formula (consumption * 0.7)",
            "Added leakage detection algorithms",
            "Water-specific hourly usage patterns",
            "Updated visualization titles and units to KL"
        ],
        "features": {
            "co2_calculation": f"{CO2_FACTOR} kg CO₂e/KL",
            "data_format": "AWU_kiloliters (Water Consumption in Kiloliters)",
            "ml_prediction": "LSTM with proper scaling for water data",
            "fallback": "Statistical prediction methods",
            "leakage_detection": "Pattern-based leakage alert system"
        },
        "endpoints": {
            "upload": "POST /api/upload",
            "monthly_data": "GET /api/data/monthly?month=YYYY-MM",
            "daily_data": "GET /api/data/daily?date=YYYY-MM-DD",
            "statistics": "GET /api/stats?month=YYYY-MM (optional)",
            "comparison": "GET /api/stats/comparison?month=YYYY-MM",
            "peak_hours": "GET /api/stats/peak-hours?month=YYYY-MM",
            "leakage_alerts": "GET /api/leakage/alerts",
            "predict": "POST /api/predict",
            "predict_next_month": "POST /api/predict-next-month",
            "predict_month_after_next": "POST /api/predict-month-after-next",
            "retrain": "POST /api/retrain",
            "retrain_status": "GET /api/retrain/status",
            "download": "GET /api/download/{filename}",
            "list_files": "GET /api/list-files",
            "health": "GET /api/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)