import os
import json
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps, lru_cache
import traceback
from datetime import datetime
import threading
import psutil

# Import your healthcare AI components
from model import HealthcareAISystem

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}})

# --- Apple M3 Optimization Settings ---
print("🍎 Detecting Apple Silicon optimizations...")

# Check for MPS (Metal Performance Shaders) support
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ MPS (Metal Performance Shaders) detected - using Apple Silicon GPU")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ CUDA detected")
else:
    DEVICE = "cpu"
    print("⚠️  Falling back to CPU")

print(f"🔥 Using device: {DEVICE}")

# M3 specific optimizations
if DEVICE == "mps":
    # Set memory fraction for M3
    torch.mps.set_per_process_memory_fraction(0.8)
    print("✅ MPS memory optimization enabled")

# Performance settings
ENABLE_MODEL_CACHING = True
MAX_CACHE_SIZE = 50  # Reduced for M3 memory management
REQUEST_TIMEOUT = 15  # Faster on M3

# --- Security ---
API_KEY = os.getenv("FLASK_API_KEY")

def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if not API_KEY:
            return view_function(*args, **kwargs)
        received_key = request.headers.get('X-API-Key')
        if not received_key or received_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return view_function(*args, **kwargs)
    return decorated_function

# --- M3 Optimized Healthcare AI ---
class M3OptimizedHealthcareAI:
    def __init__(self):
        # The model_name should ideally match the fine-tuned model name you're deploying in Vertex AI
        self.healthcare_ai = HealthcareAISystem(model_name="final-nutrional")
        self.is_loading = False
        self.load_lock = threading.Lock()
        self.response_cache = {}
        self.last_used = {}
        # Ensure _is_initialized is set appropriately after a successful initialization
        self._is_initialized = False

    def initialize(self):
        """Initialize with M3 optimizations. This should run once at startup."""
        with self.load_lock:
            if self._is_initialized: # Use the internal flag
                print("AI System already initialized.")
                return True

            if self.is_loading:
                print("AI System is currently loading...")
                return False

            self.is_loading = True
            try:
                # The load_model call for HealthcareAISystem should happen here once.
                # It's important that HealthcareAISystem().load_model() successfully handles
                # the Vertex AI/Gemini model loading without errors.
                model_loaded = self.healthcare_ai.load_model()
                if model_loaded:
                    self._is_initialized = True
                    print("✅ Healthcare AI System initialized and model loaded!")
                    return True
                else:
                    print("⚠️ Healthcare AI System failed to load model, running in fallback mode (if supported).")
                    self._is_initialized = False # Or set to True if fallback is considered initialized
                    return False # Indicate failure to load the primary model
            except Exception as e:
                print(f"❌ AI System initialization failed: {e}")
                traceback.print_exc()
                self._is_initialized = False
                return False
            finally:
                self.is_loading = False

    def _print_memory_status(self):
        """Print memory usage for M3 monitoring"""
        memory = psutil.virtual_memory()
        print(f"📊 Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

        if DEVICE == "mps" and hasattr(torch.backends.mps, 'current_allocated_memory'):
            try:
                mps_memory = torch.backends.mps.current_allocated_memory() / 1024**3
                print(f"🍎 MPS Memory: {mps_memory:.2f}GB allocated")
            except:
                pass

    def _warmup_model(self):
        """M3-optimized model warmup"""
        try:
            print("🔥 Warming up model on M3...")
            start_time = time.time()

            # Use torch.no_grad() for inference optimization
            with torch.no_grad():
                if DEVICE == "mps":
                    # Clear MPS cache before warmup
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()

                # A simple query for warmup
                result = self.healthcare_ai.generate_response(
                    "What is a common cold?", # Use a simple, non-sensitive query for warmup
                    {"age": 30, "conditions": []}
                )

            warmup_time = time.time() - start_time
            print(f"✅ M3 warmup completed in {warmup_time:.2f}s")

        except Exception as e:
            print(f"⚠️  M3 warmup failed: {e}")
            traceback.print_exc() # Print traceback for debugging warmup issues

    def generate_response_optimized(self, question, patient_info=None, domain=None):
        """M3-optimized response generation with caching"""
        # Ensure the AI system has been initialized before attempting to generate a response
        if not self._is_initialized:
            raise Exception("AI system not initialized. Please wait for startup initialization to complete.")

        # Create cache key
        cache_key = f"{question}:{json.dumps(patient_info, sort_keys=True)}:{domain}"

        # Check cache first
        if ENABLE_MODEL_CACHING and cache_key in self.response_cache:
            print("💨 Cache hit - returning cached response")
            self.last_used[cache_key] = time.time()
            return self.response_cache[cache_key]

        # Generate new response with M3 optimizations
        start_time = time.time()

        try:
            with torch.no_grad():  # Disable gradients for inference
                if DEVICE == "mps":
                    # Clear MPS cache if needed before a new inference call if cache is large
                    if len(self.response_cache) > MAX_CACHE_SIZE:
                        self._cleanup_cache()
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            torch.backends.mps.empty_cache()

                # This calls the generate_response method in model.py.
                # Ensure this method in model.py is correctly implemented to
                # interact with your fine-tuned Gemini model via Vertex AI.
                result = self.healthcare_ai.generate_response(
                    question=question,
                    patient_info=patient_info or {},
                    domain=domain
                )

            inference_time = time.time() - start_time
            print(f"⚡ M3 inference completed in {inference_time:.2f}s")

            # Cache the result
            if ENABLE_MODEL_CACHING and len(self.response_cache) < MAX_CACHE_SIZE:
                self.response_cache[cache_key] = result
                self.last_used[cache_key] = time.time()

            return result

        except Exception as e:
            print(f"❌ M3 inference error: {e}")
            raise

    def _cleanup_cache(self):
        """Clean up old cache entries"""
        if len(self.response_cache) <= MAX_CACHE_SIZE // 2:
            return

        # Remove oldest entries
        sorted_items = sorted(self.last_used.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) - MAX_CACHE_SIZE // 2

        for key, _ in sorted_items[:items_to_remove]:
            self.response_cache.pop(key, None)
            self.last_used.pop(key, None)

        print(f"🧹 Cache cleaned up - {items_to_remove} items removed")

# Initialize the optimized AI system
ai_system = M3OptimizedHealthcareAI()

# --- API Endpoints ---
@app.route('/api/query', methods=['POST'])
@require_api_key
def handle_query():
    """M3-optimized query handler"""
    # Check if the AI system has been initialized.
    # The initialization happens once in a background thread at startup.
    if not ai_system._is_initialized:
        return jsonify({
            "error": "AI system not available",
            "message": "System is still initializing or failed to start. Please wait."
        }), 503

    try:
        request_start = time.time()

        # Validate request
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No JSON data provided"}), 400

        question = request_data.get('question', '').strip()
        if not question:
            return jsonify({"error": "Question is required"}), 400

        patient_info = request_data.get('patient_info', {})
        domain = request_data.get('domain')

        print(f"🔍 Processing query: {question[:50]}...")

        # Generate response with M3 optimizations
        # Removed redundant ai_system.initialize() and ai_system.healthcare_ai.load_model()
        # as they should be handled once during startup initialization.
        result = ai_system.generate_response_optimized(
            question=question,
            patient_info=patient_info,
            domain=domain
        )

        total_time = time.time() - request_start
        print(f"✅ Total request time: {total_time:.2f}s")

        # Add performance metrics
        result['performance'] = {
            'total_time': round(total_time, 2),
            'device': DEVICE,
            'cached': cache_key in ai_system.response_cache if 'cache_key' in locals() else False
        }

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Query error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Query processing failed",
            "message": str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status_check():
    """M3-optimized status check"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "device": DEVICE,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends.mps, 'is_available') else False,
    }

    # Add memory info
    memory = psutil.virtual_memory()
    status["memory"] = {
        "percent_used": memory.percent,
        "available_gb": round(memory.available / 1024**3, 1)
    }

    # Check the actual initialization status of the AI system
    if ai_system._is_initialized:
        status.update({
            "status": "healthy",
            "message": "Healthcare AI running on Apple M3",
            "cache_size": len(ai_system.response_cache)
        })
        return jsonify(status), 200
    else:
        status.update({
            "status": "unhealthy",
            "message": "AI system not initialized or failed during startup"
        })
        return jsonify(status), 503

@app.route('/api/cache/clear', methods=['POST'])
@require_api_key
def clear_cache():
    """Clear response cache"""
    ai_system.response_cache.clear()
    ai_system.last_used.clear()

    if DEVICE == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
        torch.backends.mps.empty_cache()

    return jsonify({"message": "Cache cleared", "device": DEVICE}), 200

# --- Startup ---
def initialize_on_startup():
    """Initialize AI system in background"""
    print("🚀 Starting M3 Healthcare AI initialization...")
    success = ai_system.initialize()
    if success:
        # After successful initialization, perform warmup if desired
        ai_system._warmup_model()
        print("✅ M3 Healthcare AI ready!")
    else:
        print("❌ M3 Healthcare AI initialization failed!")

if __name__ == '__main__':
    # Start initialization in background
    init_thread = threading.Thread(target=initialize_on_startup)
    init_thread.daemon = True
    init_thread.start()

    print("🍎 Starting Flask API optimized for Apple M3...")
    print(f"🔥 Device: {DEVICE}")
    print(f"🔐 API Security: {'Enabled' if API_KEY else 'Disabled'}")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=8000,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent re-initialization
    )