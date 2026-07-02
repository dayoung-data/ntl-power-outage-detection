# =============================================================================
# config.py — Google Earth Engine Initialization
# =============================================================================
# Setup:
#   1. Replace GEE_PROJECT with your own GEE project ID
#   2. Run ee.Authenticate() once in terminal if not yet authenticated
#   3. Import this file at the top of any pipeline script
#
# To authenticate:
#   >> import ee
#   >> ee.Authenticate()
# =============================================================================

import ee
import os

# ── User Settings ─────────────────────────────────────────────────────────────
GEE_PROJECT = "your-gee-project-id"   # Replace with your GEE project ID
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ── GEE Initialization ────────────────────────────────────────────────────────
def initialize_gee():
    """Initialize Google Earth Engine. Run ee.Authenticate() first if needed."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized (project: {GEE_PROJECT})")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("Try running: import ee; ee.Authenticate()")
        raise
