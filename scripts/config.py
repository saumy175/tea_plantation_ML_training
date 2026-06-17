from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_AGRI_API_KEY = os.getenv(
    "GOOGLE_AGRI_API_KEY"
)

GEE_PROJECT_ID = os.getenv(
    "GEE_PROJECT_ID"
)

UTM_CRS = os.getenv(
    "UTM_ZONE",
    "EPSG:32646"
)

if GOOGLE_AGRI_API_KEY is None:
    raise RuntimeError(
        "Missing GOOGLE_AGRI_API_KEY"
    )

if GEE_PROJECT_ID is None:
    raise RuntimeError(
        "Missing GEE_PROJECT_ID"
    )