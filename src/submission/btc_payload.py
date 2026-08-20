import requests
import json
import time
from typing import Dict, Any, Optional
from config import SUBMISSION_URL, SESSION_ID

def build_submission_payload(
    metadata: Dict[str, Any],
    query: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build official BTC submission payload.
    Adjust fields here according to official BTC contest documentation / video guideline.
    """
    global_id = metadata.get("global_id")
    video_id = metadata.get("video_id")
    frame_idx = metadata.get("frame_idx")
    frame_id = metadata.get("frame_id")
    timestamp = metadata.get("timestamp")

    # Standard competition format
    payload = {
        "session_id": session_id or SESSION_ID,
        "submission_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query or "",
        "item": {
            "global_id": global_id,
            "video_name": f"{video_id}.mp4" if video_id and not video_id.endswith(".mp4") else video_id,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
    }
    return payload


def submit_to_btc(
    payload: Dict[str, Any],
    target_url: str = SUBMISSION_URL,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Submits payload to BTC Evaluation Server.
    """
    try:
        response = requests.post(target_url, json=payload, timeout=timeout)
        try:
            res_data = response.json()
        except Exception:
            res_data = {"text": response.text}
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": res_data,
            "submitted_payload": payload
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 503,
            "error": f"Failed to connect to BTC server ({str(e)})",
            "submitted_payload": payload
        }
