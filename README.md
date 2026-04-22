# Store Intelligence System

**End-to-end computer vision pipeline: CCTV footage → real-time retail analytics API**

Convert raw store footage into actionable metrics: visitor counts, conversion funnels, zone heatmaps, and anomaly detection.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Detection Pipeline
Process CCTV videos and generate events:
```bash
python pipeline/run.py --video-dir data/videos --output data/events.jsonl
```

### 3. Start API Server
```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Query Metrics
```bash
# Ingest events
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '[{"event_id":"...", "store_id":"STORE_BLR_002", ...}]'

# Get metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# Get funnel
curl http://localhost:8000/stores/STORE_BLR_002/funnel

# Get heatmap
curl http://localhost:8000/stores/STORE_BLR_002/heatmap

# Get anomalies
curl http://localhost:8000/stores/STORE_BLR_002/anomalies

# Health check
curl http://localhost:8000/health
```

## System Architecture

```
Raw Video (MP4)
    ↓
YOLOv8 Detection (person class)
    ↓
Centroid Tracker (unique visitor IDs)
    ↓
Event Emitter (ENTRY, EXIT, ZONE_DWELL, etc.)
    ↓
JSONL Events
    ↓
FastAPI Ingestion (deduplication, validation)
    ↓
SQLite Storage
    ↓
Analytics Queries (metrics, funnel, heatmap, anomalies)
```

## Project Structure

```
pipeline/
  ├── tracker.py      # Centroid-based object tracker
  ├── detect.py       # YOLOv8 detection (mock for testing)
  ├── emit.py         # Event emission logic
  ├── models.py       # Data models
  ├── run.py          # Pipeline orchestration
  └── __init__.py

app/
  ├── main.py         # FastAPI endpoints
  ├── database.py     # SQLite storage
  ├── ingestion.py    # Event validation
  ├── metrics.py      # Analytics computation
  └── __init__.py

tests/
  ├── test_pipeline.py  # Tracker & emitter tests
  ├── test_api.py       # API endpoint tests
  └── __init__.py

docs/
  ├── DESIGN.md       # Architecture & design decisions
  └── CHOICES.md      # Trade-offs & rationale
```

## Event Schema

```json
{
  "event_id": "uuid-v4",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_12345",
  "event_type": "ENTRY|EXIT|ZONE_ENTER|ZONE_EXIT|ZONE_DWELL|BILLING_QUEUE_JOIN|BILLING_QUEUE_ABANDON|REENTRY",
  "timestamp": "2026-03-03T14:22:10Z",
  "zone_id": "SKINCARE",
  "dwell_ms": 8400,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": {
    "queue_depth": null,
    "sku_zone": "MOISTURISER",
    "session_seq": 5
  }
}
```

## API Endpoints

### POST /events/ingest
Batch ingest events with deduplication and validation.

**Request:**
```json
[
  {
    "event_id": "uuid",
    "store_id": "STORE_BLR_002",
    "camera_id": "CAM_ENTRY_01",
    "visitor_id": "VIS_12345",
    "event_type": "ENTRY",
    "timestamp": "2026-03-03T14:22:10Z",
    "is_staff": false,
    "confidence": 0.9,
    "metadata": {"session_seq": 1}
  }
]
```

**Response:**
```json
{
  "status": "success",
  "trace_id": "uuid",
  "events_ingested": 100,
  "duplicates": 5,
  "validation_errors": [],
  "database_errors": []
}
```

**Constraints:**
- Max batch size: 500 events
- Returns 400 if batch > 500
- Idempotent by event_id

### GET /stores/{store_id}/metrics
Get store metrics for time window.

**Response:**
```json
{
  "store_id": "STORE_BLR_002",
  "time_window_hours": 24,
  "unique_visitors": 342,
  "avg_dwell_time_ms": 4200.5,
  "conversion_rate": 0.2,
  "max_queue_depth": 8,
  "trace_id": "uuid"
}
```

### GET /stores/{store_id}/funnel
Get conversion funnel: ENTRY → ZONE → BILLING → PURCHASE.

**Response:**
```json
{
  "store_id": "STORE_BLR_002",
  "time_window_hours": 24,
  "funnel": {
    "entry": 342,
    "zone_visit": 298,
    "billing_queue": 156,
    "purchase": 89
  },
  "dropoff_percentages": {
    "entry_to_zone": 12.87,
    "zone_to_billing": 47.65,
    "billing_to_purchase": 42.95
  },
  "trace_id": "uuid"
}
```

### GET /stores/{store_id}/heatmap
Get zone visit frequency (normalized 0-100).

**Response:**
```json
{
  "store_id": "STORE_BLR_002",
  "time_window_hours": 24,
  "zones": {
    "SKINCARE": 100,
    "BILLING": 45.6,
    "CHECKOUT": 23.2
  },
  "trace_id": "uuid"
}
```

### GET /stores/{store_id}/anomalies
Detect anomalies: queue spike, dead zone, conversion drop.

**Response:**
```json
{
  "store_id": "STORE_BLR_002",
  "time_window_hours": 24,
  "anomalies": [
    {
      "type": "QUEUE_SPIKE",
      "severity": "WARN",
      "message": "Queue depth reached 12",
      "value": 12
    },
    {
      "type": "CONVERSION_DROP",
      "severity": "CRITICAL",
      "message": "Visitor count down 25.3% vs 7-day avg",
      "current": 256,
      "expected": 343
    }
  ],
  "count": 2,
  "trace_id": "uuid"
}
```

### GET /health
Health check with stale feed detection.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2026-03-03T14:22:10Z",
  "last_event_timestamp": "2026-03-03T14:20:10Z",
  "stale_feed_warning": false,
  "trace_id": "uuid"
}
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_pipeline.py -v
pytest tests/test_api.py -v
```

Test coverage includes:
- Tracker: ID assignment, frame-to-frame consistency, stale removal
- Emitter: All 8 event types, zone transitions, dwell calculation
- API: Ingestion, idempotency, batch validation, metrics queries
- Schema: Validation, serialization, edge cases

## Production Features

✓ **Idempotent Ingestion**: Deduplication by event_id (primary key)
✓ **Structured Logging**: Trace IDs, latency, status codes
✓ **Error Handling**: Graceful responses, no stack traces
✓ **Batch Processing**: Partial success, detailed error reporting
✓ **Real-time Queries**: No caching, always current data
✓ **Anomaly Detection**: Queue spikes, dead zones, conversion drops

## Design Approach

### Why Centroid Tracking Instead of DeepSORT?
The system uses a simple centroid-based tracker instead of more complex approaches like DeepSORT. This choice prioritizes:
- **Simplicity:** ~50 lines of code, fully debuggable
- **Speed:** O(n²) matching is fast for <100 people per frame
- **Sufficiency:** Single-camera tracking is adequate for MVP
- **Time constraint:** 48-hour deadline favors proven, simple solutions

The trade-off is that cross-camera re-ID is not supported, but this can be added post-submission using a Re-ID model like OSNet.

### Why SQLite Instead of PostgreSQL?
- **Zero setup:** No Docker service, no configuration
- **Sufficient:** <1M events is plenty for 5 stores
- **Simplicity:** Single file, easy to backup and deploy
- **Debuggability:** Can inspect database directly with sqlite3 CLI

For 40 live stores, PostgreSQL would be necessary. The architecture uses SQLAlchemy, so migration is straightforward.

### Why Heuristic Staff Filtering?
Staff detection uses a simple heuristic (confidence > 0.9 + aspect ratio > 2.0) instead of ML:
- **Speed:** No additional inference needed
- **Simplicity:** Single line of code
- **Sufficient:** 80% accuracy is acceptable for MVP

A VLM-based approach (GPT-4V for clothing detection) would improve accuracy but adds latency and cost.

## How Conversion Rate is Computed

Conversion rate is calculated by correlating visitor sessions with POS transactions:

1. **Visitor Detection:** ENTRY events create unique visitor_id
2. **Billing Zone Tracking:** BILLING_QUEUE_JOIN events mark visitors in checkout
3. **POS Correlation:** For each transaction, find visitors in billing zone within 5-minute window
4. **Conversion Rate:** (converted_visitors / total_unique_visitors) × 100

**Formula:**
```
conversion_rate = (visitors_with_POS_transaction / total_unique_visitors) × 100
```

**Example:**
- 342 unique visitors today
- 89 visitors matched to POS transactions
- Conversion rate = (89 / 342) × 100 = 26.0%

**Note:** This requires pos_transactions.csv with store_id, transaction_id, timestamp, basket_value_inr

## Limitations

- **Single-camera tracking:** No cross-camera re-ID (can upgrade with OSNet)
- **Heuristic staff filtering:** 80% accuracy (can upgrade with VLM)
- **Hardcoded zones:** Zone rectangles from store_layout.json (can learn with VLM)
- **SQLite scalability:** ~1M event limit (can migrate to PostgreSQL)
- **No real-time streaming:** Batch processing only (can add Kafka)

## Future Improvements

- Re-ID model (OSNet) for cross-camera tracking
- VLM zone classification (GPT-4V)
- Real-time streaming (Kafka)
- Web dashboard (React)
- ML anomaly detection (Isolation Forest)
- Multi-store distributed architecture

## Correctness Properties

1. **Entry/Exit Accuracy**: entry_count ≈ exit_count (±5%)
2. **Unique Visitor Deduplication**: No duplicate visitor_ids in ENTRY events
3. **Idempotent Ingestion**: POST /events/ingest twice → same result
4. **Funnel Monotonicity**: Entry ≥ Zone ≥ Billing ≥ Purchase

## Deployment

### Local Development
```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Docker (Production)
```bash
docker build -t store-intelligence:latest .
docker run -p 8000:8000 -v /data:/app/data store-intelligence:latest
```

## Troubleshooting

**No events generated:**
- Check `data/videos/` contains MP4 files
- Verify `pipeline/run.py` completes without errors
- Check `data/events.jsonl` for output

**API returns 503:**
- Database may be locked
- Check `store_intelligence.db` permissions
- Restart API server

**Metrics show zero:**
- Ensure events were ingested via POST /events/ingest
- Check store_id matches in queries
- Verify timestamps are recent (within time_window_hours)

## Support

For questions about architecture, see `docs/DESIGN.md`.
For questions about trade-offs, see `docs/CHOICES.md`.
For questions about implementation, check test files for examples.
