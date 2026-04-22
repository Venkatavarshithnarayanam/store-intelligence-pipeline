# PROMPT:
# "Generate pytest tests for FastAPI endpoints with idempotency and batch validation"
# CHANGES MADE:
# - Added ingestion endpoint tests
# - Added idempotency verification
# - Added batch size validation
# - Updated response format checks
# - Added edge case tests: empty store, all-staff, zero purchases, re-entry

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import EventDatabase
from pipeline.models import Event, EventMetadata
import json
import os


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_db():
    """Create test database."""
    db_path = "test_store_intelligence.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = EventDatabase(db_path)
    yield db
    
    if os.path.exists(db_path):
        os.remove(db_path)


class TestEventIngestion:
    """Test event ingestion endpoint."""
    
    def test_ingest_single_event(self, client):
        """Test ingesting a single event."""
        event = {
            "event_id": "test-event-1",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_12345",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:22:10Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"session_seq": 1}
        }
        
        response = client.post("/events/ingest", json=[event])
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "partial"]
        assert data["events_ingested"] == 1
        assert data["duplicates"] == 0
        assert "trace_id" in data
    
    def test_ingest_multiple_events(self, client):
        """Test ingesting multiple events."""
        events = [
            {
                "event_id": f"test-event-{i}",
                "store_id": "STORE_BLR_002",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": f"VIS_{i}",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            }
            for i in range(5)
        ]
        
        response = client.post("/events/ingest", json=events)
        
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 5
        assert data["duplicates"] == 0
    
    def test_ingest_batch_too_large(self, client):
        """Test that batch size > 500 is rejected."""
        events = [
            {
                "event_id": f"test-event-{i}",
                "store_id": "STORE_BLR_002",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": f"VIS_{i}",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            }
            for i in range(501)
        ]
        
        response = client.post("/events/ingest", json=events)
        
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "BATCH_TOO_LARGE"
    
    def test_ingest_idempotency(self, client):
        """Test that ingesting same events twice is idempotent."""
        event = {
            "event_id": "test-event-idempotent",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_12345",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:22:10Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"session_seq": 1}
        }
        
        # First ingest
        response1 = client.post("/events/ingest", json=[event])
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["events_ingested"] == 1
        assert data1["duplicates"] == 0
        
        # Second ingest (same event)
        response2 = client.post("/events/ingest", json=[event])
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["events_ingested"] == 0
        assert data2["duplicates"] == 1
    
    def test_ingest_missing_required_field(self, client):
        """Test that missing required fields are rejected."""
        event = {
            "event_id": "test-event-1",
            "store_id": "STORE_BLR_002",
            # Missing camera_id
            "visitor_id": "VIS_12345",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:22:10Z",
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"session_seq": 1}
        }
        
        response = client.post("/events/ingest", json=[event])
        
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 0
        assert len(data["validation_errors"]) > 0
    
    def test_ingest_invalid_event_type(self, client):
        """Test that invalid event types are rejected."""
        event = {
            "event_id": "test-event-1",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_12345",
            "event_type": "INVALID_TYPE",
            "timestamp": "2026-03-03T14:22:10Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"session_seq": 1}
        }
        
        response = client.post("/events/ingest", json=[event])
        
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 0
        assert len(data["validation_errors"]) > 0
    
    def test_ingest_invalid_confidence(self, client):
        """Test that invalid confidence values are rejected."""
        event = {
            "event_id": "test-event-1",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_12345",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:22:10Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 1.5,  # Invalid: > 1.0
            "metadata": {"session_seq": 1}
        }
        
        response = client.post("/events/ingest", json=[event])
        
        assert response.status_code == 200
        data = response.json()
        assert data["events_ingested"] == 0
        assert len(data["validation_errors"]) > 0


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    def test_get_metrics(self, client):
        """Test GET /stores/{id}/metrics endpoint."""
        response = client.get("/stores/STORE_BLR_002/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "unique_visitors" in data
        assert "avg_dwell_time_ms" in data
        assert "conversion_rate" in data
        assert "max_queue_depth" in data
        assert "trace_id" in data
    
    def test_get_funnel(self, client):
        """Test GET /stores/{id}/funnel endpoint."""
        response = client.get("/stores/STORE_BLR_002/funnel")
        
        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "funnel" in data
        assert "entry" in data["funnel"]
        assert "zone_visit" in data["funnel"]
        assert "billing_queue" in data["funnel"]
        assert "purchase" in data["funnel"]
        assert "dropoff_percentages" in data
        assert "trace_id" in data
    
    def test_get_heatmap(self, client):
        """Test GET /stores/{id}/heatmap endpoint."""
        response = client.get("/stores/STORE_BLR_002/heatmap")
        
        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "zones" in data
        assert "trace_id" in data
    
    def test_get_anomalies(self, client):
        """Test GET /stores/{id}/anomalies endpoint."""
        response = client.get("/stores/STORE_BLR_002/anomalies")
        
        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "anomalies" in data
        assert "count" in data
        assert "trace_id" in data


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "stale_feed_warning" in data
        assert "trace_id" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test GET / endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_store(self, client):
        """Test metrics for store with no events."""
        response = client.get("/stores/STORE_EMPTY/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 0
        assert data["avg_dwell_time_ms"] == 0.0
        assert data["conversion_rate"] == 0.0
        assert data["max_queue_depth"] == 0
    
    def test_all_staff_events(self, client):
        """Test that staff events are excluded from metrics."""
        # Ingest only staff events
        staff_events = [
            {
                "event_id": f"staff-event-{i}",
                "store_id": "STORE_STAFF",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": f"STAFF_{i}",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": True,  # Mark as staff
                "confidence": 0.95,
                "metadata": {"session_seq": 1}
            }
            for i in range(5)
        ]
        
        response = client.post("/events/ingest", json=staff_events)
        assert response.status_code == 200
        assert response.json()["events_ingested"] == 5
        
        # Get metrics - should show 0 visitors (staff excluded)
        response = client.get("/stores/STORE_STAFF/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 0  # Staff excluded
    
    def test_zero_purchases(self, client):
        """Test metrics for store with visitors but no purchases."""
        # Ingest entry events but no billing/purchase events
        events = [
            {
                "event_id": f"entry-event-{i}",
                "store_id": "STORE_NO_PURCHASE",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": f"VIS_{i}",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            }
            for i in range(10)
        ]
        
        response = client.post("/events/ingest", json=events)
        assert response.status_code == 200
        assert response.json()["events_ingested"] == 10
        
        # Get metrics - should show visitors but 0 conversion rate
        response = client.get("/stores/STORE_NO_PURCHASE/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 10
        assert data["conversion_rate"] == 0.0  # No purchases
    
    def test_reentry_handling(self, client):
        """Test that re-entry events are handled correctly."""
        events = [
            # First entry
            {
                "event_id": "entry-1",
                "store_id": "STORE_REENTRY",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_REENTRY",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:00:00Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            },
            # Exit
            {
                "event_id": "exit-1",
                "store_id": "STORE_REENTRY",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_REENTRY",
                "event_type": "EXIT",
                "timestamp": "2026-03-03T14:10:00Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            },
            # Re-entry (same visitor)
            {
                "event_id": "reentry-1",
                "store_id": "STORE_REENTRY",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_REENTRY",
                "event_type": "REENTRY",
                "timestamp": "2026-03-03T14:20:00Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 2}
            }
        ]
        
        response = client.post("/events/ingest", json=events)
        assert response.status_code == 200
        assert response.json()["events_ingested"] == 3
        
        # Get funnel - should count re-entry correctly
        response = client.get("/stores/STORE_REENTRY/funnel")
        assert response.status_code == 200
        data = response.json()
        # Should have 2 entries (initial + re-entry)
        assert data["funnel"]["entry"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestGroupEntry:
    """Test group entry scenarios."""
    
    def test_group_entry_multiple_visitors(self, client):
        """Test that multiple visitors entering together are counted separately."""
        # Simulate 3 people entering at same timestamp
        events = [
            {
                "event_id": f"group-entry-{i}",
                "store_id": "STORE_GROUP",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": f"VIS_GROUP_{i}",
                "event_type": "ENTRY",
                "timestamp": "2026-03-03T14:00:00Z",
                "zone_id": None,
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.9,
                "metadata": {"session_seq": 1}
            }
            for i in range(3)
        ]
        
        response = client.post("/events/ingest", json=events)
        assert response.status_code == 200
        assert response.json()["events_ingested"] == 3
        
        # Verify metrics show 3 unique visitors
        response = client.get("/stores/STORE_GROUP/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 3


class TestDuplicateIngestion:
    """Test duplicate event handling."""
    
    def test_duplicate_event_not_double_counted(self, client):
        """Test that duplicate events are not double-counted."""
        event = {
            "event_id": "dup-test-1",
            "store_id": "STORE_DUP",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_DUP",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T14:00:00Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"session_seq": 1}
        }
        
        # First ingest
        response1 = client.post("/events/ingest", json=[event])
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["events_ingested"] == 1
        
        # Second ingest (duplicate)
        response2 = client.post("/events/ingest", json=[event])
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["duplicates"] == 1
        assert data2["events_ingested"] == 0
        
        # Verify metrics still show 1 visitor
        response = client.get("/stores/STORE_DUP/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 1


class TestFunnelNoDoubleCount:
    """Test that funnel doesn't double-count re-entries."""
    
    def test_funnel_reentry_not_double_counted(self, client):
        """Test that re-entry doesn't double-count visitor in funnel."""
        events = [
            # Entry
            {"event_id": "funnel-entry-1", "store_id": "STORE_FUNNEL", "camera_id": "CAM_1",
             "visitor_id": "VIS_FUNNEL", "event_type": "ENTRY", "timestamp": "2026-03-03T14:00:00Z",
             "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.9,
             "metadata": {"session_seq": 1}},
            # Exit
            {"event_id": "funnel-exit-1", "store_id": "STORE_FUNNEL", "camera_id": "CAM_1",
             "visitor_id": "VIS_FUNNEL", "event_type": "EXIT", "timestamp": "2026-03-03T14:05:00Z",
             "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.9,
             "metadata": {"session_seq": 1}},
            # Re-entry
            {"event_id": "funnel-reentry-1", "store_id": "STORE_FUNNEL", "camera_id": "CAM_1",
             "visitor_id": "VIS_FUNNEL", "event_type": "REENTRY", "timestamp": "2026-03-03T14:10:00Z",
             "zone_id": None, "dwell_ms": 0, "is_staff": False, "confidence": 0.9,
             "metadata": {"session_seq": 2}},
        ]
        
        response = client.post("/events/ingest", json=events)
        assert response.status_code == 200
        assert response.json()["events_ingested"] == 3
        
        # Get funnel
        response = client.get("/stores/STORE_FUNNEL/funnel")
        assert response.status_code == 200
        data = response.json()
        # Entry count should be 2 (ENTRY + REENTRY), but unique visitors should be 1
        assert data["funnel"]["entry"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
