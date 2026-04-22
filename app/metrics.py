"""
Metrics computation: derives analytics from stored events.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class MetricsService:
    """Computes analytics metrics from events."""
    
    def __init__(self, db):
        """
        Initialize metrics service.
        
        Args:
            db: EventDatabase instance
        """
        self.db = db
    
    def get_store_metrics(self, store_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get overall metrics for a store.
        
        Args:
            store_id: Store identifier
            hours: Time window in hours
        
        Returns:
            Dict with metrics
        """
        unique_visitors = self.db.get_unique_visitors(store_id, hours=hours)
        avg_dwell = self.db.get_avg_dwell_time(store_id, hours=hours)
        max_queue = self.db.get_max_queue_depth(store_id, hours=hours)
        
        # Get conversion rate from POS correlation
        from app.pos_correlation import POSCorrelationService
        pos_service = POSCorrelationService()
        
        # Get billing zone events for POS correlation
        billing_events = self.db.get_events(store_id, event_type="BILLING_QUEUE_JOIN", limit=10000)
        
        # Calculate conversion rate
        pos_metrics = pos_service.get_conversion_rate(store_id, unique_visitors, billing_events)
        conversion_rate = pos_metrics.get("conversion_rate", 0.0)
        
        return {
            "store_id": store_id,
            "time_window_hours": hours,
            "unique_visitors": unique_visitors,
            "avg_dwell_time_ms": round(avg_dwell, 2),
            "conversion_rate": round(conversion_rate, 4),
            "max_queue_depth": max_queue
        }
    
    def get_funnel(self, store_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get conversion funnel: ENTRY -> ZONE -> BILLING -> PURCHASE.
        
        Args:
            store_id: Store identifier
            hours: Time window in hours
        
        Returns:
            Dict with funnel stages and drop-off percentages
        """
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
        
        # Get event counts per stage
        events = self.db.get_events(store_id, limit=10000)
        
        # Filter by time window
        events = [e for e in events if e['timestamp'] > cutoff_time]
        
        # Count unique visitors per stage (using sets to avoid double-counting)
        entry_visitors = set()
        zone_visitors = set()
        billing_visitors = set()
        purchase_visitors = set()
        
        for event in events:
            visitor_id = event['visitor_id']
            event_type = event['event_type']
            
            # ENTRY and REENTRY both count as entry stage
            if event_type in ['ENTRY', 'REENTRY']:
                entry_visitors.add(visitor_id)
            elif event_type == 'ZONE_ENTER':
                zone_visitors.add(visitor_id)
            elif event_type == 'BILLING_QUEUE_JOIN':
                billing_visitors.add(visitor_id)
            elif event_type == 'ZONE_EXIT' and event.get('zone_id') == 'BILLING':
                # Simplified: assume zone exit from billing = purchase
                purchase_visitors.add(visitor_id)
        
        entry_count = len(entry_visitors)
        zone_count = len(zone_visitors)
        billing_count = len(billing_visitors)
        purchase_count = len(purchase_visitors)
        
        # Calculate drop-off percentages
        entry_to_zone_dropoff = 0.0
        zone_to_billing_dropoff = 0.0
        billing_to_purchase_dropoff = 0.0
        
        if entry_count > 0:
            entry_to_zone_dropoff = ((entry_count - zone_count) / entry_count) * 100
        if zone_count > 0:
            zone_to_billing_dropoff = ((zone_count - billing_count) / zone_count) * 100
        if billing_count > 0:
            billing_to_purchase_dropoff = ((billing_count - purchase_count) / billing_count) * 100
        
        return {
            "store_id": store_id,
            "time_window_hours": hours,
            "funnel": {
                "entry": entry_count,
                "zone_visit": zone_count,
                "billing_queue": billing_count,
                "purchase": purchase_count
            },
            "dropoff_percentages": {
                "entry_to_zone": round(entry_to_zone_dropoff, 2),
                "zone_to_billing": round(zone_to_billing_dropoff, 2),
                "billing_to_purchase": round(billing_to_purchase_dropoff, 2)
            }
        }
    
    def get_heatmap(self, store_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get zone visit frequency heatmap.
        
        Args:
            store_id: Store identifier
            hours: Time window in hours
        
        Returns:
            Dict with zone visit frequencies normalized 0-100
        """
        zone_visits = self.db.get_zone_visits(store_id, hours=hours)
        
        if not zone_visits:
            return {
                "store_id": store_id,
                "time_window_hours": hours,
                "zones": {}
            }
        
        # Normalize to 0-100
        max_visits = max(zone_visits.values()) if zone_visits else 1
        normalized = {
            zone_id: round((count / max_visits) * 100, 2)
            for zone_id, count in zone_visits.items()
        }
        
        return {
            "store_id": store_id,
            "time_window_hours": hours,
            "zones": normalized
        }
    
    def get_anomalies(self, store_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Detect anomalies: queue spike, dead zone, conversion drop.
        
        Args:
            store_id: Store identifier
            hours: Time window in hours
        
        Returns:
            Dict with detected anomalies
        """
        anomalies = []
        
        # Queue spike: max queue depth > 5
        max_queue = self.db.get_max_queue_depth(store_id, hours=hours)
        if max_queue > 5:
            anomalies.append({
                "type": "QUEUE_SPIKE",
                "severity": "WARN",
                "message": f"Queue depth reached {max_queue}",
                "value": max_queue,
                "suggested_action": "Open additional checkout lanes or call for backup staff"
            })
        
        # Dead zone: no events in last 30 minutes
        last_event = self.db.get_last_event_timestamp(store_id)
        if last_event:
            try:
                last_event_dt = datetime.fromisoformat(last_event.replace('Z', '+00:00'))
                now = datetime.utcnow().replace(tzinfo=None)
                last_event_dt = last_event_dt.replace(tzinfo=None)
                
                if (now - last_event_dt).total_seconds() > 1800:  # 30 minutes
                    anomalies.append({
                        "type": "DEAD_ZONE",
                        "severity": "WARN",
                        "message": "No events in last 30 minutes",
                        "last_event_timestamp": last_event,
                        "suggested_action": "Check camera feed and detector status"
                    })
            except:
                pass
        
        # Conversion drop: compare to 7-day average
        current_visitors = self.db.get_unique_visitors(store_id, hours=24)
        week_ago_visitors = self.db.get_unique_visitors(store_id, hours=24*7)
        
        if week_ago_visitors > 0:
            avg_daily = week_ago_visitors / 7
            if current_visitors < (avg_daily * 0.8):  # 20% drop
                anomalies.append({
                    "type": "CONVERSION_DROP",
                    "severity": "CRITICAL",
                    "message": f"Visitor count down {round((1 - current_visitors/avg_daily)*100, 1)}% vs 7-day avg",
                    "current": current_visitors,
                    "expected": round(avg_daily, 0),
                    "suggested_action": "Review store operations, staffing, and promotions"
                })
        
        return {
            "store_id": store_id,
            "time_window_hours": hours,
            "anomalies": anomalies,
            "count": len(anomalies)
        }
