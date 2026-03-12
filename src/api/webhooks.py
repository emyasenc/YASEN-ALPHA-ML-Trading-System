"""
Simple webhook storage for YASEN-ALPHA API
"""
import json
import time
import os
from typing import Dict, List
import threading
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebhookManager:
    def __init__(self, storage_file='data/webhooks.json'):
        self.storage_file = storage_file
        self.webhooks = {}  # user_id -> list of webhooks
        self.lock = threading.Lock()
        self.load()
    
    def load(self):
        """Load webhooks from file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.webhooks = json.load(f)
                logger.info(f"✅ Loaded {sum(len(v) for v in self.webhooks.values())} webhooks")
        except Exception as e:
            logger.error(f"Error loading webhooks: {e}")
            self.webhooks = {}
    
    def save(self):
        """Save webhooks to file"""
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            with open(self.storage_file, 'w') as f:
                json.dump(self.webhooks, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving webhooks: {e}")
    
    def register(self, user_id: str, url: str, events: List[str], secret: str = None):
        """Register a new webhook"""
        with self.lock:
            if user_id not in self.webhooks:
                self.webhooks[user_id] = []
            
            webhook = {
                "id": f"wh_{int(time.time())}_{len(self.webhooks[user_id])}",
                "url": url,
                "events": events,  # ["signal_change", "price_alert", "level_break"]
                "secret": secret,
                "created_at": datetime.now().isoformat(),
                "last_triggered": None,
                "active": True
            }
            self.webhooks[user_id].append(webhook)
            self.save()
            return webhook
    
    def unregister(self, user_id: str, webhook_id: str):
        """Remove a webhook"""
        with self.lock:
            if user_id in self.webhooks:
                self.webhooks[user_id] = [w for w in self.webhooks[user_id] if w['id'] != webhook_id]
                self.save()
    
    def get_user_webhooks(self, user_id: str) -> List[Dict]:
        """Get all webhooks for a user"""
        return self.webhooks.get(user_id, [])
    
    def trigger_event(self, event_type: str, data: Dict):
        """Trigger webhooks for an event"""
        triggered = 0
        for user_id, webhooks in self.webhooks.items():
            for webhook in webhooks:
                if webhook['active'] and event_type in webhook['events']:
                    try:
                        # Send webhook in background thread
                        threading.Thread(
                            target=self._send_webhook,
                            args=(webhook, event_type, data),
                            daemon=True
                        ).start()
                        triggered += 1
                    except Exception as e:
                        logger.error(f"Error triggering webhook {webhook['id']}: {e}")
        
        if triggered > 0:
            logger.info(f"🔥 Triggered {triggered} webhooks for {event_type}")
        return triggered
    
    def _send_webhook(self, webhook: Dict, event_type: str, data: Dict):
        """Send webhook to URL"""
        try:
            payload = {
                "event": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            headers = {"Content-Type": "application/json"}
            if webhook.get('secret'):
                headers["X-Webhook-Secret"] = webhook['secret']
            
            response = requests.post(
                webhook['url'],
                json=payload,
                headers=headers,
                timeout=5
            )
            
            if response.status_code < 300:
                logger.info(f"✅ Webhook {webhook['id']} sent successfully")
                # Update last triggered
                with self.lock:
                    for w in self.webhooks.get(self._find_user(webhook['id']), []):
                        if w['id'] == webhook['id']:
                            w['last_triggered'] = datetime.now().isoformat()
                            self.save()
            else:
                logger.warning(f"⚠️ Webhook {webhook['id']} failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ Webhook {webhook['id']} error: {e}")
    
    def _find_user(self, webhook_id: str) -> str:
        """Find user ID for a webhook"""
        for user_id, webhooks in self.webhooks.items():
            if any(w['id'] == webhook_id for w in webhooks):
                return user_id
        return None

# Global webhook manager
webhook_manager = WebhookManager()