"""
Fixtures Scheduler - Background job to refresh fixtures from API.
Runs once daily to minimize API calls.
"""
import threading
import time
import schedule
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FixturesScheduler:
    """Background scheduler for fixtures refresh."""
    
    def __init__(self):
        self._running = False
        self._thread = None
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        logger.info("Fixtures scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Fixtures scheduler stopped")
    
    def _run_scheduler(self):
        """Run the schedule loop."""
        # Schedule refresh every 6 hours (06:00, 12:00, 18:00, 00:00)
        # 4 calls/day * 30 days = 120 calls/month (Safe limit for 500-call quota)
        schedule.every(6).hours.do(self._refresh_job)
        
        # Also run at startup if cache is empty or stale
        self._check_and_refresh()
        
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _refresh_job(self):
        """The scheduled refresh job."""
        logger.info("Running scheduled fixtures refresh...")
        try:
            # Step 1: Fetch fixtures from The Odds API
            from data.fetchers.fixtures_cache import refresh_fixtures_from_api
            result = refresh_fixtures_from_api()
            logger.info(f"Fixtures refresh complete: {result}")
            
            # Step 2: Fetch form data from Football-Data.org (after 5s delay)
            logger.info("Waiting 5s before fetching form data...")
            time.sleep(5)
            self._refresh_form_data()
            
        except Exception as e:
            logger.error(f"Scheduled refresh failed: {e}")
    
    def _refresh_form_data(self):
        """Fetch team form data from Football-Data.org."""
        try:
            from data.fetchers.form_fetcher import refresh_all_form_data
            result = refresh_all_form_data()
            logger.info(f"Form data refresh complete: {result}")
        except Exception as e:
            logger.error(f"Form data refresh failed: {e}")
    
    def _check_and_refresh(self):
        """Check if cache needs refresh on startup."""
        try:
            from data.fetchers.fixtures_cache import get_fixtures_cache, refresh_fixtures_from_api
            
            cache = get_fixtures_cache()
            last_fetch = cache.get_last_fetch_time()
            
            # Refresh if never fetched or older than 12 hours
            should_refresh = False
            if last_fetch is None:
                logger.info("No cached fixtures - will fetch now")
                should_refresh = True
            elif (datetime.now() - last_fetch).total_seconds() > 12 * 3600:
                logger.info("Cache older than 12 hours - will refresh")
                should_refresh = True
            else:
                logger.info(f"Cache is fresh (last fetch: {last_fetch})")
            
            if should_refresh:
                result = refresh_fixtures_from_api()
                logger.info(f"Initial refresh complete: {result}")
                
        except Exception as e:
            logger.error(f"Startup cache check failed: {e}")
    
    def manual_refresh(self) -> dict:
        """Manually trigger a refresh."""
        logger.info("Manual fixtures refresh triggered...")
        try:
            from data.fetchers.fixtures_cache import refresh_fixtures_from_api
            return refresh_fixtures_from_api()
        except Exception as e:
            logger.error(f"Manual refresh failed: {e}")
            return {'error': str(e)}


# Singleton
_scheduler = None

def get_scheduler() -> FixturesScheduler:
    """Get the scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = FixturesScheduler()
    return _scheduler


def start_scheduler():
    """Start the fixtures scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the fixtures scheduler."""
    scheduler = get_scheduler()
    scheduler.stop()
