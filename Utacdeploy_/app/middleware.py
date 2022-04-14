from flask import request
from prometheus_client import Counter, Histogram, Gauge
import time

REQUEST_COUNT = Counter(
    'request_count', 'Prediction Request Count',
    ['app_name', 'method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram('request_latency', 'Request latency',
    ['app_name', 'endpoint']
)
REQUEST_LATENCY_SECONDS = Gauge('request_latency_seconds', 'Request latency seconds',
    ['app_name', 'endpoint']
)

def start_timer():
    request.start_time = time.time()
    global start_time
    start_time = time.time()

def stop_timer(response):
    resp_time = time.time() - request.start_time
    resp_time_seconds = time.time() - start_time
    REQUEST_LATENCY.labels('UTAC', request.path).observe(resp_time)
    REQUEST_LATENCY_SECONDS.labels('UTAC', request.path).set(resp_time_seconds)
    return response

def record_request_data(response):
    REQUEST_COUNT.labels('UTAC', request.method, request.path,
            response.status_code).inc()
    return response

def setup_metrics(app):
    app.before_request(start_timer)
    # The order here matters since we want stop_timer
    # to be executed first
    app.after_request(record_request_data)
    app.after_request(stop_timer)