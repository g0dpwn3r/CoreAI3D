"""
test_beta_email.py - Tests for the beta key email delivery path.

The Resend transport is exercised against a local HTTP server that mimics the
Resend REST API, so the tests never touch the network.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PyQt6.QtWidgets import QApplication


class _ResendHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the Resend ``/emails`` endpoint."""

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode('utf-8'))
        except ValueError:
            payload = {'raw': body.decode('utf-8', 'replace')}

        self.server.requests.append({
            'path': self.path,
            'authorization': self.headers.get('Authorization'),
            'content_type': self.headers.get('Content-Type'),
            'user_agent': self.headers.get('User-Agent'),
            'payload': payload,
        })

        if self.path.startswith('/fail'):
            response = json.dumps({'message': 'domain not verified'}).encode()
            self.send_response(422)
        else:
            response = json.dumps({'id': 'email-123'}).encode()
            self.send_response(200)

        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        """Keep the test output clean."""


@pytest.fixture
def resend_server():
    """Run a local HTTP server that records the Resend requests."""
    server = HTTPServer(('127.0.0.1', 0), _ResendHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class TestBetaEmailDelivery:
    """Tests for sending a beta key by email."""

    def test_unconfigured_delivery_reports_error(self, dashboard, monkeypatch):
        """Without any provider the caller gets a clear error."""
        monkeypatch.delenv('RESEND_API_KEY', raising=False)
        monkeypatch.delenv('SMTP_HOST', raising=False)
        dashboard.config.resend_api_key = ''
        dashboard.config.resend_from_email = ''

        sent, message = dashboard._send_beta_key_email('user@example.com', 'key-1')

        assert sent is False
        assert 'not configured' in message.lower()

    def test_resend_delivery_sends_key_and_instructions(self, dashboard, resend_server):
        """The key and the usage instructions reach the Resend API."""
        dashboard.config.resend_api_key = 're_test_key'
        dashboard.config.resend_from_email = 'noreply@coreai3d.local'
        dashboard.RESEND_API_URL = (
            f'http://127.0.0.1:{resend_server.server_address[1]}/emails'
        )

        sent, message = dashboard._send_beta_key_email(
            'user@example.com', 'secret-key-1', 'demo key')

        assert sent is True
        assert 'user@example.com' in message
        assert len(resend_server.requests) == 1

        request = resend_server.requests[0]
        assert request['authorization'] == 'Bearer re_test_key'
        assert request['content_type'] == 'application/json'
        # Resend requires a User-Agent header; without it the API returns
        # 403 error code 1010 before the request ever reaches the API.
        assert request['user_agent'] == 'CoreAI3D-LinuxGUI/1.3.0'
        assert request['payload']['to'] == ['user@example.com']
        assert request['payload']['from'] == 'noreply@coreai3d.local'
        assert 'secret-key-1' in request['payload']['text']
        assert 'coreai3d.nl/register' in request['payload']['text']
        # The email must contain only the beta key, not API key material.
        assert 'COREAI3D_API_KEY' not in request['payload']['text']
        assert 'COREAI3D_API_URL' not in request['payload']['text']
        # The HTML part should also carry the beta key.
        assert 'secret-key-1' in request['payload']['html']

    def test_resend_error_is_reported(self, dashboard, resend_server):
        """A rejected request surfaces the provider error."""
        dashboard.config.resend_api_key = 're_test_key'
        dashboard.RESEND_API_URL = (
            f'http://127.0.0.1:{resend_server.server_address[1]}/fail'
        )

        sent, message = dashboard._send_beta_key_email('user@example.com', 'key-1')

        assert sent is False
        assert '422' in message
        assert 'domain not verified' in message

    def test_smtp_used_as_fallback(self, dashboard, resend_server, monkeypatch):
        """When Resend fails and SMTP is configured, SMTP takes over."""
        monkeypatch.setenv('SMTP_HOST', '127.0.0.1')
        monkeypatch.setenv('SMTP_PORT', '2525')
        dashboard.config.resend_api_key = 're_test_key'
        dashboard.RESEND_API_URL = (
            f'http://127.0.0.1:{resend_server.server_address[1]}/fail'
        )

        smtp_calls = []

        def fake_smtp(recipient, key, description=''):
            smtp_calls.append((recipient, key))
            return True, f'Beta key sent to {recipient}.'

        dashboard._send_beta_key_email_smtp = fake_smtp

        sent, message = dashboard._send_beta_key_email('user@example.com', 'key-1')

        assert sent is True
        assert smtp_calls == [('user@example.com', 'key-1')]

    def test_gui_send_runs_on_worker_thread(self, dashboard, resend_server):
        """The GUI send button delivers the key and reports back on the GUI thread."""
        dashboard.config.resend_api_key = 're_test_key'
        dashboard.config.resend_from_email = 'noreply@coreai3d.local'
        dashboard.RESEND_API_URL = (
            f'http://127.0.0.1:{resend_server.server_address[1]}/emails'
        )
        dashboard._selected_beta_key = lambda: (0, 'gui-secret-key', 7)

        notifications = []
        dashboard._notify = lambda title, msg, level='info': notifications.append(
            (title, msg, level))

        dashboard.beta_email_input.setText('gui@example.com')
        dashboard.on_send_beta_key_clicked()

        deadline = time.time() + 15
        while time.time() < deadline and not notifications:
            QApplication.instance().processEvents()
            time.sleep(0.05)

        assert notifications, "the email worker never reported back"
        title, message, level = notifications[0]
        assert title == 'Beta Key Emailed'
        assert 'gui@example.com' in message

        assert len(resend_server.requests) == 1
        payload = resend_server.requests[0]['payload']
        assert payload['to'] == ['gui@example.com']
        assert 'gui-secret-key' in payload['text']
