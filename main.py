import json
import os

import requests
import werkzeug
import werkzeug.datastructures
from flask import Flask, jsonify, request
from slack import WebClient
from slack.signature import SignatureVerifier
from bot.predict import get_bot_response

app = Flask(__name__)

signature_verifier = SignatureVerifier(os.environ["SLACK_SIGNING_SECRET"])

slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])


def is_bot_event(event):
    return (
        event.get("bot_id")
        or event.get("message", {}).get("bot_id")
        or "subtype" in event
    )


# example for block response
def get_response(user_text):
    return [
        {
            "type": "image",
            "image_url": "https://media.giphy.com/media/mIZ9rPeMKefm0/source.gif",
            "alt_text": "inspiration",
        },
        {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"'{user_text}' back to you",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Press this button to create a reaction",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Click Me",
                    "emoji": True,
                },
                "value": "click_me_123",
                "action_id": "display-reaction",
            },
        },
    ]


@app.route("/slack_deploy_bot_interaction", methods=["POST"])
def slack_deploy_bot_interaction():
    data = json.loads(request.form["payload"])

    if data["token"] != os.environ["SLACK_VERIFICATION_TOKEN"]:
        return "", 200

    channel_id = data["channel"]["id"]
    response_url = data["response_url"]
    intent = data["actions"][0]["action_id"]

    print(f"Interaction, channel: {channel_id}, intent: {intent}")

    requests.post(
        url=response_url,
        json={"text": "That's how you respond to an interaction"},
    )
    return "", 200


@app.route("/slack_deploy_bot", methods=["POST"])
def slack_deploy_bot():
    is_valid_signature = signature_verifier.is_valid_request(
        request.data, request.headers
    )
    if not is_valid_signature:
        return ""

    body = json.loads(request.data)
    event = body.get("event", {})
    channel_id = event.get("channel")
    event.get("client_msg_id")
    text = event.get("text")

    if body["type"] == "url_verification":
        return jsonify({"challenge": body["challenge"]})

    if not is_bot_event(event) and "X-Slack-Retry-Num" not in request.headers:
        if body["type"] == "event_callback":
            slack_client.chat_postMessage(
                channel=channel_id, text=get_bot_response(text)["answer"]
            )

    return "", 200


def slack_bot_main(request):
    with app.app_context():
        headers = werkzeug.datastructures.Headers()
        for key, value in request.headers.items():
            headers.add(key, value)
        with app.test_request_context(
            method=request.method,
            base_url=request.base_url,
            path=request.path,
            query_string=request.query_string,
            headers=headers,
            data=request.data or request.form,
        ):
            try:
                rv = app.preprocess_request()
                if rv is None:
                    rv = app.dispatch_request()
            except Exception as e:
                rv = app.handle_user_exception(e)
            response = app.make_response(rv)
            return app.process_response(response)
