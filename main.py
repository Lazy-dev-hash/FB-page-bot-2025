
import os
import json
import requests
from flask import Flask, request, jsonify
import openai
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - you'll need to set these in Secrets
VERIFY_TOKEN = os.getenv('FACEBOOK_VERIFY_TOKEN', 'your_verify_token_here')
PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class FacebookBot:
    def __init__(self):
        self.api_version = 'v18.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
        
    def send_message(self, recipient_id, message_text):
        """Send a message to a Facebook user"""
        url = f'{self.base_url}/me/messages'
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'recipient': {'id': recipient_id},
            'message': {'text': message_text},
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Message sent successfully to {recipient_id}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def send_typing_indicator(self, recipient_id):
        """Send typing indicator to show bot is processing"""
        url = f'{self.base_url}/me/messages'
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'recipient': {'id': recipient_id},
            'sender_action': 'typing_on',
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        try:
            requests.post(url, headers=headers, json=data)
        except:
            pass  # Non-critical if this fails
    
    def get_ai_response(self, user_message, user_name="User"):
        """Get AI response using OpenAI"""
        if not OPENAI_API_KEY:
            return "I'm sorry, but I'm not configured with AI capabilities at the moment. Please contact the page administrator."
        
        try:
            # Create a conversational prompt
            prompt = f"""You are a helpful and friendly Facebook page bot assistant. 
            You should provide helpful, accurate, and engaging responses to user questions.
            Keep responses concise but informative, suitable for Facebook messaging.
            
            User {user_name} asked: {user_message}
            
            Please provide a helpful response:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Facebook page bot that provides accurate and friendly responses to user questions."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I'm having trouble processing your request right now. Please try again later or contact our support team."
    
    def handle_message(self, sender_id, message_text, sender_name=None):
        """Process incoming message and generate response"""
        logger.info(f"Processing message from {sender_id}: {message_text}")
        
        # Send typing indicator
        self.send_typing_indicator(sender_id)
        
        # Handle special commands
        if message_text.lower() in ['hi', 'hello', 'hey']:
            response = f"Hello{' ' + sender_name if sender_name else ''}! 👋 I'm here to help answer any questions you have. What would you like to know?"
        elif message_text.lower() in ['help', '/help']:
            response = """I'm an AI-powered bot that can help answer your questions! 🤖

Here's what I can do:
• Answer general questions
• Provide information and explanations
• Help with various topics
• Have conversations

Just type your question and I'll do my best to help!

Type 'about' to learn more about this page."""
        elif message_text.lower() == 'about':
            response = """🤖 About This Bot:

I'm an AI-powered assistant for this Facebook page. I use advanced language models to provide helpful and accurate responses to your questions.

Features:
✅ 24/7 availability
✅ Instant responses
✅ Wide knowledge base
✅ Friendly conversation

Feel free to ask me anything!"""
        else:
            # Get AI response for regular messages
            response = self.get_ai_response(message_text, sender_name)
        
        # Send the response
        self.send_message(sender_id, response)

# Initialize bot
bot = FacebookBot()

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Verify webhook for Facebook"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return challenge
    else:
        logger.warning("Webhook verification failed")
        return 'Verification failed', 403

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming Facebook messages"""
    try:
        data = request.get_json()
        
        if data.get('object') == 'page':
            for entry in data.get('entry', []):
                for messaging_event in entry.get('messaging', []):
                    sender_id = messaging_event.get('sender', {}).get('id')
                    
                    # Handle regular messages
                    if messaging_event.get('message'):
                        message_text = messaging_event['message'].get('text', '')
                        
                        # Skip if no text (could be attachment, etc.)
                        if not message_text:
                            continue
                        
                        # Get sender information
                        sender_name = get_user_name(sender_id)
                        
                        # Process the message
                        bot.handle_message(sender_id, message_text, sender_name)
                    
                    # Handle postbacks (button clicks)
                    elif messaging_event.get('postback'):
                        payload = messaging_event['postback'].get('payload')
                        if payload == 'GET_STARTED':
                            welcome_message = """Welcome! 🎉 

I'm an AI-powered bot ready to help answer your questions and assist you with various topics.

Type 'help' to see what I can do, or just ask me anything!"""
                            bot.send_message(sender_id, welcome_message)
        
        return 'OK', 200
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return 'Error', 500

def get_user_name(user_id):
    """Get user's name from Facebook Graph API"""
    try:
        url = f'https://graph.facebook.com/{user_id}'
        params = {
            'fields': 'first_name',
            'access_token': PAGE_ACCESS_TOKEN
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('first_name', '')
    except:
        pass
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_configured': bool(PAGE_ACCESS_TOKEN and VERIFY_TOKEN),
        'ai_configured': bool(OPENAI_API_KEY)
    })

@app.route('/', methods=['GET'])
def home():
    """Home page with bot information"""
    return """
    <html>
    <head><title>Facebook AI Bot</title></head>
    <body>
        <h1>🤖 Facebook AI Page Bot</h1>
        <p>This is an AI-powered Facebook page bot that can answer user questions.</p>
        
        <h2>Status:</h2>
        <ul>
            <li>Bot configured: {}</li>
            <li>AI configured: {}</li>
        </ul>
        
        <h2>Features:</h2>
        <ul>
            <li>✅ 24/7 automated responses</li>
            <li>✅ AI-powered question answering</li>
            <li>✅ Natural conversation</li>
            <li>✅ Special commands (help, about)</li>
            <li>✅ Typing indicators</li>
            <li>✅ User name recognition</li>
        </ul>
        
        <p>Webhook URL: <code>{}/webhook</code></p>
    </body>
    </html>
    """.format(
        "✅" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "❌",
        "✅" if OPENAI_API_KEY else "❌",
        request.url_root.rstrip('/')
    )

if __name__ == '__main__':
    logger.info("Starting Facebook AI Bot...")
    
    # Check configuration
    if not PAGE_ACCESS_TOKEN:
        logger.warning("FACEBOOK_PAGE_ACCESS_TOKEN not set")
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set - AI features will be limited")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False)
