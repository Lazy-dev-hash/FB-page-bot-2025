
import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import openai
from datetime import datetime
import logging
import time
import random
import threading
import sqlite3
from contextlib import contextmanager
import hashlib
import uuid
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO
from PIL import Image

# Import Gemini with better error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini successfully imported")
except ImportError as e:
    print(f"⚠️ Gemini import failed: {e}")
    GEMINI_AVAILABLE = False
    genai = None
except Exception as e:
    print(f"⚠️ Gemini initialization error: {e}")
    GEMINI_AVAILABLE = False
    genai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
VERIFY_TOKEN = os.getenv('FACEBOOK_VERIFY_TOKEN', 'your_verify_token_here')
PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Bot Configuration
BOT_NAME = "SUNNEL's Ultimate AI"
BOT_VERSION = "3.1.0"
REQUIRED_POST_ID = "761320392916522"
PAGE_ID = "100071491013161"

# Initialize APIs
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini configured successfully")
    except Exception as e:
        print(f"⚠️ Gemini configuration failed: {e}")
        GEMINI_AVAILABLE = False

if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
        print("✅ OpenAI configured successfully")
    except Exception as e:
        print(f"⚠️ OpenAI configuration failed: {e}")

# Database setup
@contextmanager
def get_db():
    conn = sqlite3.connect('bot_analytics.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_database():
    """Initialize database with required tables"""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                profile_pic TEXT,
                locale TEXT,
                timezone INTEGER,
                gender TEXT,
                verification_status TEXT DEFAULT 'unverified',
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                total_images INTEGER DEFAULT 0,
                user_rating REAL DEFAULT 0.0,
                conversation_memory TEXT DEFAULT '',
                preference_settings TEXT DEFAULT '{}'
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                interaction_type TEXT,
                user_message TEXT,
                bot_response TEXT,
                ai_provider TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                error_message TEXT,
                image_url TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feature_name TEXT,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS bot_stats (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total_users INTEGER DEFAULT 0,
                verified_users INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                total_images INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 100.0,
                avg_response_time REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Initialize bot stats if not exists
        conn.execute('''
            INSERT OR IGNORE INTO bot_stats (id) VALUES (1)
        ''')

class SunnelBot:
    def __init__(self):
        self.page_access_token = PAGE_ACCESS_TOKEN
        self.verify_token = VERIFY_TOKEN
        self.conversation_memory = {}
        self.start_time = datetime.now()
        
    def verify_webhook(self, token, challenge):
        """Verify webhook with Facebook"""
        if token == self.verify_token:
            return challenge
        return None
    
    def send_message(self, sender_id: str, message: str) -> bool:
        """Send message to user"""
        try:
            url = f"https://graph.facebook.com/v18.0/me/messages"
            headers = {'Content-Type': 'application/json'}
            
            data = {
                'recipient': {'id': sender_id},
                'message': {'text': message},
                'access_token': self.page_access_token
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                logger.info(f"✅ Message sent successfully to {sender_id}")
                return True
            else:
                logger.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return False
    
    def send_typing_indicator(self, sender_id: str):
        """Send typing indicator"""
        try:
            url = f"https://graph.facebook.com/v18.0/me/messages"
            headers = {'Content-Type': 'application/json'}
            
            data = {
                'recipient': {'id': sender_id},
                'sender_action': 'typing_on',
                'access_token': self.page_access_token
            }
            
            requests.post(url, headers=headers, json=data)
        except Exception as e:
            logger.error(f"Typing indicator error: {e}")
    
    def get_user_info(self, sender_id: str) -> Dict[str, Any]:
        """Get user information from Facebook"""
        try:
            url = f"https://graph.facebook.com/v18.0/{sender_id}"
            params = {
                'fields': 'first_name,last_name,profile_pic,locale,timezone,gender',
                'access_token': self.page_access_token
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get user info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Get user info error: {e}")
            return {}
    
    def update_user_database(self, sender_id: str, **kwargs):
        """Update user information in database"""
        try:
            with get_db() as conn:
                # Check if user exists
                user = conn.execute('SELECT user_id FROM users WHERE user_id = ?', (sender_id,)).fetchone()
                
                if not user:
                    # Get user info from Facebook
                    user_info = self.get_user_info(sender_id)
                    
                    conn.execute('''
                        INSERT INTO users (user_id, first_name, last_name, profile_pic, locale, timezone, gender)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        sender_id,
                        user_info.get('first_name', ''),
                        user_info.get('last_name', ''),
                        user_info.get('profile_pic', ''),
                        user_info.get('locale', ''),
                        user_info.get('timezone', 0),
                        user_info.get('gender', '')
                    ))
                
                # Update user data
                if kwargs:
                    set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
                    values = list(kwargs.values()) + [sender_id]
                    conn.execute(f'UPDATE users SET {set_clause}, last_interaction = CURRENT_TIMESTAMP WHERE user_id = ?', values)
                
        except Exception as e:
            logger.error(f"Database update error: {e}")
    
    def update_conversation_memory(self, sender_id: str, user_message: str, bot_response: str):
        """Update conversation memory"""
        try:
            if sender_id not in self.conversation_memory:
                self.conversation_memory[sender_id] = []
            
            self.conversation_memory[sender_id].append({
                'user': user_message,
                'bot': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 conversations
            if len(self.conversation_memory[sender_id]) > 10:
                self.conversation_memory[sender_id] = self.conversation_memory[sender_id][-10:]
            
            # Update database
            memory_json = json.dumps(self.conversation_memory[sender_id])
            with get_db() as conn:
                conn.execute('UPDATE users SET conversation_memory = ? WHERE user_id = ?', (memory_json, sender_id))
                
        except Exception as e:
            logger.error(f"Memory update error: {e}")
    
    def log_interaction(self, sender_id: str, interaction_type: str, user_message: str, 
                       bot_response: str, ai_provider: str, processing_time: float, 
                       error_message: str = None, image_url: str = None):
        """Log interaction to database"""
        try:
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO interactions (user_id, interaction_type, user_message, bot_response, 
                                            ai_provider, processing_time, error_message, image_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (sender_id, interaction_type, user_message, bot_response, ai_provider, processing_time, error_message, image_url))
        except Exception as e:
            logger.error(f"Interaction logging error: {e}")
    
    def track_feature_usage(self, sender_id: str, feature_name: str):
        """Track feature usage"""
        try:
            with get_db() as conn:
                # Check if feature usage exists
                existing = conn.execute(
                    'SELECT usage_count FROM feature_usage WHERE user_id = ? AND feature_name = ?',
                    (sender_id, feature_name)
                ).fetchone()
                
                if existing:
                    conn.execute('''
                        UPDATE feature_usage SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND feature_name = ?
                    ''', (sender_id, feature_name))
                else:
                    conn.execute('''
                        INSERT INTO feature_usage (user_id, feature_name, usage_count)
                        VALUES (?, ?, 1)
                    ''', (sender_id, feature_name))
        except Exception as e:
            logger.error(f"Feature tracking error: {e}")
    
    def analyze_image_with_ai(self, image_url: str, user_message: str = "", user_name: str = "User") -> tuple[str, str]:
        """Analyze image using available AI providers"""
        try:
            # Try Gemini first (supports image analysis)
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                try:
                    # Download image
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""You are {BOT_NAME}, an advanced AI assistant created by SUNNEL. 
Analyze this image thoroughly and provide a detailed, helpful response.

User's name: {user_name}
User's message: {user_message if user_message else "Please analyze this image"}

Please provide:
1. A detailed description of what you see
2. Any relevant information or insights
3. Answer any specific questions the user may have

Be engaging, informative, and use emojis appropriately."""
                        
                        response = model.generate_content([prompt, image])
                        
                        if response and response.text:
                            return response.text.strip(), "Gemini Vision"
                
                except Exception as e:
                    logger.error(f"Gemini Vision error: {e}")
            
            # Try OpenAI Vision API
            if OPENAI_API_KEY:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are {BOT_NAME}, a helpful AI assistant created by SUNNEL. Analyze images thoroughly and provide detailed, engaging responses with appropriate emojis."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Hello {user_name}! {user_message if user_message else 'Please analyze this image in detail.'}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url}
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip(), "OpenAI Vision"
                
                except Exception as e:
                    logger.error(f"OpenAI Vision error: {e}")
            
            # Fallback response
            return f"📸 Hi {user_name}! I can see you've shared an image with me. While I'm experiencing some technical difficulties with image analysis right now, I'm here to help! Could you describe what's in the image or let me know what you'd like to know about it? 😊", "Fallback"
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"📸 Hello {user_name}! I received your image but I'm having trouble analyzing it right now. Please try again or describe what you'd like to know about the image! 🤖", "Error"
    
    def get_smart_ai_response(self, user_message: str, user_name: str = "User", 
                            conversation_history: str = None) -> tuple[str, Optional[str]]:
        """Get AI response with improved provider management"""
        start_time = time.time()
        
        try:
            # Try Gemini first
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Create enhanced prompt
                    system_prompt = f"""You are {BOT_NAME}, a highly intelligent and friendly AI assistant created by SUNNEL. 

Key traits:
- Helpful, knowledgeable, and conversational
- Use emojis appropriately to make conversations engaging
- Provide detailed, accurate, and useful responses
- Be creative and show personality
- Always be respectful and supportive

User's name: {user_name}
"""
                    
                    if conversation_history:
                        full_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_history}\n\nCurrent message: {user_message}\n\nPlease respond helpfully:"
                    else:
                        full_prompt = f"{system_prompt}\n\nUser message: {user_message}\n\nPlease respond helpfully:"
                    
                    response = model.generate_content(full_prompt)
                    
                    if response and response.text:
                        processing_time = time.time() - start_time
                        return response.text.strip(), "Gemini"
                    
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
            
            # Fallback to OpenAI
            if OPENAI_API_KEY:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    
                    messages = [
                        {
                            "role": "system",
                            "content": f"You are {BOT_NAME}, a helpful AI assistant created by SUNNEL. Be conversational, helpful, and use emojis appropriately."
                        }
                    ]
                    
                    if conversation_history:
                        messages.append({"role": "assistant", "content": f"Previous conversation context: {conversation_history}"})
                    
                    messages.append({"role": "user", "content": user_message})
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        processing_time = time.time() - start_time
                        return response.choices[0].message.content.strip(), "OpenAI"
                    
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
            
            # Final fallback response
            processing_time = time.time() - start_time
            fallback_responses = [
                f"Hello {user_name}! 👋 I'm {BOT_NAME}, your AI assistant. How can I help you today?",
                f"Hi there! ✨ I'm here to help you with any questions or tasks you have!",
                f"Greetings! 🌟 I'm {BOT_NAME}, ready to assist you. What would you like to know?",
                f"Hello! 😊 I'm your friendly AI assistant. Feel free to ask me anything!"
            ]
            
            return random.choice(fallback_responses), "Fallback"
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            processing_time = time.time() - start_time
            return f"Hello! I'm {BOT_NAME} 🤖 I'm experiencing some technical difficulties right now, but I'm here to help! Please try again in a moment.", "Error"
    
    def handle_message(self, sender_id: str, message_text: str):
        """Handle incoming text messages"""
        try:
            start_time = time.time()
            logger.info(f"📥 Processing message from {sender_id}: {message_text[:50]}...")
            
            # Update user database
            self.update_user_database(sender_id, verification_status="verified")
            
            # Track message count
            with get_db() as conn:
                conn.execute(
                    'UPDATE users SET total_messages = total_messages + 1 WHERE user_id = ?',
                    (sender_id,)
                )
            
            # Track feature usage
            self.track_feature_usage(sender_id, "text_message")
            
            # Send typing indicator
            self.send_typing_indicator(sender_id)
            
            # Get user info
            user_info = self.get_user_info(sender_id)
            user_name = user_info.get('first_name', 'Friend')
            
            # Get conversation history
            conversation_history = None
            if sender_id in self.conversation_memory:
                recent_conversations = self.conversation_memory[sender_id][-3:]  # Last 3 conversations
                conversation_history = "\n".join([
                    f"User: {conv['user']}\nBot: {conv['bot']}" 
                    for conv in recent_conversations
                ])
            
            # Get AI response
            response, provider = self.get_smart_ai_response(message_text, user_name, conversation_history)
            
            # Update conversation memory
            self.update_conversation_memory(sender_id, message_text, response)
            
            # Send response
            success = self.send_message(sender_id, response)
            
            # Log interaction
            processing_time = time.time() - start_time
            self.log_interaction(
                sender_id, "text", message_text, response, provider, 
                processing_time, None if success else "Send failed"
            )
            
            logger.info(f"✅ Message processed in {processing_time:.2f}s using {provider}")
            
        except Exception as e:
            error_msg = f"Message handling error: {e}"
            logger.error(error_msg)
            
            # Send error response
            self.send_message(
                sender_id,
                "🤖 I encountered a technical issue. Please try again! I'm here to help you. 💙"
            )
    
    def handle_image_message(self, sender_id: str, image_url: str, message_text: str = "", user_name: str = "User"):
        """Handle incoming image messages with AI analysis"""
        try:
            start_time = time.time()
            logger.info(f"📸 Processing image from {sender_id}: {image_url}")
            
            # Update user database
            self.update_user_database(sender_id, verification_status="verified")
            
            # Track image count
            with get_db() as conn:
                conn.execute(
                    'UPDATE users SET total_images = total_images + 1 WHERE user_id = ?',
                    (sender_id,)
                )
            
            # Track feature usage
            self.track_feature_usage(sender_id, "image_analysis")
            
            # Send typing indicator
            self.send_typing_indicator(sender_id)
            
            # Analyze image with AI
            response, provider = self.analyze_image_with_ai(image_url, message_text, user_name)
            
            # Update conversation memory
            image_context = f"[Image shared] {message_text}" if message_text else "[Image shared]"
            self.update_conversation_memory(sender_id, image_context, response)
            
            # Send response
            success = self.send_message(sender_id, response)
            
            # Log interaction
            processing_time = time.time() - start_time
            self.log_interaction(
                sender_id, "image", message_text, response, provider, 
                processing_time, None if success else "Send failed", image_url
            )
            
            logger.info(f"✅ Image processed in {processing_time:.2f}s using {provider}")
            
        except Exception as e:
            error_msg = f"Image handling error: {e}"
            logger.error(error_msg)
            
            # Send error response
            self.send_message(
                sender_id,
                "📸 I received your image but I'm having trouble analyzing it right now. Please try again or describe what you'd like to know about the image! 🤖"
            )
    
    def handle_postback(self, sender_id: str, payload: str):
        """Handle postback events"""
        try:
            logger.info(f"📬 Postback from {sender_id}: {payload}")
            
            if payload == "GET_STARTED":
                welcome_message = f"""🌟 **Welcome to {BOT_NAME}!** 🌟

Hello! I'm your intelligent AI assistant, powered by advanced AI technology and created with ❤️ by SUNNEL.

🚀 **What I can do:**
• 💬 Have natural conversations about any topic
• 📸 Analyze and understand images you share
• 🎓 Help with learning and education
• 💡 Provide creative ideas and solutions
• 🔍 Answer questions and explain concepts
• 🎯 Assist with planning and problem-solving
• 🌍 Discuss current events and general knowledge

✨ **Just send me any message or image to get started!**

I'm here 24/7 to help make your day better! 🤖💙

*Developed with ❤️ by SUNNEL*"""
                
                self.send_message(sender_id, welcome_message)
                self.update_user_database(sender_id, verification_status="verified")
                self.track_feature_usage(sender_id, "get_started")
            
        except Exception as e:
            logger.error(f"Postback handling error: {e}")

# Initialize bot
bot = SunnelBot()

def get_user_name(user_id: str) -> Optional[str]:
    """Get user's name from Facebook Graph API with caching"""
    try:
        # Try from database first
        with get_db() as conn:
            user = conn.execute('SELECT first_name FROM users WHERE user_id = ?', (user_id,)).fetchone()
            if user and user['first_name']:
                return user['first_name']
        
        # Fetch from API if not in database
        url = f'https://graph.facebook.com/{user_id}'
        params = {
            'fields': 'first_name',
            'access_token': PAGE_ACCESS_TOKEN
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('first_name', '')
    except Exception as e:
        logger.error(f"Error getting user name: {e}")
    return None

@app.route('/')
def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{BOT_NAME} v{BOT_VERSION}</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 0; padding: 20px; color: white; }}
            .container {{ max-width: 800px; margin: 0 auto; text-align: center; }}
            .header {{ background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; margin-bottom: 20px; }}
            .status {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 10px 0; }}
            .btn {{ background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; display: inline-block; }}
            .btn:hover {{ background: #45a049; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 {BOT_NAME} v{BOT_VERSION}</h1>
                <p>Advanced AI Assistant with Image Analysis</p>
                <p>✅ Bot is running successfully!</p>
            </div>
            
            <div class="status">
                <h3>🔧 System Status</h3>
                <p>📱 Facebook Integration: {'✅ Active' if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else '❌ Not Configured'}</p>
                <p>🤖 OpenAI API: {'✅ Ready' if OPENAI_API_KEY else '❌ Not Configured'}</p>
                <p>🌟 Gemini API: {'✅ Ready' if GEMINI_API_KEY and GEMINI_AVAILABLE else '❌ Not Available'}</p>
            </div>
            
            <div class="status">
                <h3>🚀 Features Available</h3>
                <p>💬 Text Conversations</p>
                <p>📸 Image Analysis & Understanding</p>
                <p>🧠 Conversation Memory</p>
                <p>📊 User Analytics</p>
                <p>⚡ Real-time Processing</p>
            </div>
            
            <a href="/dashboard" class="btn">📊 View Dashboard</a>
            <a href="/health" class="btn">🏥 Health Check</a>
            
            <div style="margin-top: 30px;">
                <p><em>Created with ❤️ by SUNNEL</em></p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Webhook verification endpoint"""
    try:
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        if token == bot.verify_token:
            logger.info("✅ Webhook verified successfully")
            return challenge
        else:
            logger.warning("❌ Webhook verification failed")
            return 'Verification failed', 403
    except Exception as e:
        logger.error(f"Webhook verification error: {e}")
        return 'Error', 500

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Enhanced webhook handler with comprehensive error handling"""
    try:
        data = request.get_json()
        
        if data.get('object') == 'page':
            for entry in data.get('entry', []):
                for messaging_event in entry.get('messaging', []):
                    sender_id = messaging_event.get('sender', {}).get('id')
                    
                    if messaging_event.get('message'):
                        message_data = messaging_event['message']
                        message_text = message_data.get('text', '')
                        sender_name = get_user_name(sender_id)
                        
                        # Handle attachments
                        if message_data.get('attachments'):
                            for attachment in message_data['attachments']:
                                if attachment.get('type') == 'image':
                                    image_url = attachment.get('payload', {}).get('url')
                                    if image_url:
                                        threading.Thread(
                                            target=bot.handle_image_message,
                                            args=(sender_id, image_url, message_text, sender_name)
                                        ).start()
                                        break
                        elif message_text:
                            threading.Thread(
                                target=bot.handle_message,
                                args=(sender_id, message_text)
                            ).start()
                    
                    elif messaging_event.get('postback'):
                        payload = messaging_event['postback']['payload']
                        threading.Thread(
                            target=bot.handle_postback,
                            args=(sender_id, payload)
                        ).start()
        
        return 'OK', 200
        
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
        return 'Error', 500

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    try:
        with get_db() as conn:
            # Basic statistics
            user_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN verification_status = 'verified' THEN 1 END) as verified_users,
                    SUM(total_messages) as total_messages,
                    SUM(total_images) as total_images,
                    AVG(user_rating) as avg_rating,
                    COUNT(CASE WHEN last_interaction > datetime('now', '-24 hours') THEN 1 END) as active_24h,
                    COUNT(CASE WHEN last_interaction > datetime('now', '-7 days') THEN 1 END) as active_7d
                FROM users
            ''').fetchone()
            
            # Performance metrics
            perf_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_interactions,
                    AVG(processing_time) as avg_response_time,
                    COUNT(CASE WHEN error_message IS NULL THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(CASE WHEN interaction_type = 'image' THEN 1 END) as image_interactions
                FROM interactions
                WHERE timestamp > datetime('now', '-24 hours')
            ''').fetchone()
            
            # Recent interactions
            recent_interactions = conn.execute('''
                SELECT u.first_name, i.interaction_type, i.user_message, i.timestamp, i.ai_provider
                FROM interactions i
                JOIN users u ON i.user_id = u.user_id
                ORDER BY i.timestamp DESC
                LIMIT 10
            ''').fetchall()
        
        dashboard_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{BOT_NAME} Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                              gap: 20px; margin-bottom: 20px; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .section {{ background: white; padding: 20px; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .interaction-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
                .image-tag {{ background: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
                .text-tag {{ background: #f3e5f5; color: #7b1fa2; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 {BOT_NAME} Dashboard</h1>
                    <p>Real-time analytics and performance metrics</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_users'] or 0}</div>
                        <div class="stat-label">Total Users</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['verified_users'] or 0}</div>
                        <div class="stat-label">Verified Users</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_messages'] or 0}</div>
                        <div class="stat-label">Text Messages</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_images'] or 0}</div>
                        <div class="stat-label">Images Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['active_24h'] or 0}</div>
                        <div class="stat-label">Active (24h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{perf_stats['image_interactions'] or 0}</div>
                        <div class="stat-label">Images (24h)</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Performance Metrics (24h)</h2>
                    <p><strong>Success Rate:</strong> {perf_stats['success_rate'] or 100:.1f}%</p>
                    <p><strong>Average Response Time:</strong> {perf_stats['avg_response_time'] or 0:.2f}s</p>
                    <p><strong>Total Interactions:</strong> {perf_stats['total_interactions'] or 0}</p>
                    <p><strong>Image Analysis:</strong> {perf_stats['image_interactions'] or 0} images processed</p>
                </div>
                
                <div class="section">
                    <h2>💬 Recent Interactions</h2>
                    {''.join([f'<div class="interaction-item"><strong>{row["first_name"] or "Unknown"}</strong>: <span class="{"image-tag" if row["interaction_type"] == "image" else "text-tag"}">{row["interaction_type"].upper()}</span> {(row["user_message"] or "Image shared")[:100]}{"..." if len(row["user_message"] or "") > 100 else ""}<br><span class="timestamp">{row["timestamp"]} • {row["ai_provider"] or "Unknown"}</span></div>' for row in recent_interactions]) if recent_interactions else '<p>No recent interactions</p>'}
                </div>
                
                <div class="section">
                    <h2>🎯 AI Features</h2>
                    <p>✅ Text Conversations with Context Memory</p>
                    <p>✅ Advanced Image Analysis & Understanding</p>
                    <p>✅ Multi-Provider AI (Gemini + OpenAI)</p>
                    <p>✅ Real-time Performance Monitoring</p>
                    <p>✅ Comprehensive User Analytics</p>
                </div>
                
                <div class="section">
                    <p><em>Created with ❤️ by SUNNEL | Version {BOT_VERSION}</em></p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        return dashboard_html
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard Error: {str(e)}", 500

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        with get_db() as conn:
            db_status = "healthy"
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    SUM(total_messages) as total_messages,
                    AVG(CASE WHEN last_interaction > datetime('now', '-24 hours') THEN 1.0 ELSE 0.0 END) * 100 as success_rate
                FROM users
            ''').fetchone()
        
        # Test AI services
        ai_status = {}
        
        if OPENAI_API_KEY:
            ai_status['openai'] = 'configured'
        else:
            ai_status['openai'] = 'not_configured'
        
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            ai_status['gemini'] = 'configured'
        else:
            ai_status['gemini'] = 'not_configured' if not GEMINI_API_KEY else 'library_error'
        
        uptime_duration = datetime.now() - bot.start_time
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'bot_info': {
                'name': BOT_NAME,
                'version': BOT_VERSION,
                'uptime_hours': round(uptime_duration.total_seconds() / 3600, 2),
                'uptime_days': round(uptime_duration.total_seconds() / 86400, 2)
            },
            'services': {
                'database': db_status,
                'facebook_api': 'configured' if PAGE_ACCESS_TOKEN else 'not_configured',
                'webhook': 'configured' if VERIFY_TOKEN else 'not_configured',
                'ai_services': ai_status
            },
            'statistics': {
                'total_users': stats['total_users'] if stats else 0,
                'total_messages': stats['total_messages'] if stats else 0,
                'success_rate': stats['success_rate'] if stats else 100.0
            },
            'features': {
                'text_conversations': True,
                'image_analysis': bool(ai_status.get('openai') == 'configured' or ai_status.get('gemini') == 'configured'),
                'conversation_memory': True,
                'user_analytics': True,
                'error_recovery': True,
                'performance_monitoring': True,
                'multi_ai_providers': True
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Start the Flask app
    print(f"🚀 Starting {BOT_NAME} v{BOT_VERSION}")
    print("🔗 Bot will be available at: https://your-repl-url.replit.dev")
    print("📊 Dashboard available at: https://your-repl-url.replit.dev/dashboard")
    print("🏥 Health check available at: https://your-repl-url.replit.dev/health")
    print("\n🎯 Features Available:")
    print("  • 💬 Advanced text conversations")
    print("  • 📸 Complete image analysis & understanding")
    print("  • 🧠 Conversation memory & context")
    print("  • 📊 Real-time analytics & monitoring")
    print("  • 🤖 Multi-AI provider support (Gemini + OpenAI)")
    print("  • ⚡ High-performance processing")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
