
import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string
import openai
from datetime import datetime
import logging
import base64
import io
from PIL import Image
import time
import random
import threading
import sqlite3
from contextlib import contextmanager

# Only import Gemini if available (fix for grpc error)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Gemini not available: {e}")
    GEMINI_AVAILABLE = False
    genai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - you'll need to set these in Secrets
VERIFY_TOKEN = os.getenv('FACEBOOK_VERIFY_TOKEN', 'your_verify_token_here')
PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Specific post that users need to like
REQUIRED_POST_ID = "761320392916522"  # From the URL provided
PAGE_ID = "100071157053751"

# Initialize AI providers
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

# Database setup for user tracking
def init_database():
    """Initialize SQLite database for user tracking"""
    with sqlite3.connect('bot_analytics.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                first_name TEXT,
                last_name TEXT,
                profile_pic TEXT,
                first_interaction TIMESTAMP,
                last_interaction TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                total_images INTEGER DEFAULT 0,
                verification_status TEXT DEFAULT 'unverified',
                verification_date TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message_type TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                ai_provider TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bot_stats (
                id INTEGER PRIMARY KEY,
                total_users INTEGER DEFAULT 0,
                verified_users INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                uptime_start TIMESTAMP,
                last_updated TIMESTAMP
            )
        ''')
        
        # Initialize bot stats if not exists
        conn.execute('''
            INSERT OR IGNORE INTO bot_stats (id, uptime_start, last_updated) 
            VALUES (1, ?, ?)
        ''', (datetime.now(), datetime.now()))

# Initialize database on startup
init_database()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect('bot_analytics.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class EnhancedFacebookBot:
    def __init__(self):
        self.api_version = 'v18.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
        self.conversation_memory = {}
        self.user_preferences = {}
        self.verified_users = set()
        self.user_states = {}
        self.start_time = datetime.now()
        
        # Start auto uptime checker
        self.start_uptime_monitor()
        
    def start_uptime_monitor(self):
        """Start background thread for uptime monitoring"""
        def uptime_checker():
            while True:
                try:
                    # Update bot stats every 5 minutes
                    self.update_bot_stats()
                    time.sleep(300)  # 5 minutes
                except Exception as e:
                    logger.error(f"Uptime monitor error: {e}")
                    time.sleep(60)  # Retry in 1 minute on error
        
        uptime_thread = threading.Thread(target=uptime_checker, daemon=True)
        uptime_thread.start()
        logger.info("🔄 Auto uptime monitor started")
    
    def update_bot_stats(self):
        """Update bot statistics in database"""
        try:
            with get_db() as conn:
                # Count total and verified users
                total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
                verified_users = conn.execute(
                    'SELECT COUNT(*) as count FROM users WHERE verification_status = ?', 
                    ('verified',)
                ).fetchone()['count']
                total_messages = conn.execute('SELECT COUNT(*) as count FROM interactions').fetchone()['count']
                
                conn.execute('''
                    UPDATE bot_stats 
                    SET total_users = ?, verified_users = ?, total_messages = ?, last_updated = ?
                    WHERE id = 1
                ''', (total_users, verified_users, total_messages, datetime.now()))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating bot stats: {e}")
    
    def log_user_interaction(self, user_id, message_type, content, ai_provider=None):
        """Log user interaction to database"""
        try:
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO interactions (user_id, message_type, content, timestamp, ai_provider)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, message_type, content[:500], datetime.now(), ai_provider))
                
                # Update user message count
                conn.execute('''
                    UPDATE users SET 
                    total_messages = total_messages + 1,
                    last_interaction = ?
                    WHERE user_id = ?
                ''', (datetime.now(), user_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_user_profile(self, user_id):
        """Get detailed user profile from Facebook"""
        try:
            url = f'{self.base_url}/{user_id}'
            params = {
                'fields': 'first_name,last_name,name,profile_pic',
                'access_token': PAGE_ACCESS_TOKEN
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
        return None
    
    def update_user_database(self, user_id, profile_data=None, verification_status=None):
        """Update or create user in database"""
        try:
            if not profile_data:
                profile_data = self.get_user_profile(user_id)
            
            if not profile_data:
                profile_data = {'first_name': 'Unknown', 'name': 'Unknown User'}
            
            with get_db() as conn:
                # Check if user exists
                user = conn.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,)).fetchone()
                
                if user:
                    # Update existing user
                    update_data = {
                        'name': profile_data.get('name', 'Unknown'),
                        'first_name': profile_data.get('first_name', 'Unknown'),
                        'last_name': profile_data.get('last_name', ''),
                        'profile_pic': profile_data.get('profile_pic', ''),
                        'last_interaction': datetime.now()
                    }
                    
                    if verification_status:
                        update_data['verification_status'] = verification_status
                        if verification_status == 'verified':
                            update_data['verification_date'] = datetime.now()
                    
                    set_clause = ', '.join([f"{k} = ?" for k in update_data.keys()])
                    values = list(update_data.values()) + [user_id]
                    
                    conn.execute(f'UPDATE users SET {set_clause} WHERE user_id = ?', values)
                else:
                    # Create new user
                    conn.execute('''
                        INSERT INTO users 
                        (user_id, name, first_name, last_name, profile_pic, first_interaction, last_interaction, verification_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id,
                        profile_data.get('name', 'Unknown'),
                        profile_data.get('first_name', 'Unknown'),
                        profile_data.get('last_name', ''),
                        profile_data.get('profile_pic', ''),
                        datetime.now(),
                        datetime.now(),
                        verification_status or 'unverified'
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating user database: {e}")
        
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

    def send_image_message(self, recipient_id, image_url, message_text=None):
        """Send an image message to a Facebook user"""
        url = f'{self.base_url}/me/messages'
        headers = {'Content-Type': 'application/json'}
        
        message_data = {
            'attachment': {
                'type': 'image',
                'payload': {
                    'url': image_url,
                    'is_reusable': True
                }
            }
        }
        
        if message_text:
            message_data['text'] = message_text
        
        data = {
            'recipient': {'id': recipient_id},
            'message': message_data,
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Image message sent successfully to {recipient_id}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send image message: {e}")
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
            pass

    def check_user_follows_page(self, user_id):
        """Check if user follows the page"""
        try:
            url = f'{self.base_url}/{user_id}'
            params = {
                'fields': 'subscribed_field',
                'access_token': PAGE_ACCESS_TOKEN
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('subscribed_field', False)
        except Exception as e:
            logger.error(f"Error checking follow status: {e}")
        return False

    def check_user_liked_specific_post(self, user_id):
        """Check if user liked the specific required post"""
        try:
            # Check if user liked the specific post
            likes_url = f'{self.base_url}/{REQUIRED_POST_ID}/likes'
            params = {
                'access_token': PAGE_ACCESS_TOKEN,
                'limit': 1000  # Increase limit to check more likes
            }
            
            response = requests.get(likes_url, params=params)
            if response.status_code == 200:
                likes_data = response.json()
                likes = likes_data.get('data', [])
                
                # Check if user_id is in the likes
                for like in likes:
                    if like.get('id') == user_id:
                        return True
                        
                # Check pagination if there are more likes
                while likes_data.get('paging', {}).get('next'):
                    next_url = likes_data['paging']['next']
                    response = requests.get(next_url)
                    if response.status_code == 200:
                        likes_data = response.json()
                        likes = likes_data.get('data', [])
                        for like in likes:
                            if like.get('id') == user_id:
                                return True
                    else:
                        break
            
        except Exception as e:
            logger.error(f"Error checking specific post like: {e}")
        
        return False

    def verify_user_access(self, user_id):
        """Verify if user has followed page and liked the specific post"""
        if user_id in self.verified_users:
            return True, "verified"
        
        # Check follow status
        follows_page = self.check_user_follows_page(user_id)
        
        # Check if user liked the specific post
        liked_specific_post = self.check_user_liked_specific_post(user_id)
        
        if follows_page and liked_specific_post:
            self.verified_users.add(user_id)
            return True, "newly_verified"
        elif follows_page and not liked_specific_post:
            return False, "need_like"
        elif not follows_page and liked_specific_post:
            return False, "need_follow"
        else:
            return False, "need_both"

    def send_verification_request(self, user_id, status):
        """Send verification request message with specific post link"""
        post_link = f"https://www.facebook.com/{PAGE_ID}/posts/{REQUIRED_POST_ID}/?mibextid=rS40aB7S9Ucbxw6v"
        
        if status == "need_both":
            message = f"""🔐 **Welcome! Access Required** 🔐

To unlock full access to this amazing AI bot, please:

1️⃣ **Follow this page** 👍
2️⃣ **Like this specific post** ❤️
{post_link}

Once you complete both steps, you'll get full access to:
🤖 Advanced AI conversations (ChatGPT + Gemini)
🔍 Image analysis capabilities  
💭 Smart conversation memory
⚡ Instant responses 24/7

Please complete these steps and send any message to verify! 🚀"""

        elif status == "need_follow":
            message = """🔐 **Almost There!** 🔐

Thank you for liking our specific post! ❤️

To complete verification, please:
1️⃣ **Follow this page** 👍

Once you follow, you'll have full access to all AI features! 🚀"""

        elif status == "need_like":
            message = f"""🔐 **Almost There!** 🔐

Thank you for following our page! 👍

To complete verification, please:
2️⃣ **Like this specific post** ❤️
{post_link}

Once you like our post, you'll have full access to all AI features! 🚀"""

        self.send_message(user_id, message)

    def send_congratulations_with_image(self, user_id):
        """Send congratulations message with the custom image"""
        congrats_message = """🎉 **CONGRATULATIONS!** 🎉

✅ **Verification Complete!** ✅

You now have full access to our Ultimate AI Bot! 🚀

🌟 **Available Features:**
🤖 ChatGPT & Gemini AI conversations
🔍 Advanced image analysis
💭 Smart conversation memory
⚡ Instant 24/7 responses
🎨 Creative AI assistance

**Thank you for following and supporting us!** 

Developed by SUNNEL 🤍

You can now ask me anything or send images for analysis! 🍓🥰"""

        # Send the congratulations message first
        self.send_message(user_id, congrats_message)
        
        # Send the custom image hosted locally
        base_url = request.url_root.rstrip('/')
        image_url = f"{base_url}/static/images/congratulations.png"
        
        self.send_image_message(user_id, image_url, "🌟 Developed by SUNNEL - Thank you for using our AI bot! 🌟")
    
    def download_image(self, attachment_url):
        """Download image from Facebook attachment URL"""
        try:
            response = requests.get(f"{attachment_url}&access_token={PAGE_ACCESS_TOKEN}")
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
        return None
    
    def analyze_image_with_gemini(self, image_data, user_question="What's in this image?"):
        """Analyze image using Google Gemini Vision"""
        if not GEMINI_API_KEY or not GEMINI_AVAILABLE:
            return None
            
        try:
            image = Image.open(io.BytesIO(image_data))
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""🔍 **Image Analysis Request**

User asked: "{user_question}"

Please provide a detailed, friendly, and insightful analysis of this image. Include:
- What you see in the image
- Any notable details, colors, objects, people, or scenes
- Context or interesting observations
- Answer the user's specific question if applicable

Keep the response engaging and conversational! 📸✨"""
            
            response = model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return None
    
    def analyze_image_with_openai(self, image_data, user_question="What's in this image?"):
        """Analyze image using OpenAI GPT-4 Vision"""
        if not OPENAI_API_KEY:
            return None
            
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"🔍 **Image Analysis Request**\n\nUser asked: '{user_question}'\n\nPlease analyze this image and provide a detailed, friendly response. Include what you see, any interesting details, and answer their specific question!"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {e}")
            return None
    
    def get_smart_ai_response(self, user_message, user_name="User", conversation_history=None):
        """Get AI response using multiple providers with smart routing"""
        if not OPENAI_API_KEY and not (GEMINI_API_KEY and GEMINI_AVAILABLE):
            return "🤖 I'm sorry, but I'm not configured with AI capabilities at the moment. Please contact the page administrator."
        
        use_gemini = self.should_use_gemini(user_message)
        provider_used = None
        
        try:
            if use_gemini and GEMINI_API_KEY and GEMINI_AVAILABLE:
                provider_used = "Gemini"
                return self.get_gemini_response(user_message, user_name, conversation_history), provider_used
            elif OPENAI_API_KEY:
                provider_used = "OpenAI"
                return self.get_openai_response(user_message, user_name, conversation_history), provider_used
            elif GEMINI_API_KEY and GEMINI_AVAILABLE:
                provider_used = "Gemini"
                return self.get_gemini_response(user_message, user_name, conversation_history), provider_used
            else:
                return "🤖 AI services are currently unavailable. Please try again later!", None
                
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "🤖 I'm having trouble processing your request right now. Please try again later!", None
    
    def should_use_gemini(self, message):
        """Determine if we should use Gemini based on message content"""
        gemini_keywords = ['creative', 'story', 'poem', 'imagine', 'brainstorm', 'idea', 'art', 'design']
        return any(keyword in message.lower() for keyword in gemini_keywords)
    
    def get_gemini_response(self, user_message, user_name="User", conversation_history=None):
        """Get response from Google Gemini"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""🌟 **AI Assistant Powered by Gemini** 🌟

You are a helpful, friendly, and intelligent Facebook page bot. You should:
- Be conversational and engaging
- Use emojis appropriately to make responses more lively
- Provide accurate and helpful information
- Be creative and insightful
- Keep responses concise but informative for messaging

User: {user_name}
{f"Previous context: {conversation_history}" if conversation_history else ""}

User's message: {user_message}

Please provide a helpful and engaging response:"""
            
            response = model.generate_content(prompt)
            return f"🌟 {response.text}"
            
        except Exception as e:
            logger.error(f"Error with Gemini: {e}")
            raise e
    
    def get_openai_response(self, user_message, user_name="User", conversation_history=None):
        """Get response from OpenAI"""
        try:
            messages = [
                {"role": "system", "content": f"""🤖 You are an advanced AI assistant powered by ChatGPT, integrated into a Facebook page bot. 

Personality:
- Friendly, helpful, and engaging
- Use emojis appropriately to make conversations lively ✨
- Provide accurate, detailed responses
- Be conversational and natural
- Keep responses suitable for Facebook messaging (concise but informative)

Current user: {user_name}
{f"Context from previous messages: {conversation_history}" if conversation_history else ""}"""},
                {"role": "user", "content": user_message}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )
            
            return f"🤖 {response.choices[0].message.content.strip()}"
            
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
            raise e
    
    def update_conversation_memory(self, user_id, message, response):
        """Update conversation memory for context awareness"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            'user_message': message,
            'bot_response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversation_memory[user_id]) > 5:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-5:]
    
    def get_conversation_context(self, user_id):
        """Get recent conversation context"""
        if user_id in self.conversation_memory:
            recent = self.conversation_memory[user_id][-2:]
            return " | ".join([f"User: {ex['user_message']} Bot: {ex['bot_response'][:50]}..." for ex in recent])
        return None
    
    def handle_message(self, sender_id, message_text, sender_name=None):
        """Process incoming message with security verification"""
        logger.info(f"Processing message from {sender_id}: {message_text}")
        
        # Update user in database
        self.update_user_database(sender_id)
        
        # Check user verification status
        is_verified, status = self.verify_user_access(sender_id)
        
        if not is_verified:
            self.send_verification_request(sender_id, status)
            self.log_user_interaction(sender_id, "unverified_attempt", message_text)
            return
        
        # If newly verified, send congratulations
        if status == "newly_verified":
            self.update_user_database(sender_id, verification_status="verified")
            self.send_congratulations_with_image(sender_id)
            self.log_user_interaction(sender_id, "verification_complete", "User verified successfully")
            return
        
        # User is verified, proceed with normal AI response
        self.send_typing_indicator(sender_id)
        context = self.get_conversation_context(sender_id)
        response, provider = self.get_smart_ai_response(message_text, sender_name, context)
        self.update_conversation_memory(sender_id, message_text, response)
        self.send_message(sender_id, response)
        
        # Log interaction
        self.log_user_interaction(sender_id, "message", message_text, provider)
    
    def handle_image_message(self, sender_id, attachment_url, message_text="", sender_name=None):
        """Handle image messages with security verification"""
        logger.info(f"Processing image from {sender_id}")
        
        # Update user in database
        self.update_user_database(sender_id)
        
        # Check user verification status
        is_verified, status = self.verify_user_access(sender_id)
        
        if not is_verified:
            self.send_verification_request(sender_id, status)
            self.log_user_interaction(sender_id, "unverified_image_attempt", message_text)
            return
        
        # If newly verified, send congratulations
        if status == "newly_verified":
            self.update_user_database(sender_id, verification_status="verified")
            self.send_congratulations_with_image(sender_id)
            self.log_user_interaction(sender_id, "verification_complete", "User verified successfully")
            return
        
        # User is verified, proceed with image analysis
        self.send_typing_indicator(sender_id)
        
        image_data = self.download_image(attachment_url)
        
        if not image_data:
            self.send_message(sender_id, "🖼️ Sorry, I couldn't download your image. Please try sending it again!")
            return
        
        user_question = message_text if message_text.strip() else "What's in this image?"
        
        response = None
        provider = None
        
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            response = self.analyze_image_with_gemini(image_data, user_question)
            provider = "Gemini Vision"
        
        if not response and OPENAI_API_KEY:
            response = self.analyze_image_with_openai(image_data, user_question)
            provider = "OpenAI Vision"
        
        if not response:
            response = "🖼️ I can see you sent an image, but I'm having trouble analyzing it right now. Please try again later or describe what you'd like to know about it!"
            provider = "Error"
        
        final_response = f"🔍 **Image Analysis** 📸\n\n{response}"
        self.update_conversation_memory(sender_id, f"[Image] {user_question}", final_response)
        self.send_message(sender_id, final_response)
        
        # Log interaction and update image count
        self.log_user_interaction(sender_id, "image", user_question, provider)
        with get_db() as conn:
            conn.execute(
                'UPDATE users SET total_images = total_images + 1 WHERE user_id = ?',
                (sender_id,)
            )
            conn.commit()

# Initialize enhanced bot
bot = EnhancedFacebookBot()

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
                    
                    if messaging_event.get('message'):
                        message_data = messaging_event['message']
                        message_text = message_data.get('text', '')
                        sender_name = get_user_name(sender_id)
                        
                        if message_data.get('attachments'):
                            for attachment in message_data['attachments']:
                                if attachment.get('type') == 'image':
                                    image_url = attachment.get('payload', {}).get('url')
                                    if image_url:
                                        bot.handle_image_message(sender_id, image_url, message_text, sender_name)
                                        break
                        elif message_text:
                            bot.handle_message(sender_id, message_text, sender_name)
                    
                    elif messaging_event.get('postback'):
                        payload = messaging_event['postback'].get('payload')
                        if payload == 'GET_STARTED':
                            # Check verification status for get started
                            bot.update_user_database(sender_id)
                            is_verified, status = bot.verify_user_access(sender_id)
                            if not is_verified:
                                bot.send_verification_request(sender_id, status)
                            elif status == "newly_verified":
                                bot.update_user_database(sender_id, verification_status="verified")
                                bot.send_congratulations_with_image(sender_id)
                            else:
                                welcome_message = """🌟 **Welcome Back!** 🚀

You're already verified! I'm your Ultimate AI Assistant powered by:
🤖 ChatGPT for intelligent conversations
🌟 Google Gemini for creative tasks
🔍 Advanced image analysis capabilities

Just send me any message or image, and I'll help you! 🍓🥰"""
                                bot.send_message(sender_id, welcome_message)
        
        return 'OK', 200
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return 'Error', 500

@app.route('/dashboard')
def dashboard():
    """Enhanced analytics dashboard"""
    try:
        with get_db() as conn:
            # Get bot stats
            bot_stats = conn.execute('SELECT * FROM bot_stats WHERE id = 1').fetchone()
            
            # Get user statistics
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            verified_users = conn.execute(
                'SELECT COUNT(*) as count FROM users WHERE verification_status = ?', 
                ('verified',)
            ).fetchone()['count']
            unverified_users = total_users - verified_users
            
            # Get recent users (last 10)
            recent_users = conn.execute('''
                SELECT user_id, name, first_name, profile_pic, first_interaction, 
                       last_interaction, total_messages, total_images, verification_status
                FROM users 
                ORDER BY last_interaction DESC 
                LIMIT 10
            ''').fetchall()
            
            # Get interaction stats
            total_messages = conn.execute('SELECT COUNT(*) as count FROM interactions').fetchone()['count']
            total_images = conn.execute(
                'SELECT COUNT(*) as count FROM interactions WHERE message_type = ?', 
                ('image',)
            ).fetchone()['count']
            
            # Get recent interactions
            recent_interactions = conn.execute('''
                SELECT i.*, u.name, u.first_name, u.profile_pic
                FROM interactions i
                LEFT JOIN users u ON i.user_id = u.user_id
                ORDER BY i.timestamp DESC
                LIMIT 20
            ''').fetchall()
            
            # Calculate uptime
            if bot_stats:
                uptime_start = datetime.fromisoformat(bot_stats['uptime_start'])
                uptime_duration = datetime.now() - uptime_start
                uptime_hours = uptime_duration.total_seconds() / 3600
            else:
                uptime_hours = 0
            
            # AI provider stats
            openai_usage = conn.execute(
                'SELECT COUNT(*) as count FROM interactions WHERE ai_provider LIKE ?', 
                ('%OpenAI%',)
            ).fetchone()['count']
            gemini_usage = conn.execute(
                'SELECT COUNT(*) as count FROM interactions WHERE ai_provider LIKE ?', 
                ('%Gemini%',)
            ).fetchone()['count']
        
        return render_template_string(DASHBOARD_HTML, **{
            'total_users': total_users,
            'verified_users': verified_users,
            'unverified_users': unverified_users,
            'total_messages': total_messages,
            'total_images': total_images,
            'recent_users': recent_users,
            'recent_interactions': recent_interactions,
            'uptime_hours': round(uptime_hours, 1),
            'openai_usage': openai_usage,
            'gemini_usage': gemini_usage,
            'required_post_id': REQUIRED_POST_ID,
            'page_id': PAGE_ID
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard error: {e}", 500

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
    """Health check endpoint with enhanced stats"""
    try:
        with get_db() as conn:
            stats = conn.execute('SELECT * FROM bot_stats WHERE id = 1').fetchone()
            uptime_start = datetime.fromisoformat(stats['uptime_start']) if stats else bot.start_time
            uptime_duration = datetime.now() - uptime_start
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': round(uptime_duration.total_seconds() / 3600, 2),
            'bot_configured': bool(PAGE_ACCESS_TOKEN and VERIFY_TOKEN),
            'openai_configured': bool(OPENAI_API_KEY),
            'gemini_configured': bool(GEMINI_API_KEY and GEMINI_AVAILABLE),
            'database_status': 'connected',
            'total_users': stats['total_users'] if stats else 0,
            'verified_users': stats['verified_users'] if stats else 0,
            'total_messages': stats['total_messages'] if stats else 0,
            'required_post_id': REQUIRED_POST_ID,
            'features': {
                'smart_conversations': True,
                'image_analysis': bool(OPENAI_API_KEY or (GEMINI_API_KEY and GEMINI_AVAILABLE)),
                'multi_ai_providers': bool(OPENAI_API_KEY and GEMINI_API_KEY and GEMINI_AVAILABLE),
                'conversation_memory': True,
                'auto_responses': True,
                'security_verification': True,
                'user_tracking': True,
                'analytics_dashboard': True,
                'auto_uptime': True
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/static/images/<filename>')
def serve_image(filename):
    """Serve static images"""
    try:
        return app.send_static_file(f'images/{filename}')
    except:
        return "Image not found", 404

@app.route('/', methods=['GET'])
def home():
    """Enhanced home page with bot information"""
    try:
        with get_db() as conn:
            stats = conn.execute('SELECT * FROM bot_stats WHERE id = 1').fetchone()
            uptime_start = datetime.fromisoformat(stats['uptime_start']) if stats else bot.start_time
            uptime_duration = datetime.now() - uptime_start
    except:
        stats = None
        uptime_duration = datetime.now() - bot.start_time
    
    return render_template_string(HOME_HTML, **{
        'bot_status': "✅" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "❌",
        'bot_text': "Active & Secured" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "Not Configured",
        'openai_status': "🤖" if OPENAI_API_KEY else "❌",
        'openai_text': "Connected" if OPENAI_API_KEY else "Not Configured",
        'gemini_status': "🌟" if GEMINI_API_KEY and GEMINI_AVAILABLE else "❌",
        'gemini_text': "Connected" if GEMINI_API_KEY and GEMINI_AVAILABLE else "Not Available",
        'webhook_url': f"{request.url_root.rstrip('/')}/webhook",
        'dashboard_url': f"{request.url_root.rstrip('/')}/dashboard",
        'uptime_hours': round(uptime_duration.total_seconds() / 3600, 1),
        'total_users': stats['total_users'] if stats else 0,
        'verified_users': stats['verified_users'] if stats else 0,
        'total_messages': stats['total_messages'] if stats else 0,
        'required_post_id': REQUIRED_POST_ID
    })

# HTML Templates
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Ultimate AI Facebook Bot - Analytics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 10px;
        }
        .security-banner {
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
            color: white;
            font-weight: bold;
        }
        .dashboard-button {
            display: block;
            width: 100%;
            max-width: 300px;
            margin: 30px auto;
            padding: 15px 30px;
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        .dashboard-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(68, 160, 141, 0.4);
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        .feature {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4ecdc4;
        }
        .security-feature {
            border-left: 4px solid #ff6b6b;
        }
        .uptime-banner {
            background: linear-gradient(45deg, #43cea2, #185a9d);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            font-weight: bold;
        }
        .webhook-info {
            background: rgba(0, 0, 0, 0.2);
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            text-align: center;
        }
        code {
            background: rgba(0, 0, 0, 0.3);
            padding: 5px 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #4ade80; }
        .status-offline { background-color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Ultimate AI Facebook Bot</h1>
        <p class="subtitle">Advanced Analytics & AI-Powered Conversations 🚀</p>
        
        <div class="uptime-banner">
            <span class="status-indicator status-online"></span>
            🔄 Auto Uptime: {{ uptime_hours }} hours | 🎯 Monitoring Active
        </div>
        
        <div class="security-banner">
            🔐 SECURED ACCESS SYSTEM ACTIVE 🔐<br>
            Users must follow page & like specific post (ID: {{ required_post_id }})
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_users }}</div>
                <h3>Total Users</h3>
                <p>Registered in database</p>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ verified_users }}</div>
                <h3>Verified Users</h3>
                <p>Full access granted</p>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_messages }}</div>
                <h3>Total Messages</h3>
                <p>AI conversations</p>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ uptime_hours }}h</div>
                <h3>Uptime</h3>
                <p>Continuous operation</p>
            </div>
        </div>
        
        <a href="{{ dashboard_url }}" class="dashboard-button">
            📊 View Full Analytics Dashboard
        </a>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div style="font-size: 2em; margin-bottom: 10px;">{{ bot_status }}</div>
                <h3>Bot Status</h3>
                <p>{{ bot_text }}</p>
            </div>
            <div class="stat-card">
                <div style="font-size: 2em; margin-bottom: 10px;">{{ openai_status }}</div>
                <h3>ChatGPT</h3>
                <p>{{ openai_text }}</p>
            </div>
            <div class="stat-card">
                <div style="font-size: 2em; margin-bottom: 10px;">{{ gemini_status }}</div>
                <h3>Google Gemini</h3>
                <p>{{ gemini_text }}</p>
            </div>
        </div>
        
        <h2 style="text-align: center; margin: 30px 0;">🌟 Enhanced Features</h2>
        <div class="features">
            <div class="feature security-feature">
                <strong>🔐 Advanced Security</strong><br>
                Specific post verification system with auto-tracking
            </div>
            <div class="feature">
                <strong>📊 Real-time Analytics</strong><br>
                Live user tracking and interaction monitoring
            </div>
            <div class="feature">
                <strong>🤖 Multi-AI Intelligence</strong><br>
                ChatGPT + Gemini with smart provider routing
            </div>
            <div class="feature">
                <strong>🔍 Advanced Image Analysis</strong><br>
                AI-powered image understanding and description
            </div>
            <div class="feature">
                <strong>🔄 Auto Uptime Monitor</strong><br>
                Continuous operation tracking and health monitoring
            </div>
            <div class="feature">
                <strong>💾 User Database</strong><br>
                Complete user profiles and interaction history
            </div>
            <div class="feature">
                <strong>🎨 Aesthetic Dashboard</strong><br>
                Beautiful real-time analytics interface
            </div>
            <div class="feature">
                <strong>⚡ Smart Responses</strong><br>
                Context-aware AI conversations with memory
            </div>
        </div>
        
        <div class="webhook-info">
            <h3>📡 System Configuration</h3>
            <p>Webhook URL: <code>{{ webhook_url }}</code></p>
            <p>Analytics Dashboard: <code>{{ dashboard_url }}</code></p>
            <p>Ready to receive and analyze secured messages!</p>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>📊 Bot Analytics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-number {
            font-size: 3em;
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 10px;
        }
        .stat-label {
            font-size: 1.1em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .section h2 {
            margin-bottom: 20px;
            color: #4ecdc4;
            font-size: 1.8em;
        }
        .user-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            border-left: 4px solid #4ecdc4;
        }
        .user-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
        }
        .user-info {
            flex: 1;
        }
        .user-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .user-stats {
            opacity: 0.8;
            font-size: 0.9em;
        }
        .verification-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .verified {
            background: #4ade80;
            color: #065f46;
        }
        .unverified {
            background: #fbbf24;
            color: #92400e;
        }
        .interaction-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #6366f1;
        }
        .interaction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .interaction-user {
            font-weight: bold;
            color: #4ecdc4;
        }
        .interaction-time {
            font-size: 0.8em;
            opacity: 0.7;
        }
        .interaction-content {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .ai-provider {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: bold;
            margin-left: 10px;
        }
        .openai { background: #10b981; color: white; }
        .gemini { background: #8b5cf6; color: white; }
        .auto-refresh {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px 15px;
            border-radius: 25px;
            font-size: 0.9em;
        }
        .back-button {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .back-button:hover {
            transform: translateY(-2px);
        }
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }
            .user-card {
                flex-direction: column;
                text-align: center;
            }
        }
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => {
            location.reload();
        }, 30000);
    </script>
</head>
<body>
    <div class="auto-refresh">
        🔄 Auto-refresh: 30s
    </div>
    
    <div class="container">
        <a href="/" class="back-button">← Back to Home</a>
        
        <div class="header">
            <h1>📊 Bot Analytics Dashboard</h1>
            <p>Real-time monitoring and user analytics</p>
            <p><strong>Uptime:</strong> {{ uptime_hours }} hours | <strong>Required Post:</strong> {{ required_post_id }}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_users }}</div>
                <div class="stat-label">Total Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ verified_users }}</div>
                <div class="stat-label">Verified Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ unverified_users }}</div>
                <div class="stat-label">Unverified Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_messages }}</div>
                <div class="stat-label">Total Messages</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_images }}</div>
                <div class="stat-label">Images Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ openai_usage + gemini_usage }}</div>
                <div class="stat-label">AI Interactions</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 AI Provider Usage</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{{ openai_usage }}</div>
                    <div class="stat-label">OpenAI/ChatGPT</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ gemini_usage }}</div>
                    <div class="stat-label">Google Gemini</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>👥 Recent Users</h2>
            {% for user in recent_users %}
            <div class="user-card">
                <div class="user-avatar">
                    {% if user.profile_pic %}
                        <img src="{{ user.profile_pic }}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">
                    {% else %}
                        {{ user.first_name[0] if user.first_name else '?' }}
                    {% endif %}
                </div>
                <div class="user-info">
                    <div class="user-name">
                        {{ user.name or user.first_name or 'Unknown User' }}
                        <span class="verification-badge {{ 'verified' if user.verification_status == 'verified' else 'unverified' }}">
                            {{ '✅ Verified' if user.verification_status == 'verified' else '⏳ Unverified' }}
                        </span>
                    </div>
                    <div class="user-stats">
                        💬 {{ user.total_messages }} messages | 🖼️ {{ user.total_images }} images | 
                        📅 Last seen: {{ user.last_interaction[:19] if user.last_interaction else 'Never' }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>💬 Recent Interactions</h2>
            {% for interaction in recent_interactions %}
            <div class="interaction-item">
                <div class="interaction-header">
                    <span class="interaction-user">
                        {{ interaction.name or interaction.first_name or 'Unknown User' }}
                    </span>
                    <span class="interaction-time">
                        {{ interaction.timestamp[:19] if interaction.timestamp else '' }}
                        {% if interaction.ai_provider %}
                            <span class="ai-provider {{ 'openai' if 'OpenAI' in interaction.ai_provider else 'gemini' }}">
                                {{ interaction.ai_provider }}
                            </span>
                        {% endif %}
                    </span>
                </div>
                <div class="interaction-content">
                    <strong>{{ interaction.message_type.title() }}:</strong> 
                    {{ interaction.content[:100] }}{{ '...' if interaction.content|length > 100 else '' }}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    logger.info("🚀 Starting Ultimate AI Facebook Bot with Analytics...")
    
    if not PAGE_ACCESS_TOKEN:
        logger.warning("❌ FACEBOOK_PAGE_ACCESS_TOKEN not set")
    if not OPENAI_API_KEY:
        logger.warning("❌ OPENAI_API_KEY not set")
    if not GEMINI_API_KEY:
        logger.warning("❌ GEMINI_API_KEY not set")
    if not GEMINI_AVAILABLE:
        logger.warning("⚠️ Gemini libraries not available - using OpenAI only")
    
    logger.info(f"🎯 Required post ID: {REQUIRED_POST_ID}")
    logger.info(f"📊 Analytics dashboard available at: /dashboard")
    logger.info(f"🔄 Auto uptime monitoring: ACTIVE")
    
    if OPENAI_API_KEY and GEMINI_API_KEY and GEMINI_AVAILABLE:
        logger.info("🌟 Multi-AI configuration detected - Full features available!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
