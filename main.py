
import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory
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
import hashlib
import uuid
from typing import Optional, Dict, Any, List

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
BOT_VERSION = "3.0.0"
REQUIRED_POST_ID = "761320392916522"
PAGE_ID = "100071157053751"

# Initialize AI providers with better error handling
if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
        logger.info("✅ OpenAI configured successfully")
    except Exception as e:
        logger.error(f"❌ OpenAI configuration failed: {e}")
        OPENAI_API_KEY = None

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini configured successfully")
    except Exception as e:
        logger.error(f"❌ Gemini configuration failed: {e}")
        GEMINI_AVAILABLE = False

# Enhanced Database Schema
def init_database():
    """Initialize enhanced SQLite database"""
    with sqlite3.connect('bot_analytics.db') as conn:
        # Users table with more fields
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
                verification_status TEXT DEFAULT 'verified',
                verification_date TIMESTAMP,
                preferred_ai TEXT DEFAULT 'gemini',
                conversation_count INTEGER DEFAULT 0,
                last_seen_feature TEXT,
                user_rating REAL DEFAULT 5.0,
                total_errors INTEGER DEFAULT 0
            )
        ''')
        
        # Enhanced interactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message_type TEXT,
                content TEXT,
                response_text TEXT,
                timestamp TIMESTAMP,
                ai_provider TEXT,
                processing_time REAL,
                error_message TEXT,
                satisfaction_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Bot analytics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bot_stats (
                id INTEGER PRIMARY KEY,
                total_users INTEGER DEFAULT 0,
                verified_users INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                total_images INTEGER DEFAULT 0,
                uptime_start TIMESTAMP,
                last_updated TIMESTAMP,
                success_rate REAL DEFAULT 100.0,
                avg_response_time REAL DEFAULT 0.0,
                popular_features TEXT,
                error_count INTEGER DEFAULT 0
            )
        ''')
        
        # Features usage tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT,
                user_id TEXT,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP,
                success_rate REAL DEFAULT 100.0
            )
        ''')
        
        # Initialize or update bot stats
        conn.execute('''
            INSERT OR REPLACE INTO bot_stats (id, uptime_start, last_updated) 
            VALUES (1, ?, ?)
        ''', (datetime.now(), datetime.now()))

init_database()

@contextmanager
def get_db():
    """Enhanced database connection with error handling"""
    conn = None
    try:
        conn = sqlite3.connect('bot_analytics.db', timeout=30)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

class EnhancedFacebookBot:
    def __init__(self):
        self.api_version = 'v19.0'  # Updated to latest version
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
        self.conversation_memory = {}
        self.user_preferences = {}
        self.verified_users = set()
        self.user_states = {}
        self.start_time = datetime.now()
        self.message_queue = []
        self.error_counts = {}
        self.feature_stats = {}
        
        # Start enhanced monitoring
        self.start_monitoring_systems()
        
    def start_monitoring_systems(self):
        """Start enhanced background monitoring"""
        def monitoring_worker():
            while True:
                try:
                    self.update_analytics()
                    self.cleanup_old_data()
                    self.monitor_system_health()
                    time.sleep(300)  # 5 minutes
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitor_thread.start()
        logger.info("🔄 Enhanced monitoring system started")
    
    def update_analytics(self):
        """Update comprehensive analytics"""
        try:
            with get_db() as conn:
                # Calculate comprehensive stats
                stats = conn.execute('''
                    SELECT 
                        COUNT(DISTINCT user_id) as total_users,
                        COUNT(DISTINCT CASE WHEN verification_status = 'verified' THEN user_id END) as verified_users,
                        SUM(total_messages) as total_messages,
                        SUM(total_images) as total_images,
                        AVG(user_rating) as avg_rating
                    FROM users
                ''').fetchone()
                
                # Calculate success rate
                success_rate = conn.execute('''
                    SELECT 
                        (COUNT(CASE WHEN error_message IS NULL THEN 1 END) * 100.0 / COUNT(*)) as success_rate,
                        AVG(processing_time) as avg_response_time
                    FROM interactions
                    WHERE timestamp > datetime('now', '-24 hours')
                ''').fetchone()
                
                # Update bot stats
                conn.execute('''
                    UPDATE bot_stats SET 
                        total_users = ?, verified_users = ?, total_messages = ?, 
                        total_images = ?, last_updated = ?, success_rate = ?, 
                        avg_response_time = ?
                    WHERE id = 1
                ''', (
                    stats['total_users'], stats['verified_users'], stats['total_messages'],
                    stats['total_images'], datetime.now(), 
                    success_rate['success_rate'] or 100.0, 
                    success_rate['avg_response_time'] or 0.0
                ))
                
        except Exception as e:
            logger.error(f"Analytics update error: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        try:
            with get_db() as conn:
                # Keep only last 30 days of interactions
                conn.execute('''
                    DELETE FROM interactions 
                    WHERE timestamp < datetime('now', '-30 days')
                ''')
                logger.info("🧹 Old data cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def monitor_system_health(self):
        """Monitor system health and performance"""
        try:
            # Check AI service availability
            health_status = {
                'openai': bool(OPENAI_API_KEY),
                'gemini': bool(GEMINI_API_KEY and GEMINI_AVAILABLE),
                'database': True,
                'facebook_api': bool(PAGE_ACCESS_TOKEN),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            }
            
            # Log health status every hour
            if int(time.time()) % 3600 < 300:  # Every hour
                logger.info(f"🏥 System Health: {health_status}")
                
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    def log_interaction(self, user_id: str, message_type: str, content: str, 
                       response: str = "", ai_provider: str = None, 
                       processing_time: float = 0.0, error: str = None):
        """Enhanced interaction logging"""
        try:
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO interactions 
                    (user_id, message_type, content, response_text, timestamp, 
                     ai_provider, processing_time, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, message_type, content[:500], response[:500], 
                      datetime.now(), ai_provider, processing_time, error))
                
                # Update user stats
                conn.execute('''
                    UPDATE users SET 
                        total_messages = total_messages + 1,
                        last_interaction = ?,
                        conversation_count = conversation_count + 1,
                        total_errors = total_errors + ?
                    WHERE user_id = ?
                ''', (datetime.now(), 1 if error else 0, user_id))
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def track_feature_usage(self, user_id: str, feature_name: str, success: bool = True):
        """Track feature usage for analytics"""
        try:
            with get_db() as conn:
                # Update or insert feature usage
                conn.execute('''
                    INSERT OR REPLACE INTO feature_usage 
                    (feature_name, user_id, usage_count, last_used, success_rate)
                    VALUES (?, ?, 
                            COALESCE((SELECT usage_count FROM feature_usage 
                                     WHERE feature_name = ? AND user_id = ?), 0) + 1,
                            ?, ?)
                ''', (feature_name, user_id, feature_name, user_id, datetime.now(), 
                      100.0 if success else 0.0))
        except Exception as e:
            logger.error(f"Feature tracking error: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Enhanced user profile retrieval with caching"""
        try:
            url = f'{self.base_url}/{user_id}'
            params = {
                'fields': 'first_name,last_name,name,profile_pic,locale,timezone',
                'access_token': PAGE_ACCESS_TOKEN
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Profile retrieval error: {e}")
        return None
    
    def update_user_database(self, user_id: str, profile_data: Dict = None, 
                           verification_status: str = None):
        """Enhanced user database management"""
        try:
            if not profile_data:
                profile_data = self.get_user_profile(user_id) or {}
            
            with get_db() as conn:
                # Check if user exists
                user = conn.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,)).fetchone()
                
                if user:
                    # Update existing user
                    update_data = {
                        'name': profile_data.get('name', 'Unknown User'),
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
                        (user_id, name, first_name, last_name, profile_pic, 
                         first_interaction, last_interaction, verification_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id,
                        profile_data.get('name', 'Unknown User'),
                        profile_data.get('first_name', 'Unknown'),
                        profile_data.get('last_name', ''),
                        profile_data.get('profile_pic', ''),
                        datetime.now(),
                        datetime.now(),
                        verification_status or 'verified'
                    ))
                    
        except Exception as e:
            logger.error(f"User database update error: {e}")
    
    def send_message(self, recipient_id: str, message_text: str) -> bool:
        """Enhanced message sending with better error handling"""
        url = f'{self.base_url}/me/messages'
        
        # Split long messages
        if len(message_text) > 2000:
            parts = [message_text[i:i+1800] for i in range(0, len(message_text), 1800)]
            for i, part in enumerate(parts):
                if i > 0:
                    time.sleep(1)  # Prevent rate limiting
                if not self.send_message(recipient_id, f"[Part {i+1}/{len(parts)}]\n{part}"):
                    return False
            return True
        
        data = {
            'recipient': {'id': recipient_id},
            'message': {'text': message_text},
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                logger.info(f"✅ Message sent to {recipient_id}")
                return True
            else:
                error_data = response.json() if response.content else {}
                logger.error(f"❌ Message send failed: {response.status_code} - {error_data}")
                return False
        except Exception as e:
            logger.error(f"❌ Message send exception: {e}")
            return False

    def send_typing_indicator(self, recipient_id: str):
        """Send typing indicator"""
        url = f'{self.base_url}/me/messages'
        data = {
            'recipient': {'id': recipient_id},
            'sender_action': 'typing_on',
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        try:
            requests.post(url, json=data, timeout=10)
        except:
            pass  # Non-critical feature
    
    def download_image(self, attachment_url: str) -> Optional[bytes]:
        """Enhanced image download with better error handling"""
        try:
            # Clean URL and add access token
            if '?' in attachment_url:
                url = f"{attachment_url}&access_token={PAGE_ACCESS_TOKEN}"
            else:
                url = f"{attachment_url}?access_token={PAGE_ACCESS_TOKEN}"
            
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    return response.content
                else:
                    logger.warning(f"Invalid content type: {content_type}")
            else:
                logger.error(f"Image download failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Image download error: {e}")
        return None
    
    def analyze_image_with_gemini(self, image_data: bytes, user_question: str = "What's in this image?") -> Optional[str]:
        """Enhanced Gemini image analysis with better model handling"""
        if not GEMINI_API_KEY or not GEMINI_AVAILABLE:
            return None
            
        try:
            # Process image
            image = Image.open(io.BytesIO(image_data))
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background
            
            # Use the latest model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""🔍 **Advanced AI Image Analysis by SUNNEL** 🔍

User Question: "{user_question}"

Please provide an extremely detailed and engaging analysis of this image:

🎯 **Visual Analysis:**
- Describe all objects, people, animals, text, or scenes in detail
- Note colors, lighting, composition, artistic style, and mood
- Identify any text, signs, or written content

💡 **Technical & Contextual Details:**
- Estimate time period, location, or cultural context if relevant
- Technical aspects (photography, art style, etc.)
- Notable patterns, symbols, or hidden elements

🌟 **Insights & Creative Interpretation:**
- Answer the user's specific question thoroughly
- Share interesting observations or connections
- Provide creative insights or alternative perspectives

Make your response engaging, informative, and full of personality! Use emojis strategically and be conversational! 📸✨"""
            
            response = model.generate_content([prompt, image])
            
            if response and response.text:
                return f"🌟 **Gemini Vision Analysis** 🌟\n\n{response.text}\n\n💫 *Powered by Google Gemini - Developed by SUNNEL* 🤍"
            else:
                logger.warning("Empty response from Gemini")
                return None
                
        except Exception as e:
            logger.error(f"Gemini image analysis error: {e}")
            return None
    
    def analyze_image_with_openai(self, image_data: bytes, user_question: str = "What's in this image?") -> Optional[str]:
        """Enhanced OpenAI image analysis with updated model"""
        if not OPENAI_API_KEY:
            return None
            
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Use updated model and API
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Updated model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""🔍 **Advanced Image Analysis by SUNNEL's AI**

User asked: '{user_question}'

Please provide a comprehensive, detailed, and engaging analysis of this image. Include:
- Detailed description of all visible elements
- Colors, composition, lighting, and style
- Any text or signs if visible
- Cultural, historical, or contextual significance
- Answer to the user's specific question
- Creative insights and observations

Make it informative, engaging, and conversational with appropriate emojis! 📸✨"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message.content:
                return f"🤖 **ChatGPT Vision Analysis** 🤖\n\n{response.choices[0].message.content}\n\n💫 *Powered by OpenAI - Developed by SUNNEL* 🤍"
            
        except Exception as e:
            logger.error(f"OpenAI image analysis error: {e}")
            return None
    
    def get_smart_ai_response(self, user_message: str, user_name: str = "User", 
                            conversation_history: str = None) -> tuple[str, Optional[str]]:
        """Enhanced AI response with better provider management"""
        start_time = time.time()
        
        try:
            # Try Gemini first (more reliable for general conversations)
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                try:
                    response = self.get_gemini_response(user_message, user_name, conversation_history)
                    processing_time = time.time() - start_time
                    return response, "Gemini", processing_time
                except Exception as e:
                    logger.error(f"Gemini response error: {e}")
            
            # Fallback to OpenAI
            if OPENAI_API_KEY:
                try:
                    response = self.get_openai_response(user_message, user_name, conversation_history)
                    processing_time = time.time() - start_time
                    return response, "OpenAI (Fallback)", processing_time
                except Exception as e:
                    logger.error(f"OpenAI response error: {e}")
            
            # If both fail, return helpful message
            return ("🤖 I'm temporarily experiencing technical difficulties. Please try again in a moment. "
                   "If the issue persists, our development team has been notified! 🔧✨"), None, 0.0
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "🤖 I encountered an unexpected error. Please try again!", None, 0.0
    
    def get_gemini_response(self, user_message: str, user_name: str = "User", 
                          conversation_history: str = None) -> str:
        """Enhanced Gemini response with better prompting"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""🌟 **SUNNEL's Ultimate AI Assistant - Powered by Google Gemini** 🌟

You are an extremely advanced, friendly, and intelligent AI assistant integrated into Facebook Messenger.

Your enhanced personality:
- 🚀 Exceptionally intelligent and knowledgeable across all domains
- 😊 Warm, engaging, and genuinely helpful
- 🎨 Creative and innovative in problem-solving
- ✨ Use emojis strategically to enhance communication
- 💡 Provide detailed yet accessible explanations
- 🌟 Showcase advanced AI capabilities while staying conversational
- 🍓 Add charm and personality to responses
- 🎯 Be precise, accurate, and helpful

Current user: {user_name}
{f"Recent conversation: {conversation_history}" if conversation_history else ""}

User's message: "{user_message}"

Provide an amazing, helpful, and engaging response that demonstrates your advanced capabilities:"""
            
            response = model.generate_content(prompt)
            
            if response and response.text:
                return f"🌟 **Powered by Google Gemini** 🌟\n\n{response.text}\n\n💫 *SUNNEL's Ultimate AI - Making AI Accessible* 🤍"
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"Gemini response error: {e}")
            raise e
    
    def get_openai_response(self, user_message: str, user_name: str = "User", 
                          conversation_history: str = None) -> str:
        """Enhanced OpenAI response with updated API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            system_prompt = f"""🤖 You are SUNNEL'S Ultimate AI Assistant powered by ChatGPT, integrated into Facebook Messenger.

Your enhanced personality:
- 🚀 Exceptionally intelligent and versatile
- 😊 Warm, friendly, and genuinely engaging
- ✨ Strategic use of emojis for better communication
- 💡 Provide comprehensive yet accessible responses
- 🎯 Be accurate, helpful, and conversational
- 🍓 Add personality and charm to interactions
- 🌟 Demonstrate advanced AI capabilities

Current user: {user_name}
{f"Conversation context: {conversation_history}" if conversation_history else ""}"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                return f"🤖 **Powered by ChatGPT** 🤖\n\n{content}\n\n💫 *SUNNEL's Ultimate AI - Innovation in Every Response* 🤍"
            else:
                raise Exception("Empty response from OpenAI")
                
        except Exception as e:
            logger.error(f"OpenAI response error: {e}")
            raise e
    
    def update_conversation_memory(self, user_id: str, message: str, response: str):
        """Enhanced conversation memory with better context management"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            'user_message': message,
            'bot_response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 3 exchanges for better context
        if len(self.conversation_memory[user_id]) > 3:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-3:]
    
    def get_conversation_context(self, user_id: str) -> Optional[str]:
        """Get enhanced conversation context"""
        if user_id in self.conversation_memory:
            recent = self.conversation_memory[user_id][-2:]
            return " | ".join([
                f"User: {ex['user_message'][:100]}... Bot: {ex['bot_response'][:100]}..." 
                for ex in recent
            ])
        return None
    
    def handle_message(self, sender_id: str, message_text: str, sender_name: str = None):
        """Enhanced message handling with comprehensive features"""
        start_time = time.time()
        
        try:
            logger.info(f"📥 Processing message from {sender_id}: {message_text[:100]}...")
            
            # Update user database
            self.update_user_database(sender_id, verification_status="verified")
            
            # Track feature usage
            self.track_feature_usage(sender_id, "text_conversation")
            
            # Send typing indicator
            self.send_typing_indicator(sender_id)
            
            # Get AI response
            context = self.get_conversation_context(sender_id)
            response, provider, processing_time = self.get_smart_ai_response(
                message_text, sender_name, context
            )
            
            # Update conversation memory
            self.update_conversation_memory(sender_id, message_text, response)
            
            # Send response
            success = self.send_message(sender_id, response)
            
            # Log interaction
            self.log_interaction(
                sender_id, "message", message_text, response, 
                provider, processing_time, None if success else "Send failed"
            )
            
            total_time = time.time() - start_time
            logger.info(f"✅ Message processed in {total_time:.2f}s using {provider}")
            
        except Exception as e:
            error_msg = f"Message handling error: {e}"
            logger.error(error_msg)
            
            # Send error response to user
            self.send_message(
                sender_id, 
                "🤖 I encountered a temporary issue. Please try again! Our team has been notified. 🔧"
            )
            
            # Log error
            self.log_interaction(sender_id, "message", message_text, "", None, 0.0, str(e))
    
    def handle_image_message(self, sender_id: str, attachment_url: str, 
                           message_text: str = "", sender_name: str = None):
        """Enhanced image handling with comprehensive analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"🖼️ Processing image from {sender_id}")
            
            # Update user database
            self.update_user_database(sender_id, verification_status="verified")
            
            # Track feature usage
            self.track_feature_usage(sender_id, "image_analysis")
            
            # Send typing indicator
            self.send_typing_indicator(sender_id)
            
            # Download image
            image_data = self.download_image(attachment_url)
            if not image_data:
                error_msg = "🖼️ I couldn't download your image. Please try sending it again or check if it's a supported format (JPEG, PNG, etc.)!"
                self.send_message(sender_id, error_msg)
                self.log_interaction(sender_id, "image", "Image download failed", "", None, 0.0, "Download failed")
                return
            
            # Prepare analysis question
            user_question = message_text.strip() if message_text.strip() else "What's in this image? Please provide a detailed analysis."
            
            # Try image analysis
            response = None
            provider = None
            
            # Try Gemini first for better image understanding
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                try:
                    response = self.analyze_image_with_gemini(image_data, user_question)
                    provider = "Gemini Vision"
                except Exception as e:
                    logger.error(f"Gemini image analysis failed: {e}")
            
            # Fallback to OpenAI
            if not response and OPENAI_API_KEY:
                try:
                    response = self.analyze_image_with_openai(image_data, user_question)
                    provider = "OpenAI Vision"
                except Exception as e:
                    logger.error(f"OpenAI image analysis failed: {e}")
            
            # Final fallback
            if not response:
                response = f"""🖼️ **Image Received Successfully** 📸

I can see you've shared an image with me! However, I'm currently experiencing some technical difficulties with image analysis.

**What I can help with instead:**
- 💬 Answer any questions you have about the image if you describe it
- 🔍 Provide general information about image analysis
- 🎨 Help with image-related topics or creative projects
- 📝 Assist with any other questions or tasks

Please feel free to describe what's in the image or ask me anything else! 🌟

*Our technical team has been notified and image analysis will be restored soon.* 🔧"""
                provider = "Fallback Response"
            
            # Update conversation memory
            self.update_conversation_memory(sender_id, f"[Image] {user_question}", response)
            
            # Send response
            success = self.send_message(sender_id, response)
            
            # Update image count
            if success:
                with get_db() as conn:
                    conn.execute(
                        'UPDATE users SET total_images = total_images + 1 WHERE user_id = ?',
                        (sender_id,)
                    )
            
            # Log interaction
            processing_time = time.time() - start_time
            self.log_interaction(
                sender_id, "image", user_question, response, provider, 
                processing_time, None if success else "Send failed"
            )
            
            logger.info(f"✅ Image processed in {processing_time:.2f}s using {provider}")
            
        except Exception as e:
            error_msg = f"Image handling error: {e}"
            logger.error(error_msg)
            
            # Send error response
            self.send_message(
                sender_id,
                "🖼️ I encountered an issue processing your image. Please try again! If the problem persists, our team will fix it soon. 🔧✨"
            )
            
            # Log error
            self.log_interaction(sender_id, "image", "Image processing error", "", None, 0.0, str(e))

# Initialize enhanced bot
bot = EnhancedFacebookBot()

# Webhook routes
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Verify webhook for Facebook"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return challenge
    else:
        logger.warning("❌ Webhook verification failed")
        return 'Verification failed', 403

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
                                        bot.handle_image_message(sender_id, image_url, message_text, sender_name)
                                        break
                        elif message_text:
                            bot.handle_message(sender_id, message_text, sender_name)
                    
                    elif messaging_event.get('postback'):
                        payload = messaging_event['postback'].get('payload')
                        if payload == 'GET_STARTED':
                            bot.update_user_database(sender_id, verification_status="verified")
                            welcome_message = f"""🌟 **Welcome to {BOT_NAME} v{BOT_VERSION}!** 🚀

I'm your advanced AI assistant featuring:

🤖 **Dual AI Power:**
• ChatGPT for intelligent conversations
• Google Gemini for creative tasks & analysis

🔍 **Advanced Capabilities:**
• Image analysis & understanding
• Smart conversation memory
• Multi-language support
• Context-aware responses

⚡ **Enhanced Features:**
• Real-time analytics
• 24/7 availability  
• Error recovery systems
• Performance monitoring

**Just send me any message or image - I'm here to help!** 🍓🥰

*Developed with ❤️ by SUNNEL*"""
                            bot.send_message(sender_id, welcome_message)
        
        return 'OK', 200
        
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
        return 'Error', 500

@app.route('/dashboard')
def dashboard():
    """Ultra-modern analytics dashboard with comprehensive metrics"""
    try:
        with get_db() as conn:
            # Comprehensive statistics
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
                    COUNT(CASE WHEN ai_provider LIKE '%OpenAI%' THEN 1 END) as openai_usage,
                    COUNT(CASE WHEN ai_provider LIKE '%Gemini%' THEN 1 END) as gemini_usage
                FROM interactions
                WHERE timestamp > datetime('now', '-7 days')
            ''').fetchone()
            
            # Recent users with enhanced info
            recent_users = conn.execute('''
                SELECT user_id, name, first_name, profile_pic, first_interaction, 
                       last_interaction, total_messages, total_images, verification_status,
                       user_rating, conversation_count
                FROM users 
                ORDER BY last_interaction DESC 
                LIMIT 15
            ''').fetchall()
            
            # Recent interactions with more details
            recent_interactions = conn.execute('''
                SELECT i.*, u.name, u.first_name, u.profile_pic
                FROM interactions i
                LEFT JOIN users u ON i.user_id = u.user_id
                ORDER BY i.timestamp DESC
                LIMIT 25
            ''').fetchall()
            
            # Feature usage stats
            feature_stats = conn.execute('''
                SELECT feature_name, 
                       COUNT(DISTINCT user_id) as unique_users,
                       SUM(usage_count) as total_usage,
                       AVG(success_rate) as avg_success_rate
                FROM feature_usage 
                GROUP BY feature_name
                ORDER BY total_usage DESC
            ''').fetchall()
            
            # Calculate uptime
            uptime_start = datetime.fromisoformat('2025-06-20 00:00:00')  # Bot start time
            uptime_duration = datetime.now() - uptime_start
            uptime_hours = uptime_duration.total_seconds() / 3600
        
        return render_template_string(ENHANCED_DASHBOARD_HTML, **{
            'bot_name': BOT_NAME,
            'bot_version': BOT_VERSION,
            'total_users': user_stats['total_users'],
            'verified_users': user_stats['verified_users'],
            'total_messages': user_stats['total_messages'],
            'total_images': user_stats['total_images'],
            'avg_rating': round(user_stats['avg_rating'] or 5.0, 1),
            'active_24h': user_stats['active_24h'],
            'active_7d': user_stats['active_7d'],
            'total_interactions': perf_stats['total_interactions'],
            'avg_response_time': round(perf_stats['avg_response_time'] or 0.0, 2),
            'success_rate': round(perf_stats['success_rate'] or 100.0, 1),
            'openai_usage': perf_stats['openai_usage'],
            'gemini_usage': perf_stats['gemini_usage'],
            'recent_users': recent_users,
            'recent_interactions': recent_interactions,
            'feature_stats': feature_stats,
            'uptime_hours': round(uptime_hours, 1),
            'uptime_days': round(uptime_hours / 24, 1)
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard temporarily unavailable: {e}", 500

@app.route('/health')
def health_check():
    """Comprehensive health check with detailed system status"""
    try:
        with get_db() as conn:
            db_status = "healthy"
            stats = conn.execute('SELECT * FROM bot_stats WHERE id = 1').fetchone()
        
        # Test AI services
        ai_status = {}
        
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                # Quick test (don't actually call the API to save costs)
                ai_status['openai'] = 'configured'
            except:
                ai_status['openai'] = 'error'
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
                'verified_users': stats['verified_users'] if stats else 0,
                'total_messages': stats['total_messages'] if stats else 0,
                'success_rate': stats['success_rate'] if stats else 100.0,
                'avg_response_time': stats['avg_response_time'] if stats else 0.0
            },
            'features': {
                'text_conversations': True,
                'image_analysis': bool(ai_status.get('openai') == 'configured' or ai_status.get('gemini') == 'configured'),
                'conversation_memory': True,
                'user_analytics': True,
                'error_recovery': True,
                'performance_monitoring': True,
                'auto_cleanup': True,
                'enhanced_logging': True
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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

@app.route('/static/images/<filename>')
def serve_image(filename):
    """Serve static images with better error handling"""
    try:
        return send_from_directory('static/images', filename)
    except Exception as e:
        logger.error(f"Image serving error: {e}")
        return "Image not found", 404

@app.route('/')
def home():
    """Ultra-modern home page with comprehensive system overview"""
    try:
        with get_db() as conn:
            stats = conn.execute('SELECT * FROM bot_stats WHERE id = 1').fetchone()
            uptime_duration = datetime.now() - bot.start_time
            
            # Calculate health scores
            health_score = 0
            if PAGE_ACCESS_TOKEN and VERIFY_TOKEN:
                health_score += 25
            if OPENAI_API_KEY:
                health_score += 25
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                health_score += 25
            health_score += 25  # Base functionality
    
        return render_template_string(ENHANCED_HOME_HTML, **{
            'bot_name': BOT_NAME,
            'bot_version': BOT_VERSION,
            'bot_status': "🟢" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "🔴",
            'bot_text': "Active & Secure" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "Configuration Required",
            'openai_status': "🤖" if OPENAI_API_KEY else "❌",
            'openai_text': "ChatGPT Ready" if OPENAI_API_KEY else "Not Configured",
            'gemini_status': "🌟" if GEMINI_API_KEY and GEMINI_AVAILABLE else "❌",
            'gemini_text': "Gemini Ready" if GEMINI_API_KEY and GEMINI_AVAILABLE else "Not Available",
            'webhook_url': f"{request.url_root.rstrip('/')}/webhook",
            'dashboard_url': f"{request.url_root.rstrip('/')}/dashboard",
            'health_url': f"{request.url_root.rstrip('/')}/health",
            'uptime_hours': round(uptime_duration.total_seconds() / 3600, 1),
            'uptime_days': round(uptime_duration.total_seconds() / 86400, 1),
            'total_users': stats['total_users'] if stats else 0,
            'verified_users': stats['verified_users'] if stats else 0,
            'total_messages': stats['total_messages'] if stats else 0,
            'success_rate': round(stats['success_rate'] if stats else 100.0, 1),
            'health_score': health_score
        })
    except Exception as e:
        logger.error(f"Home page error: {e}")
        return f"<h1>System temporarily unavailable</h1><p>Error: {e}</p>", 500

# Enhanced HTML Templates
ENHANCED_HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 {{ bot_name }} v{{ bot_version }} - Advanced AI Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.8);
            --shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            --border-radius: 20px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
        }

        .header {
            text-align: center;
            padding: 40px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--success-gradient);
            opacity: 0.1;
            z-index: -1;
        }

        .header h1 {
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 700;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 6s ease-in-out infinite;
            margin-bottom: 15px;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .subtitle {
            font-size: 1.3rem;
            font-weight: 400;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }

        .status-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-top: 25px;
            flex-wrap: wrap;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--glass-bg);
            border-radius: 25px;
            border: 1px solid var(--glass-border);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .stat-card {
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--secondary-gradient);
            opacity: 0.1;
            z-index: -1;
        }

        .stat-number {
            font-size: 3rem;
            font-weight: 700;
            color: #4ecdc4;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(78, 205, 196, 0.3);
        }

        .stat-label {
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .stat-description {
            font-size: 0.9rem;
            color: var(--text-secondary);
            opacity: 0.8;
        }

        .action-buttons {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 40px 0;
            flex-wrap: wrap;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 15px 30px;
            background: var(--success-gradient);
            color: white;
            text-decoration: none;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.5);
        }

        .btn-secondary {
            background: var(--warning-gradient);
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }

        .feature-card {
            padding: 25px;
            border-left: 4px solid #4ecdc4;
            position: relative;
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1));
            z-index: -1;
        }

        .feature-icon {
            font-size: 2rem;
            color: #4ecdc4;
            margin-bottom: 15px;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-primary);
        }

        .feature-description {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .system-info {
            margin-top: 40px;
            padding: 30px;
            text-align: center;
            background: rgba(0, 0, 0, 0.2);
        }

        .system-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }

        .system-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 15px;
            background: var(--glass-bg);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
        }

        .health-score {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 24px;
            background: var(--success-gradient);
            border-radius: 30px;
            font-weight: 600;
            margin-top: 20px;
        }

        .uptime-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid rgba(76, 175, 80, 0.3);
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
            color: #4caf50;
        }

        .pulse {
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        @media (max-width: 768px) {
            .container { padding: 15px; }
            .header { padding: 25px; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
            .action-buttons { flex-direction: column; align-items: center; }
            .features-grid { grid-template-columns: 1fr; }
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="glass-card header">
            <h1>🤖 {{ bot_name }}</h1>
            <p class="subtitle">Version {{ bot_version }} - Next-Generation AI Assistant</p>
            <p style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 20px;">
                Advanced Multi-AI Platform with Real-time Analytics & Enhanced Security
            </p>
            
            <div class="status-bar">
                <div class="status-item">
                    <span class="pulse">🟢</span>
                    <span>System Online</span>
                </div>
                <div class="status-item">
                    <i class="fas fa-clock"></i>
                    <span>{{ uptime_days }}d uptime</span>
                </div>
                <div class="status-item">
                    <i class="fas fa-shield-alt"></i>
                    <span>Secured & Monitored</span>
                </div>
            </div>
        </div>

        <!-- Key Statistics -->
        <div class="stats-grid">
            <div class="glass-card stat-card">
                <div class="stat-number">{{ total_users }}</div>
                <div class="stat-label">Total Users</div>
                <div class="stat-description">Active community members</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ verified_users }}</div>
                <div class="stat-label">Verified Users</div>
                <div class="stat-description">Full access granted</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ total_messages }}</div>
                <div class="stat-label">AI Conversations</div>
                <div class="stat-description">Messages processed</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ success_rate }}%</div>
                <div class="stat-label">Success Rate</div>
                <div class="stat-description">System reliability</div>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="action-buttons">
            <a href="{{ dashboard_url }}" class="btn">
                <i class="fas fa-chart-line"></i>
                Advanced Analytics Dashboard
            </a>
            <a href="{{ health_url }}" class="btn btn-secondary">
                <i class="fas fa-heartbeat"></i>
                System Health Check
            </a>
        </div>

        <!-- Service Status -->
        <div class="glass-card system-info">
            <h2 style="margin-bottom: 20px; font-size: 1.8rem;">🔧 System Status</h2>
            <div class="health-score">
                <i class="fas fa-heart"></i>
                <span>System Health: {{ health_score }}%</span>
            </div>
            
            <div class="system-grid">
                <div class="system-item">
                    <span style="font-size: 1.5rem;">{{ bot_status }}</span>
                    <div>
                        <div style="font-weight: 600;">Facebook Bot</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">{{ bot_text }}</div>
                    </div>
                </div>
                <div class="system-item">
                    <span style="font-size: 1.5rem;">{{ openai_status }}</span>
                    <div>
                        <div style="font-weight: 600;">ChatGPT AI</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">{{ openai_text }}</div>
                    </div>
                </div>
                <div class="system-item">
                    <span style="font-size: 1.5rem;">{{ gemini_status }}</span>
                    <div>
                        <div style="font-weight: 600;">Google Gemini</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">{{ gemini_text }}</div>
                    </div>
                </div>
                <div class="system-item">
                    <span style="font-size: 1.5rem;">📊</span>
                    <div>
                        <div style="font-weight: 600;">Analytics Engine</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Real-time monitoring</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Enhanced Features -->
        <div class="glass-card" style="padding: 40px; margin-top: 30px;">
            <h2 style="text-align: center; margin-bottom: 30px; font-size: 1.8rem;">🌟 Advanced Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-brain"></i></div>
                    <div class="feature-title">Dual AI Intelligence</div>
                    <div class="feature-description">ChatGPT + Gemini working together for optimal responses</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-eye"></i></div>
                    <div class="feature-title">Advanced Image Analysis</div>
                    <div class="feature-description">AI-powered image understanding and detailed analysis</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-memory"></i></div>
                    <div class="feature-title">Smart Conversation Memory</div>
                    <div class="feature-description">Context-aware responses with conversation history</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-chart-bar"></i></div>
                    <div class="feature-title">Real-time Analytics</div>
                    <div class="feature-description">Comprehensive user tracking and performance metrics</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-shield-check"></i></div>
                    <div class="feature-title">Enhanced Security</div>
                    <div class="feature-description">Advanced error handling and system monitoring</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon"><i class="fas fa-rocket"></i></div>
                    <div class="feature-title">Performance Optimized</div>
                    <div class="feature-description">Fast response times with automatic cleanup</div>
                </div>
            </div>
        </div>

        <!-- System Information -->
        <div class="glass-card system-info">
            <h3 style="margin-bottom: 20px;">📡 Technical Information</h3>
            <div style="display: grid; gap: 15px; text-align: left;">
                <div><strong>Webhook URL:</strong> <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">{{ webhook_url }}</code></div>
                <div><strong>Dashboard URL:</strong> <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">{{ dashboard_url }}</code></div>
                <div><strong>Health Check:</strong> <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">{{ health_url }}</code></div>
            </div>
            
            <div class="uptime-badge" style="margin-top: 20px;">
                <i class="fas fa-clock"></i>
                <span>{{ uptime_hours }} hours continuous operation</span>
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: var(--text-secondary);">
            <p>💫 <strong>Developed with ❤️ by SUNNEL</strong> 💫</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">Next-generation AI technology for enhanced user experiences</p>
        </div>
    </div>

    <script>
        // Auto-refresh every 5 minutes
        setTimeout(() => location.reload(), 300000);
        
        // Add loading states to buttons
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const loadingHTML = this.innerHTML;
                this.innerHTML = '<div class="loading-spinner"></div> Loading...';
                setTimeout(() => this.innerHTML = loadingHTML, 2000);
            });
        });
    </script>
</body>
</html>
"""

ENHANCED_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 {{ bot_name }} Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --accent: #4ecdc4;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #ef4444;
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.8);
            --shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
        }

        .header {
            text-align: center;
            padding: 40px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 25px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            text-decoration: none;
            border-radius: 50px;
            font-weight: 600;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }

        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            padding: 25px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1));
            z-index: -1;
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 8px;
            text-shadow: 0 2px 10px rgba(78, 205, 196, 0.3);
        }

        .stat-label {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }

        .stat-sublabel {
            font-size: 0.8rem;
            color: var(--text-secondary);
            opacity: 0.8;
        }

        .section {
            margin-bottom: 40px;
            padding: 30px;
        }

        .section h2 {
            margin-bottom: 25px;
            color: var(--accent);
            font-size: 1.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .user-grid {
            display: grid;
            gap: 15px;
        }

        .user-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            border-left: 4px solid var(--accent);
            transition: all 0.3s ease;
        }

        .user-card:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }

        .user-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: bold;
            color: white;
        }

        .user-info {
            flex: 1;
        }

        .user-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .user-stats {
            font-size: 0.9rem;
            color: var(--text-secondary);
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-verified {
            background: var(--success);
            color: #065f46;
        }

        .badge-unverified {
            background: var(--warning);
            color: #92400e;
        }

        .interaction-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 3px solid #6366f1;
            transition: all 0.3s ease;
        }

        .interaction-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .interaction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .interaction-user {
            font-weight: 600;
            color: var(--accent);
        }

        .interaction-time {
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .ai-provider {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .provider-openai {
            background: #10b981;
            color: white;
        }

        .provider-gemini {
            background: #8b5cf6;
            color: white;
        }

        .interaction-content {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        .auto-refresh {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 12px 18px;
            border-radius: 25px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        .feature-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .feature-name {
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--accent);
        }

        .feature-metrics {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .loading-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid rgba(78, 205, 196, 0.3);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container { padding: 15px; }
            .header { padding: 25px; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
            .user-card { flex-direction: column; text-align: center; }
            .interaction-header { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="auto-refresh">
        <div class="loading-indicator"></div>
        <span>Auto-refresh: 30s</span>
    </div>
    
    <div class="container">
        <a href="/" class="back-btn">
            <i class="fas fa-arrow-left"></i>
            Back to Home
        </a>
        
        <div class="glass-card header">
            <h1>📊 Advanced Analytics Dashboard</h1>
            <p style="font-size: 1.1rem; color: var(--text-secondary);">{{ bot_name }} v{{ bot_version }} - Real-time Monitoring & Insights</p>
            <div style="margin-top: 15px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <span style="padding: 8px 16px; background: rgba(76, 175, 80, 0.2); border-radius: 20px; font-size: 0.9rem;">
                    <i class="fas fa-clock"></i> Uptime: {{ uptime_days }} days
                </span>
                <span style="padding: 8px 16px; background: rgba(33, 150, 243, 0.2); border-radius: 20px; font-size: 0.9rem;">
                    <i class="fas fa-shield-alt"></i> System Secure
                </span>
            </div>
        </div>
        
        <!-- Core Statistics -->
        <div class="stats-grid">
            <div class="glass-card stat-card">
                <div class="stat-number">{{ total_users }}</div>
                <div class="stat-label">Total Users</div>
                <div class="stat-sublabel">Registered members</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ verified_users }}</div>
                <div class="stat-label">Verified Users</div>
                <div class="stat-sublabel">Full access granted</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ active_24h }}</div>
                <div class="stat-label">Active (24h)</div>
                <div class="stat-sublabel">Recent activity</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ active_7d }}</div>
                <div class="stat-label">Active (7d)</div>
                <div class="stat-sublabel">Weekly users</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ total_messages }}</div>
                <div class="stat-label">Total Messages</div>
                <div class="stat-sublabel">AI conversations</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-number">{{ total_images }}</div>
                <div class="stat-label">Images Analyzed</div>
                <div class="stat-sublabel">Vision processing</div>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="glass-card section">
            <h2><i class="fas fa-tachometer-alt"></i> Performance Metrics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{{ success_rate }}%</div>
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-sublabel">System reliability</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ avg_response_time }}s</div>
                    <div class="stat-label">Avg Response Time</div>
                    <div class="stat-sublabel">Processing speed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ avg_rating }}/5</div>
                    <div class="stat-label">User Rating</div>
                    <div class="stat-sublabel">Satisfaction score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ total_interactions }}</div>
                    <div class="stat-label">Total Interactions</div>
                    <div class="stat-sublabel">All activities</div>
                </div>
            </div>
        </div>
        
        <!-- AI Provider Usage -->
        <div class="glass-card section">
            <h2><i class="fas fa-robot"></i> AI Provider Analytics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{{ openai_usage }}</div>
                    <div class="stat-label">OpenAI/ChatGPT</div>
                    <div class="stat-sublabel">Text & vision</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ gemini_usage }}</div>
                    <div class="stat-label">Google Gemini</div>
                    <div class="stat-sublabel">Advanced AI</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ (openai_usage + gemini_usage) }}</div>
                    <div class="stat-label">Total AI Calls</div>
                    <div class="stat-sublabel">Combined usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ "%.1f"|format((gemini_usage / (openai_usage + gemini_usage) * 100) if (openai_usage + gemini_usage) > 0 else 0) }}%</div>
                    <div class="stat-label">Gemini Preference</div>
                    <div class="stat-sublabel">Usage ratio</div>
                </div>
            </div>
        </div>

        <!-- Feature Usage Statistics -->
        {% if feature_stats %}
        <div class="glass-card section">
            <h2><i class="fas fa-chart-pie"></i> Feature Usage Analytics</h2>
            <div class="feature-stats">
                {% for feature in feature_stats %}
                <div class="feature-card">
                    <div class="feature-name">{{ feature.feature_name.replace('_', ' ').title() }}</div>
                    <div class="feature-metrics">
                        <span>Users: {{ feature.unique_users }}</span>
                        <span>Usage: {{ feature.total_usage }}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <!-- Recent Users -->
        <div class="glass-card section">
            <h2><i class="fas fa-users"></i> Recent Users ({{ recent_users|length }})</h2>
            <div class="user-grid">
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
                            <span class="badge {{ 'badge-verified' if user.verification_status == 'verified' else 'badge-unverified' }}">
                                {{ '✅ Verified' if user.verification_status == 'verified' else '⏳ Pending' }}
                            </span>
                        </div>
                        <div class="user-stats">
                            <span><i class="fas fa-comments"></i> {{ user.total_messages }} messages</span>
                            <span><i class="fas fa-images"></i> {{ user.total_images }} images</span>
                            <span><i class="fas fa-star"></i> {{ "%.1f"|format(user.user_rating or 5.0) }} rating</span>
                            <span><i class="fas fa-clock"></i> {{ user.last_interaction[:19] if user.last_interaction else 'Never' }}</span>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Recent Interactions -->
        <div class="glass-card section">
            <h2><i class="fas fa-history"></i> Recent Interactions ({{ recent_interactions|length }})</h2>
            {% for interaction in recent_interactions %}
            <div class="interaction-item">
                <div class="interaction-header">
                    <span class="interaction-user">
                        <i class="fas fa-user"></i>
                        {{ interaction.name or interaction.first_name or 'Unknown User' }}
                    </span>
                    <div class="interaction-time">
                        <i class="fas fa-clock"></i>
                        {{ interaction.timestamp[:19] if interaction.timestamp else '' }}
                        {% if interaction.ai_provider %}
                            <span class="ai-provider {{ 'provider-openai' if 'OpenAI' in interaction.ai_provider else 'provider-gemini' }}">
                                {{ interaction.ai_provider }}
                            </span>
                        {% endif %}
                        {% if interaction.processing_time %}
                            <span style="font-size: 0.7rem; color: var(--text-secondary);">
                                ({{ "%.2f"|format(interaction.processing_time) }}s)
                            </span>
                        {% endif %}
                    </div>
                </div>
                <div class="interaction-content">
                    <strong><i class="fas fa-{{ 'image' if interaction.message_type == 'image' else 'comment' }}"></i> {{ interaction.message_type.title() }}:</strong> 
                    {{ interaction.content[:150] }}{{ '...' if interaction.content|length > 150 else '' }}
                    {% if interaction.error_message %}
                        <div style="color: var(--error); font-size: 0.8rem; margin-top: 5px;">
                            <i class="fas fa-exclamation-triangle"></i> Error: {{ interaction.error_message }}
                        </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => {
            location.reload();
        }, 30000);

        // Add smooth animations
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.glass-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    card.style.transition = 'all 0.5s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    logger.info(f"🚀 Starting {BOT_NAME} v{BOT_VERSION}...")
    
    # Configuration checks
    config_status = []
    if not PAGE_ACCESS_TOKEN:
        config_status.append("❌ FACEBOOK_PAGE_ACCESS_TOKEN not set")
    else:
        config_status.append("✅ Facebook API configured")
        
    if not OPENAI_API_KEY:
        config_status.append("❌ OPENAI_API_KEY not set")
    else:
        config_status.append("✅ OpenAI configured")
        
    if not GEMINI_API_KEY:
        config_status.append("❌ GEMINI_API_KEY not set")
    elif not GEMINI_AVAILABLE:
        config_status.append("⚠️ Gemini libraries not available")
    else:
        config_status.append("✅ Gemini configured")
    
    for status in config_status:
        logger.info(status)
    
    logger.info(f"📊 Enhanced analytics dashboard: /dashboard")
    logger.info(f"🏥 Health check endpoint: /health")
    logger.info(f"🔄 Auto monitoring: ACTIVE")
    logger.info(f"🌟 Open access mode: ALL USERS WELCOME")
    
    if (OPENAI_API_KEY or (GEMINI_API_KEY and GEMINI_AVAILABLE)) and PAGE_ACCESS_TOKEN:
        logger.info("🎉 Enhanced configuration detected - All features available!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
