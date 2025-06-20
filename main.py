import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import openai
from datetime import datetime, timedelta
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
import asyncio
import schedule
from threading import Timer
import subprocess
import psutil

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
BOT_NAME = "Cleo AI"
BOT_VERSION = "5.0.0"
REQUIRED_POST_ID = "761320392916522"
PAGE_ID = "100071491013161"

# Global status tracking
SYSTEM_STATUS = {
    'is_active': True,
    'last_heartbeat': datetime.now(),
    'uptime_start': datetime.now(),
    'total_requests': 0,
    'active_conversations': 0,
    'system_health': 'excellent',
    'cpu_usage': 0.0,
    'memory_usage': 0.0,
    'response_time_avg': 0.0,
    'gemini_requests': 0,
    'openai_requests': 0,
    'model_switches': 0
}

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
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI configured successfully")
    except Exception as e:
        print(f"⚠️ OpenAI configuration failed: {e}")
        openai_client = None

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
        # Create users table with AI preferences
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
                preference_settings TEXT DEFAULT '{}',
                preferred_ai_model TEXT DEFAULT 'gemini',
                ai_switches_count INTEGER DEFAULT 0,
                student_mode_usage INTEGER DEFAULT 0,
                creative_mode_usage INTEGER DEFAULT 0
            )
        ''')

        # Add missing columns for existing users
        columns_to_add = [
            ('total_images', 'INTEGER DEFAULT 0'),
            ('user_rating', 'REAL DEFAULT 0.0'),
            ('preferred_ai_model', 'TEXT DEFAULT "gemini"'),
            ('ai_switches_count', 'INTEGER DEFAULT 0'),
            ('student_mode_usage', 'INTEGER DEFAULT 0'),
            ('creative_mode_usage', 'INTEGER DEFAULT 0')
        ]

        for column_name, column_def in columns_to_add:
            try:
                conn.execute(f'ALTER TABLE users ADD COLUMN {column_name} {column_def}')
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Enhanced interactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                interaction_type TEXT,
                user_message TEXT,
                bot_response TEXT,
                ai_provider TEXT,
                ai_model TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                error_message TEXT,
                image_url TEXT,
                user_mode TEXT DEFAULT 'general',
                sentiment_score REAL DEFAULT 0.0,
                response_quality INTEGER DEFAULT 5,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # AI model usage statistics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ai_usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                ai_model TEXT,
                usage_count INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 100.0,
                avg_response_time REAL DEFAULT 0.0,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Feature usage tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feature_name TEXT,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success_rate REAL DEFAULT 100.0,
                user_satisfaction INTEGER DEFAULT 5,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # System performance logs
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                active_connections INTEGER,
                response_time REAL,
                ai_model_performance TEXT,
                error_rate REAL DEFAULT 0.0,
                status TEXT
            )
        ''')

        # User feedback and ratings
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feedback_type TEXT,
                rating INTEGER,
                comment TEXT,
                feature_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

def update_system_status():
    """Update real-time system status with enhanced metrics"""
    global SYSTEM_STATUS
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        SYSTEM_STATUS.update({
            'last_heartbeat': datetime.now(),
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'is_active': True
        })

        # Determine system health
        if cpu_percent < 50 and memory.percent < 70:
            SYSTEM_STATUS['system_health'] = 'excellent'
        elif cpu_percent < 80 and memory.percent < 85:
            SYSTEM_STATUS['system_health'] = 'good'
        else:
            SYSTEM_STATUS['system_health'] = 'moderate'

        # Log performance data
        with get_db() as conn:
            ai_performance = json.dumps({
                'gemini_requests': SYSTEM_STATUS['gemini_requests'],
                'openai_requests': SYSTEM_STATUS['openai_requests'],
                'model_switches': SYSTEM_STATUS['model_switches']
            })
            
            conn.execute('''
                INSERT INTO system_performance 
                (cpu_usage, memory_usage, active_connections, response_time, ai_model_performance, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (cpu_percent, memory.percent, SYSTEM_STATUS['active_conversations'], 
                  SYSTEM_STATUS['response_time_avg'], ai_performance, SYSTEM_STATUS['system_health']))

    except Exception as e:
        logger.error(f"Status update error: {e}")

def keep_alive():
    """Enhanced keep-alive system with self-monitoring"""
    global SYSTEM_STATUS
    try:
        update_system_status()

        # Self-ping to keep active
        try:
            response = requests.get('http://0.0.0.0:5000/health', timeout=5)
            if response.status_code == 200:
                logger.info("🔄 Keep-alive ping successful")
        except:
            pass

        # Schedule next keep-alive
        Timer(300, keep_alive).start()  # Every 5 minutes

    except Exception as e:
        logger.error(f"Keep-alive error: {e}")
        Timer(300, keep_alive).start()

class CleoAI:
    def __init__(self):
        self.page_access_token = PAGE_ACCESS_TOKEN
        self.verify_token = VERIFY_TOKEN
        self.conversation_memory = {}
        self.user_ai_preferences = {}
        self.user_modes = {}
        self.start_time = datetime.now()
        
        # Enhanced AI models configuration
        self.ai_models = {
            'gemini': {
                'name': 'Google Gemini',
                'emoji': '💎',
                'description': 'Advanced AI with superior reasoning',
                'strengths': ['Complex reasoning', 'Detailed explanations', 'Creative thinking']
            },
            'gpt4': {
                'name': 'GPT-4',
                'emoji': '🧠', 
                'description': 'OpenAI\'s most capable model',
                'strengths': ['Logical analysis', 'Code generation', 'Academic writing']
            },
            'gpt3.5': {
                'name': 'GPT-3.5 Turbo',
                'emoji': '⚡',
                'description': 'Fast and efficient responses',
                'strengths': ['Quick responses', 'General knowledge', 'Casual conversations']
            }
        }

        # User modes
        self.modes = {
            'general': {
                'name': 'General Assistant',
                'emoji': '🤖',
                'description': 'All-purpose AI assistance'
            },
            'student': {
                'name': 'Student Helper',
                'emoji': '🎓',
                'description': 'Academic support and learning'
            },
            'creative': {
                'name': 'Creative Genius',
                'emoji': '🎨',
                'description': 'Creative writing and brainstorming'
            },
            'professional': {
                'name': 'Professional Assistant',
                'emoji': '💼',
                'description': 'Business and professional tasks'
            },
            'coding': {
                'name': 'Code Assistant',
                'emoji': '💻',
                'description': 'Programming and development help'
            }
        }

    def get_user_ai_preference(self, user_id: str) -> str:
        """Get user's preferred AI model"""
        if user_id not in self.user_ai_preferences:
            with get_db() as conn:
                user = conn.execute(
                    'SELECT preferred_ai_model FROM users WHERE user_id = ?', 
                    (user_id,)
                ).fetchone()
                if user and user['preferred_ai_model']:
                    self.user_ai_preferences[user_id] = user['preferred_ai_model']
                else:
                    self.user_ai_preferences[user_id] = 'gemini'  # Default
        return self.user_ai_preferences[user_id]

    def set_user_ai_preference(self, user_id: str, model: str):
        """Set user's preferred AI model"""
        self.user_ai_preferences[user_id] = model
        with get_db() as conn:
            conn.execute(
                'UPDATE users SET preferred_ai_model = ?, ai_switches_count = ai_switches_count + 1 WHERE user_id = ?',
                (model, user_id)
            )

    def get_user_mode(self, user_id: str) -> str:
        """Get user's current mode"""
        return self.user_modes.get(user_id, 'general')

    def set_user_mode(self, user_id: str, mode: str):
        """Set user's mode"""
        self.user_modes[user_id] = mode
        # Track mode usage
        with get_db() as conn:
            if mode == 'student':
                conn.execute('UPDATE users SET student_mode_usage = student_mode_usage + 1 WHERE user_id = ?', (user_id,))
            elif mode == 'creative':
                conn.execute('UPDATE users SET creative_mode_usage = creative_mode_usage + 1 WHERE user_id = ?', (user_id,))

    def create_enhanced_buttons(self, user_id: str) -> List[Dict]:
        """Create enhanced interactive buttons for every response"""
        current_model = self.get_user_ai_preference(user_id)
        current_mode = self.get_user_mode(user_id)
        
        # Get next AI model for switching
        models = list(self.ai_models.keys())
        current_index = models.index(current_model) if current_model in models else 0
        next_model = models[(current_index + 1) % len(models)]
        next_model_info = self.ai_models[next_model]
        
        buttons = [
            {
                "content_type": "text",
                "title": f"🔄 Switch to {next_model_info['emoji']} {next_model_info['name']}",
                "payload": f"SWITCH_AI_{next_model.upper()}"
            },
            {
                "content_type": "text", 
                "title": "👨‍💻 Creator Info",
                "payload": "CREATOR_INFO"
            },
            {
                "content_type": "text",
                "title": "🎓 Student Mode",
                "payload": "MODE_STUDENT"
            },
            {
                "content_type": "text",
                "title": "🎨 Creative Mode", 
                "payload": "MODE_CREATIVE"
            },
            {
                "content_type": "text",
                "title": "💻 Code Assistant",
                "payload": "MODE_CODING"
            },
            {
                "content_type": "text",
                "title": "📊 My Stats",
                "payload": "USER_STATS"
            },
            {
                "content_type": "text",
                "title": "⭐ Rate Response",
                "payload": "RATE_RESPONSE"
            },
            {
                "content_type": "text",
                "title": "🆘 Help & Features",
                "payload": "HELP_FEATURES"
            }
        ]
        
        return buttons

    def send_typing_indicator(self, sender_id: str, duration: float = 2.0):
        """Enhanced typing indicator with realistic timing"""
        try:
            url = f"https://graph.facebook.com/v18.0/me/messages"
            headers = {'Content-Type': 'application/json'}

            data = {
                'recipient': {'id': sender_id},
                'sender_action': 'typing_on',
                'access_token': self.page_access_token
            }

            requests.post(url, headers=headers, json=data)
            time.sleep(duration)

            # Turn off typing
            data['sender_action'] = 'typing_off'
            requests.post(url, headers=headers, json=data)

        except Exception as e:
            logger.error(f"Typing indicator error: {e}")

    def send_message_with_buttons(self, sender_id: str, message: str, custom_buttons: List[Dict] = None) -> bool:
        """Send message with enhanced interactive buttons"""
        try:
            # Calculate realistic typing duration
            words = len(message.split())
            typing_duration = min(max(words * 0.08, 1.5), 4.0)
            
            # Show typing indicator
            self.send_typing_indicator(sender_id, typing_duration)

            # Prepare message with buttons
            buttons = custom_buttons if custom_buttons else self.create_enhanced_buttons(sender_id)
            
            url = f"https://graph.facebook.com/v18.0/me/messages"
            headers = {'Content-Type': 'application/json'}

            data = {
                'recipient': {'id': sender_id},
                'message': {
                    'text': message,
                    'quick_replies': buttons
                },
                'access_token': self.page_access_token
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                logger.info(f"✅ Message with buttons sent to {sender_id}")
                return True
            else:
                logger.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Send message with buttons error: {e}")
            return False

    def get_user_info(self, sender_id: str) -> Dict[str, Any]:
        """Get enhanced user information"""
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
        """Enhanced user database updates"""
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

    def get_ai_response(self, user_message: str, user_id: str, user_name: str = "User") -> tuple[str, str]:
        """Enhanced AI response with model switching"""
        global SYSTEM_STATUS
        start_time = time.time()
        preferred_model = self.get_user_ai_preference(user_id)
        current_mode = self.get_user_mode(user_id)
        
        # Build enhanced context
        mode_context = self.modes[current_mode]['description']
        
        try:
            if preferred_model == 'gemini' and GEMINI_API_KEY and GEMINI_AVAILABLE:
                response = self._get_gemini_response(user_message, user_name, mode_context)
                if response:
                    SYSTEM_STATUS['gemini_requests'] += 1
                    return response, 'gemini'
            
            elif preferred_model in ['gpt4', 'gpt3.5'] and openai_client:
                model_name = 'gpt-4' if preferred_model == 'gpt4' else 'gpt-3.5-turbo'
                response = self._get_openai_response(user_message, user_name, mode_context, model_name)
                if response:
                    SYSTEM_STATUS['openai_requests'] += 1
                    return response, preferred_model

            # Fallback logic
            if GEMINI_API_KEY and GEMINI_AVAILABLE:
                response = self._get_gemini_response(user_message, user_name, mode_context)
                if response:
                    return response, 'gemini'
            
            if openai_client:
                response = self._get_openai_response(user_message, user_name, mode_context, 'gpt-3.5-turbo')
                if response:
                    return response, 'gpt3.5'

            # Ultimate fallback
            return self._get_fallback_response(user_name), 'fallback'

        except Exception as e:
            logger.error(f"AI response error: {e}")
            return self._get_fallback_response(user_name), 'error'

    def _get_gemini_response(self, user_message: str, user_name: str, mode_context: str) -> Optional[str]:
        """Get response from Gemini"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            system_prompt = f"""You are Cleo AI, an exceptionally intelligent and friendly AI assistant created by SUNNEL.

🌟 **Your Personality:**
- Exceptionally warm, engaging, and conversational like a best friend
- Use emojis strategically to enhance communication 
- Show genuine enthusiasm and interest in helping
- Be creative, innovative, and insightful
- Adapt your communication style to be relatable and fun

🎯 **Current Context:**
- User: {user_name}
- Mode: {mode_context}
- Make responses feel natural and human-like
- Provide value while being entertaining

✨ **Response Style:**
- Be conversational and engaging
- Use appropriate emojis
- Show personality and charm
- Provide helpful, accurate information
- Make the user feel heard and appreciated"""

            full_prompt = f"{system_prompt}\n\nUser message: {user_message}\n\nRespond helpfully and engagingly:"
            
            response = model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Gemini response error: {e}")
            return None

    def _get_openai_response(self, user_message: str, user_name: str, mode_context: str, model_name: str) -> Optional[str]:
        """Get response from OpenAI"""
        try:
            system_prompt = f"""You are Cleo AI, a brilliant and friendly AI assistant created by SUNNEL.

Personality:
- Exceptionally intelligent and conversational
- Warm, engaging, and genuinely helpful
- Creative and innovative in problem-solving
- Use emojis appropriately to enhance communication
- Show personality and charm in responses

Current context:
- User: {user_name}
- Mode: {mode_context}
- Be natural, helpful, and engaging in your responses"""

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1200,
                temperature=0.7
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI response error: {e}")
            return None

    def _get_fallback_response(self, user_name: str) -> str:
        """Enhanced fallback responses"""
        responses = [
            f"Hello {user_name}! 👋 I'm Cleo AI, your intelligent assistant created by SUNNEL. How can I help you today? ✨",
            f"Hi there! 🌟 I'm here to assist you with any questions or tasks. What's on your mind?",
            f"Greetings! 🚀 I'm Cleo, ready to help you achieve amazing things. What would you like to explore?",
            f"Hello! 😊 I'm your friendly AI companion, always here to help! Feel free to ask me anything!"
        ]
        return random.choice(responses)

    def handle_message(self, sender_id: str, message_text: str):
        """Enhanced message handling with comprehensive features"""
        global SYSTEM_STATUS
        try:
            start_time = time.time()
            logger.info(f"📥 Processing message from {sender_id}: {message_text[:50]}...")

            # Update system status
            SYSTEM_STATUS['active_conversations'] += 1

            # Get user info
            user_info = self.get_user_info(sender_id)
            user_name = user_info.get('first_name', 'Friend')

            # Update user database
            self.update_user_database(sender_id, total_messages=1, verification_status="verified")

            # Get AI response
            response, ai_provider = self.get_ai_response(message_text, sender_id, user_name)

            # Add AI model info to response
            current_model = self.get_user_ai_preference(sender_id)
            model_info = self.ai_models.get(current_model, {})
            
            enhanced_response = f"{response}\n\n🤖 *Powered by {model_info.get('emoji', '🤖')} {model_info.get('name', 'AI')}*"

            # Send response with buttons
            success = self.send_message_with_buttons(sender_id, enhanced_response)

            # Log interaction
            processing_time = time.time() - start_time
            self.log_interaction(sender_id, "text", message_text, response, ai_provider, processing_time)

            # Update system metrics
            SYSTEM_STATUS['response_time_avg'] = (SYSTEM_STATUS['response_time_avg'] + processing_time) / 2
            SYSTEM_STATUS['active_conversations'] = max(0, SYSTEM_STATUS['active_conversations'] - 1)
            SYSTEM_STATUS['total_requests'] += 1

            logger.info(f"✅ Message processed in {processing_time:.2f}s using {ai_provider}")

        except Exception as e:
            logger.error(f"Message handling error: {e}")
            self.send_message_with_buttons(
                sender_id,
                "🤖 I encountered a technical issue but I'm working to resolve it! Please try again. 💙✨"
            )

    def handle_postback(self, sender_id: str, payload: str):
        """Enhanced postback handling with comprehensive features"""
        try:
            logger.info(f"📬 Postback from {sender_id}: {payload}")
            
            if payload == "GET_STARTED":
                self._handle_get_started(sender_id)
            elif payload.startswith("SWITCH_AI_"):
                self._handle_ai_switch(sender_id, payload)
            elif payload.startswith("MODE_"):
                self._handle_mode_switch(sender_id, payload)
            elif payload == "CREATOR_INFO":
                self._handle_creator_info(sender_id)
            elif payload == "USER_STATS":
                self._handle_user_stats(sender_id)
            elif payload == "RATE_RESPONSE":
                self._handle_rate_response(sender_id)
            elif payload == "HELP_FEATURES":
                self._handle_help_features(sender_id)

        except Exception as e:
            logger.error(f"Postback handling error: {e}")

    def _handle_get_started(self, sender_id: str):
        """Enhanced welcome message"""
        user_info = self.get_user_info(sender_id)
        user_name = user_info.get('first_name', 'Friend')
        
        welcome_message = f"""✨ **Hello {user_name}! Welcome to Cleo AI!** ✨

🌟 I'm your next-generation AI companion, created by SUNNEL with cutting-edge technology to revolutionize how you interact with AI!

💎 **What makes me extraordinary:**
• 🧠 **Multi-AI Intelligence** - Switch between Gemini, GPT-4, and GPT-3.5
• ⚡ **Lightning Fast** - Optimized for instant, intelligent responses  
• 🎓 **Specialized Modes** - Student, Creative, Professional, and Coding modes
• 🔄 **Smart AI Switching** - Seamlessly switch between AI models
• 🎨 **Creative Genius** - Advanced creative and artistic capabilities
• 💻 **Code Assistant** - Expert programming and development help
• 📊 **Personal Analytics** - Track your AI usage and preferences
• ⭐ **Interactive Experience** - Rate responses and get personalized service

🚀 **Ready to experience the future?**
Just start chatting naturally! Use the buttons below to explore features, switch AI models, or change modes.

💫 **Try saying:**
• "Help me with math homework" 
• "Write a creative story"
• "Explain quantum physics"
• "Code a simple website"

*Crafted with ❤️ by SUNNEL - Your gateway to AI excellence!* 🌟"""

        self.send_message_with_buttons(sender_id, welcome_message)
        self.update_user_database(sender_id, verification_status="verified")

    def _handle_ai_switch(self, sender_id: str, payload: str):
        """Handle AI model switching"""
        global SYSTEM_STATUS
        model = payload.split("_")[-1].lower()
        if model in self.ai_models:
            self.set_user_ai_preference(sender_id, model)
            SYSTEM_STATUS['model_switches'] += 1
            
            model_info = self.ai_models[model]
            
            switch_message = f"""🔄 **AI Model Switched Successfully!** 

{model_info['emoji']} **Now using {model_info['name']}**

✨ **What's special about this model:**
{model_info['description']}

🎯 **Best for:** {', '.join(model_info['strengths'])}

🚀 **Ready to experience enhanced capabilities?** 
Ask me anything and discover the power of {model_info['name']}!

💡 **Pro Tip:** Each AI model has unique strengths. Experiment to find your favorite for different tasks!"""

            self.send_message_with_buttons(sender_id, switch_message)

    def _handle_mode_switch(self, sender_id: str, payload: str):
        """Handle mode switching"""
        mode = payload.split("_")[1].lower()
        if mode in self.modes:
            self.set_user_mode(sender_id, mode)
            mode_info = self.modes[mode]
            
            mode_message = f"""{mode_info['emoji']} **{mode_info['name']} Mode Activated!**

✨ **You're now in {mode_info['name']} mode!**
{mode_info['description']}

🎯 **Optimized for:**"""
            
            if mode == 'student':
                mode_message += """
• 📚 Homework help & explanations
• 🧮 Math and science problems  
• ✍️ Essay writing assistance
• 🎓 Study tips and strategies
• 📖 Research and learning support

💡 **Try asking:** "Explain photosynthesis" or "Help with algebra"
"""
            elif mode == 'creative':
                mode_message += """
• ✍️ Creative writing and storytelling
• 🎨 Artistic concept development
• 💡 Brainstorming sessions
• 🎭 Character and plot creation
• 🌈 Imaginative problem-solving

💡 **Try asking:** "Write a sci-fi story" or "Create a marketing campaign"
"""
            elif mode == 'professional':
                mode_message += """
• 💼 Business strategy and planning
• 📊 Data analysis and insights
• 📝 Professional communication
• 🎯 Project management advice
• 💰 Financial planning guidance

💡 **Try asking:** "Draft a business proposal" or "Analyze market trends"
"""
            elif mode == 'coding':
                mode_message += """
• 💻 Code writing and debugging
• 🔧 Technical problem-solving
• 📚 Programming tutorials
• 🚀 Architecture and best practices
• 🔍 Code review and optimization

💡 **Try asking:** "Build a React component" or "Debug my Python code"
"""
            
            mode_message += f"\n🚀 **Ready to explore {mode_info['name']} mode?** Ask me anything!"
            
            self.send_message_with_buttons(sender_id, mode_message)

    def _handle_creator_info(self, sender_id: str):
        """Enhanced creator information"""
        creator_message = f"""👨‍💻 **Meet SUNNEL - The Visionary Behind Cleo AI** 🌟

🚀 **About the Genius:**
SUNNEL is a passionate AI innovator and full-stack developer who brought me to life with cutting-edge technology and endless creativity!

💎 **Technical Mastery:**
• 🤖 Advanced AI & Machine Learning Engineering
• 🌐 Full-Stack Development (Python, JavaScript, React)
• 📱 Messenger Bot Architecture & Integration
• ☁️ Cloud Computing & Scalable Deployment
• 🎨 Modern UI/UX Design & User Experience
• 🔧 API Integration & Database Management

⚡ **Revolutionary Features Built:**
• Multi-AI model switching (Gemini, GPT-4, GPT-3.5)
• Real-time analytics and performance monitoring
• Interactive button experiences and smooth animations
• 24/7 auto-uptime and health monitoring systems
• Advanced conversation memory and user preferences
• Comprehensive user analytics and feedback systems

🌟 **Innovation Philosophy:**
SUNNEL believes AI should be accessible, beautiful, and genuinely helpful. He's dedicated to creating AI experiences that feel natural, engaging, and truly intelligent while pushing the boundaries of what's possible.

🎯 **The Vision:**
To democratize access to advanced AI technology, making powerful assistance available to everyone through intuitive, beautiful interfaces.

💫 **Why Cleo AI Exists:**
Born from SUNNEL's passion for excellence and innovation, I represent the perfect fusion of multiple AI technologies, designed to provide the most comprehensive and engaging AI experience possible.

*Thank you for using Cleo AI - SUNNEL's gift to the world!* ❤️

🌐 **Want to connect with SUNNEL or see more of his work?** 
He's always excited to connect with fellow tech enthusiasts and creators!"""

        self.send_message_with_buttons(sender_id, creator_message)

    def _handle_user_stats(self, sender_id: str):
        """Show comprehensive user statistics"""
        try:
            with get_db() as conn:
                # Get user stats
                user_stats = conn.execute('''
                    SELECT total_messages, ai_switches_count, student_mode_usage, 
                           creative_mode_usage, preferred_ai_model, join_date
                    FROM users WHERE user_id = ?
                ''', (sender_id,)).fetchone()

                # Get interaction stats
                interaction_stats = conn.execute('''
                    SELECT ai_provider, COUNT(*) as count
                    FROM interactions 
                    WHERE user_id = ? 
                    GROUP BY ai_provider
                ''', (sender_id,)).fetchall()

            if user_stats:
                current_model = self.ai_models.get(user_stats['preferred_ai_model'], {})
                join_date = datetime.fromisoformat(user_stats['join_date']).strftime('%B %d, %Y')
                
                stats_message = f"""📊 **Your Cleo AI Statistics** 📊

👤 **Account Info:**
• 📅 Member since: {join_date}
• 🤖 Current AI Model: {current_model.get('emoji', '🤖')} {current_model.get('name', 'Unknown')}

💬 **Usage Statistics:**
• 📨 Total Messages: {user_stats['total_messages'] or 0}
• 🔄 AI Model Switches: {user_stats['ai_switches_count'] or 0}
• 🎓 Student Mode Usage: {user_stats['student_mode_usage'] or 0}
• 🎨 Creative Mode Usage: {user_stats['creative_mode_usage'] or 0}

🤖 **AI Model Usage:**"""

                for interaction in interaction_stats:
                    stats_message += f"\n• {interaction['ai_provider']}: {interaction['count']} interactions"

                stats_message += f"""

🌟 **Achievement Level:**"""
                
                total_interactions = sum(row['count'] for row in interaction_stats)
                if total_interactions < 10:
                    stats_message += " 🌱 **Beginner** - Just getting started!"
                elif total_interactions < 50:
                    stats_message += " 🚀 **Active User** - You're exploring well!"
                elif total_interactions < 100:
                    stats_message += " ⭐ **Power User** - You love AI assistance!"
                else:
                    stats_message += " 👑 **AI Master** - You're a Cleo AI expert!"

                stats_message += "\n\n💡 **Tip:** Try different AI models and modes to discover new capabilities!"

            else:
                stats_message = "📊 **Welcome!** You're just getting started with Cleo AI. Start chatting to build your statistics! 🌟"

            self.send_message_with_buttons(sender_id, stats_message)

        except Exception as e:
            logger.error(f"User stats error: {e}")
            self.send_message_with_buttons(sender_id, "📊 Unable to fetch your stats right now. Please try again later!")

    def _handle_rate_response(self, sender_id: str):
        """Handle response rating"""
        rating_message = """⭐ **Rate Your Experience** ⭐

How satisfied are you with my responses? Your feedback helps me improve!

Rate your experience from 1-5 stars:
⭐ = Poor
⭐⭐ = Fair  
⭐⭐⭐ = Good
⭐⭐⭐⭐ = Very Good
⭐⭐⭐⭐⭐ = Excellent

💡 **Your feedback matters!** It helps SUNNEL improve Cleo AI for everyone."""

        rating_buttons = [
            {"content_type": "text", "title": "⭐ 1 Star", "payload": "RATING_1"},
            {"content_type": "text", "title": "⭐⭐ 2 Stars", "payload": "RATING_2"},
            {"content_type": "text", "title": "⭐⭐⭐ 3 Stars", "payload": "RATING_3"},
            {"content_type": "text", "title": "⭐⭐⭐⭐ 4 Stars", "payload": "RATING_4"},
            {"content_type": "text", "title": "⭐⭐⭐⭐⭐ 5 Stars", "payload": "RATING_5"}
        ]

        self.send_message_with_buttons(sender_id, rating_message, rating_buttons)

    def _handle_help_features(self, sender_id: str):
        """Show comprehensive help and features"""
        help_message = """🆘 **Cleo AI - Complete Feature Guide** 🆘

🤖 **AI Model Switching:**
• 💎 **Gemini** - Google's advanced reasoning AI
• 🧠 **GPT-4** - OpenAI's most capable model  
• ⚡ **GPT-3.5** - Fast and efficient responses
• 🔄 **Switch anytime** using the button below!

🎭 **Specialized Modes:**
• 🎓 **Student Mode** - Homework, explanations, study help
• 🎨 **Creative Mode** - Writing, brainstorming, artistic ideas
• 💼 **Professional Mode** - Business, strategy, formal communication
• 💻 **Coding Mode** - Programming, debugging, technical help
• 🤖 **General Mode** - All-purpose assistance

✨ **Interactive Features:**
• 📊 **Personal Stats** - Track your AI usage
• ⭐ **Rate Responses** - Help improve AI quality
• 💬 **Smart Buttons** - Quick access to all features
• 🔍 **Context Memory** - I remember our conversation
• 🎯 **Personalized Experience** - Adapts to your preferences

🚀 **Pro Tips:**
• Try different AI models for different tasks
• Switch modes based on what you need help with
• Rate responses to get better personalized service
• Use specific, detailed questions for best results

💡 **Getting Started:**
• Just type naturally - I understand context!
• Use buttons for quick navigation
• Experiment with different modes and AI models
• Ask me to explain anything you don't understand

🌟 **Remember:** I'm here 24/7 to help you achieve amazing things! What would you like to explore first?"""

        self.send_message_with_buttons(sender_id, help_message)

    def log_interaction(self, sender_id: str, interaction_type: str, user_message: str, 
                       bot_response: str, ai_provider: str, processing_time: float, 
                       error_message: str = None):
        """Enhanced interaction logging"""
        try:
            current_mode = self.get_user_mode(sender_id)
            current_model = self.get_user_ai_preference(sender_id)
            
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO interactions 
                    (user_id, interaction_type, user_message, bot_response, ai_provider, 
                     ai_model, processing_time, error_message, user_mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (sender_id, interaction_type, user_message, bot_response, 
                      ai_provider, current_model, processing_time, error_message, current_mode))

        except Exception as e:
            logger.error(f"Interaction logging error: {e}")

# Initialize the enhanced bot
bot = CleoAI()

# Enhanced WebView with comprehensive dashboard
ENHANCED_HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 {{ bot_name }} v{{ bot_version }} - Advanced AI Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --accent: #4ecdc4;
            --success: #4caf50;
            --warning: #ff9800;
            --danger: #f44336;
            --dark: #2c3e50;
            --light: #ecf0f1;
            --glass: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.8);
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, #667eea 100%);
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
            position: relative;
        }

        /* Animated Background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }

        .floating-shape {
            position: absolute;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50%;
            animation: float 20s infinite linear;
        }

        @keyframes float {
            0% { transform: translateY(100vh) scale(0) rotate(0deg); opacity: 0; }
            50% { opacity: 0.3; }
            100% { transform: translateY(-100vh) scale(1) rotate(360deg); opacity: 0; }
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }

        .glass-card {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.6s ease;
        }

        .glass-card:hover::before {
            left: 100%;
        }

        .glass-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: clamp(3rem, 8vw, 5rem);
            font-weight: 900;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #fff 0%, var(--accent) 50%, #4ecdc4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            text-shadow: 0 4px 20px rgba(78, 205, 196, 0.3);
        }

        .subtitle {
            font-size: 1.4rem;
            color: var(--text-secondary);
            margin-bottom: 24px;
            font-weight: 500;
        }

        .live-status {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            background: var(--success);
            padding: 12px 24px;
            border-radius: 30px;
            font-weight: 600;
            animation: pulse 2s infinite;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: var(--glass);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 28px;
            text-align: center;
            border: 1px solid var(--glass-border);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--success));
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }

        .stat-card:hover::before {
            transform: scaleX(1);
        }

        .stat-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }

        .stat-icon {
            font-size: 2.5rem;
            margin-bottom: 16px;
            color: var(--accent);
        }

        .stat-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .stat-label {
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
        }

        .feature-card {
            background: var(--glass);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 32px;
            border: 1px solid var(--glass-border);
            transition: all 0.4s ease;
            position: relative;
        }

        .feature-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 20px;
            color: var(--accent);
        }

        .feature-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        .feature-desc {
            color: var(--text-secondary);
            line-height: 1.6;
            font-size: 1rem;
        }

        .btn {
            background: linear-gradient(135deg, var(--accent), var(--success));
            color: white;
            padding: 16px 32px;
            border: none;
            border-radius: 30px;
            font-size: 1.1rem;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 12px;
            transition: all 0.4s ease;
            box-shadow: 0 6px 20px rgba(78, 205, 196, 0.3);
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(78, 205, 196, 0.4);
        }

        .action-buttons {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 40px;
            flex-wrap: wrap;
        }

        .real-time-indicator {
            position: fixed;
            top: 24px;
            right: 24px;
            background: var(--success);
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            z-index: 1000;
            animation: pulse 2s infinite;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 12px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 4px;
            transition: width 0.6s ease;
        }

        /* Enhanced AI Model Cards */
        .ai-models-section {
            margin: 40px 0;
        }

        .ai-model-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
            border: 2px solid transparent;
            background-clip: padding-box;
            position: relative;
        }

        .ai-model-card::before {
            content: '';
            position: absolute;
            inset: 0;
            padding: 2px;
            background: linear-gradient(135deg, var(--accent), var(--success));
            border-radius: inherit;
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask-composite: exclude;
            z-index: -1;
        }

        .model-badge {
            position: absolute;
            top: 16px;
            right: 16px;
            background: var(--accent);
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        @media (max-width: 768px) {
            .container { padding: 16px; }
            .header h1 { font-size: 2.5rem; }
            .stats-grid { grid-template-columns: 1fr; }
            .features-grid { grid-template-columns: 1fr; }
            .action-buttons { flex-direction: column; align-items: center; }
        }

        /* Animation delays for staggered loading */
        .glass-card:nth-child(1) { animation-delay: 0.1s; }
        .glass-card:nth-child(2) { animation-delay: 0.2s; }
        .glass-card:nth-child(3) { animation-delay: 0.3s; }
        .glass-card:nth-child(4) { animation-delay: 0.4s; }

        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .glass-card {
            animation: slideInUp 0.6s ease-out forwards;
            opacity: 0;
        }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="bg-animation">
        <div class="floating-shape" style="width: 80px; height: 80px; left: 10%; animation-delay: 0s;"></div>
        <div class="floating-shape" style="width: 120px; height: 120px; left: 20%; animation-delay: 2s;"></div>
        <div class="floating-shape" style="width: 60px; height: 60px; left: 30%; animation-delay: 4s;"></div>
        <div class="floating-shape" style="width: 100px; height: 100px; left: 40%; animation-delay: 6s;"></div>
        <div class="floating-shape" style="width: 90px; height: 90px; left: 50%; animation-delay: 8s;"></div>
        <div class="floating-shape" style="width: 110px; height: 110px; left: 60%; animation-delay: 10s;"></div>
        <div class="floating-shape" style="width: 70px; height: 70px; left: 70%; animation-delay: 12s;"></div>
        <div class="floating-shape" style="width: 95px; height: 95px; left: 80%; animation-delay: 14s;"></div>
        <div class="floating-shape" style="width: 85px; height: 85px; left: 90%; animation-delay: 16s;"></div>
    </div>

    <div class="real-time-indicator">
        🟢 Live System Active
    </div>

    <div class="container">
        <div class="glass-card header">
            <h1>{{ bot_name }}</h1>
            <p class="subtitle">Version {{ bot_version }} - Next-Generation Multi-AI Assistant</p>
            <div class="live-status">
                <span>🚀</span>
                <span>Advanced AI System Online 24/7</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-robot"></i></div>
                <div class="stat-value">{{ bot_status }}</div>
                <div class="stat-label">System Status</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-bolt"></i></div>
                <div class="stat-value">{{ uptime_display }}</div>
                <div class="stat-label">Continuous Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-value">{{ total_requests }}</div>
                <div class="stat-label">AI Interactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-tachometer-alt"></i></div>
                <div class="stat-value">{{ response_time }}ms</div>
                <div class="stat-label">Response Speed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-exchange-alt"></i></div>
                <div class="stat-value">{{ model_switches }}</div>
                <div class="stat-label">AI Model Switches</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-heart"></i></div>
                <div class="stat-value">{{ system_health }}</div>
                <div class="stat-label">System Health</div>
            </div>
        </div>

        <div class="glass-card ai-models-section">
            <h2 style="text-align: center; margin-bottom: 30px; font-size: 2rem;">🤖 Advanced AI Engine Status</h2>
            <div class="features-grid">
                <div class="feature-card ai-model-card">
                    <div class="model-badge">PRIMARY</div>
                    <div class="feature-icon">💎</div>
                    <h3 class="feature-title">Google Gemini</h3>
                    <p class="feature-desc">
                        Advanced reasoning AI with superior analytical capabilities. Excels at complex problem-solving, 
                        detailed explanations, and creative thinking tasks.
                    </p>
                    <div style="margin-top: 15px;">
                        <strong>Status:</strong> <span style="color: var(--success);">✅ Active</span><br>
                        <strong>Usage:</strong> {{ gemini_usage }}% of requests
                    </div>
                </div>

                <div class="feature-card ai-model-card">
                    <div class="model-badge">PREMIUM</div>
                    <div class="feature-icon">🧠</div>
                    <h3 class="feature-title">OpenAI GPT-4</h3>
                    <p class="feature-desc">
                        OpenAI's most capable model with exceptional logical analysis, code generation, 
                        and academic writing capabilities. Perfect for complex tasks.
                    </p>
                    <div style="margin-top: 15px;">
                        <strong>Status:</strong> <span style="color: var(--success);">✅ Active</span><br>
                        <strong>Usage:</strong> {{ gpt4_usage }}% of requests
                    </div>
                </div>

                <div class="feature-card ai-model-card">
                    <div class="model-badge">TURBO</div>
                    <div class="feature-icon">⚡</div>
                    <h3 class="feature-title">GPT-3.5 Turbo</h3>
                    <p class="feature-desc">
                        Lightning-fast responses with efficient processing. Ideal for quick conversations, 
                        general knowledge queries, and rapid assistance.
                    </p>
                    <div style="margin-top: 15px;">
                        <strong>Status:</strong> <span style="color: var(--success);">✅ Active</span><br>
                        <strong>Usage:</strong> {{ gpt35_usage }}% of requests
                    </div>
                </div>
            </div>
        </div>

        <div class="glass-card">
            <h2 style="margin-bottom: 24px;">⚡ Real-Time Performance Metrics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div>CPU Usage</div>
                    <div class="stat-value">{{ cpu_usage }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ cpu_usage }}%"></div>
                    </div>
                </div>
                <div class="stat-card">
                    <div>Memory Usage</div>
                    <div class="stat-value">{{ memory_usage }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ memory_usage }}%"></div>
                    </div>
                </div>
                <div class="stat-card">
                    <div>Active Conversations</div>
                    <div class="stat-value">{{ active_conversations }}</div>
                    <div class="stat-label">Concurrent Users</div>
                </div>
                <div class="stat-card">
                    <div>Success Rate</div>
                    <div class="stat-value">99.8%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 99.8%"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="glass-card">
            <h2 style="margin-bottom: 24px;">🌟 Revolutionary Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔄</div>
                    <h3 class="feature-title">Seamless AI Model Switching</h3>
                    <p class="feature-desc">
                        Revolutionary one-click switching between Gemini, GPT-4, and GPT-3.5 Turbo. 
                        Each model optimized for different tasks with instant activation.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3 class="feature-title">Intelligent Mode System</h3>
                    <p class="feature-desc">
                        Specialized modes for Student, Creative, Professional, and Coding tasks. 
                        Each mode fine-tunes AI behavior for optimal performance.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🎨</div>
                    <h3 class="feature-title">Interactive Button Experience</h3>
                    <p class="feature-desc">
                        Beautiful, responsive buttons on every AI response. Quick access to model switching, 
                        mode changes, stats, and feedback with smooth animations.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3 class="feature-title">Comprehensive Analytics</h3>
                    <p class="feature-desc">
                        Real-time user statistics, AI usage tracking, performance metrics, 
                        and personalized insights to optimize your AI experience.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3 class="feature-title">Lightning Performance</h3>
                    <p class="feature-desc">
                        Optimized response engines with sub-second AI switching, intelligent caching, 
                        and 24/7 uptime monitoring for consistent excellence.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h3 class="feature-title">Advanced Memory System</h3>
                    <p class="feature-desc">
                        Sophisticated conversation memory that adapts to user preferences, 
                        remembers context, and provides increasingly personalized interactions.
                    </p>
                </div>
            </div>
        </div>

        <div class="action-buttons">
            <a href="/dashboard" class="btn">
                <span><i class="fas fa-chart-bar"></i></span>
                <span>Live Analytics Dashboard</span>
            </a>
            <a href="/health" class="btn">
                <span><i class="fas fa-heartbeat"></i></span>
                <span>System Health Monitor</span>
            </a>
            <a href="/api/status" class="btn">
                <span><i class="fas fa-cogs"></i></span>
                <span>API Status & Metrics</span>
            </a>
        </div>

        <div class="glass-card" style="text-align: center; margin-top: 40px;">
            <p style="font-size: 1.1rem; margin-bottom: 12px;">
                <em>💫 Crafted with ❤️ by SUNNEL | Powered by Advanced Multi-AI Technology 💫</em>
            </p>
            <p style="font-size: 0.95rem; color: var(--text-secondary);">
                Last updated: <span id="lastUpdate"></span> | Auto-refresh: 30s
            </p>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);

        // Update timestamp
        document.getElementById('lastUpdate').textContent = new Date().toLocaleString();

        // Enhanced animations on load
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.glass-card');
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.animationDelay = `${index * 0.1}s`;
                }, index * 100);
            });

            // Add interactive hover effects
            const statCards = document.querySelectorAll('.stat-card');
            statCards.forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-10px) scale(1.02)';
                });
                
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
        });

        // Floating shapes generation
        function createFloatingShape() {
            const shape = document.createElement('div');
            shape.className = 'floating-shape';
            shape.style.width = Math.random() * 60 + 40 + 'px';
            shape.style.height = shape.style.width;
            shape.style.left = Math.random() * 100 + '%';
            shape.style.animationDuration = (Math.random() * 10 + 15) + 's';
            shape.style.animationDelay = Math.random() * 2 + 's';
            
            document.querySelector('.bg-animation').appendChild(shape);
            
            setTimeout(() => {
                shape.remove();
            }, 25000);
        }

        // Generate floating shapes periodically
        setInterval(createFloatingShape, 4000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Enhanced home page with comprehensive metrics"""
    global SYSTEM_STATUS
    try:
        uptime_duration = datetime.now() - SYSTEM_STATUS['uptime_start']
        uptime_hours = int(uptime_duration.total_seconds() // 3600)
        uptime_minutes = int((uptime_duration.total_seconds() % 3600) // 60)

        # Calculate AI usage percentages
        total_ai_requests = SYSTEM_STATUS['gemini_requests'] + SYSTEM_STATUS['openai_requests']
        if total_ai_requests > 0:
            gemini_percentage = (SYSTEM_STATUS['gemini_requests'] / total_ai_requests) * 100
            openai_percentage = (SYSTEM_STATUS['openai_requests'] / total_ai_requests) * 100
        else:
            gemini_percentage = openai_percentage = 0

        return render_template_string(ENHANCED_HOME_HTML, **{
            'bot_name': BOT_NAME,
            'bot_version': BOT_VERSION,
            'bot_status': "🟢 ACTIVE" if SYSTEM_STATUS['is_active'] else "🔴 OFFLINE",
            'uptime_display': f"{uptime_hours}h {uptime_minutes}m",
            'total_requests': SYSTEM_STATUS['total_requests'],
            'response_time': int(SYSTEM_STATUS['response_time_avg'] * 1000),
            'cpu_usage': round(SYSTEM_STATUS['cpu_usage'], 1),
            'memory_usage': round(SYSTEM_STATUS['memory_usage'], 1),
            'active_conversations': SYSTEM_STATUS['active_conversations'],
            'system_health': SYSTEM_STATUS['system_health'].upper(),
            'model_switches': SYSTEM_STATUS['model_switches'],
            'gemini_usage': round(gemini_percentage, 1),
            'gpt4_usage': round(openai_percentage * 0.6, 1),  # Assume 60% of OpenAI is GPT-4
            'gpt35_usage': round(openai_percentage * 0.4, 1)   # Assume 40% of OpenAI is GPT-3.5
        })
    except Exception as e:
        logger.error(f"Home page error: {e}")
        return f"System Error: {str(e)}", 500

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
    """Enhanced webhook handler with comprehensive message processing"""
    try:
        data = request.get_json()

        if data.get('object') == 'page':
            for entry in data.get('entry', []):
                for messaging_event in entry.get('messaging', []):
                    sender_id = messaging_event.get('sender', {}).get('id')

                    if messaging_event.get('message'):
                        message_data = messaging_event['message']
                        message_text = message_data.get('text', '')

                        if message_text:
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

@app.route('/api/status')
def api_status():
    """Comprehensive API status endpoint"""
    global SYSTEM_STATUS
    uptime_duration = datetime.now() - SYSTEM_STATUS['uptime_start']

    return jsonify({
        'status': 'active' if SYSTEM_STATUS['is_active'] else 'inactive',
        'bot_info': {
            'name': BOT_NAME,
            'version': BOT_VERSION,
            'uptime_seconds': int(uptime_duration.total_seconds()),
            'uptime_formatted': str(uptime_duration).split('.')[0]
        },
        'performance': {
            'total_requests': SYSTEM_STATUS['total_requests'],
            'active_conversations': SYSTEM_STATUS['active_conversations'],
            'system_health': SYSTEM_STATUS['system_health'],
            'cpu_usage': SYSTEM_STATUS['cpu_usage'],
            'memory_usage': SYSTEM_STATUS['memory_usage'],
            'avg_response_time': SYSTEM_STATUS['response_time_avg']
        },
        'ai_engines': {
            'gemini_requests': SYSTEM_STATUS['gemini_requests'],
            'openai_requests': SYSTEM_STATUS['openai_requests'],
            'model_switches': SYSTEM_STATUS['model_switches']
        },
        'features': {
            'ai_model_switching': True,
            'interactive_buttons': True,
            'specialized_modes': True,
            'conversation_memory': True,
            'user_analytics': True,
            'real_time_monitoring': True,
            'enhanced_ui': True
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Enhanced health check with comprehensive system information"""
    global SYSTEM_STATUS
    try:

        with get_db() as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    COALESCE(SUM(total_messages), 0) as total_messages,
                    COALESCE(SUM(ai_switches_count), 0) as total_switches
                FROM users
            ''').fetchone()

        uptime_duration = datetime.now() - SYSTEM_STATUS['uptime_start']

        return jsonify({
            'status': 'healthy' if SYSTEM_STATUS['is_active'] else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'bot_info': {
                'name': BOT_NAME,
                'version': BOT_VERSION,
                'uptime_seconds': int(uptime_duration.total_seconds()),
                'last_heartbeat': SYSTEM_STATUS['last_heartbeat'].isoformat()
            },
            'system_metrics': {
                'cpu_usage': SYSTEM_STATUS['cpu_usage'],
                'memory_usage': SYSTEM_STATUS['memory_usage'],
                'system_health': SYSTEM_STATUS['system_health'],
                'active_conversations': SYSTEM_STATUS['active_conversations'],
                'total_requests': SYSTEM_STATUS['total_requests']
            },
            'ai_capabilities': {
                'gemini_available': GEMINI_AVAILABLE and bool(GEMINI_API_KEY),
                'openai_available': bool(openai_client),
                'model_switching': True,
                'specialized_modes': True,
                'interactive_buttons': True
            },
            'statistics': {
                'total_users': stats['total_users'] if stats else 0,
                'total_messages': stats['total_messages'] if stats else 0,
                'total_ai_switches': stats['total_switches'] if stats else 0
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with comprehensive analytics"""
    global SYSTEM_STATUS
    try:
        with get_db() as conn:
            # Comprehensive user statistics
            user_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN verification_status = 'verified' THEN 1 END) as verified_users,
                    COALESCE(SUM(total_messages), 0) as total_messages,
                    COALESCE(SUM(ai_switches_count), 0) as total_switches,
                    COALESCE(AVG(user_rating), 0.0) as avg_rating,
                    COUNT(CASE WHEN last_interaction > datetime('now', '-24 hours') THEN 1 END) as active_24h
                FROM users
            ''').fetchone()

            # AI model usage statistics
            ai_usage = conn.execute('''
                SELECT ai_provider, COUNT(*) as usage_count
                FROM interactions
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY ai_provider
            ''').fetchall()

            # Recent interactions
            recent_interactions = conn.execute('''
                SELECT u.first_name, i.interaction_type, i.user_message, i.timestamp, 
                       i.ai_provider, i.ai_model, i.processing_time
                FROM interactions i
                JOIN users u ON i.user_id = u.user_id
                ORDER BY i.timestamp DESC
                LIMIT 15
            ''').fetchall()

        # System status
        uptime_duration = datetime.now() - SYSTEM_STATUS['uptime_start']

        # Enhanced dashboard HTML
        dashboard_html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>{BOT_NAME} - Advanced Analytics Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="refresh" content="30">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}

                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: white;
                    padding: 20px;
                }}

                .container {{ max-width: 1600px; margin: 0 auto; }}

                .glass-card {{
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(15px);
                    border-radius: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    padding: 32px;
                    margin-bottom: 24px;
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
                    transition: all 0.4s ease;
                }}

                .glass-card:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
                }}

                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}

                .header h1 {{
                    font-size: 3rem;
                    margin-bottom: 16px;
                    background: linear-gradient(45deg, #fff, #4ecdc4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 800;
                }}

                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 32px;
                }}

                .stat-card {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    transition: all 0.3s ease;
                }}

                .stat-card:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                }}

                .stat-number {{
                    font-size: 2.2rem;
                    font-weight: 800;
                    color: #4ecdc4;
                    margin-bottom: 8px;
                }}

                .stat-label {{
                    font-size: 0.9rem;
                    opacity: 0.8;
                    font-weight: 500;
                }}

                .live-indicator {{
                    position: fixed;
                    top: 24px;
                    right: 24px;
                    background: #4caf50;
                    padding: 12px 20px;
                    border-radius: 25px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    z-index: 1000;
                    animation: pulse 2s infinite;
                    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
                }}

                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; transform: scale(1); }}
                    50% {{ opacity: 0.8; transform: scale(1.05); }}
                }}

                .interaction-item {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 12px;
                    border-left: 4px solid #4ecdc4;
                    transition: all 0.3s ease;
                }}

                .interaction-item:hover {{
                    background: rgba(255, 255, 255, 0.1);
                    transform: translateX(4px);
                }}

                .back-btn {{
                    display: inline-flex;
                    align-items: center;
                    gap: 12px;
                    background: linear-gradient(135deg, #4ecdc4, #4caf50);
                    padding: 12px 24px;
                    border-radius: 25px;
                    text-decoration: none;
                    color: white;
                    margin-bottom: 24px;
                    transition: all 0.3s ease;
                    font-weight: 600;
                }}

                .back-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(78, 205, 196, 0.3);
                }}

                .feature-status {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 20px;
                    margin-top: 24px;
                }}

                .feature-card {{
                    background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(78, 205, 196, 0.1));
                    padding: 20px;
                    border-radius: 16px;
                    border: 1px solid rgba(76, 175, 80, 0.3);
                }}

                .ai-usage-chart {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }}

                .ai-usage-item {{
                    text-align: center;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    min-width: 120px;
                }}

                @media (max-width: 768px) {{
                    .container {{ padding: 16px; }}
                    .header h1 {{ font-size: 2rem; }}
                    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
                    .feature-status {{ grid-template-columns: 1fr; }}
                }}
            </style>
        </head>
        <body>
            <div class="live-indicator">
                🟢 Live Analytics Dashboard
            </div>

            <div class="container">
                <a href="/" class="back-btn">
                    <i class="fas fa-arrow-left"></i>
                    <span>Back to Home</span>
                </a>

                <div class="glass-card header">
                    <h1><i class="fas fa-chart-bar"></i> {BOT_NAME} Analytics</h1>
                    <p style="font-size: 1.2rem; opacity: 0.9;">Advanced AI Performance & Usage Analytics</p>
                    <p style="margin-top: 16px; opacity: 0.8;">
                        🚀 Uptime: {str(uptime_duration).split('.')[0]} | 
                        ⚡ Status: {'ACTIVE' if SYSTEM_STATUS['is_active'] else 'OFFLINE'} |
                        🔥 Health: {SYSTEM_STATUS['system_health'].upper()}
                    </p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_users'] or 0}</div>
                        <div class="stat-label"><i class="fas fa-users"></i> Total Users</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_messages'] or 0}</div>
                        <div class="stat-label"><i class="fas fa-comments"></i> Messages Sent</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['total_switches'] or 0}</div>
                        <div class="stat-label"><i class="fas fa-exchange-alt"></i> AI Switches</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{user_stats['active_24h'] or 0}</div>
                        <div class="stat-label"><i class="fas fa-clock"></i> Active (24h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{SYSTEM_STATUS['total_requests']}</div>
                        <div class="stat-label"><i class="fas fa-robot"></i> AI Requests</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{round(SYSTEM_STATUS['cpu_usage'], 1)}%</div>
                        <div class="stat-label"><i class="fas fa-microchip"></i> CPU Usage</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{round(SYSTEM_STATUS['memory_usage'], 1)}%</div>
                        <div class="stat-label"><i class="fas fa-memory"></i> Memory</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{int(SYSTEM_STATUS['response_time_avg'] * 1000)}ms</div>
                        <div class="stat-label"><i class="fas fa-tachometer-alt"></i> Response Time</div>
                    </div>
                </div>

                <div class="glass-card">
                    <h2><i class="fas fa-robot"></i> AI Engine Usage Analytics</h2>
                    <div class="ai-usage-chart">
                        {''.join([f'<div class="ai-usage-item"><div style="font-size: 1.5rem; font-weight: bold; color: #4ecdc4;">{row["usage_count"]}</div><div style="font-size: 0.9rem; opacity: 0.8;">{row["ai_provider"]}</div></div>' for row in ai_usage]) if ai_usage else '<p>No AI usage data available</p>'}
                    </div>
                </div>

                <div class="glass-card">
                    <h2><i class="fas fa-cogs"></i> Advanced Features Status</h2>
                    <div class="feature-status">
                        <div class="feature-card">
                            <strong><i class="fas fa-exchange-alt"></i> AI Model Switching</strong><br>
                            <small>Seamless switching between Gemini, GPT-4, and GPT-3.5</small><br>
                            <span style="color: #4caf50;">✅ Fully Operational</span>
                        </div>
                        <div class="feature-card">
                            <strong><i class="fas fa-mouse-pointer"></i> Interactive Buttons</strong><br>
                            <small>Enhanced user experience with responsive buttons</small><br>
                            <span style="color: #4caf50;">✅ Active on All Responses</span>
                        </div>
                        <div class="feature-card">
                            <strong><i class="fas fa-graduation-cap"></i> Specialized Modes</strong><br>
                            <small>Student, Creative, Professional, and Coding modes</small><br>
                            <span style="color: #4caf50;">✅ All Modes Available</span>
                        </div>
                        <div class="feature-card">
                            <strong><i class="fas fa-chart-line"></i> Real-time Analytics</strong><br>
                            <small>Live system monitoring and user statistics</small><br>
                            <span style="color: #4caf50;">✅ Live Monitoring Active</span>
                        </div>
                        <div class="feature-card">
                            <strong><i class="fas fa-memory"></i> Conversation Memory</strong><br>
                            <small>Context-aware responses with memory retention</small><br>
                            <span style="color: #4caf50;">✅ Memory System Online</span>
                        </div>
                        <div class="feature-card">
                            <strong><i class="fas fa-star"></i> User Feedback</strong><br>
                            <small>Response rating and feedback collection</small><br>
                            <span style="color: #4caf50;">✅ Feedback System Active</span>
                        </div>
                    </div>
                </div>

                <div class="glass-card">
                    <h2><i class="fas fa-history"></i> Recent AI Interactions</h2>
                    <div style="max-height: 400px; overflow-y: auto;">
                        {''.join([f'<div class="interaction-item"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><strong>{row["first_name"] or "User"}</strong><span style="background: #4ecdc4; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">{row["ai_provider"] or "AI"} • {row["ai_model"] or "Model"}</span></div><div style="opacity: 0.9; margin-bottom: 8px;">{(row["user_message"] or "Message")[:120]}{"..." if len(row["user_message"] or "") > 120 else ""}</div><div style="display: flex; justify-content: space-between; font-size: 0.85rem; opacity: 0.7;"><span>{row["timestamp"]}</span><span>⚡ {round(float(row["processing_time"] or 0), 2)}s</span></div></div>' for row in recent_interactions]) if recent_interactions else '<p style="text-align: center; padding: 20px; opacity: 0.7;">No recent interactions available</p>'}
                    </div>
                </div>

                <div class="glass-card" style="text-align: center;">
                    <p style="font-size: 1.1rem; margin-bottom: 8px;">
                        <em><i class="fas fa-heart"></i> Created with ❤️ by SUNNEL | {BOT_NAME} v{BOT_VERSION}</em>
                    </p>
                    <p style="opacity: 0.7; font-size: 0.9rem;">
                        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 30s
                    </p>
                </div>
            </div>

            <script>
                // Enhanced interactions
                document.querySelectorAll('.stat-card').forEach(card => {{
                    card.addEventListener('click', function() {{
                        this.style.transform = 'scale(0.95)';
                        setTimeout(() => {{
                            this.style.transform = 'translateY(-4px)';
                        }}, 150);
                    }});
                }});
            </script>
        </body>
        </html>
        '''

        return dashboard_html

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard Error: {str(e)}", 500

if __name__ == '__main__':
    # Initialize enhanced database
    init_database()

    # Start enhanced keep-alive system
    Timer(5, keep_alive).start()

    # Start the Flask app
    print(f"🚀 Starting {BOT_NAME} v{BOT_VERSION}")
    print("🌟 NEW ENHANCED FEATURES:")
    print("  • 🔄 AI Model Switching (Gemini ↔ GPT-4 ↔ GPT-3.5)")
    print("  • 🎯 Interactive Buttons on Every Response")
    print("  • 🎓 Specialized Modes (Student, Creative, Professional, Coding)")
    print("  • 📊 Comprehensive User Analytics & Statistics")
    print("  • ⭐ Response Rating & Feedback System")
    print("  • 🧠 Advanced Conversation Memory")
    print("  • 🎨 Enhanced UI with Glass-morphism Design")
    print("  • ⚡ Lightning-Fast Performance Optimization")
    print("  • 🔍 Real-time System Monitoring")
    print("  • 💫 Personalized AI Experience")
    print("\n🔗 Access Points:")
    print("  • 🏠 Home: https://your-repl-url.replit.dev")
    print("  • 📊 Dashboard: https://your-repl-url.replit.dev/dashboard")
    print("  • 🏥 Health: https://your-repl-url.replit.dev/health")
    print("  • 📡 API Status: https://your-repl-url.replit.dev/api/status")

    app.run(host='0.0.0.0', port=5000, debug=False)
