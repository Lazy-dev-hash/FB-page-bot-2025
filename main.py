
import os
import json
import requests
from flask import Flask, request, jsonify
import openai
from datetime import datetime
import logging
import base64
import io
from PIL import Image
import time
import random

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

# Initialize AI providers
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

class EnhancedFacebookBot:
    def __init__(self):
        self.api_version = 'v18.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
        self.conversation_memory = {}
        self.user_preferences = {}
        self.verified_users = set()  # Store verified users
        self.user_states = {}  # Track user verification state
        
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

    def check_user_liked_recent_posts(self, user_id):
        """Check if user liked recent posts (simplified check)"""
        try:
            # Get recent posts from the page
            url = f'{self.base_url}/me/posts'
            params = {
                'fields': 'id,likes.summary(true)',
                'limit': 5,
                'access_token': PAGE_ACCESS_TOKEN
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                posts_data = response.json()
                posts = posts_data.get('data', [])
                
                for post in posts:
                    # Check if user liked this post
                    post_id = post.get('id')
                    likes_url = f'{self.base_url}/{post_id}/likes/{user_id}'
                    like_params = {'access_token': PAGE_ACCESS_TOKEN}
                    
                    like_response = requests.get(likes_url, params=like_params)
                    if like_response.status_code == 200:
                        like_data = like_response.json()
                        if like_data.get('data'):
                            return True
            
        except Exception as e:
            logger.error(f"Error checking like status: {e}")
        
        return False

    def verify_user_access(self, user_id):
        """Verify if user has followed page and liked posts"""
        if user_id in self.verified_users:
            return True, "verified"
        
        # Check follow status
        follows_page = self.check_user_follows_page(user_id)
        
        # Check like status  
        liked_posts = self.check_user_liked_recent_posts(user_id)
        
        if follows_page and liked_posts:
            self.verified_users.add(user_id)
            return True, "newly_verified"
        elif follows_page and not liked_posts:
            return False, "need_like"
        elif not follows_page and liked_posts:
            return False, "need_follow"
        else:
            return False, "need_both"

    def send_verification_request(self, user_id, status):
        """Send verification request message based on status"""
        if status == "need_both":
            message = """🔐 **Welcome! Access Required** 🔐

To unlock full access to this amazing AI bot, please:

1️⃣ **Follow this page** 👍
2️⃣ **Like our recent posts** ❤️

Once you complete both steps, you'll get full access to:
🤖 Advanced AI conversations (ChatGPT + Gemini)
🔍 Image analysis capabilities  
💭 Smart conversation memory
⚡ Instant responses 24/7

Please complete these steps and send any message to verify! 🚀"""

        elif status == "need_follow":
            message = """🔐 **Almost There!** 🔐

Thank you for liking our posts! ❤️

To complete verification, please:
1️⃣ **Follow this page** 👍

Once you follow, you'll have full access to all AI features! 🚀"""

        elif status == "need_like":
            message = """🔐 **Almost There!** 🔐

Thank you for following our page! 👍

To complete verification, please:
2️⃣ **Like our recent posts** ❤️

Once you like our posts, you'll have full access to all AI features! 🚀"""

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

You can now ask me anything or send images for analysis! 🍓🥰"""

        # Send the congratulations message first
        self.send_message(user_id, congrats_message)
        
        # Send the custom image (you'll need to host this image publicly)
        # For now, I'll use a placeholder - you should upload your image to a public URL
        image_url = "https://i.imgur.com/YourImageHere.png"  # Replace with your hosted image URL
        
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
        
        try:
            if use_gemini and GEMINI_API_KEY and GEMINI_AVAILABLE:
                return self.get_gemini_response(user_message, user_name, conversation_history)
            elif OPENAI_API_KEY:
                return self.get_openai_response(user_message, user_name, conversation_history)
            elif GEMINI_API_KEY and GEMINI_AVAILABLE:
                return self.get_gemini_response(user_message, user_name, conversation_history)
            else:
                return "🤖 AI services are currently unavailable. Please try again later!"
                
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "🤖 I'm having trouble processing your request right now. Please try again later!"
    
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
        
        # Check user verification status
        is_verified, status = self.verify_user_access(sender_id)
        
        if not is_verified:
            self.send_verification_request(sender_id, status)
            return
        
        # If newly verified, send congratulations
        if status == "newly_verified":
            self.send_congratulations_with_image(sender_id)
            return
        
        # User is verified, proceed with normal AI response
        self.send_typing_indicator(sender_id)
        context = self.get_conversation_context(sender_id)
        response = self.get_smart_ai_response(message_text, sender_name, context)
        self.update_conversation_memory(sender_id, message_text, response)
        self.send_message(sender_id, response)
    
    def handle_image_message(self, sender_id, attachment_url, message_text="", sender_name=None):
        """Handle image messages with security verification"""
        logger.info(f"Processing image from {sender_id}")
        
        # Check user verification status
        is_verified, status = self.verify_user_access(sender_id)
        
        if not is_verified:
            self.send_verification_request(sender_id, status)
            return
        
        # If newly verified, send congratulations
        if status == "newly_verified":
            self.send_congratulations_with_image(sender_id)
            return
        
        # User is verified, proceed with image analysis
        self.send_typing_indicator(sender_id)
        
        image_data = self.download_image(attachment_url)
        
        if not image_data:
            self.send_message(sender_id, "🖼️ Sorry, I couldn't download your image. Please try sending it again!")
            return
        
        user_question = message_text if message_text.strip() else "What's in this image?"
        
        response = None
        
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            response = self.analyze_image_with_gemini(image_data, user_question)
        
        if not response and OPENAI_API_KEY:
            response = self.analyze_image_with_openai(image_data, user_question)
        
        if not response:
            response = "🖼️ I can see you sent an image, but I'm having trouble analyzing it right now. Please try again later or describe what you'd like to know about it!"
        
        final_response = f"🔍 **Image Analysis** 📸\n\n{response}"
        self.update_conversation_memory(sender_id, f"[Image] {user_question}", final_response)
        self.send_message(sender_id, final_response)

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
                            is_verified, status = bot.verify_user_access(sender_id)
                            if not is_verified:
                                bot.send_verification_request(sender_id, status)
                            elif status == "newly_verified":
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
        'openai_configured': bool(OPENAI_API_KEY),
        'gemini_configured': bool(GEMINI_API_KEY and GEMINI_AVAILABLE),
        'features': {
            'smart_conversations': True,
            'image_analysis': bool(OPENAI_API_KEY or (GEMINI_API_KEY and GEMINI_AVAILABLE)),
            'multi_ai_providers': bool(OPENAI_API_KEY and GEMINI_API_KEY and GEMINI_AVAILABLE),
            'conversation_memory': True,
            'auto_responses': True,
            'security_verification': True
        }
    })

@app.route('/', methods=['GET'])
def home():
    """Enhanced home page with bot information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Ultimate AI Facebook Bot - Secured</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                color: white;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }}
            h1 {{
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .subtitle {{
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }}
            .security-banner {{
                background: linear-gradient(45deg, #ff6b6b, #feca57);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
                color: white;
                font-weight: bold;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .status-card {{
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            .status-icon {{
                font-size: 2em;
                margin-bottom: 10px;
            }}
            .features {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }}
            .feature {{
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #4ecdc4;
            }}
            .security-feature {{
                border-left: 4px solid #ff6b6b;
            }}
            .webhook-info {{
                background: rgba(0, 0, 0, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                text-align: center;
            }}
            code {{
                background: rgba(0, 0, 0, 0.3);
                padding: 5px 10px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
            }}
            .emoji {{
                font-size: 1.2em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Ultimate AI Facebook Bot</h1>
            <p class="subtitle">Secured & Powered by ChatGPT, Gemini & Advanced AI 🚀</p>
            
            <div class="security-banner">
                🔐 SECURED ACCESS SYSTEM ACTIVE 🔐<br>
                Users must follow page & like posts to access AI features
            </div>
            
            <div class="status-grid">
                <div class="status-card">
                    <div class="status-icon">{bot_status}</div>
                    <h3>Bot Status</h3>
                    <p>{bot_text}</p>
                </div>
                <div class="status-card">
                    <div class="status-icon">{openai_status}</div>
                    <h3>ChatGPT</h3>
                    <p>{openai_text}</p>
                </div>
                <div class="status-card">
                    <div class="status-icon">{gemini_status}</div>
                    <h3>Google Gemini</h3>
                    <p>{gemini_text}</p>
                </div>
            </div>
            
            <h2>🌟 Enhanced Features</h2>
            <div class="features">
                <div class="feature security-feature">
                    <strong class="emoji">🔐</strong> <strong>Security Verification</strong><br>
                    Users must follow page and like posts to access AI features
                </div>
                <div class="feature">
                    <strong class="emoji">🤖</strong> <strong>Multi-AI Intelligence</strong><br>
                    Powered by both ChatGPT and Google Gemini for the best responses
                </div>
                <div class="feature">
                    <strong class="emoji">🔍</strong> <strong>Advanced Image Analysis</strong><br>
                    Send any image and ask questions - I'll analyze and explain everything!
                </div>
                <div class="feature">
                    <strong class="emoji">💭</strong> <strong>Conversation Memory</strong><br>
                    I remember our conversation context for more natural interactions
                </div>
                <div class="feature">
                    <strong class="emoji">⚡</strong> <strong>Smart Auto-Responses</strong><br>
                    No commands needed! Just talk naturally and I'll understand
                </div>
                <div class="feature">
                    <strong class="emoji">🎨</strong> <strong>Creative AI Tasks</strong><br>
                    Brainstorming, creative writing, ideas, and artistic discussions
                </div>
                <div class="feature">
                    <strong class="emoji">🎉</strong> <strong>Welcome Image System</strong><br>
                    Custom congratulations image sent to verified users
                </div>
                <div class="feature">
                    <strong class="emoji">🌍</strong> <strong>24/7 Availability</strong><br>
                    Always online, always ready to help verified users
                </div>
            </div>
            
            <div class="webhook-info">
                <h3>📡 Webhook Configuration</h3>
                <p>Webhook URL: <code>{webhook_url}</code></p>
                <p>Ready to receive and process secured messages from Facebook!</p>
            </div>
        </div>
    </body>
    </html>
    """.format(
        bot_status="✅" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "❌",
        bot_text="Active & Secured" if PAGE_ACCESS_TOKEN and VERIFY_TOKEN else "Not Configured",
        openai_status="🤖" if OPENAI_API_KEY else "❌",
        openai_text="Connected" if OPENAI_API_KEY else "Not Configured",
        gemini_status="🌟" if GEMINI_API_KEY and GEMINI_AVAILABLE else "❌",
        gemini_text="Connected" if GEMINI_API_KEY and GEMINI_AVAILABLE else "Not Available",
        webhook_url=f"{request.url_root.rstrip('/')}/webhook"
    )

if __name__ == '__main__':
    logger.info("🚀 Starting Ultimate AI Facebook Bot with Security...")
    
    if not PAGE_ACCESS_TOKEN:
        logger.warning("❌ FACEBOOK_PAGE_ACCESS_TOKEN not set")
    if not OPENAI_API_KEY:
        logger.warning("❌ OPENAI_API_KEY not set")
    if not GEMINI_API_KEY:
        logger.warning("❌ GEMINI_API_KEY not set")
    if not GEMINI_AVAILABLE:
        logger.warning("⚠️ Gemini libraries not available - using OpenAI only")
    
    if OPENAI_API_KEY and GEMINI_API_KEY and GEMINI_AVAILABLE:
        logger.info("🌟 Multi-AI configuration detected - Full features available!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
