import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import re
import warnings
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
import subprocess
import threading
import time
import platform
import urllib.request
import zipfile
import shutil
import stat
import json
from collections import defaultdict, Counter
import uuid
warnings.filterwarnings("ignore")

# --- CORREÇÃO PARA CLOUDFLARE PROXY ---
class ProxyFix(object):
    """Corrige headers quando atrás de proxy (Cloudflare)"""
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        scheme = environ.get('HTTP_X_FORWARDED_PROTO', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)

# --- 1. Configurações de Segurança ---
CONFIG_FILE = "user_config.json"
CLOUDFLARED_DIR = "cloudflared"
FAISS_INDEX_PATH = "my_index.faiss"
NOTES_STORAGE_PATH = "notes.pkl"
CATEGORIES_PATH = "categories.pkl"
ANALYTICS_PATH = "analytics.pkl"

app = Flask(__name__, template_folder='.', static_folder='.')
app.wsgi_app = ProxyFix(app.wsgi_app)  # CORREÇÃO APLICADA
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Mudado de 'Strict' para 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

def load_user_config():
    """Carrega configurações do usuário ou cria configuração inicial."""
    import json
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("\n" + "="*50)
        print("🔐 CONFIGURAÇÃO INICIAL DE SEGURANÇA")
        print("="*50)
        print("Para acessar sua aplicação remotamente, você precisa definir uma senha.")
        print("Esta senha protegerá suas memórias pessoais.\n")
        
        while True:
            password = input("Digite uma senha forte (mín. 8 caracteres): ").strip()
            if len(password) >= 8:
                break
            print("❌ Senha muito curta! Use pelo menos 8 caracteres.")
        
        confirm = input("Confirme a senha: ").strip()
        if password != confirm:
            print("❌ Senhas não coincidem! Reinicie a aplicação.")
            exit(1)
        
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        
        config = {
            "password_hash": password_hash.hex(),
            "salt": salt,
            "created_at": datetime.now().isoformat(),
            "remote_access": True,
            "cloudflare_tunnel": True
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuração salva! Sua aplicação estará protegida.")
        print("🌐 Acesso remoto habilitado com Cloudflare Tunnel")
        print("="*50 + "\n")
        
        return config

def verify_password(password, config):
    """Verifica se a senha está correta."""
    salt = config['salt']
    stored_hash = config['password_hash']
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return password_hash.hex() == stored_hash

def download_cloudflared():
    """Baixa o cloudflared automaticamente baseado no sistema operacional."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if "64" in machine or "amd64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            filename = "cloudflared.exe"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"
            filename = "cloudflared.exe"
    elif system == "darwin":
        if "arm" in machine or "m1" in machine or "m2" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
        filename = "cloudflared"
    else:
        if "arm" in machine:
            if "64" in machine:
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
            else:
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"
        elif "64" in machine or "amd64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-386"
        filename = "cloudflared"
    
    if not os.path.exists(CLOUDFLARED_DIR):
        os.makedirs(CLOUDFLARED_DIR)
    
    cloudflared_path = os.path.join(CLOUDFLARED_DIR, filename)
    
    if os.path.exists(cloudflared_path):
        return cloudflared_path
    
    print("🌐 Baixando Cloudflare Tunnel...")
    try:
        urllib.request.urlretrieve(url, cloudflared_path)
        
        if system != "windows":
            st = os.stat(cloudflared_path)
            os.chmod(cloudflared_path, st.st_mode | stat.S_IEXEC)
        
        print("✅ Cloudflare Tunnel baixado com sucesso!")
        return cloudflared_path
    except Exception as e:
        print(f"❌ Erro ao baixar cloudflared: {e}")
        return None

def start_cloudflare_tunnel():
    """Inicia o túnel do Cloudflare em uma thread separada."""
    cloudflared_path = download_cloudflared()
    
    if not cloudflared_path:
        print("❌ Não foi possível baixar o Cloudflare Tunnel")
        return None
    
    def run_tunnel():
        try:
            print("🚀 Iniciando Cloudflare Tunnel...")
            cmd = [cloudflared_path, "tunnel", "--url", "http://localhost:5000"]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True
            )
            
            print("⏳ Aguardando URL do túnel...")
            tunnel_found = False
            
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    
                    if "tunnel" in line.lower() or "cloudflare" in line.lower():
                        print(f"🛠 Debug: {line}")
                    
                    import re
                    urls = re.findall(r'https://[a-zA-Z0-9\-]+\.(?:trycloudflare\.com|cfargotunnel\.com|argotunnel\.com)', line)
                    
                    if urls and not tunnel_found:
                        tunnel_url = urls[0]
                        tunnel_found = True
                        print(f"\n🌐 TÚNEL CLOUDFLARE ATIVO!")
                        print(f"🔗 URL Externa: {tunnel_url}")
                        print(f"📱 Acesse de qualquer lugar: {tunnel_url}")
                        print(f"🔒 Protegido com senha")
                        print("="*60)
                        
                        with open("tunnel_url.txt", "w") as f:
                            f.write(tunnel_url)
                        
                        break
                    
                    if "error" in line.lower() or "failed" in line.lower():
                        print(f"⚠️ Possível erro: {line}")
                        
                except Exception as e:
                    print(f"⚠️ Erro ao ler linha: {e}")
                    break
            
            if not tunnel_found:
                print("⚠️ URL do túnel não encontrada. Verifique a conexão.")
                    
        except Exception as e:
            print(f"❌ Erro no túnel Cloudflare: {e}")
            print("💡 Tentativa manual: Execute no terminal separado:")
            print(f"   {cloudflared_path} tunnel --url http://localhost:5000")
    
    tunnel_thread = threading.Thread(target=run_tunnel, daemon=True)
    tunnel_thread.start()
    time.sleep(5)
    return tunnel_thread

def require_auth(f):
    """Decorator que exige autenticação para acessar rotas protegidas."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_api_request = any(request.path.startswith(p) for p in ['/get-', '/add-', '/ask', '/ai-', '/stats', '/edit-', '/delete-', '/categorize'])

        if 'authenticated' not in session or not session['authenticated']:
            if is_api_request or request.is_json:
                return jsonify({"error": "Não autenticado", "redirect": "/login"}), 401
            return redirect(url_for('login'))
        
        if 'login_time' in session:
            login_time = datetime.fromisoformat(session['login_time'])
            if datetime.now() - login_time > app.config['PERMANENT_SESSION_LIFETIME']:
                session.clear()
                if is_api_request or request.is_json:
                    return jsonify({"error": "Sessão expirada", "redirect": "/login"}), 401
                flash("Sua sessão expirou. Por favor, faça login novamente.")
                return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

user_config = load_user_config()

# --- 2. Configuração Melhorada da IA ---
AI_MODEL_NAME = "microsoft/DialoGPT-medium"
ai_tokenizer = None
ai_model = None
AI_AVAILABLE = False

FALLBACK_MODEL = "gpt2"

def setup_local_ai():
    """Configura a IA local usando modelos mais avançados."""
    global ai_tokenizer, ai_model, AI_AVAILABLE
    try:
        print("🤖 Carregando modelo de IA melhorado...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        try:
            ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME, padding_side='left')
            ai_model = AutoModelForCausalLM.from_pretrained(AI_MODEL_NAME)
            print(f"✅ IA avançada carregada: {AI_MODEL_NAME}")
        except Exception as e:
            print(f"⚠️ Modelo principal falhou, tentando fallback: {e}")
            ai_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, padding_side='left')
            ai_model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL)
            print(f"✅ IA fallback carregada: {FALLBACK_MODEL}")
        
        if ai_tokenizer.pad_token is None:
            ai_tokenizer.pad_token = ai_tokenizer.eos_token
        AI_AVAILABLE = True
        
    except (ImportError, OSError) as e:
        print(f"❌ Modelo de IA não pôde ser carregado: {e}. IA indisponível.")
        AI_AVAILABLE = False

print("📚 Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modelo de embeddings carregado.")

setup_local_ai()

d = 384

# --- 3. Sistema de Categorização Inteligente ---
class IntelligentCategorizer:
    def __init__(self):
        self.categories = {}
        self.category_embeddings = {}
        self.load_categories()
    
    def load_categories(self):
        """Carrega categorias salvas ou cria as iniciais."""
        if os.path.exists(CATEGORIES_PATH):
            try:
                with open(CATEGORIES_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.categories = data.get('categories', {})
                    self.category_embeddings = data.get('embeddings', {})
                print(f"📂 {len(self.categories)} categorias carregadas.")
            except Exception as e:
                print(f"⚠️ Erro ao carregar categorias: {e}")
                self._create_initial_categories()
        else:
            self._create_initial_categories()
    
    def _create_initial_categories(self):
        """Cria categorias iniciais baseadas em análise semântica."""
        initial_categories = {
            "pessoal": {
                "keywords": ["família", "amigo", "casa", "vida", "pessoa", "relacionamento", "saúde"],
                "color": "#8b5cf6",
                "description": "Vida pessoal e relacionamentos"
            },
            "trabalho": {
                "keywords": ["trabalho", "emprego", "projeto", "reunião", "empresa", "carreira", "negócio"],
                "color": "#3b82f6", 
                "description": "Assuntos profissionais"
            },
            "conhecimento": {
                "keywords": ["estudar", "aprender", "livro", "curso", "técnico", "conhecimento", "skill"],
                "color": "#10b981",
                "description": "Aprendizado e conhecimento"
            },
            "finanças": {
                "keywords": ["dinheiro", "conta", "banco", "investimento", "economia", "comprar", "pagar"],
                "color": "#f59e0b",
                "description": "Questões financeiras"
            },
            "tecnologia": {
                "keywords": ["código", "programa", "software", "app", "tecnologia", "computador", "internet"],
                "color": "#ef4444",
                "description": "Tecnologia e programação"
            },
            "criatividade": {
                "keywords": ["arte", "música", "criativo", "design", "pintura", "desenho", "fotografia"],
                "color": "#f97316",
                "description": "Atividades criativas"
            }
        }
        
        self.categories = initial_categories
        
        for cat_name, cat_data in self.categories.items():
            keywords_text = " ".join(cat_data["keywords"])
            embedding = model.encode([keywords_text])[0]
            self.category_embeddings[cat_name] = embedding
        
        self.save_categories()
        print("🎯 Categorias iniciais criadas.")
    
    def save_categories(self):
        """Salva categorias no disco."""
        try:
            data = {
                'categories': self.categories,
                'embeddings': self.category_embeddings
            }
            with open(CATEGORIES_PATH, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"❌ Erro ao salvar categorias: {e}")
    
    def categorize_memory(self, text):
        """Categoriza uma memória usando análise melhorada."""
        text_lower = text.lower()
        
        category_scores = {}
        
        for cat_name, cat_data in self.categories.items():
            score = 0.0
            keywords = cat_data.get("keywords", [])
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 2.0
                    
            related_words = {
                "pessoal": ["amigo", "amiga", "família", "pai", "mãe", "irmão", "irmã", "namorado", "namorada", "esposo", "esposa", "filho", "filha", "primo", "prima", "tio", "tia", "avô", "avó", "friend", "brother", "sister"],
                "trabalho": ["trabalho", "emprego", "chefe", "colega", "projeto", "reunião", "empresa", "job", "work", "office", "meeting"],
                "conhecimento": ["estudar", "aprender", "livro", "curso", "universidade", "escola", "prova", "estudo", "learn", "study", "book"],
                "tecnologia": ["programar", "código", "software", "app", "python", "javascript", "html", "css", "computer", "tech"],
                "finanças": ["dinheiro", "banco", "conta", "pagar", "comprar", "vender", "investir", "money", "bank"],
                "criatividade": ["desenhar", "pintar", "música", "arte", "design", "criar", "art", "music", "create"]
            }
            
            if cat_name in related_words:
                for related in related_words[cat_name]:
                    if related in text_lower:
                        score += 1.5
            
            category_scores[cat_name] = score
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        if best_category[1] > 0.5:
            return best_category[0], min(0.95, best_category[1] / 3.0)
        
        memory_embedding = model.encode([text])[0]
        best_score = 0.0
        best_cat = "geral"
        
        for cat_name, cat_embedding in self.category_embeddings.items():
            similarity = np.dot(memory_embedding, cat_embedding) / (
                np.linalg.norm(memory_embedding) * np.linalg.norm(cat_embedding)
            )
            
            if similarity > best_score and similarity > 0.25:
                best_score = similarity
                best_cat = cat_name
        
        return best_cat if best_score > 0.25 else "geral", best_score
    
    def _generate_color(self):
        """Gera cor aleatória para nova categoria."""
        colors = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#f97316", "#6366f1", "#14b8a6", "#f472b6", "#84cc16"]
        used_colors = [cat["color"] for cat in self.categories.values()]
        available = [c for c in colors if c not in used_colors]
        return available[0] if available else f"#{secrets.token_hex(3)}"

categorizer = IntelligentCategorizer()

# --- 4. Sistema de Persistência Melhorado ---
def save_data(index, storage):
    print("💾 Salvando dados no disco...")
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(NOTES_STORAGE_PATH, 'wb') as f:
            pickle.dump(storage, f)
        print("✅ Dados salvos com sucesso.")
    except Exception as e:
        print(f"💥 Erro ao salvar dados: {e}")

def load_data():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(NOTES_STORAGE_PATH):
        print("📂 Carregando dados existentes do disco...")
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(NOTES_STORAGE_PATH, 'rb') as f:
                storage = pickle.load(f)
            print(f"👍 Dados carregados: {index.ntotal} notas encontradas.")
            return index, storage
        except Exception as e:
            print(f"💥 Erro ao carregar dados: {e}. Criando nova base.")
    print("⏳ Nenhum dado salvo encontrado. Criando nova base.")
    index = faiss.IndexFlatL2(d)
    storage = {}
    return index, storage

index, note_storage = load_data()

# --- 5. Sistema de Análise e Conexões Melhorado ---
def calculate_text_similarity(text1, text2):
    try:
        def preprocess_text(text):
            return re.sub(r'[^\w\s]', '', text.lower())
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)
        if len(processed_text1.split()) < 2 or len(processed_text2.split()) < 2:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        return cosine_similarity(tfidf_matrix)[0][1]
    except Exception:
        return 0.0

def find_memory_connections(note_storage, threshold=0.25):
    """Encontra conexões entre memórias com análise de categorias."""
    connections = []
    note_ids = list(note_storage.keys())
    
    print(f"\n🔗 ANALISANDO CONEXÕES entre {len(note_ids)} memórias...")
    
    for i, id1 in enumerate(note_ids):
        for id2 in note_ids[i+1:]:
            memory1 = note_storage[id1]
            memory2 = note_storage[id2]
            
            text1 = memory1.get('content', memory1) if isinstance(memory1, dict) else memory1
            text2 = memory2.get('content', memory2) if isinstance(memory2, dict) else memory2
            
            cat1, score1 = categorizer.categorize_memory(text1)
            cat2, score2 = categorizer.categorize_memory(text2)
            
            category_penalty = 0.2 if cat1 != cat2 and score1 > 0.4 and score2 > 0.4 else 0
            category_bonus = 0.3 if cat1 == cat2 and score1 > 0.3 and score2 > 0.3 else 0
            
            embedding1 = model.encode([text1])
            embedding2 = model.encode([text2])
            
            similarity = np.dot(embedding1[0], embedding2[0]) / (
                np.linalg.norm(embedding1[0]) * np.linalg.norm(embedding2[0])
            )
            
            words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
            words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))
            common_words = words1.intersection(words2)
            stop_words = {'muito', 'mais', 'para', 'como', 'está', 'são', 'tem', 'foi', 'ser', 'ter'}
            significant_common = common_words - stop_words
            word_similarity = len(significant_common) / max(len(words1.union(words2)), 1)
            
            tfidf_similarity = calculate_text_similarity(text1, text2)
            
            combined_similarity = (
                similarity * 0.4 + 
                tfidf_similarity * 0.25 + 
                word_similarity * 0.25 + 
                category_bonus - 
                category_penalty
            )
            
            if combined_similarity > threshold:
                connections.append({
                    'id1': id1,
                    'id2': id2,
                    'similarity': float(combined_similarity),
                    'content1': text1,
                    'content2': text2,
                    'category1': cat1,
                    'category2': cat2
                })
    
    print(f"\n🎯 RESULTADO: {len(connections)} conexões encontradas.")
    return connections

def generate_ai_response(query, relevant_memories):
    """Gera resposta melhorada da IA com debug."""
    if not AI_AVAILABLE:
        return "IA não está disponível no momento."
    
    if not relevant_memories:
        return f"Não encontrei memórias específicas sobre '{query}'. Pode tentar reformular a pergunta?"
    
    try:
        print(f"🤖 Gerando resposta para: '{query}'")
        print(f"📚 Memórias relevantes: {len(relevant_memories)}")
        
        context_parts = []
        for i, memory in enumerate(relevant_memories[:3]):
            content = memory.get('content', memory) if isinstance(memory, dict) else memory
            context_parts.append(f"Memória {i+1}: {content}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Contexto das minhas memórias:
{context}

Pergunta: {query}

Resposta direta e útil (baseada apenas nas memórias acima):"""
        
        print(f"🔍 Prompt gerado: {prompt[:100]}...")
        
        inputs = ai_tokenizer.encode(prompt, return_tensors='pt', max_length=400, truncation=True)
        
        print(f"🔤 Tokens de entrada: {inputs.shape[1]}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            outputs = ai_model.generate(
                inputs, 
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                top_p=0.85,
                top_k=40,
                pad_token_id=ai_tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        response = ai_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        
        print(f"🎯 Resposta bruta: '{response}'")
        
        response = response.replace(prompt, "").strip()
        response = re.sub(r'Resposta[^:]*:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n+', ' ', response)
        response = response.strip()
        
        print(f"✨ Resposta limpa: '{response}'")
        
        if len(response) < 10:
            print("⚠️ Resposta muito curta, usando fallback")
            response = ""
        
        if not response or len(set(response.split())) < 3:
            print("⚠️ Resposta inadequada, gerando fallback inteligente")
            
            memory_keywords = []
            for memory in relevant_memories[:2]:
                content = memory.get('content', memory) if isinstance(memory, dict) else memory
                words = re.findall(r'\b\w{4,}\b', content.lower())
                memory_keywords.extend(words[:3])
            
            if memory_keywords:
                unique_keywords = list(dict.fromkeys(memory_keywords))[:5]
                keywords_str = ", ".join(unique_keywords)
                response = f"Baseado nas suas memórias sobre {keywords_str}, posso elaborar mais detalhes se especificar o que precisa saber."
            else:
                response = f"Encontrei informações sobre '{query}' nas suas memórias. Que aspecto específico gostaria de explorar?"
        
        print(f"🎉 Resposta final: '{response}'")
        return response
        
    except Exception as e:
        print(f"💥 Erro detalhado na IA: {e}")
        print(f"🔍 Tipo do erro: {type(e)}")
        import traceback
        traceback.print_exc()
        
        return f"Encontrei {len(relevant_memories)} memória(s) relacionada(s) a '{query}'. A IA teve dificuldades técnicas, mas você pode ver as memórias nas fontes abaixo."

# --- 6. Rotas de Autenticação ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '').strip()
        
        if verify_password(password, user_config):
            session['authenticated'] = True
            session['login_time'] = datetime.now().isoformat()
            session.permanent = True
            print(f"✅ Login bem-sucedido às {datetime.now().strftime('%H:%M:%S')}")
            return redirect(url_for('home'))
        else:
            print(f"❌ Tentativa de login inválida às {datetime.now().strftime('%H:%M:%S')}")
            flash('Senha incorreta!')
            
    return render_template('login_template.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Você foi desconectado.")
    print(f"ℹ️ Logout realizado às {datetime.now().strftime('%H:%M:%S')}")
    return redirect(url_for('login'))

# --- 7. Rotas da API Melhoradas ---
@app.route('/')
@require_auth
def home():
    return render_template('twomind.html')

@app.route('/get-all-notes', methods=['GET'])
@require_auth
def get_all_notes():
    """Retorna todas as notas com metadados melhorados."""
    notes_list = []
    for note_id, note_data in note_storage.items():
        if isinstance(note_data, dict):
            content = note_data.get('content', '')
            category, score = categorizer.categorize_memory(content)
            notes_list.append({
                "id": note_id,
                "content": content,
                "category": category,
                "category_score": float(score),
                "category_color": categorizer.categories.get(category, {}).get('color', '#64748b'),
                "created_at": note_data.get('created_at', datetime.now().isoformat()),
                "updated_at": note_data.get('updated_at', datetime.now().isoformat())
            })
        else:
            category, score = categorizer.categorize_memory(note_data)
            notes_list.append({
                "id": note_id,
                "content": note_data,
                "category": category,
                "category_score": float(score),
                "category_color": categorizer.categories.get(category, {}).get('color', '#64748b'),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
    
    notes_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(notes_list)

@app.route('/get-categories', methods=['GET'])
@require_auth
def get_categories():
    """Retorna todas as categorias disponíveis."""
    return jsonify({
        "status": "success",
        "categories": categorizer.categories
    })

@app.route('/get-connections', methods=['GET'])
@require_auth
def get_connections():
    try:
        connections = find_memory_connections(note_storage, threshold=0.25)
        return jsonify({"status": "success", "connections": connections})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/add-note', methods=['POST'])
@require_auth
def add_note():
    global index, note_storage
    data = request.json
    note_text = data.get('content')
    if not note_text:
        return jsonify({"status": "error", "message": "Conteúdo vazio."}), 400

    note_text = ' '.join(note_text.strip().split())
    note_vector = model.encode([note_text])
    index.add(note_vector)
    current_id = index.ntotal - 1
    
    category, score = categorizer.categorize_memory(note_text)
    
    note_data = {
        'content': note_text,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'category': category,
        'category_score': float(score),
        'id': str(uuid.uuid4())
    }
    
    note_storage[current_id] = note_data
    save_data(index, note_storage)
    
    print(f"➕ Nota adicionada (ID: {current_id}, Categoria: {category}): {note_text[:50]}...")
    
    return jsonify({
        "status": "success", 
        "id": current_id, 
        "content": note_text,
        "category": category,
        "category_color": categorizer.categories.get(category, {}).get('color', '#64748b'),
        "category_score": float(score),
        "created_at": note_data['created_at']
    })

@app.route('/edit-note', methods=['POST'])
@require_auth
def edit_note():
    """Nova funcionalidade: editar memória existente."""
    global index, note_storage
    data = request.json
    note_id = data.get('id')
    new_content = data.get('content', '').strip()
    
    if not new_content or note_id is None:
        return jsonify({"status": "error", "message": "Dados inválidos."}), 400
    
    try:
        note_id = int(note_id)
        if note_id not in note_storage:
            return jsonify({"status": "error", "message": "Nota não encontrada."}), 404
        
        category, score = categorizer.categorize_memory(new_content)
        
        if isinstance(note_storage[note_id], dict):
            note_storage[note_id].update({
                'content': new_content,
                'updated_at': datetime.now().isoformat(),
                'category': category,
                'category_score': float(score)
            })
        else:
            note_storage[note_id] = {
                'content': new_content,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'category': category,
                'category_score': float(score),
                'id': str(uuid.uuid4())
            }
        
        save_data(index, note_storage)
        
        print(f"✏️ Nota editada (ID: {note_id}): {new_content[:50]}...")
        
        return jsonify({
            "status": "success",
            "id": note_id,
            "content": new_content,
            "category": category,
            "category_color": categorizer.categories.get(category, {}).get('color', '#64748b'),
            "updated_at": note_storage[note_id]['updated_at']
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Erro ao editar: {str(e)}"}), 500

@app.route('/delete-note', methods=['POST'])
@require_auth
def delete_note():
    """Nova funcionalidade: deletar memória."""
    global index, note_storage
    data = request.json
    note_id = data.get('id')
    
    if note_id is None:
        return jsonify({"status": "error", "message": "ID não fornecido."}), 400
    
    try:
        note_id = int(note_id)
        if note_id not in note_storage:
            return jsonify({"status": "error", "message": "Nota não encontrada."}), 404
        
        del note_storage[note_id]
        save_data(index, note_storage)
        
        print(f"🗑️ Nota deletada (ID: {note_id})")
        
        return jsonify({"status": "success", "message": "Nota deletada com sucesso."})
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Erro ao deletar: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
@require_auth
def ask():
    data = request.json
    query_text = data.get('query')
    category_filter = data.get('category')
    sort_by = data.get('sort', 'relevance')
    
    top_k = min(20, index.ntotal)

    if not query_text or index.ntotal == 0:
        return jsonify({"results": [], "scores": []})

    print(f"\n🔍 BUSCA INICIADA: '{query_text}' (Filtro: {category_filter}, Ordem: {sort_by})")
    
    query_words = set(re.findall(r'\b\w{3,}\b', query_text.lower()))
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    candidates = []
    for i, distance in zip(indices[0], distances[0]):
        if i != -1 and i in note_storage:
            note_data = note_storage[i]
            content = note_data.get('content', note_data) if isinstance(note_data, dict) else note_data
            
            if category_filter and category_filter != 'all':
                note_category = note_data.get('category', 'geral') if isinstance(note_data, dict) else categorizer.categorize_memory(content)[0]
                if note_category != category_filter:
                    continue
            
            content_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
            exact_matches = query_words.intersection(content_words)
            context_overlap = len(exact_matches) / len(query_words) if query_words else 0
            embedding_score = max(0, 1 - (distance / 2.0))
            combined_score = (embedding_score * 0.6) + (context_overlap * 0.4)
            
            candidate_data = {
                'content': content,
                'combined_score': combined_score,
                'context_overlap': context_overlap,
                'embedding_score': embedding_score,
                'id': i
            }
            
            if isinstance(note_data, dict):
                candidate_data['created_at'] = note_data.get('created_at', '')
                candidate_data['category'] = note_data.get('category', 'geral')
            
            candidates.append(candidate_data)

    filtered_results = []
    for candidate in candidates:
        accept = False
        if candidate['embedding_score'] > 0.7 and candidate['context_overlap'] > 0.2:
            accept = True
        elif candidate['context_overlap'] > 0.6:
            accept = True
        elif candidate['combined_score'] > 0.6:
            accept = True
        
        if accept:
            filtered_results.append(candidate)

    if sort_by == 'date':
        filtered_results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_by == 'alphabetical':
        filtered_results.sort(key=lambda x: x['content'].lower())
    else:
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    final_results = filtered_results[:5]
    
    results = [r['content'] for r in final_results]
    scores = [r['combined_score'] for r in final_results]
    categories = [r.get('category', 'geral') for r in final_results]
    
    print(f"📊 RESULTADO: {len(results)} memórias relevantes encontradas.")
    return jsonify({
        "results": results, 
        "scores": scores, 
        "categories": categories,
        "query": query_text,
        "total_matches": len(filtered_results)
    })

@app.route('/ai-status', methods=['GET'])
@require_auth
def ai_status():
    return jsonify({
        "ai_available": AI_AVAILABLE,
        "model": AI_MODEL_NAME if AI_AVAILABLE else None,
    })

@app.route('/ask-ai', methods=['POST'])
@require_auth
def ask_ai():
    data = request.json
    query_text = data.get('query')
    if not query_text: 
        return jsonify({"status": "error", "message": "Query vazia"}), 400
    if not AI_AVAILABLE: 
        return jsonify({"status": "error", "message": "IA indisponível"}), 503
    
    top_k = min(10, index.ntotal)
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    candidates = []
    for i, d in zip(indices[0], distances[0]):
        if i in note_storage and (1 - d/2) > 0.3:
            note_data = note_storage[i]
            content = note_data.get('content', note_data) if isinstance(note_data, dict) else note_data
            score = float(1 - d/2)
            
            candidate = {'content': content, 'score': score}
            if isinstance(note_data, dict):
                candidate['category'] = note_data.get('category', 'geral')
            
            candidates.append(candidate)
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    relevant_memories = candidates[:3]
    
    if relevant_memories:
        ai_response = generate_ai_response(query_text, relevant_memories)
    else:
        ai_response = "Não encontrei memórias relevantes para responder à sua pergunta."
        
    return jsonify({
        "status": "success", 
        "query": query_text, 
        "ai_response": ai_response,
        "sources": candidates[:5]
    })

@app.route('/stats', methods=['GET'])
@require_auth
def get_stats():
    """Nova rota: estatísticas do sistema."""
    try:
        total_memories = len(note_storage)
        categories_count = {}
        
        for note_data in note_storage.values():
            if isinstance(note_data, dict):
                category = note_data.get('category', 'geral')
            else:
                category, _ = categorizer.categorize_memory(note_data)
            
            categories_count[category] = categories_count.get(category, 0) + 1
        
        connections = find_memory_connections(note_storage, threshold=0.25)
        avg_connections = len(connections) / max(total_memories, 1)
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_memories": total_memories,
                "total_categories": len(categorizer.categories),
                "categories_distribution": categories_count,
                "total_connections": len(connections),
                "avg_connections_per_memory": round(avg_connections, 2),
                "ai_status": "Ativo" if AI_AVAILABLE else "Inativo"
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- 8. Execução do Servidor ---
if __name__ == '__main__':
    # Criar template de login melhorado
    with open("login_template.html", "w", encoding="utf-8") as f:
        f.write('''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TwoMind - Login</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .gradient-bg {
            background: radial-gradient(ellipse at center, #0f172a 0%, #020617 100%);
        }
    </style>
</head>
<body class="gradient-bg min-h-screen flex items-center justify-center text-white">
    <div class="bg-slate-800/50 backdrop-blur-xl p-8 rounded-3xl shadow-2xl w-full max-w-md border border-slate-700/50">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold text-cyan-400 mb-2">🧠 TwoMind</h1>
            <p class="text-slate-400">Acesso seguro às suas memórias</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-xl">
                    {% for message in messages %}
                        <p class="text-red-300 text-sm text-center font-semibold">{{ message }}</p>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <form method="post" action="/login" class="space-y-6">
            <div>
                <label for="password" class="block text-sm font-medium text-slate-300 mb-2">
                    Senha de acesso:
                </label>
                <input type="password" id="password" name="password" required
                       class="w-full px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl text-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                       placeholder="Digite sua senha">
            </div>
            
            <button type="submit" 
                    class="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-bold py-3 px-6 rounded-xl transition-all duration-300 hover:scale-105 shadow-lg">
                Entrar
            </button>
        </form>
        
        <div class="mt-6 text-center">
            <p class="text-xs text-slate-500">
                🔒 Conexão protegida via Cloudflare.<br>
                Sessão válida por 24 horas.
            </p>
        </div>
    </div>
</body>
</html>''')

    print(f"\n{'='*60}")
    print("🧠 TwoMind - SERVIDOR MELHORADO")
    print("="*60)
    print(f"📚 Memórias carregadas: {len(note_storage)}")
    print(f"🎯 Categorias disponíveis: {len(categorizer.categories)}")
    print(f"🤖 IA disponível: {'Sim' if AI_AVAILABLE else 'Não'}")
    print(f"🔐 Autenticação: Habilitada")
    print(f"🌐 Túnel Cloudflare: Iniciando...")
    
    # Inicia o túnel Cloudflare em background
    tunnel_thread = start_cloudflare_tunnel()
    
    host = '0.0.0.0'  # MUDANÇA IMPORTANTE: aceita conexões externas
    port = 5000
    
    print(f"🏠 Acesso local: http://localhost:{port}")
    print(f"⏳ Aguardando túnel Cloudflare...")
    print("="*60)
    
    try:
        # Adicionado processes=1 para evitar conflitos
        app.run(host=host, port=port, debug=False, threaded=True, processes=1)
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        print("🔧 Limpando recursos...")