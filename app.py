import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
warnings.filterwarnings("ignore")

# --- 1. Configurações de Segurança ---

# Arquivo de configuração do usuário
CONFIG_FILE = "user_config.json"
CLOUDFLARED_DIR = "cloudflared"

# Define os caminhos para os arquivos de salvamento
FAISS_INDEX_PATH = "my_index.faiss"
NOTES_STORAGE_PATH = "notes.pkl"

# Inicializa o servidor web Flask
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app, supports_credentials=True)

# Configurações de segurança
app.config['SECRET_KEY'] = secrets.token_hex(32)  # Chave secreta aleatória
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # Sessão dura 24h
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Cookie não acessível via JavaScript
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'  # Proteção CSRF
app.config['SESSION_COOKIE_SECURE'] = False  # HTTP para simplicidade

def load_user_config():
    """Carrega configurações do usuário ou cria configuração inicial."""
    import json
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Primeira execução - cria configuração
        print("\n" + "="*50)
        print("🔒 CONFIGURAÇÃO INICIAL DE SEGURANÇA")
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
        
        # Hash da senha
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
    
    # Determina a URL de download baseada no OS e arquitetura
    if system == "windows":
        if "64" in machine or "amd64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            filename = "cloudflared.exe"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"
            filename = "cloudflared.exe"
    elif system == "darwin":  # macOS
        if "arm" in machine or "m1" in machine or "m2" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
        filename = "cloudflared"
    else:  # Linux
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
        
        # Torna executável no Linux/macOS
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
            # Comando para criar um túnel temporário
            cmd = [cloudflared_path, "tunnel", "--url", "http://localhost:5000"]
            
            # Inicia o processo
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Redireciona stderr para stdout
                text=True,
                bufsize=0,  # Sem buffer
                universal_newlines=True
            )
            
            print("⏳ Aguardando URL do túnel...")
            tunnel_found = False
            
            # Monitora a saída linha por linha
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    
                    # Debug: mostra as linhas para identificar o problema
                    if "tunnel" in line.lower() or "cloudflare" in line.lower():
                        print(f"🐛 Debug: {line}")
                    
                    # Procura por URLs do túnel com regex mais abrangente
                    import re
                    urls = re.findall(r'https://[a-zA-Z0-9\-]+\.(?:trycloudflare\.com|cfargotunnel\.com|argotunnel\.com)', line)
                    
                    if urls and not tunnel_found:
                        tunnel_url = urls[0]
                        tunnel_found = True
                        print(f"\n🌍 TÚNEL CLOUDFLARE ATIVO!")
                        print(f"🔗 URL Externa: {tunnel_url}")
                        print(f"📱 Acesse de qualquer lugar: {tunnel_url}")
                        print(f"🔒 Protegido com senha")
                        print("="*60)
                        
                        # Salva a URL em um arquivo para referência
                        with open("tunnel_url.txt", "w") as f:
                            f.write(tunnel_url)
                        
                        break
                    
                    # Verifica se há erro
                    if "error" in line.lower() or "failed" in line.lower():
                        print(f"⚠️  Possível erro: {line}")
                        
                except Exception as e:
                    print(f"⚠️  Erro ao ler linha: {e}")
                    break
            
            if not tunnel_found:
                print("⚠️  URL do túnel não encontrada. Tentando comando alternativo...")
                # Tenta comando mais simples
                process.terminate()
                time.sleep(1)
                
                # Comando alternativo
                cmd_alt = [cloudflared_path, "tunnel", "--hello-world"]
                process_alt = subprocess.Popen(cmd_alt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                for line in process_alt.stdout:
                    line = line.strip()
                    print(f"🐛 Alt Debug: {line}")
                    if "https://" in line:
                        urls = re.findall(r'https://[a-zA-Z0-9\-\.]+', line)
                        if urls:
                            print(f"🌍 Túnel alternativo encontrado: {urls[0]}")
                            break
                    
        except Exception as e:
            print(f"❌ Erro no túnel Cloudflare: {e}")
            print("💡 Tentativa manual: Execute no terminal separado:")
            print(f"   {cloudflared_path} tunnel --url http://localhost:5000")
    
    # Inicia o túnel em uma thread separada
    tunnel_thread = threading.Thread(target=run_tunnel, daemon=True)
    tunnel_thread.start()
    
    # Aguarda mais tempo para o túnel inicializar
    time.sleep(5)
    
    return tunnel_thread

def require_auth(f):
    """Decorator que exige autenticação para acessar rotas protegidas."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_api_request = any(request.path.startswith(p) for p in ['/get-', '/add-', '/ask', '/ai-', '/stats'])

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

# Carrega configuração do usuário
user_config = load_user_config()

# --- Configuração da IA e Modelo de Embeddings ---
AI_MODEL_NAME = "microsoft/DialoGPT-small"
ai_tokenizer = None
ai_model = None
AI_AVAILABLE = False

def setup_local_ai():
    """Configura a IA local usando transformers."""
    global ai_tokenizer, ai_model, AI_AVAILABLE
    try:
        print("🤖 Carregando modelo de IA local...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME, padding_side='left')
        ai_model = AutoModelForCausalLM.from_pretrained(AI_MODEL_NAME)
        if ai_tokenizer.pad_token is None:
            ai_tokenizer.pad_token = ai_tokenizer.eos_token
        AI_AVAILABLE = True
        print(f"✅ IA local carregada: {AI_MODEL_NAME}")
    except (ImportError, OSError) as e:
        print(f"❌ Modelo de IA não pôde ser carregado: {e}. IA indisponível.")
        AI_AVAILABLE = False

print("📚 Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modelo de embeddings carregado.")

setup_local_ai()

d = 384 # Dimensão dos vetores do modelo

# --- Funções de Persistência e Processamento de Dados ---

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
    """Encontra conexões entre memórias baseadas em similaridade semântica E contextual"""
    connections = []
    note_ids = list(note_storage.keys())
    
    print(f"\n🔗 ANALISANDO CONEXÕES entre {len(note_ids)} memórias...")
    
    for i, id1 in enumerate(note_ids):
        for id2 in note_ids[i+1:]:
            text1 = note_storage[id1]
            text2 = note_storage[id2]
            
            # Análise de domínio/categoria
            def get_content_category(text):
                text_lower = text.lower()
                categories = {
                    'senhas': ['senha', 'password', 'login', 'acesso', 'email', 'banco'],
                    'pets': ['gato', 'gata', 'cachorro', 'pet', 'animal', 'naomi'],
                    'trabalho': ['trabalho', 'empresa', 'projeto', 'reunião'],
                    'pessoal': ['pessoal', 'família', 'casa', 'amigo'],
                    'finanças': ['dinheiro', 'banco', 'conta', 'cartão', 'pagar']
                }
                
                category_scores = {}
                
                for category, keywords in categories.items():
                    matches = len([w for w in keywords if w in text_lower])
                    # Evita divisão por zero se a lista de keywords for vazia
                    if len(keywords) > 0:
                        category_scores[category] = matches / len(keywords)
                    else:
                        category_scores[category] = 0
                
                # Retorna a categoria com maior score, ou 'geral' se todos forem 0
                if all(score == 0 for score in category_scores.values()):
                    return 'geral', 0
                
                return max(category_scores.items(), key=lambda x: x[1])
            
            category1, score1 = get_content_category(text1)
            category2, score2 = get_content_category(text2)
            
            # Se são categorias diferentes com scores altos, não conecta
            if category1 != category2 and score1 > 0.3 and score2 > 0.3:
                continue
            
            # Calcula similaridade usando embeddings do modelo
            embedding1 = model.encode([text1])
            embedding2 = model.encode([text2])
            
            # Similaridade do cosseno entre embeddings
            similarity = np.dot(embedding1[0], embedding2[0]) / (
                np.linalg.norm(embedding1[0]) * np.linalg.norm(embedding2[0])
            )
            
            # Análise de palavras-chave em comum
            words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
            words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))
            common_words = words1.intersection(words2)
            stop_words = {'muito', 'mais', 'para', 'como', 'está', 'são', 'tem', 'foi', 'ser', 'ter'}
            significant_common = common_words - stop_words
            word_similarity = len(significant_common) / max(len(words1.union(words2)), 1)
            
            # TF-IDF similarity
            tfidf_similarity = calculate_text_similarity(text1, text2)
            
            # Bônus se mesma categoria
            category_bonus = 0.2 if category1 == category2 and score1 > 0.2 and score2 > 0.2 else 0
            
            # Combina métricas
            combined_similarity = (
                similarity * 0.4 + 
                tfidf_similarity * 0.3 + 
                word_similarity * 0.3 + 
                category_bonus
            )
            
            if combined_similarity > threshold:
                connections.append({
                    'id1': id1,
                    'id2': id2,
                    'similarity': float(combined_similarity),
                    'content1': text1,
                    'content2': text2
                })
    
    print(f"\n🎯 RESULTADO: {len(connections)} conexões encontradas.")
    return connections

def generate_ai_response(query, relevant_memories):
    if not AI_AVAILABLE or not relevant_memories:
        return None
    try:
        context = " | ".join(relevant_memories[:3])
        prompt = f"Contexto: {context}\nPergunta: {query}\nResposta útil:"
        inputs = ai_tokenizer.encode(prompt, return_tensors='pt', max_length=300, truncation=True)
        outputs = ai_model.generate(
            inputs, max_new_tokens=100, temperature=0.7, do_sample=True,
            pad_token_id=ai_tokenizer.eos_token_id, no_repeat_ngram_size=3
        )
        response = ai_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        return response if len(response) > 10 else f"Baseado nas suas memórias, encontrei informações sobre: {', '.join([mem[:50] + '...' for mem in relevant_memories[:2]])}."
    except Exception as e:
        print(f"💥 Erro ao gerar resposta da IA: {e}")
        return f"Encontrei {len(relevant_memories)} memória(s) relacionada(s). Veja as fontes abaixo."

# --- 2. Rotas de Autenticação ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Correção: usar request.form para formulários HTML padrão
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
            
    # Renderiza template de login
    return render_template('login_template.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Você foi desconectado.")
    print(f"ℹ️ Logout realizado às {datetime.now().strftime('%H:%M:%S')}")
    return redirect(url_for('login'))

# --- 3. Rotas Protegidas da API ---

@app.route('/')
@require_auth
def home():
    return render_template('twomind.html')

@app.route('/get-all-notes', methods=['GET'])
@require_auth
def get_all_notes():
    notes_list = [{"id": note_id, "content": content} for note_id, content in note_storage.items()]
    return jsonify(notes_list)

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
    note_storage[current_id] = note_text
    save_data(index, note_storage)
    
    print(f"➕ Nota adicionada (ID: {current_id}): {note_text[:50]}...")
    return jsonify({"status": "success", "id": current_id, "content": note_text})

@app.route('/ask', methods=['POST'])
@require_auth
def ask():
    data = request.json
    query_text = data.get('query')
    top_k = min(20, index.ntotal)

    if not query_text or index.ntotal == 0:
        return jsonify({"results": [], "scores": []})

    print(f"\n🔍 BUSCA INICIADA: '{query_text}'")
    query_words = set(re.findall(r'\b\w{3,}\b', query_text.lower()))
    
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    candidates = []
    for i, distance in zip(indices[0], distances[0]):
        if i != -1 and i in note_storage:
            content = note_storage[i]
            content_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
            
            exact_matches = query_words.intersection(content_words)
            context_overlap = len(exact_matches) / len(query_words) if query_words else 0
            embedding_score = max(0, 1 - (distance / 2.0))
            combined_score = (embedding_score * 0.6) + (context_overlap * 0.4)
            
            candidates.append({
                'content': content,
                'combined_score': combined_score,
                'context_overlap': context_overlap,
                'embedding_score': embedding_score
            })

    # Filtros para resultados mais relevantes
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

    filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
    final_results = filtered_results[:5]
    
    results = [r['content'] for r in final_results]
    scores = [r['combined_score'] for r in final_results]
    
    print(f"📊 RESULTADO: {len(results)} memórias relevantes encontradas.")
    return jsonify({"results": results, "scores": scores})

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
    if not query_text: return jsonify({"status": "error", "message": "Query vazia"}), 400
    if not AI_AVAILABLE: return jsonify({"status": "error", "message": "IA indisponível"}), 503
    
    top_k = min(10, index.ntotal)
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    candidates = [{'content': note_storage[i], 'score': float(1 - d/2)} 
                  for i, d in zip(indices[0], distances[0]) 
                  if i in note_storage and (1 - d/2) > 0.3]
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    relevant_memories = [c['content'] for c in candidates[:3]]
    
    if relevant_memories:
        ai_response = generate_ai_response(query_text, relevant_memories)
    else:
        ai_response = "Não encontrei memórias relevantes para responder."
        
    return jsonify({
        "status": "success", "query": query_text, "ai_response": ai_response,
        "sources": candidates[:5]
    })

# --- 4. Execução do Servidor ---
if __name__ == '__main__':
    # Criar template de login
    with open("login_template.html", "w", encoding="utf-8") as f:
        f.write('''
    <!DOCTYPE html>
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
    </html>
    ''')

    print(f"\n{'='*60}")
    print("🧠 TwoMind - SERVIDOR GLOBAL")
    print("="*60)
    print(f"🔍 Memórias carregadas: {len(note_storage)}")
    print(f"🤖 IA disponível: {'Sim' if AI_AVAILABLE else 'Não'}")
    print(f"🔒 Autenticação: Habilitada")
    print(f"🌐 Túnel Cloudflare: Iniciando...")
    
    # Inicia o túnel Cloudflare em background
    tunnel_thread = start_cloudflare_tunnel()
    
    host = '127.0.0.1'  # Apenas localhost para segurança
    port = 5000
    
    print(f"🏠 Acesso local: http://{host}:{port}")
    print(f"⏳ Aguardando túnel Cloudflare...")
    print("="*60)
    
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        print("🔧 Limpando recursos...")
        # O processo do cloudflared será terminado automaticamente