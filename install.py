#!/usr/bin/env python3
"""
🧠 SEGUNDO CÉREBRO - Instalador Automático
Instala todas as dependências e prepara a aplicação para uso.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Executa um comando e exibe o progresso."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro em {description}:")
        print(f"   {e.stderr}")
        return False

def check_python_version():
    """Verifica se a versão do Python é compatível."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}")
        print("   Por favor, atualize o Python e tente novamente.")
        return False
    print(f"✅ Python {version.major}.{version.minor} - Compatível!")
    return True

def install_dependencies():
    """Instala as dependências do requirements.txt."""
    
    # Verifica se pip está disponível
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip não encontrado! Instale o pip primeiro.")
        return False
    
    # Atualiza pip
    print("\n📦 Atualizando pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                  capture_output=True)
    
    # Instala dependências
    if os.path.exists("requirements.txt"):
        command = f'"{sys.executable}" -m pip install -r requirements.txt'
        return run_command(command, "Instalando dependências Python")
    else:
        print("❌ Arquivo requirements.txt não encontrado!")
        return False

def create_desktop_shortcut():
    """Cria um atalho na área de trabalho."""
    system = platform.system()
    script_dir = os.getcwd()
    
    if system == "Windows":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "Segundo Cérebro.lnk")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{os.path.join(script_dir, "app.py")}"'
            shortcut.WorkingDirectory = script_dir
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("✅ Atalho criado na área de trabalho!")
            return True
        except ImportError:
            print("⚠️  Para criar atalho no Windows, instale: pip install winshell pywin32")
            return False
    
    elif system == "Darwin":  # macOS
        app_script = f'''#!/bin/bash
cd "{script_dir}"
"{sys.executable}" app.py
'''
        app_path = os.path.expanduser("~/Desktop/Segundo Cérebro.command")
        with open(app_path, 'w') as f:
            f.write(app_script)
        os.chmod(app_path, 0o755)
        print("✅ Atalho criado na área de trabalho!")
        return True
    
    elif system == "Linux":
        desktop_entry = f'''[Desktop Entry]
Version=1.0
Type=Application
Name=Segundo Cérebro
Comment=Sistema de Memória Inteligente
Exec="{sys.executable}" "{os.path.join(script_dir, "app.py")}"
Icon=applications-science
Path={script_dir}
Terminal=true
StartupNotify=false
'''
        desktop_path = os.path.expanduser("~/Desktop/Segundo Cérebro.desktop")
        with open(desktop_path, 'w') as f:
            f.write(desktop_entry)
        os.chmod(desktop_path, 0o755)
        print("✅ Atalho criado na área de trabalho!")
        return True
    
    return False

def main():
    print("🧠 SEGUNDO CÉREBRO - INSTALADOR")
    print("="*50)
    
    # Verifica Python
    if not check_python_version():
        input("\nPressione Enter para sair...")
        return
    
    # Instala dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação das dependências!")
        input("Pressione Enter para sair...")
        return
    
    # Baixa modelos (primeiro uso pode demorar)
    print("\n🤖 Preparando modelos de IA...")
    print("   (Isso pode demorar alguns minutos na primeira vez)")
    
    try:
        # Testa importação dos modelos principais
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Modelo de embeddings carregado!")
        
        # Tenta carregar modelo de IA (opcional)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
            print("✅ Modelo de IA carregado!")
        except:
            print("⚠️  Modelo de IA não pôde ser carregado (funcionalidade opcional)")
            
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        print("   A aplicação pode ainda funcionar com funcionalidade limitada.")
    
    # Cria atalho
    create_desktop_shortcut()
    
    print("\n" + "="*50)
    print("🎉 INSTALAÇÃO CONCLUÍDA!")
    print("="*50)
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Execute: python app.py")
    print("2. Configure sua senha na primeira execução")
    print("3. Aguarde o túnel Cloudflare ser criado")
    print("4. Acesse de qualquer lugar com a URL fornecida")
    print("\n🔒 SEGURANÇA:")
    print("• Sua aplicação ficará protegida por senha")
    print("• Acesso global via túnel Cloudflare criptografado")
    print("• Dados salvos localmente no seu computador")
    print("\n💡 DICA:")
    print("• Use o atalho criado na área de trabalho")
    print("• Mantenha o terminal aberto enquanto usar")
    
    input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Instalação cancelada pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        input("Pressione Enter para sair...")